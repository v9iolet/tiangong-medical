# src/agents/knowledge/ingestion/doc_ingestion.py
"""
知识文档入库：解析文档 → 分块 → 向量化 → 写入 Milvus knowledge_docs collection，供 doc_rag 检索。

两个入口：
- ingest_file：导入单个上传文档，优先 MinerU 解析，失败回退 LlamaIndex
- ingest_drug_instructions：从 PostgreSQL drug_details 表批量导入药品说明书

同一文档重复导入时先完成解析、向量化和变体切分，再 upsert 新分块、删除多余的旧分块，
解析、向量化或切分失败不会改动已有分块；Milvus upsert 与旧尾块清理之间可能短暂出现新旧混合。
上传文档除 original 外，可按 strategies 额外写入 parent_child / semantic 变体（以结构分块为父单元）。
MilvusClient 和文档解析都是同步调用，统一通过 asyncio.to_thread 放到线程中执行，避免阻塞事件循环。
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING
from loguru import logger
from langchain_core.embeddings import Embeddings
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
)

from src.agents.knowledge.retrieval.doc_rag import COLLECTION_NAME
from src.agents.knowledge.retrieval.variant_schema import (
    ORIGINAL_FILTER, VARIANT_FIELDS, original_chunk_metadata,
)
from src.agents.knowledge.ingestion import artifact_store
from src.agents.knowledge.ingestion.markdown_assets import (
    conservative_clean, ensure_no_base64, extract_images_and_clean_markdown,
)
from src.agents.knowledge.ingestion.structured_chunking import chunk_documents
from src.agents.knowledge.ingestion.variant_chunking import (
    PARENT_CHILD, SEMANTIC, VARIANT_STRATEGIES, ParentUnit, VariantChunk,
    build_parent_child_chunks, build_semantic_chunks, build_semantic_embed_model,
)
from src.core.config import get_settings

# 仅供类型检查和 IDE 推断使用，运行时仍在函数内延迟导入 llama_index
if TYPE_CHECKING:
    from llama_index.core.node_parser import SentenceSplitter

# 向量维度，必须与嵌入模型的输出维度一致（DashScope 文本向量默认 1024 维）
EMBEDDING_DIM = 1024
# 分块大小与重叠长度，按 SentenceSplitter 的 token 计
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
# 当前 Embedding 接口每次最多接受 20 条文本；在适配器内部拆批前先限制输入数量。
EMBED_BATCH_SIZE = 20
# 按 doc_id 列表批量删除时，每批的 id 数量，避免过滤表达式过长
DELETE_BATCH_SIZE = 200
DRUG_DOC_TYPE = "drug_instruction"
ProgressCallback = Callable[[str], Awaitable[None]]


async def _run_blocking(function, *args, **kwargs):
    """取消时等待同步写入结束，随后再释放文档锁，避免新旧任务交错。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _report_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        await progress(message)


def build_knowledge_schema():
    """knowledge_docs 的 schema；ensure_knowledge_collection 和重建脚本共用。"""
    # auto_id=False：主键由入库代码生成（{doc_id}_{chunk_index}），便于按文档覆盖和定位分块
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
    schema.add_field("category", DataType.VARCHAR, max_length=100)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    # 65535 是 Milvus VARCHAR 的长度上限；BM25 需要 text 开启分词，中文用 chinese analyzer
    schema.add_field(
        "text", DataType.VARCHAR, max_length=65535,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    # 稀疏向量由 BM25 Function 在服务端从 text 生成，写入时不用提供
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    # 分块元数据非空：original 行也要写 chunk_strategy / source_id，parent_* 用空字符串表示不适用
    for name, max_length in VARIANT_FIELDS.items():
        schema.add_field(name, DataType.VARCHAR, max_length=max_length)
    schema.add_function(Function(
        name="text_bm25", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["sparse_embedding"],
    ))
    return schema


def build_knowledge_index_params():
    """knowledge_docs 的索引；dense 的 metric_type 必须与 doc_rag 检索时的 search_params 一致。"""
    index_params = MilvusClient.prepare_index_params()
    # IVF_FLAT：把向量聚成 nlist 个桶，查询时只扫描 nprobe 个桶，适合中小规模数据
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    for name in ("chunk_strategy", "doc_id", "doc_type"):
        index_params.add_index(field_name=name, index_type="INVERTED")
    return index_params


def validate_knowledge_schema(milvus_client: MilvusClient, collection_name: str = COLLECTION_NAME) -> None:
    """校验已有集合是否符合新 schema；不符合就报错，不追加字段。"""
    info = milvus_client.describe_collection(collection_name)
    fields = {field["name"]: field for field in info["fields"]}
    problems = []
    for name, max_length in VARIANT_FIELDS.items():
        field = fields.get(name)
        if field is None:
            problems.append(f"缺少字段 {name}")
        elif field.get("nullable") or int(field["params"].get("max_length", 0)) < max_length:
            problems.append(f"字段 {name} 必须非空且长度不小于 {max_length}")
    sparse = fields.get("sparse_embedding")
    if sparse is None or int(sparse["type"]) != int(DataType.SPARSE_FLOAT_VECTOR):
        problems.append("缺少稀疏向量字段 sparse_embedding")
    # BM25 Function 要求输入字段开启分词，所以有这个 Function 即说明 text 已开 analyzer
    if not any(
        int(function["type"]) == int(FunctionType.BM25)
        and list(function["input_field_names"]) == ["text"]
        and list(function["output_field_names"]) == ["sparse_embedding"]
        for function in info.get("functions") or []
    ):
        problems.append("缺少 text → sparse_embedding 的 BM25 Function")
    if "index_version" in fields:
        problems.append("仍保留已废弃的 index_version 字段")
    if problems:
        raise ValueError(
            f"集合 '{collection_name}' 的 schema 与新版本不一致（{'；'.join(problems)}）。"
            "请先运行 python -m scripts.rebuild_knowledge_collection --execute 重建集合"
        )


def ensure_knowledge_collection(milvus_client: MilvusClient) -> None:
    """
    确保 knowledge_docs collection 存在。

    集合不存在时按新 schema 创建；已存在时只校验，不追加字段、不回填或重写已有记录。
    """
    if milvus_client.has_collection(COLLECTION_NAME):
        validate_knowledge_schema(milvus_client)
        return

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=build_knowledge_schema(),
        index_params=build_knowledge_index_params(),
    )
    logger.info(f"Collection '{COLLECTION_NAME}' 创建成功")


async def _parse_with_mineru(
    file_path: str, file_name: str, *, strict: bool = False,
    progress: ProgressCallback | None = None,
) -> str | None:
    """
    尝试用 MinerU 解析文档，失败则返回 None。

    任何异常（服务不可用、超时、配置缺失）都被吞掉并记录告警，由调用方回退到 LlamaIndex。
    strict=True 时保留异常，供需要确认 MinerU 解析链路的离线入库场景使用。
    """
    try:
        # 延迟导入：MinerU 是可选依赖服务，模块加载时不强制读取其配置
        from src.agents.knowledge.ingestion.mineru_client import parse_document
        md_text = await parse_document(file_path, file_name)
        # 不足 10 个字符视为解析无效（如扫描件未识别出文字），同样走回退
        if md_text and len(md_text.strip()) > 10:
            logger.info(f"MinerU 解析成功: {file_name} ({len(md_text)} chars)")
            return md_text
        if strict:
            raise ValueError(f"MinerU 未解析出有效 Markdown: {file_name}")
    except Exception as e:
        if strict:
            raise
        logger.warning(f"MinerU 解析失败，回退到 LlamaIndex: {e}")
        await _report_progress(progress, "MinerU 解析失败，改用 LlamaIndex 解析；异常详情见后端日志")
    return None


async def _parse_with_llamaindex(file_path: str) -> list:
    """
    用 LlamaIndex 解析文档（兜底方案）。

    返回 LlamaIndex Document 列表；PDF 按页拆成多个 Document，页码在 metadata["page_label"] 中。
    load_data 是同步调用，放到线程中执行。
    """
    # 延迟导入：llama_index 体积较大，只在回退时才加载
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_files=[file_path])
    return await _run_blocking(reader.load_data)


def _make_splitter() -> SentenceSplitter:
    """
    三条入库路径（MinerU Markdown / LlamaIndex 解析结果 / 药品说明书）共用的分块器。

    按双换行识别段落，优先在段落、句子边界（含中文。？！）处切分；相邻块之间重叠 CHUNK_OVERLAP 个 token。
    """
    from llama_index.core.node_parser import SentenceSplitter
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )


async def _replace_doc_chunks(
    milvus_client: MilvusClient,
    doc_id: str,
    data: list[dict],
) -> None:
    """
    用新分块覆盖某个文档在 Milvus 中的数据。

    1. upsert：分块主键为 {doc_id}_{chunk_index}，同序号的旧分块被直接覆盖，文档全程可检索
    2. 删除 chunk_index >= 新分块数的旧分块（新版本分块变少时的残留）
    两步之间检索可能短暂看到新旧分块混合，可以接受。
    """
    # 上传文档与药品说明书共用此入口，后续覆盖入库也不能把元数据写回 NULL。
    data = [{**row, **original_chunk_metadata(str(row['id']))} for row in data]
    await _run_blocking(
        milvus_client.upsert, collection_name=COLLECTION_NAME, data=data,
    )
    await _run_blocking(
        milvus_client.delete,
        collection_name=COLLECTION_NAME,
        filter=f'({ORIGINAL_FILTER}) and doc_id == "{doc_id}" and chunk_index >= {len(data)}',
    )


def _normalize_strategies(strategies: Iterable[str]) -> tuple[str, ...]:
    """校验并去重（保持传入顺序）；original 总是写入，不允许出现在这里。"""
    if isinstance(strategies, str):
        raise ValueError(f"strategies 必须是策略名的序列，而不是单个字符串：{strategies!r}")
    unique = tuple(dict.fromkeys(strategies))
    invalid = [name for name in unique if name not in VARIANT_STRATEGIES]
    if invalid:
        raise ValueError(f"strategies 只允许 {VARIANT_STRATEGIES}（original 总是写入）：{invalid}")
    return unique


async def _build_variant_chunks(
    strategy: str,
    parents: list[ParentUnit],
    embedding_model: Embeddings,
) -> list[VariantChunk]:
    """在结构块内切出某种变体的子块；只切分，不写 Milvus。"""
    if strategy == PARENT_CHILD:
        return await _run_blocking(build_parent_child_chunks, parents)
    # 语义分块要对每个父单元做句子级 embedding，用 LlamaIndex 包装同一个向量模型
    return await build_semantic_chunks(parents, build_semantic_embed_model(embedding_model))


async def _replace_doc_variants(
    milvus_client: MilvusClient,
    embedding_model: Embeddings,
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
    variants: dict[str, list[VariantChunk]],
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """
    先删该文档的全部变体行，再写本次勾选的策略，返回各策略写入的行数。

    先删全部再写，重新上传时少勾的策略也不会留下旧行。
    """
    # 延迟导入：variant_writer 在模块顶层依赖本模块，直接导入会循环
    from src.agents.knowledge.ingestion.variant_writer import write_variant_chunks

    await _report_progress(progress, "清理该文档旧的父子 / 语义分块")
    await _run_blocking(
        milvus_client.delete,
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == {json.dumps(doc_id)} and chunk_strategy != {json.dumps("original")}',
    )
    written = {}
    for strategy, chunks in variants.items():
        written[strategy] = await write_variant_chunks(
            milvus_client, embedding_model, chunks, strategy=strategy,
            doc_id=doc_id, doc_name=doc_name, doc_type=doc_type, category=category, progress=progress,
        )
    return written


async def _embed_chunks(
    texts: list[str],
    embedding_model: Embeddings,
    progress: ProgressCallback | None = None,
) -> list[list[float]]:
    """分批向量化，返回与 texts 一一对应的向量；任一批失败直接抛出，由调用方放弃本次写入。"""
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        await _report_progress(progress, f"向量化第 {i // EMBED_BATCH_SIZE + 1} 批：{len(batch)} 个分块（共 {len(texts)} 个）")
        embeddings.extend(await embedding_model.aembed_documents(batch))
        await _report_progress(progress, f"向量化完成 {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)} 个分块")
    return embeddings


@asynccontextmanager
async def _document_lock(doc_id: str):
    # 同名文档的磁盘产物和 Milvus 覆盖过程共用锁，API/Gradio 不会交错写入。
    directory = Path(tempfile.gettempdir()) / "tiangong-document-locks"
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(directory / f"{doc_id}.lock", thread_local=False)
    while True:
        try:
            lock.acquire(timeout=0)
            break
        except Timeout:
            await asyncio.sleep(0.1)
    try:
        yield
    finally:
        lock.release()


async def _prepare_documents(
    file_path: str, doc_name: str, doc_id: str, mineru_only: bool,
    save_artifacts: bool,
    progress: ProgressCallback | None,
):
    """复用现有解析接口；产物以同一字节集合保存本地和 MinIO，保留页码。"""
    from llama_index.core import Document

    paths = artifact_store.artifact_paths(doc_id)
    fingerprint = await _run_blocking(artifact_store.source_hash, file_path)
    await _report_progress(progress, "提交 MinerU 解析任务，等待本地服务返回 Markdown")
    md_text = await _parse_with_mineru(file_path, doc_name, strict=mineru_only, progress=progress)
    if md_text:
        parser, documents = "mineru", [Document(text=md_text)]
        await _report_progress(progress, f"MinerU 解析完成：{len(md_text)} 字符")
    else:
        await _report_progress(progress, "使用 LlamaIndex 解析文档")
        parser, documents = "llamaindex", await _parse_with_llamaindex(file_path)
    parsed = [{"text": document.text, "metadata": document.metadata} for document in documents]
    raw = "\n\n".join(document.text for document in documents)
    files: dict[str, bytes] = {}
    if save_artifacts:
        files = {
            "raw.md": raw.encode("utf-8"),
            "documents.json": json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
            f"source{Path(file_path).suffix.lower()}": await _run_blocking(Path(file_path).read_bytes),
            "manifest.json": b'{"complete": false}',
        }
        await _run_blocking(artifact_store.persist_artifacts, paths, files, reset=True)
        await _report_progress(progress, "原始解析正文已保存，开始提取图片")
    cleaned_documents, display_parts, assets = [], [], {}
    for document in documents:
        try:
            result = await _run_blocking(extract_images_and_clean_markdown, document.text, paths)
            cleaned, display = result.retrieval_markdown, result.document_markdown
            for asset in result.assets:
                existing = assets.setdefault(asset["asset_id"], asset)
                existing["captions"] = list(dict.fromkeys([*existing["captions"], *asset["captions"]]))
            if save_artifacts:
                files.update(result.image_files)
        except Exception:
            logger.opt(exception=True).warning("图片结构处理失败，尝试保守清理")
            await _report_progress(progress, "图片处理降级：保留图注，移除图片编码")
            cleaned = await _run_blocking(conservative_clean, document.text)
            display = cleaned
        ensure_no_base64(cleaned)
        cleaned_documents.append(Document(text=cleaned, metadata=document.metadata))
        display_parts.append(display)
    retrieval = "\n\n".join(document.text for document in cleaned_documents)
    await _report_progress(progress, f"图片清理完成：{len(assets)} 张图片；正文 {len(raw)} → {len(retrieval)} 字符")
    if save_artifacts:
        settings = get_settings()
        manifest = {
            "complete": True, "doc_id": doc_id, "doc_name": doc_name,
            "source_hash": fingerprint, "parser": parser,
            "documents_hash": hashlib.sha256(json.dumps(parsed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "mineru_tier": settings.MINERU_TIER, "mineru_url": settings.MINERU_API_URL,
            "processing_version": artifact_store.PROCESSING_VERSION,
            "image_count": len(assets), "raw_chars": len(raw), "retrieval_chars": len(retrieval),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        files.update({
            "document.md": "\n\n".join(display_parts).encode("utf-8"),
            "retrieval.md": retrieval.encode("utf-8"),
            "assets.json": json.dumps(list(assets.values()), ensure_ascii=False, indent=2).encode("utf-8"),
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        })
        status = await _run_blocking(artifact_store.persist_artifacts, paths, files)
        await _report_progress(progress, "解析产物已同步 MinIO" if status["minio"] else "MinIO 不可用：本地产物已保存，继续入库")
        if not status["local"]:
            await _report_progress(progress, "本地目录不可写：产物仅保存在 MinIO，本地预览暂不可用")
    return cleaned_documents, paths if save_artifacts else None, files


async def ingest_file(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    mineru_only: bool = False,
    progress: ProgressCallback | None = None,
    *,
    save_artifacts: bool = True,
    strategies: Iterable[str] = (PARENT_CHILD,),
) -> int:
    """复用解析 → 保存产物/清理图片 → 结构分块 → 原有向量化和覆盖写入 → 变体分块写入。

    同名仍使用已有 doc_id，每次上传重新解析；产物仅用于预览。结构处理失败时对清理
    后的正文使用 SentenceSplitter；MinIO 不可用时保留本地产物继续入库。

    original 总是写入。strategies 选择额外写入的变体（parent_child / semantic，可为空），
    父单元就是结构分块的结果。写完 original 后先删该文档的全部变体行再写本次勾选的策略，
    重新上传时少勾的策略不会留下旧行。返回 original 分块数。
    """
    # 先校验：非法策略不能解析文档，更不能写任何数据
    strategies = _normalize_strategies(strategies)
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]
    async with _document_lock(doc_id):
        documents, paths, files = await _prepare_documents(
            file_path, doc_name, doc_id, mineru_only, save_artifacts,
            progress,
        )
        await _report_progress(progress, "开始 Markdown 结构分块")
        splitter = _make_splitter()
        try:
            nodes = await _run_blocking(chunk_documents, documents, splitter)
        except Exception:
            logger.opt(exception=True).warning("结构分块失败，回退 SentenceSplitter")
            await _report_progress(progress, "结构分块降级：使用 SentenceSplitter")
            nodes = await _run_blocking(splitter.get_nodes_from_documents, documents)
        texts = [node.get_content() for node in nodes if node.get_content().strip()]
        nodes = [node for node in nodes if node.get_content().strip()]
        if not texts:
            await _report_progress(progress, "文档无有效检索正文，本次不写入 Milvus")
            return 0
        for text in texts:
            ensure_no_base64(text)
        if paths:
            chunks = [{"chunk_index": i, "text": node.get_content(), "metadata": node.metadata} for i, node in enumerate(nodes)]
            files["chunks.jsonl"] = ("\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n").encode("utf-8")
            status = await _run_blocking(artifact_store.persist_artifacts, paths, files)
            await _report_progress(progress, f"分块预览已保存：{paths.root / 'chunks.jsonl'}" if status["local"] else "分块预览已保存到 MinIO，本地预览暂不可用")
        await _report_progress(progress, f"分块完成：共 {len(texts)} 个分块")
        embeddings = await _embed_chunks(texts, embedding_model, progress=progress)
        if len(embeddings) != len(texts):
            raise ValueError("向量数量与文档分块数量不一致，停止写入")
        pages = []
        for node in nodes:
            try:
                pages.append(int(node.metadata.get("page_label", 0)))
            except (ValueError, TypeError):
                pages.append(0)
        all_data = [{
            "id": f"{doc_id}_{index}", "doc_id": doc_id, "doc_name": doc_name,
            "doc_type": doc_type, "category": category, "page_number": pages[index],
            "chunk_index": index, "text": text[:65000], "embedding": embedding,
        } for index, (text, embedding) in enumerate(zip(texts, embeddings))]
        # 父单元 = 结构块；source_id 取它在 original 里的主键，变体可追溯到 original 行；页码与 original 一致
        parents = [
            ParentUnit(row["id"], text, {"page_number": row["page_number"]})
            for row, text in zip(all_data, texts)
        ]
        # 变体先切分完再动 Milvus：语义分块要调 embedding，失败时 original 和旧变体都不改
        variants = {}
        for strategy in strategies:
            await _report_progress(progress, f"{strategy} 分块：{len(parents)} 个结构块")
            variants[strategy] = await _build_variant_chunks(strategy, parents, embedding_model)
            await _report_progress(progress, f"{strategy} 分块完成：共 {len(variants[strategy])} 个子块")
        await _report_progress(progress, "检查 Milvus knowledge_docs 集合")
        await _run_blocking(ensure_knowledge_collection, milvus_client)
        await _report_progress(progress, "写入 Milvus 并清理同名文档的多余旧分块")
        await _replace_doc_chunks(milvus_client, doc_id, all_data)
        await _report_progress(progress, f"original 写入完成：{len(all_data)} 个分块")
        written = await _replace_doc_variants(
            milvus_client, embedding_model, doc_id, doc_name, doc_type, category, variants, progress,
        )
        summary = "，".join(f"{strategy} {count} 个子块" for strategy, count in written.items())
        await _report_progress(progress, "Milvus 写入完成：" + "，".join(
            [f"original {len(all_data)} 个分块"] + ([summary] if summary else [])))
        logger.info("文档 '{}' 导入完成，original {} 个分块，变体 {}", doc_name, len(all_data), written)
        return len(all_data)

async def ingest_drug_instructions(
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session,
) -> int:
    """
    从 PostgreSQL drug_details 表导入药品说明书到 Milvus。

    每个药品的说明书作为一篇文档，按药品逐个覆盖更新；分类固定为“药剂科”，doc_type 固定为 drug_instruction。
    全部药品处理成功后，清理 Milvus 中已不在本次结果里的药品说明书（药品被删除或说明书被清空）。
    中途任一药品失败时直接抛出，已处理的药品保持新数据，未处理的保持旧数据，且不做清理。

    Returns:
        所有药品写入的分块总数。
    """
    # 延迟导入：避免知识模块与业务模型、llama_index 在加载时互相依赖
    from sqlalchemy import select
    from src.modules.medical.model import Drug, DrugDetail
    from llama_index.core import Document

    await _run_blocking(ensure_knowledge_collection, milvus_client)

    # 只取有完整说明书的药品
    result = await db_session.execute(
        select(Drug.name, DrugDetail.full_instruction)
        .join(DrugDetail, Drug.id == DrugDetail.drug_id)
        .where(DrugDetail.full_instruction.isnot(None))
    )
    rows = result.all()

    # 查询结果为空更可能是数据库异常，直接返回，不清理 Milvus，避免误删全部说明书
    if not rows:
        logger.warning("drug_details 表中无说明书数据")
        return 0

    splitter = _make_splitter()
    total = 0
    # 本次成功写入的药品 doc_id，用于最后清理已失效的说明书
    current_doc_ids: set[str] = set()

    for drug_name, instruction in rows:
        # 跳过只有空白字符的说明书；它不会进入 current_doc_ids，旧分块在最后被清理
        if not instruction or not instruction.strip():
            continue

        # 加 "drug_" 前缀，避免与同名的上传文档（ingest_file 用 doc_name 哈希）撞 doc_id
        doc_id = hashlib.md5(f"drug_{drug_name}".encode()).hexdigest()[:16]

        # 说明书已是纯文本，直接包装成 LlamaIndex Document 交给分块器
        doc = Document(text=instruction, metadata={"drug_name": drug_name})
        nodes = splitter.get_nodes_from_documents([doc])
        texts = [n.get_content() for n in nodes]

        if not texts:
            continue

        embeddings = await _embed_chunks(texts, embedding_model)
        data = []
        for idx, (text_content, emb) in enumerate(zip(texts, embeddings)):
            data.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": f"{drug_name}说明书",
                "doc_type": DRUG_DOC_TYPE,
                "category": "药剂科",
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

        # 每个药品处理完立即覆盖写入：中途某个药品失败时，已处理的药品不受影响
        await _replace_doc_chunks(milvus_client, doc_id, data)
        current_doc_ids.add(doc_id)
        total += len(data)

    removed = await _cleanup_stale_drug_docs(milvus_client, current_doc_ids)
    logger.info(f"药品说明书导入完成，共 {total} 个分块，清理失效说明书 {removed} 篇")
    return total


async def _cleanup_stale_drug_docs(
    milvus_client: MilvusClient,
    current_doc_ids: set[str],
) -> int:
    """
    删除 Milvus 中不在 current_doc_ids 里的药品说明书，返回删除的文档数。

    只查询 chunk_index == 0 的分块：每篇文档恰好一条，结果条数等于文档数而不是分块数。
    """
    def _list_drug_doc_ids() -> set[str]:
        # query_iterator 分页拉取，不受单次 query 返回条数上限的限制
        iterator = milvus_client.query_iterator(
            collection_name=COLLECTION_NAME,
            filter=f'({ORIGINAL_FILTER}) and doc_type == "{DRUG_DOC_TYPE}" and chunk_index == 0',
            output_fields=["doc_id"],
        )
        doc_ids = set()
        try:
            while batch := iterator.next():
                doc_ids.update(row["doc_id"] for row in batch)
        finally:
            iterator.close()
        return doc_ids

    stale = sorted(await _run_blocking(_list_drug_doc_ids) - current_doc_ids)
    # 分批删除，避免 in [...] 过滤表达式过长
    for i in range(0, len(stale), DELETE_BATCH_SIZE):
        batch = stale[i:i + DELETE_BATCH_SIZE]
        id_list = ", ".join(f'"{doc_id}"' for doc_id in batch)
        await _run_blocking(
            milvus_client.delete,
            collection_name=COLLECTION_NAME,
            filter=f"({ORIGINAL_FILTER}) and doc_id in [{id_list}]",
        )
    return len(stale)