"""把父子 / 语义子块向量化并覆盖写入 knowledge_docs；数据集脚本和上传入库共用。

主键 = VARIANT_PREFIX + 策略 + ":" + sha256(source_id:序号)，同一来源重切得到相同主键。
覆盖写 = 先删该文档该策略的旧行，再按批 insert；每批只含完整的父单元，中途失败缺的是整个父单元。
sparse_embedding 由服务端 BM25 Function 从 text 生成，这里不写。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math

from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

from src.agents.knowledge.ingestion.doc_ingestion import (
    EMBED_BATCH_SIZE, EMBEDDING_DIM, ProgressCallback, _report_progress, _run_blocking,
)
from src.agents.knowledge.ingestion.variant_chunking import (
    VARIANT_STRATEGIES, VariantChunk, gather_cancel_on_error,
)
from src.agents.knowledge.retrieval.doc_rag import COLLECTION_NAME
from src.agents.knowledge.retrieval.variant_schema import VARIANT_FIELDS, VARIANT_PREFIX, strategy_filter

# 每次 insert 的行数上限；一个父单元的块超过它时，该父单元自己一批
INSERT_BATCH_SIZE = 200
# 同时进行的 embedding 请求数的默认值，每个请求最多 EMBED_BATCH_SIZE 条文本；write_variant_chunks 可用 concurrency 覆盖
EMBED_CONCURRENCY = 4
# schema 中文档级字段的长度；与 build_knowledge_schema 保持一致
ROW_FIELD_LIMITS = {
    "id": 256, "doc_id": 128, "doc_name": 256, "doc_type": 50, "category": 100,
    "text": 65535, **VARIANT_FIELDS,
}


def variant_chunk_id(strategy: str, source_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{source_id}:{ordinal}".encode("utf-8")).hexdigest()
    return f"{VARIANT_PREFIX}{strategy}:{digest}"


def _validate_row(row: dict) -> None:
    """超过字段长度直接报错，不静默截断。Milvus 的 VARCHAR max_length 按 UTF-8 字节计，一个汉字占 3 字节。"""
    for name, limit in ROW_FIELD_LIMITS.items():
        size = len(row[name].encode("utf-8"))
        if size > limit:
            raise ValueError(f"分块 {row['id']} 的字段 {name} 有 {size} 字节，超过上限 {limit}")


def _validate_embedding(vector, chunk_id: str) -> None:
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(f"分块 {chunk_id} 的向量维度 {len(vector)} 与 EMBEDDING_DIM={EMBEDDING_DIM} 不一致")
    try:
        finite = all(math.isfinite(value) for value in vector)
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"分块 {chunk_id} 的向量含 NaN、Inf 或非数值")


async def _embed_texts(texts: list[str], embedding_model: Embeddings, concurrency: int) -> list[list[float]]:
    """按 EMBED_BATCH_SIZE 分批、最多 concurrency 个请求同时在飞，返回与 texts 一一对应的向量。"""
    semaphore = asyncio.Semaphore(concurrency)

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await embedding_model.aembed_documents(batch)

    batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    results = await gather_cancel_on_error(embed_batch(batch) for batch in batches)
    return [vector for batch_vectors in results for vector in batch_vectors]


def _group_by_parent(rows: list[dict], chunks: list[VariantChunk]) -> list[list[dict]]:
    """相邻且 source_id 相同的行属于同一个父单元；同一 source_id 不连续出现视为输入错误。"""
    groups: list[list[dict]] = []
    seen: set[str] = set()
    for row, chunk in zip(rows, chunks):
        if groups and groups[-1][0]["source_id"] == chunk.source_id:
            groups[-1].append(row)
            continue
        if chunk.source_id in seen:
            raise ValueError(f"父单元 {chunk.source_id} 的子块不连续，无法按完整父单元分批")
        seen.add(chunk.source_id)
        groups.append([row])
    return groups


def _batches(groups: list[list[dict]], size: int) -> list[list[dict]]:
    """按完整父单元打包：放不下就另起一批；单个父单元超过 size 时独占一批。"""
    batches: list[list[dict]] = []
    current: list[dict] = []
    for group in groups:
        if current and len(current) + len(group) > size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def _build_rows(
    chunks: list[VariantChunk], strategy: str, doc_id: str, doc_name: str, doc_type: str, category: str,
) -> list[dict]:
    rows, ids = [], set()
    for index, chunk in enumerate(chunks):
        if chunk.chunk_strategy != strategy:
            raise ValueError(f"分块 {chunk.source_id}#{chunk.ordinal} 的策略 {chunk.chunk_strategy} 与写入策略 {strategy} 不一致")
        if not chunk.text.strip() or not chunk.source_id:
            raise ValueError(f"分块 {chunk.source_id}#{chunk.ordinal} 的正文或 source_id 为空")
        chunk_id = variant_chunk_id(strategy, chunk.source_id, chunk.ordinal)
        if chunk_id in ids:
            raise ValueError(f"分块主键重复：{chunk.source_id}#{chunk.ordinal}")
        ids.add(chunk_id)
        row = {
            "id": chunk_id, "doc_id": doc_id, "doc_name": doc_name, "doc_type": doc_type,
            "category": category, "page_number": int(chunk.metadata.get("page_number") or 0),
            "chunk_index": index, "text": chunk.text, "chunk_strategy": strategy,
            "parent_id": chunk.parent_id, "parent_text": chunk.parent_text, "source_id": chunk.source_id,
        }
        _validate_row(row)
        rows.append(row)
    return rows


async def write_variant_chunks(
    milvus_client: MilvusClient,
    embedding_model: Embeddings,
    chunks: list[VariantChunk],
    *,
    strategy: str,
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
    progress: ProgressCallback | None = None,
    concurrency: int = EMBED_CONCURRENCY,
) -> int:
    """
    覆盖写某个文档的一种变体策略，返回写入的行数。

    chunks 按父单元连续排列（build_*_chunks 的输出即是）；chunk_index 取块在文档内的全局序号。
    第一批向量化并校验通过后才删除旧行，embedding 服务不可用时旧数据不动；
    之后逐批 embedding、校验、insert，某批失败则后面的父单元缺失，已写入的是完整父单元。
    concurrency 是每批 insert 内同时进行的 embedding 请求数。
    """
    if strategy not in VARIANT_STRATEGIES:
        raise ValueError(f"策略必须是 {VARIANT_STRATEGIES} 之一，original 由结构分块写入：{strategy}")
    if concurrency < 1:
        raise ValueError(f"concurrency 必须 >= 1：{concurrency}")
    rows = _build_rows(chunks, strategy, doc_id, doc_name, doc_type, category)
    batches = _batches(_group_by_parent(rows, chunks), INSERT_BATCH_SIZE)

    async def embed_batch(batch: list[dict]) -> list[dict]:
        vectors = await _embed_texts([row["text"] for row in batch], embedding_model, concurrency)
        if len(vectors) != len(batch):
            raise ValueError(f"向量数量 {len(vectors)} 与分块数量 {len(batch)} 不一致，停止写入")
        for row, vector in zip(batch, vectors):
            _validate_embedding(vector, row["id"])
        return [{**row, "embedding": vector} for row, vector in zip(batch, vectors)]

    # 先校验第一批再删旧行；没有任何新块时只删旧行
    first = await embed_batch(batches[0]) if batches else None
    await _run_blocking(
        milvus_client.delete, collection_name=COLLECTION_NAME,
        filter=f"doc_id == {json.dumps(doc_id)} and {strategy_filter(strategy)}",
    )
    written = 0
    for number, batch in enumerate(batches, 1):
        data = first if number == 1 else await embed_batch(batch)
        await _run_blocking(milvus_client.insert, collection_name=COLLECTION_NAME, data=data)
        written += len(data)
        await _report_progress(progress, f"{strategy} 写入 {written}/{len(rows)} 个子块")
    return written
