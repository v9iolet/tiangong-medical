"""
公共医学数据集初始化脚本。
从 HuggingFace 下载公共医学数据集，清洗后导入 Milvus 知识库。

用法：
    # 导入全部数据集
    python scripts/init_public_datasets.py

    # 只导入指定数据集
    python scripts/init_public_datasets.py --dataset cmirb
    python scripts/init_public_datasets.py --dataset dialogue
    python scripts/init_public_datasets.py --dataset medqa

    # CMIRB 语义分块（只处理 original 已入库的 doc_id；先用一个 doc_id 估算耗时和费用）
    python scripts/init_public_datasets.py --dataset cmirb --strategy semantic --doc-id <doc_id>
    python scripts/init_public_datasets.py --dataset cmirb --strategy semantic --concurrency 8 --force

    # --concurrency 是同时进行的 embedding 请求数，句子窗口预取和子块写入共用（默认 8，按接口额度调整）
    python scripts/init_public_datasets.py --dataset cmirb --strategy semantic --concurrency 4

依赖：
    pip install datasets huggingface_hub
"""

import asyncio
import argparse
import hashlib
import sys
import os
import threading
import time
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import get_embedding_model

from pymilvus import MilvusClient
from loguru import logger

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.ingestion.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.ingestion.variant_chunking import (
    SEMANTIC, ParentUnit, build_semantic_chunks, build_semantic_embed_model,
)
from src.agents.knowledge.ingestion.variant_writer import write_variant_chunks
from src.agents.knowledge.retrieval.doc_rag import COLLECTION_NAME
from src.agents.knowledge.retrieval.variant_schema import ORIGINAL_FILTER, original_chunk_metadata, strategy_filter

settings = get_settings()
BATCH_SIZE = 20

# CMIRB 固定本地缓存的版本，数据集更新后 doc_id 和 source_id 不会悄悄漂移
CMIRB_REVISION = "3844e2255fbe12c43844927e1e4930b3af911667"
# 每个 doc_id 包含的记录数
CMIRB_GROUP_SIZE = 5000
CMIRB_DOC_TYPE = "literature"
CMIRB_CATEGORY = "医学文献"
# 查询已入库的 source_id 时的分页大小，低于 Milvus 单次 16384 的上限
QUERY_BATCH_SIZE = 5000
# 语义分块的默认 embedding 并发数（句子窗口预取和子块写入共用）。
# 按接口额度 RPM 2000 / TPM 100 万估算：并发 8 约 240 RPM、70 万 TPM，留有余量
CMIRB_SEMANTIC_CONCURRENCY = 8


class BuildAbort(Exception):
    """前置核对不通过，终止构建；此时不会写入任何数据。"""


def _get_deps():
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    embedding_model = get_embedding_model()
    ensure_knowledge_collection(milvus_client)
    return milvus_client, embedding_model


def _count_doc_chunks(milvus_client: MilvusClient, doc_id: str) -> int:
    """查询某个 doc_id 已入库的分块数。"""
    res = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter=f'({ORIGINAL_FILTER}) and doc_id == "{doc_id}"',
        output_fields=["count(*)"],
    )
    return res[0]["count(*)"] if res else 0


async def _insert_texts(
    milvus_client: MilvusClient,
    embedding_model,
    texts: list[str],
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
    force: bool = False,
) -> int:
    """
    批量向量化并写入 Milvus。

    断点续传：该 doc_id 已完整入库（条数一致）时直接跳过，避免重复消耗 embedding 额度；
    force=True 时无条件删除后重新入库。
    """
    if not force and _count_doc_chunks(milvus_client, doc_id) == len(texts):
        print(f"  [跳过] {doc_name} 已完整入库")
        return len(texts)

    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'({ORIGINAL_FILTER}) and doc_id == "{doc_id}"',
    )

    all_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = await embedding_model.aembed_documents(batch)
        for j, (text_content, emb) in enumerate(zip(batch, embeddings)):
            idx = i + j
            all_data.append({
                "id": f"{doc_id}_{idx}",
                **original_chunk_metadata(f"{doc_id}_{idx}"),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "category": category,
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i:i + 1000],
            )
    return len(all_data)


# ═══════════════════════════════════════════════════════════════════════
# 数据集 1：CMIRB/MedicalRetrieval — 10万条医学检索语料
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CmirbRecord:
    """一条语料记录；source_id 与 original 入库的主键相同，语义分块的父单元以它为来源。"""
    source_id: str
    text: str


@dataclass
class CmirbGroup:
    """一个 doc_id 对应的一组记录。"""
    doc_id: str
    doc_name: str
    records: list[CmirbRecord]


def _read_cmirb_corpus():
    """读取 CMIRB 语料原始行；单独成函数，测试里替换它即可，不必下载数据集。"""
    from datasets import load_dataset

    return load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus", revision=CMIRB_REVISION)


def load_cmirb_records() -> list[CmirbGroup]:
    """
    读取 CMIRB 语料，过滤（>20 字）、截断（2000 字）后按 5000 条分组。
    original 和 semantic 共用，保证 doc_id（md5("cmirb_{组起始下标}")[:16]）
    和 source_id（{doc_id}_{组内序号}）与 original 入库的行一一对应。
    """
    texts = []
    for row in _read_cmirb_corpus():
        text = row.get("text", "") or row.get("content", "")
        if text and len(text.strip()) > 20:
            texts.append(text.strip()[:2000])

    groups = []
    for i in range(0, len(texts), CMIRB_GROUP_SIZE):
        doc_id = hashlib.md5(f"cmirb_{i}".encode()).hexdigest()[:16]
        groups.append(CmirbGroup(
            doc_id=doc_id,
            doc_name=f"CMIRB医学检索语料_{i // CMIRB_GROUP_SIZE + 1}",
            records=[
                CmirbRecord(source_id=f"{doc_id}_{idx}", text=text)
                for idx, text in enumerate(texts[i:i + CMIRB_GROUP_SIZE])
            ],
        ))
    return groups


class CountingEmbeddings(Embeddings):
    """包一层统计 embedding 请求次数和文本条数；句子窗口和子块的请求并发进行，计数加锁。"""

    # 每累计这么多次请求打印一次心跳；语义切分阶段要几分钟才开始写入，没有心跳看不出是否在跑
    HEARTBEAT_REQUESTS = 50

    def __init__(self, inner: Embeddings, label: str = ""):
        self.inner = inner
        self.label = label
        self.requests = 0
        self.texts = 0
        self._lock = threading.Lock()

    def _count(self, texts: list[str]) -> None:
        with self._lock:
            self.requests += 1
            self.texts += len(texts)
            if self.label and self.requests % self.HEARTBEAT_REQUESTS == 0:
                print(f"  [进度] {self.label} embedding {self.requests} 次（{self.texts} 条文本）", flush=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._count(texts)
        return self.inner.embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self._count(texts)
        return await self.inner.aembed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._count([text])
        return self.inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        self._count([text])
        return await self.inner.aembed_query(text)


def _select_groups(groups: list[CmirbGroup], doc_ids: list[str] | None) -> list[CmirbGroup]:
    """按 --doc-id 筛选；传了数据集里没有的 doc_id 就报错，并列出可用的。"""
    if not doc_ids:
        return groups
    known = {group.doc_id for group in groups}
    unknown = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id not in known]
    if unknown:
        raise BuildAbort(f"数据集中没有这些 doc_id：{', '.join(unknown)}；可用：{', '.join(sorted(known))}")
    wanted = set(doc_ids)
    return [group for group in groups if group.doc_id in wanted]


def _check_original_counts(milvus_client: MilvusClient, groups: list[CmirbGroup]) -> None:
    """每个 doc_id 的数据集记录数必须等于 Milvus 里 original 的行数；任一不一致就中止，此时还没写任何数据。"""
    mismatched = []
    for group in groups:
        stored = _count_doc_chunks(milvus_client, group.doc_id)
        if stored != len(group.records):
            mismatched.append(f"{group.doc_id}（数据集 {len(group.records)} 条，original {stored} 行）")
    if mismatched:
        raise BuildAbort(
            "数据集与已入库的 original 不一致：" + "；".join(mismatched)
            + "。语义分块的 source_id 要和 original 一一对应，请先用 --strategy original --force 重新入库"
        )


def _stored_semantic_sources(milvus_client: MilvusClient, doc_id: str) -> set[str]:
    """该 doc_id 已入库的 semantic 行覆盖了哪些 source_id；用 query_iterator 分页，避开单次 16384 的上限。"""
    iterator = milvus_client.query_iterator(
        collection_name=COLLECTION_NAME,
        batch_size=QUERY_BATCH_SIZE,
        filter=f'doc_id == "{doc_id}" and {strategy_filter(SEMANTIC)}',
        output_fields=["source_id"],
        consistency_level="Strong",
    )
    sources = set()
    try:
        while batch := iterator.next():
            sources.update(row["source_id"] for row in batch)
    finally:
        iterator.close()
    return sources


async def _build_semantic_group(
    milvus_client: MilvusClient, embedding_model: Embeddings, group: CmirbGroup, concurrency: int, force: bool,
) -> dict | None:
    """
    构建一个 doc_id 的语义分块：一条记录就是一个父单元。
    断点续跑：semantic 行已覆盖全部 source_id 就跳过（force 时重建），返回 None；
    否则返回耗时、父单元数、子块数和两类 embedding 请求次数。
    """
    expected = {record.source_id for record in group.records}
    covered = expected & _stored_semantic_sources(milvus_client, group.doc_id)
    if not force and covered == expected:
        print(f"  [跳过] {group.doc_name} 的语义分块已覆盖全部 {len(expected)} 条记录")
        return None
    if covered and not force:
        print(f"  [INFO] {group.doc_name} 只覆盖 {len(covered)}/{len(expected)} 条记录，整组重建")

    # 句子窗口（切分时）和子块（写入时）的 embedding 分开计数
    sentence_counter = CountingEmbeddings(embedding_model, label=f"{group.doc_name} 句子窗口")
    chunk_counter = CountingEmbeddings(embedding_model, label=f"{group.doc_name} 子块")
    started = time.monotonic()

    parents = [
        ParentUnit(source_id=record.source_id, text=record.text, metadata={"page_number": 0})
        for record in group.records
    ]
    print(f"  [INFO] {group.doc_name}：{len(parents)} 个父单元，开始语义切分...")
    chunks = await build_semantic_chunks(parents, build_semantic_embed_model(sentence_counter), concurrency)

    async def progress(message: str) -> None:
        print(f"  [进度] {group.doc_name} {message}")

    written = await write_variant_chunks(
        milvus_client, chunk_counter, chunks,
        strategy=SEMANTIC, doc_id=group.doc_id, doc_name=group.doc_name,
        doc_type=CMIRB_DOC_TYPE, category=CMIRB_CATEGORY, progress=progress, concurrency=concurrency,
    )
    stats = {
        "elapsed": time.monotonic() - started,
        "parents": len(parents),
        "chunks": written,
        "sentence_requests": sentence_counter.requests,
        "sentence_texts": sentence_counter.texts,
        "chunk_requests": chunk_counter.requests,
        "chunk_texts": chunk_counter.texts,
    }
    print(
        f"  [统计] {group.doc_id} 耗时 {stats['elapsed']:.1f}s，父单元 {stats['parents']}，子块 {stats['chunks']}；"
        f"embedding 请求：句子窗口 {stats['sentence_requests']} 次（{stats['sentence_texts']} 条文本），"
        f"子块 {stats['chunk_requests']} 次（{stats['chunk_texts']} 条文本）"
    )
    return stats


async def download_cmirb(
    milvus_client, embedding_model, force: bool = False,
    strategy: str = "original", doc_ids: list[str] | None = None, concurrency: int = CMIRB_SEMANTIC_CONCURRENCY,
):
    """
    下载 CMIRB/MedicalRetrieval 数据集并导入 Milvus。
    包含 ~101K 条中文医学文本，适合作为文档 RAG 的基础语料。

    strategy=original 写入原始分块；strategy=semantic 在 original 已入库的前提下，
    对每条记录做语义分块并写入变体。doc_ids 只处理指定的 doc_id。
    """
    print("[INFO] 下载 CMIRB/MedicalRetrieval 数据集...")
    all_groups = load_cmirb_records()
    groups = _select_groups(all_groups, doc_ids)
    total_texts = sum(len(group.records) for group in groups)

    if strategy == SEMANTIC:
        print(f"[INFO] CMIRB 共 {total_texts} 条有效文本，{len(groups)} 个 doc_id，开始语义分块...")
        _check_original_counts(milvus_client, groups)
        results = []
        for group in groups:
            stats = await _build_semantic_group(milvus_client, embedding_model, group, concurrency, force)
            if stats:
                results.append(stats)
        total = sum(stats["chunks"] for stats in results)
        print(
            f"[OK] CMIRB 语义分块完成，处理 {len(results)}/{len(groups)} 个 doc_id，共 {total} 个子块；"
            f"跳过 {len(groups) - len(results)} 个"
        )
        return total

    print(f"[INFO] CMIRB 共 {total_texts} 条有效文本，开始向量化...")

    total = 0
    done = 0
    for group in groups:
        count = await _insert_texts(
            milvus_client, embedding_model, [record.text for record in group.records],
            doc_id=group.doc_id,
            doc_name=group.doc_name,
            doc_type=CMIRB_DOC_TYPE,
            category=CMIRB_CATEGORY,
            force=force,
        )
        total += count
        done += len(group.records)
        print(f"  [进度] {done}/{total_texts}")

    print(f"[OK] CMIRB 导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 2：Chinese-medical-dialogue-data — 6科室医患对话
# ═══════════════════════════════════════════════════════════════════════

async def download_med_dialogue(milvus_client, embedding_model, force: bool = False):
    """
    下载中文医患对话数据集并导入 Milvus。
    每条记录格式化为"问：xxx 答：xxx"，作为知识文档。

    注意：BillGPT 镜像与 Toyhom 原版不同，仓库里只有一个 外科.zip（约 6 万个 JSON 小文件），
    每个 JSON 含 input（"患者:...\\n医生:..."）和预计算的 1536 维 embedding。
    该 embedding 与本项目向量模型不一致，只取 input 文本重新向量化。
    用 load_dataset 逐个解析 JSON 极慢（约 3 条/秒），这里直接下载 zip 自行读取。
    """
    import json
    import zipfile
    from huggingface_hub import hf_hub_download

    print("[INFO] 下载 Chinese-medical-dialogue-data 数据集...")
    zip_path = hf_hub_download(
        "BillGPT/Chinese-medical-dialogue-data", "外科.zip", repo_type="dataset"
    )

    texts = []
    seen = set()
    with zipfile.ZipFile(zip_path) as zf:
        # 排序保证每次分块一致，断点续传依赖 doc_id 与分块内容稳定对应
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        for name in names:
            raw = (json.loads(zf.read(name)).get("input") or "").strip()
            q, sep, a = raw.partition("\n医生:")
            q = q.removeprefix("患者:").strip()
            a = a.strip()
            if not sep or len(q) <= 5 or len(a) <= 10:
                continue
            text = f"问：{q}\n答：{a}"[:2000]
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)

    print(f"[INFO] 医患对话共 {len(texts)} 条有效记录，开始向量化...")

    chunk_size = 5000
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"med_dialogue_{i}".encode()).hexdigest()[:16]
        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"中文医患对话_{i // chunk_size + 1}",
            doc_type="literature",
            category="医患对话",
            force=force,
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] 医患对话导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 3：MedQA — 执业医师考试题（评测用，存到本地文件）
# ═══════════════════════════════════════════════════════════════════════

async def download_medqa():
    """
    下载 MedQA 中文执业医师考试题，保存到 data/eval/ 目录。
    用于评测知识 Agent 的准确率，不导入 Milvus。
    """
    from datasets import load_dataset
    import json

    print("[INFO] 下载 MedQA 数据集...")
    # 只取 test split（3426 题）做评测；train 有 2.7 万题，评测用不到
    ds = load_dataset("bigbio/med_qa", "med_qa_zh_4options_bigbio_qa", split="test")

    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    records = []
    for row in ds:
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer = row.get("answer", [])
        if question:
            records.append({
                "question": question,
                "choices": choices,
                "answer": answer,
                "split": "test",
            })

    output_path = os.path.join(eval_dir, "medqa_zh.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] MedQA 保存完成：{output_path}，共 {len(records)} 道题")
    return len(records)


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公共医学数据集初始化")
    parser.add_argument(
        "--dataset",
        choices=["cmirb", "dialogue", "medqa", "all"],
        default="all",
        help="要导入的数据集（默认 all）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已入库数据，全部删除后重新向量化",
    )
    parser.add_argument(
        "--strategy",
        choices=["original", "semantic"],
        default="original",
        help="分块策略（默认 original）；semantic 只支持 --dataset cmirb，且要求 original 已入库",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="只处理指定的 doc_id，可重复；只支持 --dataset cmirb",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CMIRB_SEMANTIC_CONCURRENCY,
        help=(
            f"语义分块同时进行的 embedding 请求数（默认 {CMIRB_SEMANTIC_CONCURRENCY}），"
            "句子窗口预取和子块写入共用，只对 --strategy semantic 生效"
        ),
    )
    args = parser.parse_args(argv)
    if args.dataset != "cmirb" and args.strategy == SEMANTIC:
        parser.error("--strategy semantic 只支持 --dataset cmirb")
    if args.dataset != "cmirb" and args.doc_ids:
        parser.error("--doc-id 只支持 --dataset cmirb")
    if args.concurrency < 1:
        parser.error("--concurrency 必须 >= 1")
    return args


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    milvus_client, embedding_model = _get_deps()

    try:
        if args.dataset in ("cmirb", "all"):
            await download_cmirb(
                milvus_client, embedding_model, force=args.force,
                strategy=args.strategy, doc_ids=args.doc_ids, concurrency=args.concurrency,
            )

        if args.dataset in ("dialogue", "all"):
            await download_med_dialogue(milvus_client, embedding_model, force=args.force)

        if args.dataset in ("medqa", "all"):
            await download_medqa()
    except BuildAbort as error:
        print(f"[中止] {error}", file=sys.stderr)
        return 1

    print("\n[DONE] 数据集初始化完成！")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))