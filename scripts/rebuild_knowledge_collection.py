"""
重建 knowledge_docs：按新 schema（中文分词 + BM25 稀疏向量 + 非空分块元数据）新建集合，
把原始行连同 embedding 原样拷过去，核对通过后再改名切换。

用法：
    # 只做预检和统计（默认），不写 Milvus
    python -m scripts.rebuild_knowledge_collection

    # 真正执行：建 knowledge_docs_rebuild → 拷贝 → 核对 → 旧集合改名为 knowledge_docs_legacy_<ts> → 新集合改名
    python -m scripts.rebuild_knowledge_collection --execute

旧集合会保留，回滚时把名字改回来即可；确认无误后再手动删除 legacy 集合。
执行期间两份数据并存，Milvus 磁盘占用约翻倍。
"""

import argparse
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymilvus import MilvusClient

from src.agents.knowledge.ingestion.doc_ingestion import (
    EMBEDDING_DIM, build_knowledge_index_params, build_knowledge_schema, validate_knowledge_schema,
)
from src.agents.knowledge.retrieval.doc_rag import COLLECTION_NAME
from src.agents.knowledge.retrieval.variant_schema import VARIANT_PREFIX
from src.core.config import get_settings

REBUILD_NAME = "knowledge_docs_rebuild"
BATCH_SIZE = 1000
# 抽样比对 embedding 的上限；每个批次取第一行
MAX_SAMPLES = 200
# 抽样 embedding 允许的最大绝对误差（同一份 float32 原样拷贝，应完全一致）
EMBEDDING_TOLERANCE = 1e-6
# 新 schema 里不存在的旧字段，拷贝时丢掉
DROPPED_FIELDS = {"index_version", "sparse_embedding"}


class RebuildAbort(Exception):
    """预检或核对不通过，终止重建；已有集合的名字不会被改动。"""


def _count(client: MilvusClient, collection: str, filter_expr: str = "") -> int:
    rows = client.query(
        collection_name=collection, filter=filter_expr,
        output_fields=["count(*)"], consistency_level="Strong",
    )
    return int(rows[0]["count(*)"]) if rows else 0


def _iterate(client: MilvusClient, collection: str, output_fields: list[str], filter_expr: str = ""):
    """分页遍历集合，按批 yield 行列表。"""
    iterator = client.query_iterator(
        collection_name=collection, batch_size=BATCH_SIZE,
        filter=filter_expr, output_fields=output_fields,
        # 刚 flush 完就统计，Bounded 一致性可能漏掉最新写入的行
        consistency_level="Strong",
    )
    try:
        while batch := iterator.next():
            yield batch
    finally:
        iterator.close()


def collect_stats(client: MilvusClient, collection: str) -> dict:
    """只读统计：总行数、每个 doc_id 的行数、ragvariant 行数、非 original 行数。"""
    per_doc: dict[str, int] = {}
    for batch in _iterate(client, collection, ["doc_id"]):
        for row in batch:
            per_doc[row["doc_id"]] = per_doc.get(row["doc_id"], 0) + 1
    return {
        "total": _count(client, collection),
        "per_doc": per_doc,
        "variant_rows": _count(client, collection, f'id like "{VARIANT_PREFIX}%"'),
        # chunk_strategy 是旧集合里的 nullable 字段，NULL 也算不满足
        "non_original_rows": _count(client, collection, 'chunk_strategy is null or chunk_strategy != "original"'),
    }


def preflight(client: MilvusClient, stats: dict, source: str, target: str) -> None:
    """预检不通过就抛出 RebuildAbort；调用前已确认源集合存在。"""
    try:
        validate_knowledge_schema(client, source)
    except ValueError:
        pass
    else:
        raise RebuildAbort(f"源集合 '{source}' 已是新 schema，无需重建")
    if client.has_collection(target):
        raise RebuildAbort(f"目标集合 '{target}' 已存在；请确认内容后手动删除再重试，脚本不会自动删除")
    if stats["variant_rows"]:
        raise RebuildAbort(f"源集合有 {stats['variant_rows']} 行 {VARIANT_PREFIX} 变体数据，只能迁移原始行")
    if stats["non_original_rows"]:
        raise RebuildAbort(f"源集合有 {stats['non_original_rows']} 行 chunk_strategy 不是 original")
    if sum(stats["per_doc"].values()) != stats["total"]:
        raise RebuildAbort(f"按 doc_id 汇总 {sum(stats['per_doc'].values())} 行，与总行数 {stats['total']} 不一致")
    dim = next(f["params"].get("dim") for f in client.describe_collection(source)["fields"] if f["name"] == "embedding")
    if int(dim) != EMBEDDING_DIM:
        raise RebuildAbort(f"源集合向量维度 {dim} 与新 schema 的 {EMBEDDING_DIM} 不一致")


def print_stats(stats: dict, source: str) -> None:
    print(f"源集合: {source}")
    print(f"总行数: {stats['total']:,}")
    print(f"doc_id 数: {len(stats['per_doc'])}")
    for doc_id, count in sorted(stats["per_doc"].items()):
        print(f"  {doc_id}: {count:,}")
    print(f"{VARIANT_PREFIX} 行数: {stats['variant_rows']:,}")
    print(f"非 original 行数: {stats['non_original_rows']:,}")


def copy_rows(client: MilvusClient, source: str, target: str) -> dict[str, dict]:
    """原样拷贝原始行和 embedding；稀疏向量由服务端从 text 生成。返回抽样行供核对。"""
    fields = [f["name"] for f in client.describe_collection(source)["fields"] if f["name"] not in DROPPED_FIELDS]
    samples: dict[str, dict] = {}
    copied = 0
    for batch in _iterate(client, source, fields):
        rows = []
        for row in batch:
            row = {name: row[name] for name in fields}
            # 旧集合的元数据字段是 nullable；新 schema 非空，空值按“不适用”处理
            row["parent_id"] = row.get("parent_id") or ""
            row["parent_text"] = row.get("parent_text") or ""
            row["source_id"] = row.get("source_id") or row["id"]
            rows.append(row)
        result = client.insert(collection_name=target, data=rows)
        if int(result["insert_count"]) != len(rows):
            raise RebuildAbort(f"写入 {len(rows)} 行，服务端只确认 {result['insert_count']} 行")
        if len(samples) < MAX_SAMPLES:
            samples[rows[0]["id"]] = rows[0]
        copied += len(rows)
        print(f"  已拷贝 {copied:,} 行", flush=True)
    return samples


def verify(client: MilvusClient, source_stats: dict, target: str, samples: dict[str, dict]) -> None:
    """核对 schema、总行数、每个 doc_id 的行数、抽样 embedding，并跑一次 BM25 冒烟；任一不通过就抛出 RebuildAbort。"""
    # 改名后入库都要过 validate_knowledge_schema，在这里先确认新集合能通过
    try:
        validate_knowledge_schema(client, target)
    except ValueError as error:
        raise RebuildAbort(f"新集合 schema 校验不通过：{error}") from error
    client.flush(target)
    target_stats = collect_stats(client, target)
    if target_stats["total"] != source_stats["total"]:
        raise RebuildAbort(f"总行数不一致：源 {source_stats['total']:,}，新 {target_stats['total']:,}")
    if target_stats["per_doc"] != source_stats["per_doc"]:
        diff = sorted(
            doc_id for doc_id in source_stats["per_doc"].keys() | target_stats["per_doc"].keys()
            if source_stats["per_doc"].get(doc_id) != target_stats["per_doc"].get(doc_id)
        )
        raise RebuildAbort(f"以下 doc_id 的行数不一致：{', '.join(diff)}")

    id_list = ", ".join(f'"{sample_id}"' for sample_id in samples)
    found = {
        row["id"]: row for row in client.query(
            collection_name=target, filter=f"id in [{id_list}]",
            output_fields=["id", "text", "embedding"], consistency_level="Strong",
        )
    }
    for sample_id, expected in samples.items():
        row = found.get(sample_id)
        if row is None:
            raise RebuildAbort(f"抽样行 {sample_id} 在新集合中不存在")
        if row["text"] != expected["text"]:
            raise RebuildAbort(f"抽样行 {sample_id} 的 text 不一致")
        if len(row["embedding"]) != len(expected["embedding"]) or any(
            not math.isclose(a, b, abs_tol=EMBEDDING_TOLERANCE)
            for a, b in zip(row["embedding"], expected["embedding"])
        ):
            raise RebuildAbort(f"抽样行 {sample_id} 的 embedding 不一致")

    # BM25 冒烟：拿一条抽样文本的开头去查稀疏向量，至少要有命中
    query_text = next(iter(samples.values()))["text"][:20]
    hits = client.search(
        collection_name=target, data=[query_text], anns_field="sparse_embedding",
        limit=3, search_params={"metric_type": "BM25"}, output_fields=["id"],
    )
    if not hits or not hits[0]:
        raise RebuildAbort("BM25 冒烟查询没有命中，稀疏向量可能未生成")
    print(f"核对通过：{target_stats['total']:,} 行，{len(target_stats['per_doc'])} 个 doc_id，"
          f"抽样 {len(samples)} 行 embedding 一致，BM25 冒烟命中 {len(hits[0])} 条")


def swap(client: MilvusClient, source: str, target: str, timestamp: str) -> str:
    """旧集合改名为 legacy，再把新集合改名为正式名；第二步失败时把旧集合改回去。"""
    legacy = f"{source}_legacy_{timestamp}"
    if client.has_collection(legacy):
        raise RebuildAbort(f"备份集合 '{legacy}' 已存在")
    client.rename_collection(source, legacy)
    try:
        client.rename_collection(target, source)
    except Exception:
        client.rename_collection(legacy, source)
        raise
    return legacy


def run(client: MilvusClient, *, execute: bool, source: str = COLLECTION_NAME, target: str = REBUILD_NAME,
        timestamp: str | None = None) -> None:
    """dry-run 只调用只读方法；execute 在预检和核对都通过后才改名。"""
    if not client.has_collection(source):
        raise RebuildAbort(f"源集合 '{source}' 不存在")
    stats = collect_stats(client, source)
    print_stats(stats, source)
    preflight(client, stats, source, target)
    print("预检通过")
    if not execute:
        print("dry-run：未写入任何数据；加 --execute 执行重建")
        return

    print(f"新建集合 {target}")
    client.create_collection(
        collection_name=target, schema=build_knowledge_schema(), index_params=build_knowledge_index_params(),
    )
    try:
        samples = copy_rows(client, source, target)
        verify(client, stats, target, samples)
    except Exception:
        print(f"重建失败，未改名；'{target}' 已保留供排查，确认后请手动删除")
        raise
    legacy = swap(client, source, target, timestamp or datetime.now().strftime("%Y%m%d%H%M%S"))
    print(f"完成：旧集合已改名为 '{legacy}'，新集合已改名为 '{source}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="按新 schema 重建 knowledge_docs 并导入原始向量数据")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只做预检和统计（默认）")
    mode.add_argument("--execute", action="store_true", help="建集合、拷数据、核对并改名")
    args = parser.parse_args()
    settings = get_settings()
    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    try:
        run(client, execute=args.execute)
    except RebuildAbort as error:
        print(f"中止：{error}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
