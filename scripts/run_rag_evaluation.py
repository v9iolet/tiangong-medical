"""独立运行真实 RAG 评估：python -m scripts.run_rag_evaluation --samples 1。"""
from __future__ import annotations

import argparse
import asyncio


CHANNELS = {
    "doc_rag": "文档 RAG",
    "graph_rag": "知识图谱 RAG",
    "fusion": "多路融合 RAG",
    "nl2sql": "SQL 数据查询",
    "prescription_review": "处方审核",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RAG 实验并将 TruLens 记录写入 PostgreSQL")
    parser.add_argument("--dataset", default="rag_eval_dataset.json", help="data/eval 中的评估集文件名")
    parser.add_argument("--samples", type=int, default=1, help="抽样题数，默认 1")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--category", action="append", help="按分类筛选，可重复指定")
    parser.add_argument("--channels", nargs="+", choices=list(CHANNELS), default=["doc_rag", "graph_rag", "fusion"])
    parser.add_argument("--label", default="manual-test", help="实验标签；版本自动附加入口、系统用户、时间和独立运行编号")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True, help="是否启用问题改写")
    parser.add_argument("--hyde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunking", choices=["original", "parent_child", "semantic"], default="original")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--context-top-k", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples 必须至少为 1")
    if not args.label.strip() or any(c in args.label for c in "/\\"):
        parser.error("--label 不能为空或包含路径分隔符")
    from src.agents.knowledge.retrieval.options import DocRetrievalOptions
    try:
        DocRetrievalOptions(chunk_strategy=args.chunking, top_k=args.top_k, context_top_k=args.context_top_k)
    except ValueError as error:
        parser.error(str(error))
    return args


async def run(args: argparse.Namespace) -> int:
    from scripts import gradio_knowledge_app as app

    final = None
    printed_lines = 0
    try:
        async for update in app.run_rag_evaluation(
            dataset=args.dataset, categories=args.category or [app.EVAL_ALL_CATEGORIES],
            sample_size=args.samples, seed=args.seed, custom_questions="",
            channel_labels=[CHANNELS[channel] for channel in args.channels],
            metric_names=["答案相关性", "上下文相关性", "有据性", "参考答案一致性"],
            role_label="医生", doc_type_label="不限制", use_rewrite=args.rewrite,
            experiment_label=args.label, run_source="cli",
            doc_hyde=getattr(args, "hyde", True), doc_rerank=getattr(args, "rerank", True), doc_chunk_strategy=getattr(args, "chunking", "original"),
            doc_parent=getattr(args, "parent", True), doc_hybrid=getattr(args, "hybrid", True),
            doc_top_k=getattr(args, "top_k", 20), doc_context_top_k=getattr(args, "context_top_k", 5),
        ):
            final = update
            lines = update[1].splitlines()
            for line in lines[printed_lines:]:
                print(line, flush=True)
            printed_lines = len(lines)
        if final is None:
            return 1
        print(final[0], flush=True)
        if final[3].empty:
            return 1
        print(final[2].to_string(index=False), flush=True)
        print("完整结果请通过 TruLens Dashboard 查看（PostgreSQL: trulens_eval）。", flush=True)
        return 0 if (
            (final[3]["评估状态"] == "complete").all()
            and not final[3]["降级"].any()
            and (final[3]["持久化状态"] == "saved").all()
        ) else 1
    finally:
        app._clear_tracked_apps()
        if app._eval_deps.cache_info().currsize:
            deps = app._eval_deps()
            await deps.neo4j_driver.close()
            deps.milvus_client.close()
            app._eval_deps.cache_clear()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
