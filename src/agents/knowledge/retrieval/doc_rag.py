# src/agents/knowledge/retrieval/doc_rag.py
"""
文档 RAG：基于 Milvus 的文档片段检索与问答生成。

流程：问题向量化（可选 HyDE）→ Milvus dense / BM25 混合召回 → Reranker 精排 → LLM 生成带引用回答
"""

from __future__ import annotations

import asyncio
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from src.agents.knowledge.prompts import DOC_QA_PROMPT
from src.agents.knowledge.retrieval.diagnostics import report_issue
from src.agents.knowledge.retrieval.options import DocRetrievalOptions
from src.agents.knowledge.retrieval.variant_schema import ORIGINAL_FILTER, strategy_filter

# 文档切块后的向量集合，字段：doc_name / doc_type / page_number / chunk_index / text / 向量
COLLECTION_NAME = "knowledge_docs"


async def search_docs_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel | None,
    options: DocRetrievalOptions,
    doc_type: str | None = None,
) -> list[dict]:
    """按 options 检索文档；索引能力由评估入口预检，不在每次查询时检查。

    HyDE 只影响 dense 查询向量，BM25 始终使用传入的问题文本。
    score 为 COSINE 或 RRF 分数，不能跨两种检索模式直接比较。
    返回值保留文档名、页码、正文与可选的 rerank_score；检索失败时记录诊断并返回空列表。
    """
    top_k, rerank_top_k = options.top_k, options.context_top_k
    if options.use_hyde and llm is None:
        raise ValueError("启用 HyDE 时必须提供生成模型")
    # 1. 生成查询向量
    if options.use_hyde:
        from src.agents.knowledge.retrieval.hyde import generate_hyde_embedding
        query_vec = await generate_hyde_embedding(question, llm, embedding_model)
    else:
        query_vec = await embedding_model.aembed_query(question)

    # 2. Milvus dense 或 dense + BM25 混合检索
    # nprobe：IVF 类索引查询时扫描的聚类桶数，越大越准但越慢
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    filter_expr = (strategy_filter(options.chunk_strategy)
                   if options.chunk_strategy != "original" else ORIGINAL_FILTER)
    if doc_type:
        filter_expr += f" and doc_type == {json.dumps(doc_type, ensure_ascii=False)}"
    fields = ["doc_name", "doc_type", "page_number", "chunk_index", "text"]
    if options.expand_parents:
        fields += ["parent_id", "parent_text"]
    try:
        if options.use_hybrid:
            dense_request = AnnSearchRequest(
                data=[query_vec], anns_field="embedding", param=search_params,
                limit=top_k, expr=filter_expr,
            )
            sparse_request = AnnSearchRequest(
                data=[question], anns_field="sparse_embedding", param={"metric_type": "BM25"},
                limit=top_k, expr=filter_expr,
            )
            results = await asyncio.to_thread(
                milvus_client.hybrid_search, collection_name=options.collection_name,
                reqs=[dense_request, sparse_request], ranker=RRFRanker(60),
                limit=top_k, output_fields=fields,
            )
        else:
            results = await asyncio.to_thread(
                milvus_client.search, collection_name=options.collection_name, data=[query_vec],
                anns_field="embedding", limit=top_k, output_fields=fields,
                search_params=search_params, filter=filter_expr,
            )
    except Exception as e:
        logger.warning(f"文档检索失败: {e}")
        report_issue("milvus_query_failed", fatal=True)
        return []

    # search 支持批量查询，返回值按查询分组；这里只传了一个向量，取 results[0]
    if not results or not results[0]:
        return []

    # dense 返回 COSINE 相似度，hybrid 返回 RRF 融合分数；均为越大越相关。
    hits = [
        {**hit["entity"], "score": hit.get("distance", 0.0)}
        for hit in results[0]
    ]

    # 无重排时仍限制上下文预算，避免“关闭重排”同时增加生成输入量。
    if options.use_rerank:
        from src.agents.knowledge.retrieval.reranker import rerank_docs
        hits = await rerank_docs(question, hits, top_k=rerank_top_k)
    else:
        hits = hits[:rerank_top_k]
    if options.expand_parents:
        expanded, seen = [], set()
        for hit in hits:
            parent_id, parent_text = hit.get("parent_id"), hit.get("parent_text")
            if not parent_id or not parent_text:
                raise ValueError("命中片段缺少父块，拒绝静默退化为普通检索")
            if parent_id not in seen:
                expanded.append({**hit, "child_text": hit["text"], "text": parent_text})
                seen.add(parent_id)
        hits = expanded
    return hits


EMPTY_ANSWER = "当前知识库中未找到与您问题相关的文档内容。"


def doc_context_parts(hits: list[dict]) -> list[str]:
    """生成和逐片段评估使用完全相同的引用格式。"""
    return [f"片段{i} [{hit['doc_name']}, 第{hit.get('page_number', '?')}页]:\n{hit['text']}"
            for i, hit in enumerate(hits, 1)]


def format_doc_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(doc_context_parts(hits))


async def answer_docs(question: str, hits: list[dict], llm: BaseChatModel,
                      role: str = "patient") -> str:
    if not hits:
        return EMPTY_ANSWER
    prompt = DOC_QA_PROMPT.format(question=question, context=format_doc_context(hits), role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.text


async def search_docs(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    role: str = "patient",
    use_hyde: bool = True,
    options: DocRetrievalOptions | None = None,
) -> str:
    """
    文档 RAG 检索 + Reranker 精排 + HyDE 增强 + 生成回答。

    混合检索默认开启；top_k / rerank_top_k 分别映射为召回与上下文预算。
    use_hyde 控制假设回答向量化；role 用于调整回答的专业深度。
    返回 LLM 生成的回答文本，未检索到内容时直接返回固定提示，不调用 LLM。
    """
    hits = await search_docs_raw(
        question, embedding_model, milvus_client,
        llm=llm, doc_type=doc_type,
        options=options or DocRetrievalOptions(
            top_k=top_k, context_top_k=rerank_top_k, use_hyde=use_hyde,
        ),
    )
    return await answer_docs(question, hits, llm, role)
