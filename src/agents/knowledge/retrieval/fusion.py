"""多通道共享业务链路：收集证据与诊断，再统一生成并检查幻觉。"""
from __future__ import annotations

import asyncio
import json

from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.knowledge.prompts import FUSION_PROMPT
from src.agents.knowledge.retrieval.diagnostics import observe_source
from src.agents.knowledge.retrieval.evidence import Evidence
from src.agents.knowledge.retrieval.options import DocRetrievalOptions

EMPTY_ANSWER = "所有检索通道均未找到与您问题相关的信息。"


async def collect_sources(
    question: str, llm: BaseChatModel, embedding_model: Embeddings,
    milvus_client: MilvusClient, neo4j_driver: AsyncDriver,
    db_session: AsyncSession | None = None, channels: list[str] | None = None,
    doc_options: DocRetrievalOptions | None = None, doc_type: str | None = None,
) -> Evidence:
    """并发检索；失败通道留诊断，不阻止其他通道作为来源。"""
    from src.agents.knowledge.retrieval import doc_rag, graph_rag, nl2sql

    channels = ["doc_rag", "graph_rag"] if channels is None else channels
    tasks = {}
    if "doc_rag" in channels:
        tasks["doc_rag"] = doc_rag.search_docs_raw(
            question, embedding_model, milvus_client, llm=llm,
            options=doc_options or DocRetrievalOptions(), doc_type=doc_type,
        )
    if "graph_rag" in channels:
        tasks["graph_rag"] = graph_rag.search_graph_raw(question, neo4j_driver, llm)
    if "nl2sql" in channels and db_session is not None:
        tasks["nl2sql"] = nl2sql.search_sql_raw(question, llm, db_session)

    evidence = Evidence(question)
    parts = []
    results = await asyncio.gather(*(observe_source(call) for call in tasks.values()))
    for name, (value, status) in zip(tasks, results):
        evidence.sources[name] = status.to_dict()
        if status.status not in {"ok", "partial"} or not value:
            continue
        if name == "doc_rag":
            evidence.contexts.extend(doc_rag.doc_context_parts(value))
            parts.append("### 文档检索结果\n" + doc_rag.format_doc_context(value))
        elif name == "graph_rag":
            text = graph_rag.format_graph_context(value)
            evidence.contexts.append(text)
            parts.append("### 知识图谱检索结果\n" + text)
        else:
            sql, rows = value
            if not rows:
                continue
            text = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
            evidence.contexts.append(f"执行的 SQL：{sql}\n查询结果：\n{text}")
            parts.append(f"### 运营数据查询结果\n执行的 SQL：{sql}\n{text}")
        evidence.used_sources.append(name)
    evidence.prompt_kwargs = {"sources": "\n\n".join(parts)}
    return evidence


async def answer_fusion(question: str, evidence: Evidence, llm: BaseChatModel,
                        role: str = "patient") -> str:
    if not evidence.contexts:
        return EMPTY_ANSWER
    sources = evidence.prompt_kwargs["sources"]
    prompt = FUSION_PROMPT.format(question=question, sources=sources, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    answer = response.text
    from src.agents.knowledge.retrieval.hallucination_check import check_hallucination
    result, status = await observe_source(check_hallucination(question, sources, answer, llm))
    evidence.sources["hallucination_check"] = status.to_dict()
    if result is not None and not result["is_grounded"]:
        claims = "、".join(result.get("unsupported_claims", []))
        answer += f"\n\n⚠️ 提示：以下内容未在检索结果中找到充分依据，请谨慎参考：{claims}"
    return answer


async def multi_channel_search(
    question: str, llm: BaseChatModel, embedding_model: Embeddings,
    milvus_client: MilvusClient, neo4j_driver: AsyncDriver,
    db_session: AsyncSession | None = None, channels: list[str] | None = None,
    role: str = "patient", doc_options: DocRetrievalOptions | None = None,
    doc_type: str | None = None,
) -> str:
    evidence = await collect_sources(question, llm, embedding_model, milvus_client, neo4j_driver,
                                     db_session, channels, doc_options, doc_type)
    return await answer_fusion(question, evidence, llm, role)
