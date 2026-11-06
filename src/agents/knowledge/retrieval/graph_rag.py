# src/agents/knowledge/retrieval/graph_rag.py
"""
GraphRAG：基于 Neo4j 医学知识图谱的检索与问答。

流程：LLM 实体提取 → LLM 生成 Cypher（执行报错时带上错误信息重试）→ 执行查询 → LLM 整合为自然语言回答。
适合“某疾病有哪些症状 / 用什么药 / 挂哪个科”这类实体关系、多跳推理问题。
"""

from __future__ import annotations
import json
from typing import LiteralString, cast
from loguru import logger
from src.agents.knowledge.retrieval.diagnostics import report_issue
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from neo4j import AsyncDriver, READ_ACCESS

from src.agents.knowledge.prompts import (
    ENTITY_EXTRACT_PROMPT, NL2CYPHER_PROMPT, GRAPH_QA_PROMPT,
)

MAX_CYPHER_RETRIES = 2  # Cypher 执行报错后的最大重试次数（总尝试次数 = MAX_CYPHER_RETRIES + 1）


async def _extract_entities(question: str, llm: BaseChatModel) -> dict:
    """
    从问题中提取医学实体，作为生成 Cypher 的提示，帮助模型定位图谱中的起始节点。

    Returns:
        {"diseases": [...], "symptoms": [...], "drugs": [...], "departments": [...], "checks": [...]}
        解析失败时返回各类别均为空的字典，不阻断后续 Cypher 生成。
    """
    prompt = ENTITY_EXTRACT_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.text.strip()
        # 兼容模型把 JSON 包在 ```json ... ``` 代码块里的情况
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        report_issue("entity_parse_fallback")
        return {"diseases": [], "symptoms": [], "drugs": [], "departments": [], "checks": []}


async def _generate_cypher(
    question: str, entities: dict, llm: BaseChatModel, error_hint: str = "",
) -> str:
    """调用 LLM 生成 Cypher；error_hint 为上一次执行的报错信息，用于让模型自我修正。"""
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 Cypher 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2CYPHER_PROMPT.format(
        question=question,
        entities=json.dumps(entities, ensure_ascii=False),
    ) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    cypher = response.text.strip()
    # 兼容模型把 Cypher 包在 ```cypher ... ``` 代码块里的情况
    if "```" in cypher:
        cypher = cypher.split("```")[1].lstrip("cypher").strip()
    return cypher


async def _execute_cypher(cypher: str, neo4j_driver: AsyncDriver) -> list[dict]:
    """
    执行 Cypher 并以字典列表形式返回结果；执行报错时异常向上抛出，由调用方决定是否重试。

    使用只读会话执行：LLM 生成的语句若包含 CREATE / DELETE / SET 等写操作，会被 Neo4j 服务端拒绝。
    """
    if not cypher:
        return []
    async with neo4j_driver.session(default_access_mode=READ_ACCESS) as session:
        # neo4j 驱动把参数标注为 LiteralString，提醒调用方用参数化查询防注入；
        # 这里的语句由 LLM 动态生成，无法参数化，安全性由只读会话兜底，因此显式 cast
        result = await session.run(cast(LiteralString, cypher))
        # data() 把每条 Record 转成 {返回列名: 值} 的字典
        return await result.data()


async def search_graph_raw(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> list[dict]:
    """
    GraphRAG 检索，返回原始图谱查询结果（不经过 LLM 生成）。
    供 fusion 多通道融合直接使用原始数据。

    Returns:
        图谱查询结果，最多 20 条；重试耗尽或无结果时返回空列表。
    """
    # 1. 实体提取（只做一次，重试时复用）
    entities = await _extract_entities(question, llm)
    logger.info(f"GraphRAG 实体提取: {entities}")

    # 2. 生成并执行 Cypher，报错时把错误信息反馈给 LLM 重新生成
    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)
        logger.info(f"GraphRAG Cypher (attempt {attempt + 1}): {cypher}")
        try:
            records = await _execute_cypher(cypher, neo4j_driver)
            # 与 NL2CYPHER_PROMPT 中“最多 20 条”的约定一致，防止模型漏写 LIMIT 时结果过多撑爆上下文
            return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                report_issue("cypher_query_failed", fatal=True)
                return []
    return []


async def search_graph(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
    role: str = "patient",
) -> str:
    """
    GraphRAG 完整流程：实体提取 → NL2Cypher（带重试）→ LLM 整合。

    role 用于调整回答的专业深度；未查到结果时直接返回固定提示，不调用 LLM。
    """
    records = await search_graph_raw(question, neo4j_driver, llm)

    return await answer_graph(question, records, llm, role)


EMPTY_ANSWER = "知识图谱中未找到与您问题相关的信息。"


def format_graph_context(records: list[dict]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2, default=str)


async def answer_graph(question: str, records: list[dict], llm: BaseChatModel,
                       role: str = "patient") -> str:
    if not records:
        return EMPTY_ANSWER

    # 3. 将图谱结果序列化后交给 LLM 组织成自然语言回答
    graph_result = format_graph_context(records)
    prompt = GRAPH_QA_PROMPT.format(
        question=question, graph_result=graph_result, role=role,
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.text
