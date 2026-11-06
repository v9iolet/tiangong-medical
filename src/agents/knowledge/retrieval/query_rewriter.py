# src/agents/knowledge/retrieval/query_rewriter.py
"""Query 理解层：将用户口语化提问改写为适合检索的规范化查询，并识别问题意图。"""

from __future__ import annotations
import json
from loguru import logger
from src.agents.knowledge.retrieval.diagnostics import report_issue
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agents.knowledge.prompts import QUERY_REWRITE_PROMPT


async def rewrite_query(
    question: str,
    llm: BaseChatModel,
    role: str = "patient",
) -> dict:
    """
    Query 改写：口语 → 医学术语，复杂问题拆分为子查询。

    Args:
        question: 用户原始问题
        llm: 用于改写的大模型
        role: 用户角色（patient / doctor / pharmacist 等），影响改写措辞

    Returns:
        {"queries": [...], "intent": "..."}
        解析失败时降级为原问题 + knowledge_qa 意图，保证下游检索始终可用。
    """
    prompt = QUERY_REWRITE_PROMPT.format(question=question, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.text.strip()
        # 兼容模型把 JSON 包在 ```json ... ``` 代码块里的情况：取第一个代码块内容并去掉语言标记
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        # 字段缺失时兜底：没有改写结果就用原问题检索，没有意图就按通用知识问答处理
        if "queries" not in result or not result["queries"]:
            result["queries"] = [question]
        if "intent" not in result:
            result["intent"] = "knowledge_qa"
        return result
    except Exception as e:
        logger.warning(f"Query 改写失败: {e}")
        report_issue("rewrite_fallback")
        return {"queries": [question], "intent": "knowledge_qa"}

async def prepare_query(question: str, llm: BaseChatModel, role: str = "patient",
                        use_rewrite: bool = True) -> str:
    """统一改写入口；关闭时不调用模型，开启时取首条检索 query。"""
    if not use_rewrite:
        return question
    result = await rewrite_query(question, llm, role)
    return result["queries"][0] if result.get("queries") else question
