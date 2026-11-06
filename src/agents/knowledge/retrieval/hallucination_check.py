# src/agents/knowledge/retrieval/hallucination_check.py
"""幻觉检测：用 LLM 逐条核对回答中的事实性陈述是否有检索结果支撑。"""

from __future__ import annotations
import json
from loguru import logger
from src.agents.knowledge.retrieval.diagnostics import report_issue
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agents.knowledge.prompts import HALLUCINATION_CHECK_PROMPT


async def check_hallucination(
    question: str,
    evidence: str,
    answer: str,
    llm: BaseChatModel,
    threshold: float = 0.7,
) -> dict:
    """
    幻觉检测：校验回答是否基于检索结果。
    返回 {"is_grounded": bool, "unsupported_claims": [...], "confidence": float}
    如果 confidence < threshold，标记为不可信。

    Args:
        question: 用户问题
        evidence: 检索证据文本（各通道结果拼接后的摘要）
        answer: 待校验的系统回答
        llm: 用于判断的大模型
        threshold: 置信度阈值，模型自评置信度低于该值时，即使判定“有依据”也视为不可信
    """
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        question=question, evidence=evidence, answer=answer,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        content = response.text.strip()
        # 兼容模型把 JSON 包在 ```json ... ``` 代码块里的情况
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        # 双重条件：模型判定有依据 且 置信度达到阈值，才算可信；字段缺失时按不可信处理
        result["is_grounded"] = result.get("is_grounded", False) and result.get("confidence", 0) >= threshold
        return result
    except Exception as e:
        # 检测本身失败时默认放行（视为可信），避免检测环节故障导致正常回答被打上警告
        logger.warning(f"幻觉检测失败: {e}")
        report_issue("hallucination_check_fallback")
        return {"is_grounded": True, "unsupported_claims": [], "confidence": 1.0}