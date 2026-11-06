# src/agents/knowledge/retrieval/hyde.py
"""HyDE 增强检索：用 LLM 生成的“假设答案”向量替代原始问题向量，缓解问句与文档表述不一致的问题。"""

from __future__ import annotations
from loguru import logger
from src.agents.knowledge.retrieval.diagnostics import report_issue
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.agents.knowledge.prompts import HYDE_PROMPT


async def generate_hyde_embedding(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
) -> list[float]:
    """
    HyDE（Hypothetical Document Embeddings）：
    1. LLM 生成假设性回答
    2. 对假设回答做向量化
    3. 用假设回答的向量去检索（比原始问题的向量召回率更高）

    原理：知识库里存的是“陈述句式”的答案文本，短问句与其在向量空间距离较远；
    假设回答即使细节不准，表述风格和用词也更接近真实文档。

    Args:
        question: 用户问题（建议传入改写后的规范化查询）
        llm: 用于生成假设回答的大模型
        embedding_model: 嵌入模型，必须与建库时使用的模型一致，否则向量不可比

    Returns:
        查询向量。LLM 调用失败时回退为原始问题的向量，不影响检索流程。
    """
    prompt = HYDE_PROMPT.format(question=question)
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        hypothetical_doc = response.text.strip()
        logger.debug(f"HyDE 假设文档: {hypothetical_doc[:100]}...")
        # 注意：假设文档只用于生成向量，不会进入最终回答的上下文，避免把编造内容带给用户
        return await embedding_model.aembed_query(hypothetical_doc)
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始查询向量: {e}")
        report_issue("hyde_fallback")
        return await embedding_model.aembed_query(question)