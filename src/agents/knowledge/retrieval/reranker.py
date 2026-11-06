# src/agents/knowledge/retrieval/reranker.py
"""
重排（Rerank）：对向量粗召回的候选文档做二次精排。

向量检索是双塔结构，query 与文档分别编码，速度快但精度有限；
Reranker 是交叉编码器，把 query 和每篇文档拼接后联合打分，精度更高但更慢，
因此只对粗召回的少量候选（如 Top-20）做精排，再取 Top-K 交给 LLM。
"""

from __future__ import annotations

import asyncio
from loguru import logger

from src.core.config import get_settings
from src.agents.knowledge.retrieval.diagnostics import report_issue

settings = get_settings()


async def rerank_docs(
    query: str,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    使用 DashScope Reranker 对检索结果精排。
    输入：query + 粗检索结果列表（每个元素含 "text" 字段）
    输出：按相关性重排后的 top_k 结果，每个元素额外带 "rerank_score" 字段

    任何异常（未安装 SDK、接口报错、网络异常）都会降级为按原向量检索顺序截取 top_k，
    保证重排失败不阻断问答流程。
    """
    if not documents:
        return []

    # 候选数不超过 top_k 时重排不会减少结果，直接返回以省去一次 API 调用
    if len(documents) <= top_k:
        return documents

    try:
        # 延迟导入：dashscope 为可选依赖，未安装时走下方 ImportError 降级分支
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = settings.DASHSCOPE_API_KEY

        texts = [doc.get("text", "") for doc in documents]
        # return_documents=False：只返回下标和分数，原文从本地 documents 中按下标取回，减少传输量
        # SDK 提供同步接口；在线程中等待重排响应，让其他协程继续执行。
        response = await asyncio.to_thread(
            TextReRank.call,
            model="qwen3.7-text-rerank",
            query=query,
            documents=texts,
            top_n=top_k,
            return_documents=False,
        )

        if response.status_code != 200:
            logger.warning(f"Reranker 调用失败: {response.message}")
            report_issue("rerank_fallback")
            return documents[:top_k]

        # 结果已按 relevance_score 降序排列，item.index 对应输入 documents 的下标
        reranked = []
        for item in response.output.results:
            idx = item.index
            doc = documents[idx].copy()  # 浅拷贝，避免修改调用方传入的原始数据
            doc["rerank_score"] = item.relevance_score
            reranked.append(doc)

        return reranked

    except ImportError:
        logger.warning("dashscope 未安装，回退到向量距离排序")
        report_issue("rerank_unavailable")
        return documents[:top_k]
    except Exception as e:
        logger.warning(f"Reranker 异常，回退到向量距离排序: {e}")
        report_issue("rerank_fallback")
        return documents[:top_k]