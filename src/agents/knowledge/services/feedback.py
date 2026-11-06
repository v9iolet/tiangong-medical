# src/agents/knowledge/services/feedback.py
"""
知识问答反馈：记录用户对回答的点赞/踩及文字意见，并提供汇总统计。

反馈数据用于评估各检索通道的回答质量，定位需要补充或修正的知识文档。
表结构见 src/agents/knowledge/model.py 中的 KnowledgeFeedback。
"""

from __future__ import annotations
from datetime import datetime
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def save_feedback(
        db: AsyncSession,
        user_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
        intent: str = "",
        channels: str = "",
) -> int:
    """
    保存用户对知识问答的反馈（点赞/踩）。
    rating: 1=有用, -1=无用, 0=未评价
    返回反馈记录 ID。

    Args:
        db: 异步数据库会话，函数内部会 commit
        user_id: 反馈用户 ID
        question / answer: 被评价的问答，入库前按字段长度截断
        rating: 评价结果，取值见上
        comment: 用户补充的文字意见
        intent: 本次问答的检索意图（路由结果），便于按意图统计满意度
        channels: 本次问答使用的检索通道，多个通道由调用方拼成字符串传入

    Returns:
        新插入记录的 ID；未取到返回行时为 0。
    """

    # 写入 knowledge_feedback 表
    # 截断长度与 KnowledgeFeedback 字段长度一致，避免超长导致插入失败
    # created_at 显式传入：表的 server_default 也能兜底，这里统一使用应用服务器时间
    # RETURNING id 为 PostgreSQL 语法，插入的同时拿回自增 ID
    result = await db.execute(
        text("""
             INSERT INTO knowledge_feedback
             (user_id, question, answer_preview, rating, comment, intent, channels, created_at)
             VALUES (:user_id, :question, :answer_preview, :rating, :comment, :intent, :channels,
                     :created_at) RETURNING id
             """),
        {
            "user_id": user_id,
            "question": question[:500],
            "answer_preview": answer[:500],
            "rating": rating,
            "comment": comment[:1000],
            "intent": intent,
            "channels": channels,
            "created_at": datetime.now(),
        },
    )
    await db.commit()
    row = result.fetchone()
    feedback_id = row[0] if row else 0
    # 同时写一条审计日志（audit=True 标记，会单独落到 logs/audit_*.log）
    logger.bind(audit=True).info(
        f"knowledge_feedback | user={user_id} | rating={rating} | "
        f"question={question[:60]} | comment={comment[:60]}"
    )
    return feedback_id


async def get_feedback_stats(db: AsyncSession) -> dict:
    """
    获取反馈统计数据。

    全表统计，未按时间或用户过滤；未评价（rating=0）的记录只计入 total。
    """

    # 返回 {"total": N, "positive": N, "negative": N}
    # COUNT(*) FILTER (WHERE ...) 为 PostgreSQL 聚合过滤语法，一次扫描同时得到三个计数
    result = await db.execute(text("""
                                   SELECT COUNT(*) AS total,
                                          COUNT(*)    FILTER (WHERE rating = 1) AS positive, COUNT(*) FILTER (WHERE rating = -1) AS negative
                                   FROM knowledge_feedback
                                   """))
    row = result.fetchone()
    if row:
        return {"total": row[0], "positive": row[1], "negative": row[2]}
    return {"total": 0, "positive": 0, "negative": 0}
