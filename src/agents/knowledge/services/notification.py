# src/agents/knowledge/services/notification.py
"""
知识文档更新通知：文档上传/更新/删除时写入通知记录，相关科室按分类拉取未读通知。

采用“写表 + 轮询查询”的方式，不做实时推送；表结构见 src/agents/knowledge/model.py 中的 KnowledgeNotification。
"""

from __future__ import annotations
from datetime import datetime
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def notify_doc_update(
    db: AsyncSession,
    doc_name: str,
    doc_type: str,
    category: str,
    action: str = "upload",
) -> int:
    """
    记录知识文档更新通知。
    文档上传/更新/删除时调用，写入通知表供相关科室查询。

    Args:
        db: 异步数据库会话，函数内部会 commit
        doc_name: 文档名称
        doc_type: 文档类型（如 pdf / docx）
        category: 文档所属分类，科室按此字段过滤自己关心的通知
        action: 操作类型，upload / update / delete

    Returns:
        新插入通知的 ID；未取到返回行时为 0。
    """
    # 新通知一律为未读；RETURNING id 为 PostgreSQL 语法，插入的同时拿回自增 ID
    result = await db.execute(
        text("""
            INSERT INTO knowledge_notifications
                (doc_name, doc_type, category, action, is_read, created_at)
            VALUES
                (:doc_name, :doc_type, :category, :action, false, :created_at)
            RETURNING id
        """),
        {
            "doc_name": doc_name,
            "doc_type": doc_type,
            "category": category,
            "action": action,
            "created_at": datetime.now(),
        },
    )
    await db.commit()
    row = result.fetchone()
    nid = row[0] if row else 0
    logger.info(f"知识更新通知 | action={action} | doc={doc_name} | category={category}")
    return nid


async def get_unread_notifications(
    db: AsyncSession,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    获取未读的知识更新通知。

    Args:
        db: 异步数据库会话
        category: 只查指定分类；为 None 时查询全部分类
        limit: 最多返回条数，按创建时间倒序取最新的

    Returns:
        通知列表；created_at 转为字符串，便于直接 JSON 序列化返回给前端。
    """
    # 两个分支只差 category 条件，分开写可以保持 SQL 为静态文本，不做字符串拼接
    if category:
        result = await db.execute(
            text("""
                SELECT id, doc_name, doc_type, category, action, created_at
                FROM knowledge_notifications
                WHERE is_read = false AND category = :category
                ORDER BY created_at DESC LIMIT :limit
            """),
            {"category": category, "limit": limit},
        )
    else:
        result = await db.execute(
            text("""
                SELECT id, doc_name, doc_type, category, action, created_at
                FROM knowledge_notifications
                WHERE is_read = false
                ORDER BY created_at DESC LIMIT :limit
            """),
            {"limit": limit},
        )
    rows = result.fetchall()
    # 按 SELECT 列顺序取值
    return [
        {
            "id": r[0], "doc_name": r[1], "doc_type": r[2],
            "category": r[3], "action": r[4],
            "created_at": str(r[5]),
        }
        for r in rows
    ]


async def mark_notifications_read(db: AsyncSession, ids: list[int]) -> None:
    """
    标记通知为已读。

    ids 为空时直接返回，省去一次无意义的数据库调用。
    注意：已读状态是全局的，不区分用户，一个人标记后其他人也看不到这条未读通知。
    """
    if not ids:
        return
    # ANY(:ids) 为 PostgreSQL 数组语法，列表作为单个数组参数绑定，避免拼接 IN (...)
    # 原生 SQL 不会触发 ORM 的 onupdate，因此 updated_at 需要手动更新
    await db.execute(
        text("""
            UPDATE knowledge_notifications
            SET is_read = true, updated_at = :updated_at
            WHERE id = ANY(:ids)
        """),
        {"ids": ids, "updated_at": datetime.now()},
    )
    await db.commit()
