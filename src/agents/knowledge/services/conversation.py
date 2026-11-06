# src/agents/knowledge/services/conversation.py
"""
知识问答多轮对话上下文：基于 Redis 保存最近几轮问答，供追问时拼进 Prompt。

与 Supervisor 的 checkpointer 短期记忆相互独立，只服务于知识问答 Agent，
按 user_id + session_id 隔离，过期后自动清除，不做持久化。

存储结构为 Redis List，每个元素是一条 JSON 序列化的消息，按时间顺序从右侧追加。
"""

from __future__ import annotations
import json
from loguru import logger


# 上下文过期时间（秒）：30 分钟无新对话即清空；每次保存都会重置过期时间
CONTEXT_TTL = 1800
# Redis 中最多保留的消息条数（一问一答算 2 条，即最近 5 轮）
MAX_HISTORY = 10


def _context_key(user_id: str, session_id: str) -> str:
    """知识对话上下文的 Redis key。"""
    return f"knowledge_ctx:{user_id}:{session_id}"


def _dump(turn: dict) -> str:
    # ensure_ascii=False：中文按原文存储，便于在 Redis 中直接排查
    return json.dumps(turn, ensure_ascii=False)


async def load_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """
    从 Redis 加载知识对话上下文。

    Args:
        redis_client: 异步 Redis 客户端
        user_id / session_id: 共同组成 Redis key，保证不同用户、不同会话互不串扰

    Returns:
        消息列表，每个元素形如 {"role": "user" | "assistant", "content": "..."}；
        key 不存在、已过期或读取异常时返回空列表，不阻断问答流程。
    """
    key = _context_key(user_id, session_id)
    try:
        raw_items = await redis_client.lrange(key, 0, -1)
        return [json.loads(item) for item in raw_items]
    except Exception as e:
        # Redis 不可用或数据损坏时降级为无上下文的单轮问答
        logger.warning(f"加载知识对话上下文失败: {e}")
    return []


async def save_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
    history: list[dict],
) -> None:
    """
    用给定历史整体覆盖 Redis 中的知识对话上下文。

    只保留最近 MAX_HISTORY 条消息，并设置 CONTEXT_TTL 过期时间；
    写入失败只记录告警，不影响本轮回答返回给用户。
    追加新一轮对话请用 append_turn，避免并发时覆盖其他请求写入的内容。
    """
    key = _context_key(user_id, session_id)
    # 截断旧消息，控制 Redis 占用和后续 Prompt 长度
    trimmed = history[-MAX_HISTORY:]
    try:
        # MULTI/EXEC 事务：删除 + 写入 + 设置过期一次性执行，读方不会看到写了一半的列表
        pipe = redis_client.pipeline(transaction=True)
        pipe.delete(key)
        if trimmed:
            pipe.rpush(key, *[_dump(turn) for turn in trimmed])
            pipe.expire(key, CONTEXT_TTL)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"保存知识对话上下文失败: {e}")


async def append_turn(
    redis_client,
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
) -> list[dict]:
    """
    追加一轮对话到上下文，返回更新后的历史。

    RPUSH + LTRIM + EXPIRE + LRANGE 在同一个 MULTI/EXEC 事务中执行，
    同一会话并发追加时各轮都会保留，不会出现“读 → 改 → 写”互相覆盖丢记录的问题。

    Returns:
        追加并截断后的历史（最多 MAX_HISTORY 条）；Redis 异常时只返回本轮的两条消息。
    """
    key = _context_key(user_id, session_id)
    new_turns = [
        {"role": "user", "content": question},
        # 回答只保留前 500 字：追问时只需知道上一轮大意，完整回答会让 Prompt 过长
        {"role": "assistant", "content": answer[:500]},
    ]
    try:
        pipe = redis_client.pipeline(transaction=True)
        pipe.rpush(key, *[_dump(turn) for turn in new_turns])
        # 只保留末尾 MAX_HISTORY 条，丢弃最早的消息
        pipe.ltrim(key, -MAX_HISTORY, -1)
        pipe.expire(key, CONTEXT_TTL)
        pipe.lrange(key, 0, -1)
        *_, raw_items = await pipe.execute()
        return [json.loads(item) for item in raw_items]
    except Exception as e:
        logger.warning(f"追加知识对话上下文失败: {e}")
        return new_turns


def format_conversation_context(history: list[dict]) -> str:
    """
    将对话历史格式化为 Prompt 可用的字符串。

    只取最近 6 条消息（3 轮），比 Redis 中保留的更少，进一步控制 Prompt 长度；
    无历史时返回空字符串，调用方可直接拼接到 Prompt 前面而无需判断。
    """
    if not history:
        return ""
    parts = []
    for turn in history[-6:]:
        role = "用户" if turn["role"] == "user" else "助手"
        parts.append(f"{role}：{turn['content']}")
    # 末尾两个换行，与后面拼接的正文 Prompt 隔开
    return "以下是之前的对话上下文：\n" + "\n".join(parts) + "\n\n"
