# src/agents/knowledge/retrieval/nl2sql.py
"""
NL2SQL：自然语言问题 → LLM 生成 SQL → 安全校验 → 执行（失败带报错重试）→ LLM 总结回答。
"""

from __future__ import annotations
import asyncio
import re
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.agents.knowledge.prompts import NL2SQL_PROMPT, SQL_QA_PROMPT
from src.agents.knowledge.retrieval.evidence import Evidence
from src.agents.knowledge.retrieval.diagnostics import SourceStatus

MAX_SQL_RETRIES = 2       # 执行报错后的最大重试次数（总尝试次数 = MAX_SQL_RETRIES + 1）
SQL_TIMEOUT_SECONDS = 10  # 单次 SQL 执行超时
MAX_ROWS = 100            # 返回结果最大行数

# 代码层安全兜底：prompt 中的规则只约束模型，这里强制拦截写操作和敏感字段
FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"\bpatients\b[^;]*\b(phone|id_card)\b", re.IGNORECASE),
]


class SQLQueryError(Exception):
    """NL2SQL 查询失败，提示语与稳定错误码分别保存。"""

    def __init__(self, message: str, code: str = "query_failed"):
        super().__init__(message)
        self.code = code


def _validate_sql(sql: str) -> tuple[bool, str]:
    """
    SQL 安全校验。

    Returns:
        (True, 补全 LIMIT 后的 SQL) 或 (False, 拦截原因)
    """
    stripped = sql.strip().rstrip(";")
    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作或字段"
    if "LIMIT" not in stripped.upper():
        stripped += f" LIMIT {MAX_ROWS}"
    return True, stripped


async def _generate_sql(
    question: str, llm: BaseChatModel, error_hint: str = "",
) -> str:
    """调用 LLM 生成 SQL；error_hint 为上一次执行的报错信息，用于让模型自我修正。"""
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2SQL_PROMPT.format(question=question) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    # content 类型为 str | list，.text 统一取出纯文本
    sql = response.text.strip()
    # 兼容模型把 SQL 包在 ```sql ... ``` 代码块里的情况
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql


async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> tuple[str, list[dict]]:
    """
    NL2SQL 检索：LLM 生成 SQL → 安全校验 → 执行（报错时带上错误信息重试）。

    Returns:
        (实际执行的 SQL, 查询结果行列表)。结果为空时返回空列表。

    Raises:
        SQLQueryError: 安全校验不通过、执行超时或重试耗尽，异常信息可直接展示给用户。
    """
    error_hint = ""
    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(question, llm, error_hint)
        logger.info(f"NL2SQL SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated = _validate_sql(raw_sql)
        if not valid:
            logger.warning(f"SQL 安全校验失败: {validated}")
            raise SQLQueryError(f"查询被安全策略拦截：{validated}。请换一种方式提问。", code="blocked")

        try:
            query_result = await asyncio.wait_for(
                db.execute(text(validated)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(f"SQL 执行超时 ({SQL_TIMEOUT_SECONDS}s): {validated}")
            raise SQLQueryError(f"查询执行超时（{SQL_TIMEOUT_SECONDS}秒），请简化查询条件后重试。", code="timeout")
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            # PostgreSQL 语句报错后事务进入 aborted 状态，不回滚的话后续重试都会失败
            await db.rollback()
            continue

        rows = query_result.mappings().all()
        return validated, [dict(row) for row in rows[:MAX_ROWS]]

    raise SQLQueryError("数据查询执行失败，请尝试换一种方式提问。")


async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> str:
    """NL2SQL 检索 + LLM 将查询结果总结为自然语言回答。"""
    evidence = await collect_sql(question, llm, db)
    if evidence.fallback is not None:
        return evidence.fallback
    return await answer_sql(question, evidence.data, llm)


EMPTY_ANSWER = "未查询到相关数据。"


async def answer_sql(question: str, result: tuple[str, list[dict]] | None, llm: BaseChatModel) -> str:
    sql, data = result or ("", [])
    if not data:
        return EMPTY_ANSWER

    # default=str：兼容 datetime、Decimal 等无法直接 JSON 序列化的字段
    result_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    prompt = SQL_QA_PROMPT.format(question=question, sql=sql, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.text



async def collect_sql(question: str, llm: BaseChatModel, db: AsyncSession | None) -> Evidence:
    """保留 SQL 策略拦截、运行失败、空结果的不同业务状态。"""
    evidence = Evidence(question)
    if db is None:
        evidence.fallback = "数据库连接不可用，无法执行查询。"
        status = SourceStatus("failed", issues=["database_unavailable"], fatal=True)
    else:
        try:
            sql, rows = await search_sql_raw(question, llm, db)
        except SQLQueryError as error:
            evidence.fallback = str(error)
            status = SourceStatus("blocked" if error.code == "blocked" else "failed",
                                  issues=[] if error.code == "blocked" else [error.code],
                                  fatal=error.code != "blocked")
        else:
            evidence.data = (sql, rows)
            status = SourceStatus("ok" if rows else "empty", len(rows))
            if rows:
                text = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
                evidence.contexts = [f"执行的 SQL：{sql}\n查询结果：\n{text}"]
                evidence.prompt_kwargs = {"sql": sql, "result": text}
                evidence.used_sources = ["nl2sql"]
    evidence.sources["nl2sql"] = status.to_dict()
    return evidence
