"""评估期间收集检索诊断；普通问答保持现有返回值和降级行为。"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable

from loguru import logger


@dataclass
class SourceStatus:
    status: str = "pending"
    count: int = 0
    issues: list[str] = field(default_factory=list)
    fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_current_source: ContextVar[SourceStatus | None] = ContextVar("rag_source_diagnostics", default=None)


def report_issue(code: str, *, fatal: bool = False) -> None:
    """只存稳定错误码，不把请求、连接凭据或异常原文写入状态。"""
    source = _current_source.get()
    if source is not None:
        if code not in source.issues:
            source.issues.append(code)
        source.fatal = source.fatal or fatal


def report_source_count(count: int, *, skipped: bool = False) -> None:
    source = _current_source.get()
    if source is not None:
        source.count = count
        source.status = "skipped" if skipped else ("ok" if count else "empty")


async def observe_source(call: Awaitable[Any]) -> tuple[Any, SourceStatus]:
    """同一检索调用的证据和诊断一起返回；ContextVar 隔离融合并行任务。"""
    source = SourceStatus()
    token = _current_source.set(source)
    try:
        try:
            value = await call
        except Exception:
            logger.exception("检索阶段调用失败")
            report_issue("call_failed", fatal=True)
            value = None
        local_evidence = source.status == "ok" and source.count > 0
        if source.status == "pending":
            source.count = len(value) if isinstance(value, list) else int(bool(value))
            source.status = "ok" if source.count else "empty"
        if source.fatal:
            # 处方校验可能保留本地规则证据，同时图谱查询失败。
            source.status = "partial" if local_evidence else "failed"
        return value, source
    finally:
        _current_source.reset(token)
