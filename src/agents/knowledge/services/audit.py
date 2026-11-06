# src/agents/knowledge/services/audit.py
"""
知识检索审计：记录每次问答的用户、意图、通道、耗时与幻觉检测结果，便于事后追溯和质量分析。

审计日志通过 loguru 的 audit=True 标记输出，除普通 sink 外还会单独写入 logs/audit_*.log（见 src/core/logger.py）。
"""

from __future__ import annotations
import time
from loguru import logger


class QueryAuditLog:
    """查询审计日志记录器。记录每次知识检索的完整链路。"""

    @staticmethod
    def log(
        user_id: str,
        role: str,
        question: str,
        intent: str,
        channels: list[str],
        answer_preview: str,
        duration_ms: float,
        hallucination_check: dict | None = None,
    ) -> None:
        """
        输出一条知识检索审计日志。

        Args:
            user_id / role: 提问用户及其角色
            question: 用户问题，日志中只保留前 80 字
            intent: 路由判定的检索意图
            channels: 实际使用的检索通道
            answer_preview: 最终回答，日志中只保留前 80 字
            duration_ms: 整条检索链路耗时（毫秒），通常取自 Timer.elapsed_ms
            hallucination_check: check_hallucination 的返回结果；未做检测时为 None，日志中记为 N/A
        """
        # 字段以 " | " 分隔、key=value 形式输出，便于日志平台按字段检索
        logger.bind(audit=True).info(
            "knowledge_query | "
            f"user={user_id} | role={role} | intent={intent} | "
            f"channels={channels} | duration={duration_ms:.0f}ms | "
            f"grounded={hallucination_check.get('is_grounded', 'N/A') if hallucination_check else 'N/A'} | "
            f"question={question[:80]} | "
            f"answer={answer_preview[:80]}"
        )


class Timer:
    """
    简单计时器，用于测量检索耗时。

    用法：
        with Timer() as t:
            ...
        QueryAuditLog.log(..., duration_ms=t.elapsed_ms)

    退出 with 块时固定结束时间，之后任意时刻读取 elapsed_ms 结果都不变；
    在 with 块内读取则返回截至当前的耗时。
    """

    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        # perf_counter 是单调时钟，不受系统时间调整影响，适合测量耗时
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *args):
        # 记录结束时间；不吞异常：返回 None，with 块内的异常照常向外抛出
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        # 未进入 with 块时返回 0，避免调用方拿到 None 参与格式化报错
        if self._start is None:
            return 0.0
        # 仍在 with 块内时以当前时间计算
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000
