"""单题执行与评分校验，通过结果对象向页面和 CLI 提供统一判断。"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import pandas as pd
from loguru import logger

from src.agents.knowledge.evaluation.results import EvaluationResult


async def evaluate_question(tracked: Any, tru_app: Any, result: EvaluationResult,
                            store: Any, app_name: str) -> AsyncIterator[str]:
    async def persist():
        try:
            await asyncio.to_thread(store.save, app_name, result)
        except Exception:
            logger.exception("评估状态写入失败")
            result.persistence_status = "failed"
            if "status_persistence_failed" not in result.error_codes:
                result.error_codes.append("status_persistence_failed")
            return False
        return True

    # 开始就记录 pending；进程中断后仍能看到未完成的题目。
    if not await persist():
        result.finish("status_persistence_failed")
        yield "状态保存失败，本题未启动"
        return
    recording = None
    started = time.perf_counter()
    try:
        with tru_app as recording:
            result.answer = await tracked.query(result.question)
    except Exception:
        logger.exception("评估问答失败")
        result.error_codes.append("generation_failed" if tracked.generation_status == "failed" else "retrieval_failed")
        result.generation_status = "failed"
    else:
        result.generation_status = tracked.generation_status
    finally:
        result.elapsed = round(time.perf_counter() - started, 2)
        result.sources = dict(tracked.last_sources)
        retrieval = tracked.last_retrieval
        result.used_sources = list(retrieval.used_sources) if retrieval else []
        result.context_count = len(retrieval.contexts) if retrieval else 0
        if recording is not None:
            result.record_ids = [record.record_id for record in recording.records]
        if result.generation_status == "failed" and not result.sources:
            result.sources = {"query": {"status": "failed", "count": 0, "issues": ["query_failed"]}}
    if not await persist():
        result.finish("status_persistence_failed")
        yield "问答已结束，但状态保存失败"
        return

    if result.generation_status == "failed" or result.status == "runtime_failed":
        result.finish("runtime_failed")
        await persist()
        yield "问答存在运行故障，评分未执行"
        return
    if len(result.record_ids) != 1:
        result.error_codes.append("invalid_record_count")
        result.finish("invalid_record_count")
        await persist()
        yield "追踪记录不完整，评分未执行"
        return

    yield f"回答完成，耗时 {result.elapsed} 秒，开始计算适用指标"
    reason = ""
    try:
        scores = (await asyncio.to_thread(tru_app.retrieve_feedback_results, result.record_ids, 300)
                  if any(m.status == "pending" for m in result.metrics.values()) else pd.DataFrame())
    except Exception:
        logger.exception("评估指标计算失败")
        reason = "evaluation_failed"
        # 某项指标异常时，仍从持久化记录中读取已经成功的评分。
        try:
            from trulens.core import TruSession
            records, columns = await asyncio.to_thread(
                TruSession().get_records_and_feedback, record_ids=result.record_ids,
            )
            scores = records.set_index("record_id")[columns] if not records.empty else pd.DataFrame()
        except Exception:
            scores = pd.DataFrame()
    result.validate_scores(scores, reason)
    result.finish()
    await persist()
    scores_text = "，".join(
        f"{name} {metric.score:.3f}" if metric.status == "complete" else f"{name} {metric.status} ({metric.reason})"
        for name, metric in result.metrics.items()
    )
    yield f"指标：{scores_text}；评估 {result.status}；降级 {result.degraded}"
