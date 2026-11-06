"""多通道评估的完成规则、评分校验和汇总；页面与 CLI 共用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any
from uuid import uuid4

import pandas as pd

GROUND_TRUTH = "参考答案一致性"


def strategy_columns(configuration: dict[str, Any]) -> dict[str, Any]:
    """显示生效的文档策略与预算；历史记录缺少配置时明确标记。"""
    options = configuration.get("doc_options", {})
    if not options:
        return {"文档策略": "不适用或未记录", "Query 改写": configuration.get("use_rewrite", "未记录")}
    strategy = options.get("chunk_strategy", "original")
    return {
        "文档策略": {"original": "原始分块（固定分块）", "parent_child": "父子分块", "semantic": "语义分块"}.get(strategy, strategy),
        "HyDE": options.get("use_hyde", "未记录"),
        "重排序": options.get("use_rerank", "未记录"),
        "混合检索": options.get("use_hybrid", "未记录"),
        "父块扩展": strategy == "parent_child" and options.get("use_parent", False),
        "Query 改写": configuration.get("use_rewrite", "未记录"),
        "top_k": options.get("top_k", "未记录"),
        "rerank_top_k / context_top_k": options.get("context_top_k", "未记录"),
        "文档集合": options.get("collection_name", "未记录"),
    }


@dataclass
class MetricResult:
    status: str = "pending"
    score: float | None = None
    reason: str = ""


@dataclass
class EvaluationResult:
    app_version: str
    channel: str
    question: str
    category: str
    reference: str
    metric_names: list[str]
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    record_ids: list[str] = field(default_factory=list)
    answer: str = ""
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    used_sources: list[str] = field(default_factory=list)
    generation_status: str = "pending"
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    elapsed: float | None = None
    context_count: int = 0
    error_codes: list[str] = field(default_factory=list)
    persistence_status: str = "pending"
    phase: str = "running"
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics:
            self.metrics = {
                name: MetricResult("not_applicable", reason="no_reference")
                if name == GROUND_TRUTH and not self.reference else MetricResult()
                for name in self.metric_names
            }

    @property
    def degraded(self) -> bool:
        return any(s.get("issues") or s.get("status") == "failed" for s in self.sources.values())

    @property
    def status(self) -> str:
        retrieval_failed = any(s.get("status") == "failed" for s in self.sources.values()) and not self.used_sources
        if self.generation_status == "failed" or retrieval_failed:
            return "runtime_failed"
        if self.phase == "running":
            return "running"
        if self.error_codes or any(m.status not in {"complete", "not_applicable"} for m in self.metrics.values()):
            return "incomplete"
        return "complete"

    @property
    def successful(self) -> bool:
        return self.status == "complete" and not self.degraded and self.persistence_status == "saved"

    def validate_scores(self, frame: pd.DataFrame, reason: str = "") -> None:
        """一次 query 应对应一条记录；按 record_id 对齐，不能随意取首行。"""
        row = None
        if len(self.record_ids) == 1 and len(frame) == 1 and str(frame.index[0]) == self.record_ids[0]:
            row = frame.iloc[0]
        for name, metric in self.metrics.items():
            if metric.status == "not_applicable":
                continue
            value = row.get(name) if row is not None else None
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = math.nan
            if math.isfinite(score) and 0 <= score <= 1:
                metric.status, metric.score, metric.reason = "complete", score, ""
            else:
                metric.status, metric.score = "failed", None
                metric.reason = reason or ("invalid_score" if value is not None else "missing_score")

    def finish(self, reason: str = "") -> None:
        for metric in self.metrics.values():
            if metric.status == "pending":
                metric.status, metric.reason = "failed", reason or "not_computed"
        self.phase = "finished"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema_version": 1, "status": self.status, "degraded": self.degraded}

    def display_row(self) -> dict[str, Any]:
        return {
            "通道": self.channel, "实验版本": self.app_version,
            **strategy_columns(self.configuration),
            "配置(JSON)": json.dumps(self.configuration, ensure_ascii=False, sort_keys=True),
            "题单编号": self.configuration.get("sample_batch_id", "")[:12],
            "问题": self.question, "分类": self.category,
            "回答": self.answer, "参考答案": self.reference,
            "检索片段数": self.context_count,
            "耗时(秒)": self.elapsed,
            **{name: metric.score for name, metric in self.metrics.items()},
            "生成状态": self.generation_status, "评估状态": self.status, "降级": self.degraded,
            "来源状态": "; ".join(f"{name}: {s['status']}" + (f" ({','.join(s.get('issues', []))})" if s.get('issues') else "") for name, s in self.sources.items()),
            "使用来源": ", ".join(self.used_sources),
            "指标状态": "; ".join(f"{name}: {m.status}" + (f" ({m.reason})" if m.reason else "") for name, m in self.metrics.items()),
            "持久化状态": self.persistence_status,
            "错误": "; ".join(self.error_codes),
            "_metric_results": {name: asdict(m) for name, m in self.metrics.items()},
            "_sources": self.sources,
            "_successful": self.successful,
        }


def summarize(rows: list[dict[str, Any]], metric_names: list[str]) -> pd.DataFrame:
    summaries = []
    for channel in dict.fromkeys(row["通道"] for row in rows):
        group = [row for row in rows if row["通道"] == channel]
        total = len(group)
        result = {
            "通道": channel,
            **{key: value for key, value in group[0].items() if key in {
                "文档策略", "HyDE", "重排序", "父块扩展", "Query 改写", "top_k",
                "rerank_top_k / context_top_k", "文档集合", "分块版本",
            }},
            "回答完成": f"{sum(row.get('生成状态') in {'generated', 'fallback'} for row in group)}/{total}",
            "评估完整": f"{sum(row.get('评估状态') == 'complete' for row in group)}/{total}",
            "降级次数": sum(bool(row.get("降级")) for row in group),
            "零召回次数": sum(any(s['status'] == 'empty' for s in row.get('_sources', {}).values()) for row in group),
            "运行失败": sum(row.get("评估状态") == "runtime_failed" for row in group),
        }
        for name in metric_names:
            metrics = [row.get("_metric_results", {}).get(name) for row in group]
            applicable = [m for m in metrics if m is not None and m["status"] != "not_applicable"]
            scores = [m["score"] for m in applicable if m["status"] == "complete"]
            result[name] = round(sum(scores) / len(scores), 3) if scores else None
            result[f"{name}有效/适用"] = f"{len(scores)}/{len(applicable)}"
        durations = [row["耗时(秒)"] for row in group if row.get("耗时(秒)") is not None]
        result["耗时(秒)"] = round(sum(durations) / len(durations), 3) if durations else None
        summaries.append(result)
    return pd.DataFrame(summaries)


def historical_summary(results: list[EvaluationResult], legacy: pd.DataFrame) -> pd.DataFrame:
    """新统计只使用有状态的记录；旧实验保留评分，完成情况标记未记录。"""
    if results:
        rows = []
        names = list(dict.fromkeys(name for item in results for name in item.metric_names))
        for item in results:
            row = item.display_row()
            row["通道"] = item.app_version
            rows.append(row)
        current = summarize(rows, names).rename(columns={"通道": "实验/通道", "耗时(秒)": "平均耗时(秒)"})
        current["状态记录数"] = [sum(item.app_version == version for item in results) for version in current["实验/通道"]]
        current["状态统计"] = "已记录（统计仅含状态记录）"
    else:
        current = pd.DataFrame()
    if legacy.empty:
        return current
    old = legacy.copy()
    if not current.empty:
        old = old[~old["实验/通道"].isin(current["实验/通道"])]
        if "成本" in legacy:
            current = current.merge(legacy[["实验/通道", "成本"]], how="left", on="实验/通道")
    old["状态统计"] = "未记录"
    old["状态记录数"] = 0
    unknown_columns = ["回答完成", "评估完整", "降级次数", "零召回次数", "运行失败"]
    unknown_columns += [name for name in current.columns if name.endswith("有效/适用")]
    for name in unknown_columns:
        old[name] = "未记录"
    return pd.DataFrame(current.to_dict("records") + old.to_dict("records"))
