"""最终评估状态与 TruLens record_id 关联保存，不改写已经结束的 span。"""
from __future__ import annotations

from datetime import datetime, timezone
import json

from sqlalchemy import Column, DateTime, Engine, JSON, MetaData, String, Table, insert, select, update

from src.agents.knowledge.evaluation.results import EvaluationResult, MetricResult


class EvaluationStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.table = Table(
            "tiangong_eval_results", MetaData(),
            Column("attempt_id", String(36), primary_key=True),
            Column("app_name", String(256), nullable=False, index=True),
            Column("app_version", String(512), nullable=False, index=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("payload", JSON, nullable=False),
        )
        self.table.metadata.create_all(engine)

    def save(self, app_name: str, result: EvaluationResult) -> None:
        result.persistence_status = "saved"
        try:
            payload = result.to_dict()
            # PostgreSQL JSON 不允许 NaN / Infinity，写入前在相同接口校验。
            json.dumps(payload, allow_nan=False)
            values = dict(app_name=app_name, app_version=result.app_version,
                          updated_at=datetime.now(timezone.utc), payload=payload)
            with self.engine.begin() as connection:
                existing = connection.execute(
                    update(self.table).where(self.table.c.attempt_id == result.attempt_id).values(**values)
                )
                if not existing.rowcount:
                    connection.execute(insert(self.table).values(attempt_id=result.attempt_id, **values))
        except Exception:
            result.persistence_status = "failed"
            raise

    def load(self, app_name: str) -> list[EvaluationResult]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(self.table.c.payload).where(self.table.c.app_name == app_name)
                .order_by(self.table.c.updated_at)
            ).scalars().all()
        results = []
        for payload in payloads:
            data = {key: value for key, value in payload.items()
                    if key not in {"schema_version", "status", "degraded"}}
            data["metrics"] = {name: MetricResult(**value) for name, value in data["metrics"].items()}
            results.append(EvaluationResult(**data))
        return results
