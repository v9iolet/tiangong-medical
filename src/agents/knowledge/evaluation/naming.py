"""评估运行的可读身份；一次运行内各通道共享编号。"""
from dataclasses import asdict, dataclass
from datetime import datetime
import getpass
import re
from uuid import uuid4


def _name_part(value: str, fallback: str, limit: int = 80) -> str:
    return re.sub(r"[^\w.-]+", "-", value.strip()).strip(".-")[:limit] or fallback


@dataclass(frozen=True)
class EvaluationRun:
    owner: str
    source: str
    label: str
    run_id: str
    started_at: str

    @classmethod
    def create(cls, label: str, source: str = "web") -> "EvaluationRun":
        if source not in {"web", "cli"}:
            raise ValueError("评估入口必须为 web 或 cli")
        now = datetime.now().astimezone()
        return cls(
            owner=_name_part(getpass.getuser(), "user", 40),
            source=source,
            label=_name_part(label or "", "manual-test"),
            run_id=f"{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}",
            started_at=now.isoformat(timespec="seconds"),
        )

    def app_version(self, channel: str, strategy: str) -> str:
        return f"{self.source}-{self.owner}/{self.label}/{self.run_id}/{channel}/{strategy}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
