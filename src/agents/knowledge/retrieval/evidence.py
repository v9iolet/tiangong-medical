"""业务检索证据：生成与评估共用，不依赖 TruLens。"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    query: str
    contexts: list[str] = field(default_factory=list)
    prompt_kwargs: dict[str, Any] = field(default_factory=dict)
    fallback: str | None = None
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    used_sources: list[str] = field(default_factory=list)
    data: Any = None
