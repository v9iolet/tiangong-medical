"""同一 knowledge_docs 中按分块策略隔离检索数据。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re


@dataclass(frozen=True)
class DocRetrievalOptions:
    collection_name: str = "knowledge_docs"
    use_hyde: bool = True
    use_rerank: bool = True
    use_hybrid: bool = True
    chunk_strategy: str = "original"
    use_parent: bool = True
    top_k: int = 20
    context_top_k: int = 5

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", self.collection_name):
            raise ValueError("集合名称只能使用字母、数字和下划线，且不能以数字开头")
        if self.chunk_strategy not in {"original", "parent_child", "semantic"}:
            raise ValueError("分块策略必须为 original、parent_child 或 semantic")
        if not 1 <= self.context_top_k <= self.top_k <= 200:
            raise ValueError("需要满足 1 ≤ 最终片段数 ≤ 召回数量 ≤ 200")

    @property
    def expand_parents(self) -> bool:
        return self.chunk_strategy == "parent_child" and self.use_parent

    def to_dict(self) -> dict:
        return asdict(self)

    def experiment_suffix(self, *, use_rewrite: bool, index_identity: str = "") -> str:
        payload = {**self.to_dict(), "use_rewrite": use_rewrite, "index_identity": index_identity}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        return (f"{self.chunk_strategy}-rw{int(use_rewrite)}-hy{int(self.use_hyde)}-rr{int(self.use_rerank)}"
                f"-hb{int(self.use_hybrid)}-pc{int(self.expand_parents)}-k{self.top_k}-n{self.context_top_k}-{digest}")
