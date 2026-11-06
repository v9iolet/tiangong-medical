"""TruLens 追踪适配器：只包装共享业务检索与生成，不复制 prompt 或融合逻辑。"""
from __future__ import annotations

import json
from typing import Any

from trulens.core.otel.instrument import instrument, SpanAttributes

from src.agents.knowledge.retrieval.diagnostics import observe_source
from src.agents.knowledge.retrieval.evidence import Evidence as Retrieval
from src.agents.knowledge.retrieval.options import DocRetrievalOptions

SpanType = SpanAttributes.SpanType
CHANNELS = {
    "doc_rag": "search_knowledge_docs",
    "graph_rag": "search_knowledge_graph",
    "nl2sql": "search_knowledge_sql",
    "fusion": "search_knowledge_multi",
    "prescription_review": "review_prescription_tool",
}


def _retrieval_attributes(ret, exception, *args, **kwargs) -> dict[str, Any]:
    return {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", ""),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: ret.contexts if isinstance(ret, Retrieval) else [],
        "tiangong.retrieval.sources": json.dumps(ret.sources, ensure_ascii=False) if isinstance(ret, Retrieval) else "{}",
        "tiangong.retrieval.used_sources": ret.used_sources if isinstance(ret, Retrieval) else [],
    }


class TrackedKnowledge:
    def __init__(self, channel, llm, embedding_model, milvus_client, neo4j_driver,
                 db_session=None, role="patient", doc_type=None, use_rewrite=True,
                 doc_options=None):
        if channel not in CHANNELS:
            raise ValueError(f"未知评估通道：{channel}")
        self.channel = channel
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.role = role
        self.doc_type = doc_type or None
        self.use_rewrite = use_rewrite
        self.doc_options = doc_options or DocRetrievalOptions()
        self.last_retrieval = None
        self.last_sources = {}
        self.generation_status = "pending"

    async def _observe(self, name, call):
        value, status = await observe_source(call)
        self.last_sources[name] = status.to_dict()
        return value

    @instrument(span_type=SpanType.RETRIEVAL, attributes=_retrieval_attributes)
    async def retrieve(self, query: str) -> Retrieval:
        from src.agents.knowledge.retrieval import doc_rag, graph_rag, nl2sql, fusion, query_rewriter
        from src.agents.knowledge.review import prescription_review

        question = query
        if self.channel in {"doc_rag", "graph_rag", "fusion"} and self.use_rewrite:
            question = await self._observe("rewrite", query_rewriter.prepare_query(
                query, self.llm, self.role, self.use_rewrite))
            if question is None:
                raise RuntimeError("问题改写未完成")
        evidence = Retrieval(question)
        if self.channel == "doc_rag":
            hits = await self._observe("doc_rag", doc_rag.search_docs_raw(
                question, self.embedding_model, self.milvus_client, llm=self.llm,
                options=self.doc_options, doc_type=self.doc_type))
            evidence.data = hits or []
            evidence.contexts = doc_rag.doc_context_parts(evidence.data)
            evidence.prompt_kwargs = {"context": doc_rag.format_doc_context(evidence.data)}
            evidence.used_sources = ["doc_rag"] if evidence.contexts else []
        elif self.channel == "graph_rag":
            records = await self._observe("graph_rag", graph_rag.search_graph_raw(question, self.neo4j_driver, self.llm))
            evidence.data = records or []
            if records:
                text = graph_rag.format_graph_context(records)
                evidence.contexts = [text]
                evidence.prompt_kwargs = {"graph_result": text}
                evidence.used_sources = ["graph_rag"]
        elif self.channel == "nl2sql":
            evidence = await nl2sql.collect_sql(question, self.llm, self.db_session)
        elif self.channel == "fusion":
            evidence = await fusion.collect_sources(
                question, self.llm, self.embedding_model, self.milvus_client, self.neo4j_driver,
                self.db_session, doc_options=self.doc_options, doc_type=self.doc_type)
        else:
            evidence = await prescription_review.collect_prescription_checks(
                question, self.llm, self.embedding_model, self.milvus_client,
                self.neo4j_driver, self.doc_options)
        self.last_sources.update(evidence.sources)
        evidence.sources = dict(self.last_sources)
        self.last_retrieval = evidence
        return evidence

    @instrument(span_type=SpanType.GENERATION)
    async def generate(self, retrieval: Retrieval) -> str:
        from src.agents.knowledge.retrieval import doc_rag, graph_rag, nl2sql, fusion
        from src.agents.knowledge.review import prescription_review

        if retrieval.fallback is not None:
            self.generation_status = "fallback"
            return retrieval.fallback
        self.generation_status = "generated" if retrieval.contexts else "fallback"
        try:
            if self.channel == "doc_rag":
                answer = await doc_rag.answer_docs(retrieval.query, retrieval.data, self.llm, self.role)
            elif self.channel == "graph_rag":
                answer = await graph_rag.answer_graph(retrieval.query, retrieval.data, self.llm, self.role)
            elif self.channel == "nl2sql":
                answer = await nl2sql.answer_sql(retrieval.query, retrieval.data, self.llm)
            elif self.channel == "fusion":
                answer = await fusion.answer_fusion(retrieval.query, retrieval, self.llm, self.role)
            else:
                answer = await prescription_review.write_prescription_report(retrieval, self.llm)
            if not answer.strip():
                raise ValueError("empty_answer")
            return answer
        except Exception:
            self.generation_status = "failed"
            raise
        finally:
            self.last_sources.update(retrieval.sources)

    @instrument()
    async def query(self, query: str) -> str:
        self.last_sources = {}
        self.last_retrieval = None
        self.generation_status = "pending"
        retrieval = await self.retrieve(query=query)
        return await self.generate(retrieval)
