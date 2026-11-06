"""父子 / 语义分块：只做切分，不碰 Milvus。

输入「父单元」（数据集的一条记录，或上传文档的一个结构块），输出带 parent_id / parent_text / source_id 的子块。
- parent_child：SentenceSplitter 在每个父单元内切成 256 token 的子块，不跨父单元
- semantic：每个父单元单独用 SemanticSplitterNodeParser 按句子窗口的向量距离断句，百分位阈值按父单元单独算；
  句子窗口的向量先跨父单元去重、打包预取，切分时从缓存读，避免每个父单元只带几条文本发一次请求

token 计数用 LlamaIndex 默认 tokenizer（tiktoken gpt-3.5-turbo），与 doc_ingestion 的 SentenceSplitter 一致。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding, SimilarityMode
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_regex
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer

from src.agents.knowledge.ingestion.structured_chunking import MAX_CHUNK_TOKENS

PARENT_CHILD = "parent_child"
SEMANTIC = "semantic"
# 变体策略；original 由 doc_ingestion 的结构分块写入，不走这里
VARIANT_STRATEGIES = (PARENT_CHILD, SEMANTIC)

# 子块大小与重叠，按 SentenceSplitter 的 token 计
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 32
# 语义分块预取句子窗口向量时，同时进行的 embedding 请求数；每个请求最多 EMBED_BATCH_SIZE 条窗口文本
SEMANTIC_CONCURRENCY = 4
# 语义断句参数：窗口前后各带 1 句；取句间距离的第 95 百分位作为断点阈值
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_PERCENTILE = 95
# 句子切分：句末标点（。！？.!?）跟在句子后面保留；数字之间的小数点（0.5g）不算句末
SENTENCE_PATTERN = r"(?:[^。！？.!?]|(?<=\d)\.(?=\d))+[。！？.!?]*|[。！？.!?]+"
_split_sentences = split_by_regex(SENTENCE_PATTERN)


@dataclass
class ParentUnit:
    """父单元：source_id 在同一文档内唯一，metadata 里的 page_number 等会原样带到子块。"""
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantChunk:
    """子块；ordinal 是它在父单元内的序号（从 0 开始），用于生成确定的主键。"""
    chunk_strategy: str
    parent_id: str
    parent_text: str
    source_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


async def gather_cancel_on_error(coroutines: Iterable[Awaitable]) -> list:
    """并发执行，结果顺序与输入一致；任一失败或被取消时，取消并等待其余任务结束再抛出。"""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class LangchainBatchEmbedding(BaseEmbedding):
    """
    把 langchain 的 Embeddings 包成 LlamaIndex 的 embed_model，批量请求一次发出。

    不用 llama_index 自带的 LangchainEmbedding：它没有实现 _aget_text_embeddings，
    异步批量会退化成「每条文本一次 aembed_documents」，一批 20 条变成 20 个请求。
    langchain 的 aembed_documents 本身是异步的（未实现时在线程池里跑同步版），不会阻塞事件循环。
    """
    _embeddings: Any = PrivateAttr()

    def __init__(self, embeddings, embed_batch_size: int):
        super().__init__(embed_batch_size=embed_batch_size, model_name=type(embeddings).__name__)
        self._embeddings = embeddings

    @classmethod
    def class_name(cls) -> str:
        return "LangchainBatchEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embeddings.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._embeddings.aembed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embeddings.embed_documents([text])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._embeddings.aembed_documents([text]))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)


def build_semantic_embed_model(embeddings):
    """把 langchain 的 Embeddings 包成 LlamaIndex 的 embed_model，供 build_semantic_chunks 使用。"""
    # 延迟导入：doc_ingestion 在模块顶层依赖本模块，直接导入会循环
    from src.agents.knowledge.ingestion.doc_ingestion import EMBED_BATCH_SIZE
    return LangchainBatchEmbedding(embeddings, embed_batch_size=EMBED_BATCH_SIZE)


class CachedEmbedding(BaseEmbedding):
    """
    带缓存的 embed_model：文本命中缓存直接返回，未命中回退到内层 embed_model 并计数。

    缓存由 build_semantic_chunks 预取填好；切分在线程里同步调用本类，命中时只是字典查找。
    相似度计算委托给内层，保证与直接使用内层 embed_model 时的结果一致。
    """
    _inner: Any = PrivateAttr()
    _cache: dict = PrivateAttr()
    _misses: int = PrivateAttr(default=0)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, inner: BaseEmbedding, cache: dict[str, list[float]]):
        super().__init__(embed_batch_size=inner.embed_batch_size, model_name=inner.model_name)
        self._inner = inner
        self._cache = cache

    @classmethod
    def class_name(cls) -> str:
        return "CachedEmbedding"

    @property
    def misses(self) -> int:
        """回退到内层 embed_model 的文本条数；预取完整时应为 0。"""
        return self._misses

    def _count_misses(self, count: int) -> None:
        with self._lock:
            self._misses += count

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._inner.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._inner.aget_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._aget_text_embeddings([text]))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._cache.get(text) for text in texts]
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing:
            self._count_misses(len(missing))
            fetched = self._inner.get_text_embedding_batch([texts[index] for index in missing])
            for index, vector in zip(missing, fetched):
                vectors[index] = vector
        return vectors

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._cache.get(text) for text in texts]
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing:
            self._count_misses(len(missing))
            fetched = await self._inner.aget_text_embedding_batch([texts[index] for index in missing])
            for index, vector in zip(missing, fetched):
                vectors[index] = vector
        return vectors

    def similarity(self, embedding1, embedding2, mode: SimilarityMode = SimilarityMode.DEFAULT) -> float:
        return self._inner.similarity(embedding1, embedding2, mode=mode)


def _to_chunks(parent: ParentUnit, pieces: list[str], strategy: str) -> list[VariantChunk]:
    """丢掉空白块，保持原文顺序；ordinal 在丢弃空白之后重新编号。"""
    texts = [piece.strip() for piece in pieces if piece.strip()]
    return [
        VariantChunk(
            chunk_strategy=strategy, parent_id=parent.source_id, parent_text=parent.text,
            source_id=parent.source_id, text=text, ordinal=ordinal, metadata=dict(parent.metadata),
        )
        for ordinal, text in enumerate(texts)
    ]


def _check_parents(parents: list[ParentUnit]) -> None:
    for parent in parents:
        if not parent.source_id:
            raise ValueError("父单元的 source_id 不能为空")


def build_parent_child_chunks(parents: list[ParentUnit]) -> list[VariantChunk]:
    """每个父单元单独切成 256 token 的子块（重叠 32），不跨父单元。"""
    _check_parents(parents)
    # paragraph_separator 与 doc_ingestion._make_splitter 一致：上传正文按双换行分段
    splitter = SentenceSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP, paragraph_separator="\n\n",
    )
    return [
        chunk for parent in parents
        for chunk in _to_chunks(parent, splitter.split_text(parent.text), PARENT_CHILD)
    ]


def _split_oversized(pieces: list[str]) -> list[str]:
    """语义块超过 MAX_CHUNK_TOKENS 时再用 SentenceSplitter 切一次，不留重叠。"""
    count = get_tokenizer()
    splitter = SentenceSplitter(
        chunk_size=MAX_CHUNK_TOKENS, chunk_overlap=0, paragraph_separator="\n\n",
    )
    result = []
    for piece in pieces:
        result.extend(splitter.split_text(piece) if len(count(piece)) > MAX_CHUNK_TOKENS else [piece])
    return result


def _make_semantic_parser(embed_model) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        embed_model=embed_model, sentence_splitter=_split_sentences,
        buffer_size=SEMANTIC_BUFFER_SIZE, breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_PERCENTILE,
    )


def _is_unsplittable(text: str) -> bool:
    """不超过两句时不可能切开：两句只有一个句间距离，百分位阈值就等于它，而断点要求严格大于阈值。"""
    return len(_split_sentences(text)) <= 2


def _sentence_windows(text: str, parser: SemanticSplitterNodeParser) -> list[str]:
    """
    parser 切分这个父单元时会请求向量的窗口文本，不超过两句的父单元返回空列表。

    切句和拼窗口用的是 parser 自己的 sentence_splitter 与 _build_sentence_groups（私有方法，
    升级 llama_index 后由测试核对「预取文本 == parser 实际请求的文本」）。
    """
    text = text.strip()
    if _is_unsplittable(text):
        return []
    groups = parser._build_sentence_groups(parser.sentence_splitter(text))
    return [group["combined_sentence"] for group in groups]


def _semantic_pieces(text: str, parser: SemanticSplitterNodeParser) -> list[str]:
    """同步切分一个父单元；parser 的 embed_model 命中缓存时纯 CPU，由调用方放到线程里执行。"""
    text = text.strip()
    # 不超过两句直接整段作为一个块，不调用 embedding
    if _is_unsplittable(text):
        return _split_oversized([text])
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return _split_oversized([node.get_content() for node in nodes])


async def _prefetch_window_embeddings(
    windows: list[str], embed_model, concurrency: int,
) -> dict[str, list[float]]:
    """窗口文本按 EMBED_BATCH_SIZE 一批，最多 concurrency 个请求同时在飞；任一批失败则取消其余并抛出。"""
    # 延迟导入：doc_ingestion 在模块顶层依赖本模块，直接导入会循环
    from src.agents.knowledge.ingestion.doc_ingestion import EMBED_BATCH_SIZE

    semaphore = asyncio.Semaphore(concurrency)

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            vectors = await embed_model.aget_text_embedding_batch(batch)
        if len(vectors) != len(batch):
            raise ValueError(f"向量数量 {len(vectors)} 与窗口文本数量 {len(batch)} 不一致，停止语义分块")
        return vectors

    batches = [windows[i:i + EMBED_BATCH_SIZE] for i in range(0, len(windows), EMBED_BATCH_SIZE)]
    results = await gather_cancel_on_error(embed_batch(batch) for batch in batches)
    return {
        text: vector
        for batch, vectors in zip(batches, results)
        for text, vector in zip(batch, vectors)
    }


def _split_all(texts: list[str], embed_model) -> list[list[str]]:
    """同步切分全部父单元；一个 parser 复用，embed_model 命中缓存时没有网络请求。"""
    parser = _make_semantic_parser(embed_model)
    return [_semantic_pieces(text, parser) for text in texts]


def _collect_windows(parents: list[ParentUnit], embed_model) -> list[str]:
    """所有父单元的窗口文本，按首次出现的顺序去重。"""
    parser = _make_semantic_parser(embed_model)
    windows: dict[str, None] = {}
    for parent in parents:
        windows.update(dict.fromkeys(_sentence_windows(parent.text, parser)))
    return list(windows)


async def build_semantic_chunks(
    parents: list[ParentUnit], embed_model, concurrency: int = SEMANTIC_CONCURRENCY,
) -> list[VariantChunk]:
    """
    每个父单元单独做语义分块，百分位阈值只看该父单元内的句间距离。

    embed_model 是 LlamaIndex 的 BaseEmbedding（生产用 build_semantic_embed_model，测试可传假实现）。
    先收集所有父单元的句子窗口文本并去重，按 EMBED_BATCH_SIZE 跨父单元打包，用 aget_text_embedding_batch
    预取向量；concurrency 是同时进行的 embedding 请求数。切分本身是纯 CPU，放到一个线程里跑，
    向量从缓存取，未命中才回退到 embed_model。断点算法与逐个父单元直接用 parser 切分完全一致。
    """
    _check_parents(parents)
    if concurrency < 1:
        raise ValueError(f"concurrency 必须 >= 1：{concurrency}")

    windows = await asyncio.to_thread(_collect_windows, parents, embed_model)
    cache = await _prefetch_window_embeddings(windows, embed_model, concurrency)
    cached_model = CachedEmbedding(embed_model, cache)
    pieces = await asyncio.to_thread(_split_all, [parent.text for parent in parents], cached_model)
    return [
        chunk for parent, parent_pieces in zip(parents, pieces)
        for chunk in _to_chunks(parent, parent_pieces, SEMANTIC)
    ]
