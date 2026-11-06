"""天宫医疗知识问答 RAG Gradio 测试工作台。

启动：
    python scripts/gradio_knowledge_app.py

页面直接复用知识 Agent 的五个工具，不经过 FastAPI，适合在本地验证：
- 知识问答：Agent 自动选择检索通道，或直接指定文档 RAG / GraphRAG / NL2SQL / 多路融合 / 处方审核
- 文档入库：MinerU 解析、分块、向量化写入 Milvus，并预览解析产物
- 文档 RAG 评估：原始片段基线与 HyDE、重排、父子/语义分块策略及组合对比
- 多通道 RAG 评估：固定文档策略，横向对比不同通道和融合效果

运行前请确认项目根目录存在 .env，并且 Milvus、Neo4j、模型服务等依赖可用；
RAG 评估另需 PostgreSQL 中已创建 trulens_eval 数据库（见 src/agents/knowledge/evaluation/trulens_config.py）。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator
from zipfile import ZipFile, ZIP_DEFLATED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Gradio 在导入时读取端口范围；保留 7861 为默认起点，并允许占用时顺延。
if __name__ == "__main__":
    os.environ.setdefault("GRADIO_SERVER_PORT", "7861")

import gradio as gr
import pandas as pd
from langchain.agents import create_agent
from pymilvus import MilvusClient
from loguru import logger
from PIL import Image

from src.agents.knowledge.ingestion.artifact_store import artifact_paths, read_manifest, source_hash
from src.agents.knowledge.ingestion.doc_ingestion import (
    ingest_file,
)
from src.agents.knowledge.ingestion.mineru_client import check_mineru_health
from src.agents.knowledge.ingestion.variant_chunking import PARENT_CHILD, SEMANTIC
from src.agents.knowledge.retrieval.doc_rag import COLLECTION_NAME
from src.agents.knowledge.retrieval.options import DocRetrievalOptions
from src.agents.knowledge.evaluation.naming import EvaluationRun
from src.agents.knowledge.retrieval.variant_schema import ORIGINAL_FILTER
from src.agents.knowledge.tools import build_knowledge_tools
from src.agents.streaming import is_agent_model_run, is_answer_token
from src.core.config import get_embedding_model, get_settings
from src.agents.workers.knowledge_agent import (
    KNOWLEDGE_SYSTEM_PROMPT,
    _build_deps,
)


MODE_AUTO = "自动选择（Knowledge Agent）"
MODE_DOCS = "文档 RAG"
MODE_GRAPH = "知识图谱 RAG"
MODE_SQL = "SQL 数据查询"
MODE_MULTI = "多路融合 RAG"
MODE_PRESCRIPTION = "处方审核"

MODE_TO_TOOL = {
    MODE_DOCS: "search_knowledge_docs",
    MODE_GRAPH: "search_knowledge_graph",
    MODE_SQL: "search_knowledge_sql",
    MODE_MULTI: "search_knowledge_multi",
    MODE_PRESCRIPTION: "review_prescription_tool",
}

ROLE_OPTIONS = {
    "患者 / 普通用户": "patient",
    "医生": "doctor",
    "药师": "pharmacist",
    "企业内部员工": "staff",
}

DOC_TYPE_OPTIONS = {
    "不限制": "",
    "临床指南": "guideline",
    "药品说明书": "drug_instruction",
    "医院制度 / SOP": "sop",
    "医学文献": "literature",
}

# RAG 评估：页面上的检索模式 → TrackedKnowledge 通道；app_version 包含入口、用户、标签、运行编号及策略
MODE_TO_CHANNEL = {
    MODE_DOCS: "doc_rag",
    MODE_GRAPH: "graph_rag",
    MODE_SQL: "nl2sql",
    MODE_MULTI: "fusion",
    MODE_PRESCRIPTION: "prescription_review",
}
EVAL_APP_NAME = "tiangong-knowledge-rag"
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_ALL_CATEGORIES = "全部"
DOC_EVAL_BASELINE = "基线（无优化）"
DOC_EVAL_CUSTOM = "自定义策略组合"

CSS = """
.gradio-container { max-width: 1440px !important; margin: auto; }
.hero { padding: 16px 4px; }
.hero .eyebrow { color: var(--link-text-color); letter-spacing: 3px; font-size: 12px; margin-bottom: 8px; }
.hero h1 { font-size: 28px; color: var(--body-text-color); margin: 0 0 8px; }
.hero p { color: var(--body-text-color-subdued); margin: 0; line-height: 1.7; }
.panel { border: 1px solid var(--border-color-primary) !important; border-radius: 14px !important; padding: 20px !important; }
.eval-workspace { align-items: flex-start; gap: 20px !important; }
.eval-card { background: var(--block-background-fill); }
.eval-card h3, .eval-run h3 { margin-top: 0 !important; color: var(--body-text-color); }
.eval-run { margin-top: 20px !important; }
.eval-note { color: var(--body-text-color-subdued); font-size: 13px; }
@media (max-width: 760px) {
  .panel { padding: 14px !important; }
  .hero { padding: 16px 4px !important; }
  .eval-workspace > .eval-card { min-width: 0 !important; flex-basis: 100% !important; }
}
.log-panel textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 12px !important; line-height: 1.8 !important; }
.doc-list { max-height: 360px; overflow: auto; }
footer { display: none !important; }
"""


def _log_line(message: str) -> str:
    return f"[{datetime.now():%H:%M:%S}] {message}"


def _backend_failure(action: str) -> str:
    # 不把可能带有凭据和请求上下文的异常文本返回给浏览器。
    logger.opt(exception=True).error(action)
    return f"{action}，详细错误见后端控制台。"



def _content_to_text(value: Any) -> str:
    """把 LangChain 消息块、工具输出或普通对象转成可展示文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return _content_to_text(value.content)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _short(value: Any, limit: int = 700) -> str:
    text = _content_to_text(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def _history_messages(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """只把用户和助手的可见消息传回 Agent，避免把 Gradio 元数据带进模型。"""
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role")
        content = _content_to_text(item.get("content"))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


@lru_cache(maxsize=1)
def _document_client() -> MilvusClient:
    """文档列表与入库共用客户端；查询列表不依赖模型配置。"""
    settings = get_settings()
    return MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        timeout=10,
    )


@lru_cache(maxsize=1)
def _ingestion_runtime() -> tuple[Any, MilvusClient]:
    """缓存文档入库使用的向量模型和 Milvus 客户端。"""
    return get_embedding_model(), _document_client()


async def _indexed_documents_markdown() -> str:
    """读取 knowledge_docs 中的文档清单，供页面确认入库结果。"""
    try:
        milvus_client = await asyncio.to_thread(_document_client)
        if not await asyncio.to_thread(milvus_client.has_collection, COLLECTION_NAME):
            return "当前 Milvus 知识库为空，首次入库时会创建集合。"
        rows = await asyncio.to_thread(
            milvus_client.query,
            collection_name=COLLECTION_NAME,
            consistency_level="Strong",
            filter=f"({ORIGINAL_FILTER}) and chunk_index == 0",
            output_fields=["doc_name", "doc_type", "category"],
            limit=500,
        )
        if not rows:
            return "当前 Milvus 知识库为空。"
        lines = ["| 文档 | 类型 | 分类 |", "|---|---|---|"]
        for row in rows:
            name = str(row.get("doc_name", "")).replace("|", "\\|")
            doc_type = str(row.get("doc_type", "")).replace("|", "\\|")
            category = str(row.get("category", "")).replace("|", "\\|")
            lines.append(f"| {name} | {doc_type} | {category} |")
        return f"显示前 {len(rows)} 篇文档（最多 500 篇）。\n\n" + "\n".join(lines)
    except Exception:
        return _backend_failure("读取 Milvus 文档清单失败")


def _uploaded_file_path(uploaded: Any) -> str:
    if isinstance(uploaded, str):
        return uploaded
    if hasattr(uploaded, "name"):
        return str(uploaded.name)
    raise ValueError("未读取到上传文件路径")


def _chunk_strategies(parent_child: bool, semantic: bool) -> tuple[str, ...]:
    """把页面上的勾选项转成 ingest_file 的 strategies；original 总是写入，不在其中。"""
    return tuple(name for name, checked in ((PARENT_CHILD, parent_child), (SEMANTIC, semantic)) if checked)


async def ingest_uploaded_document(
    uploaded: Any, doc_type_label: str, category: str, ingest_mode: str,
    parent_child: bool = True, semantic: bool = False,
):
    """逐步更新页面日志；每次请求持有独立的队列和后台任务。"""
    lines: list[str] = []
    task = None
    if not uploaded:
        yield "请选择文档。", gr.skip(), _log_line("等待选择文档")
        return
    try:
        file_path = _uploaded_file_path(uploaded)
        allowed_ext = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".xlsx", ".jpg", ".jpeg", ".png"}
        if Path(file_path).suffix.lower() not in allowed_ext:
            yield "不支持的文件格式。", gr.skip(), _log_line("文件格式校验未通过")
            return
        doc_name = Path(file_path).name
        doc_type = DOC_TYPE_OPTIONS.get(doc_type_label, "guideline") or "guideline"
        category = (category or "通用").strip() or "通用"
        strategies = _chunk_strategies(parent_child, semantic)
        queue: asyncio.Queue[str] = asyncio.Queue()
        started = time.perf_counter()

        async def progress(message: str) -> None:
            logger.info("文档入库 [{}] {}", doc_name, message)
            await queue.put(_log_line(message))

        async def run() -> int:
            await progress("初始化向量模型和 Milvus 客户端")
            embedding_model, milvus_client = await asyncio.to_thread(_ingestion_runtime)
            return await ingest_file(
                file_path=file_path, doc_name=doc_name, doc_type=doc_type, category=category,
                embedding_model=embedding_model, milvus_client=milvus_client,
                mineru_only=ingest_mode.startswith("仅使用 MinerU"), progress=progress,
                strategies=strategies,
            )

        lines.append(_log_line(f"开始入库：{doc_name}"))
        yield "正在处理文档…", gr.skip(), "\n".join(lines)
        task = asyncio.create_task(run())
        while not task.done() or not queue.empty():
            try:
                lines.append(await asyncio.wait_for(queue.get(), timeout=1))
            except asyncio.TimeoutError:
                pass
            yield f"正在处理 · 已运行 {time.perf_counter() - started:.0f} 秒", gr.skip(), "\n".join(lines)
        chunks = await task
        if chunks == 0:
            status = "文档解析后无有效分块，本次未写入 Milvus。"
        else:
            status = f"**入库成功** · {doc_name} · {chunks} 个分块 · {time.perf_counter() - started:.1f} 秒"
        lines.append(_log_line("任务完成，刷新文档清单"))
        docs = await _indexed_documents_markdown()
        yield status, docs, "\n".join(lines)
    except Exception:
        status = _backend_failure("文档入库失败")
        lines.append(_log_line(status))
        yield status, gr.skip(), "\n".join(lines)
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def mineru_status() -> str:
    result = await check_mineru_health()
    if result.get("status") == "unavailable":
        logger.error("MinerU 健康检查失败：{}", result.get("error", "未知错误"))
        return "MinerU 服务不可用，详细错误见后端控制台。"
    return f"✅ MinerU 服务正常：\n\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"


@lru_cache(maxsize=32)
def _runtime(user_id: str, role: str) -> tuple[Any, dict[str, Any]]:
    """按用户和角色缓存一次 Agent 与工具，避免每次提问都重复初始化模型和客户端。"""
    deps = _build_deps(db_session=None, user_id=user_id, role=role)
    tools = build_knowledge_tools(deps)
    agent = create_agent(
        model=deps.llm,
        tools=tools,
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )
    return agent, {tool.name: tool for tool in tools}


def _debug_header(mode: str, role_label: str, session_id: str) -> list[str]:
    return [
        f"模式：{mode}",
        f"角色：{role_label}",
        f"会话：{session_id}",
        "",
    ]


async def _stream_auto(
    agent: Any,
    messages: list[dict[str, str]],
    debug_lines: list[str],
) -> AsyncIterator[str]:
    """消费 LangGraph 事件，边输出回答边记录工具调用。

    每次 yield 当前完整的回答文本（不是增量），这样 Agent 决定调用工具时可以撤回它之前说的开场白；
    工具内部 LLM 和子 Agent 的 token 不进入回答。
    """
    answer = ""
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        event_name = event.get("event", "")
        event_data = event.get("data", {}) or {}
        event_name_label = event.get("name", "")
        metadata = event.get("metadata", {}) or {}

        if event_name == "on_tool_start":
            debug_lines.append(
                _log_line(f"调用工具 {event_name_label}，输入：{_short(event_data.get('input'), 500)}")
            )
            yield answer
        elif event_name == "on_tool_end":
            debug_lines.append(
                _log_line(f"工具 {event_name_label} 执行完成")
            )
            yield answer
        elif event_name == "on_chat_model_stream":
            chunk = event_data.get("chunk")
            if not is_answer_token(chunk, metadata):
                continue
            token = _content_to_text(chunk)
            if token:
                answer += token
                yield answer
        elif event_name == "on_chat_model_end" and is_agent_model_run(metadata):
            # 这一轮输出带 tool_calls，说明前面的文本只是"我先去检索"之类的开场白，不是回答
            if getattr(event_data.get("output"), "tool_calls", None) and answer:
                debug_lines.append(_log_line(f"撤回工具调用前的过程说明：{_short(answer, 200)}"))
                answer = ""
                yield answer

    # 某些兼容模型不会发 on_chat_model_stream，但会在 graph 结束事件里返回完整消息。
    if not answer:
        debug_lines.append("模型未产生可流式展示的 token，已尝试读取最终消息。")
        yield answer


async def _run_direct_tool(
    tool: Any,
    question: str,
    doc_type: str,
) -> tuple[str, list[str]]:
    started = time.perf_counter()
    debug_lines = [_log_line(f"调用工具 {tool.name}")]
    payload: dict[str, str] = {"question": question}
    if tool.name == "search_knowledge_docs":
        payload["doc_type"] = doc_type
    result = await tool.ainvoke(payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    answer = _content_to_text(result)
    debug_lines.extend([
        _log_line(f"工具执行完成，耗时 {elapsed_ms:.0f} ms"),
    ])
    return answer, debug_lines


async def answer_question(
    question: str,
    history: list[dict[str, Any]] | None,
    user_id: str,
    role_label: str,
    mode: str,
    doc_type_label: str,
    session_id: str,
):
    """Gradio 事件处理器；使用生成器让回答和调试日志逐步刷新。"""
    history = list(history or [])
    question = (question or "").strip()
    user_id = (user_id or "anonymous").strip() or "anonymous"
    session_id = (session_id or "local-session").strip() or "local-session"
    role = ROLE_OPTIONS.get(role_label, "patient")
    doc_type = DOC_TYPE_OPTIONS.get(doc_type_label, "")

    if not question:
        yield history, "请输入问题后再发送。", session_id, ""
        return

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
    debug_lines = _debug_header(mode, role_label, session_id)
    debug_lines.append(f"问题：{question}")
    debug_lines.append(f"文档类型过滤：{doc_type_label}")

    debug_lines.append(_log_line("初始化知识 Agent"))
    yield history, "\n".join(debug_lines), session_id, ""
    try:
        agent, tools = await asyncio.to_thread(_runtime, user_id, role)
        if mode == MODE_AUTO:
            messages = _history_messages(history[:-1])
            # 自动模式不强行改变 Agent 的工具选择；文档类型过滤只在显式文档模式生效。
            answer = ""
            async for answer in _stream_auto(agent, messages, debug_lines):
                history[-1]["content"] = answer
                yield history, "\n".join(debug_lines), session_id, ""
            if not answer:
                # 用一次非流式调用兜底，兼容没有流式 token 的模型网关。
                result = await agent.ainvoke({"messages": messages})
                final_messages = result.get("messages", [])
                answer = _content_to_text(final_messages[-1]) if final_messages else "未获得模型回答。"
                history[-1]["content"] = answer
        else:
            tool_name = MODE_TO_TOOL[mode]
            debug_lines.append(_log_line(f"开始执行 {tool_name}，等待检索和生成回答"))
            yield history, "\n".join(debug_lines), session_id, ""
            if mode == MODE_SQL:
                # SQL 工具需要真实的异步数据库会话；其他 RAG 工具不需要这层连接。
                from src.infra.database import AsyncSessionLocal

                async with AsyncSessionLocal() as db_session:
                    sql_deps = _build_deps(
                        db_session=db_session, user_id=user_id, role=role,
                    )
                    sql_tools = build_knowledge_tools(sql_deps)
                    sql_tool = next(tool for tool in sql_tools if tool.name == tool_name)
                    answer, direct_debug = await _run_direct_tool(sql_tool, question, doc_type)
            else:
                answer, direct_debug = await _run_direct_tool(tools[tool_name], question, doc_type)
            history[-1]["content"] = answer
            debug_lines.extend(direct_debug)

        debug_lines.append(_log_line(f"完成：回答长度 {len(history[-1]['content'])} 字符"))
        yield history, "\n".join(debug_lines), session_id, ""
    except Exception:
        error_text = _backend_failure("知识问答失败")
        history[-1]["content"] = error_text
        debug_lines.append(_log_line(error_text))
        yield history, "\n".join(debug_lines), session_id, ""



def reset_session() -> tuple[list[dict[str, str]], str, str, str]:
    """清空页面上下文并生成新的本地会话标识。"""
    import uuid

    new_id = f"gradio-{uuid.uuid4().hex[:10]}"
    return [], "已清空对话上下文。", new_id, ""


def _stage_artifact_preview(root: Path, assets: list[dict]) -> tuple[list, list[str]]:
    """只将文本归档和解码后的栅格图片复制到 Gradio 缓存，不暴露产物目录。"""
    images = []
    raster_formats = {"PNG", "JPEG", "GIF", "WEBP", "BMP"}
    image_root = (root / "images").resolve()
    # 先在系统临时目录生成安全副本；复制到 Gradio 缓存后立即清理暂存目录。
    with tempfile.TemporaryDirectory(prefix="tiangong-preview-") as temporary:
        staging = Path(temporary)
        for index, asset in enumerate(assets):
            if asset.get("format") not in {"png", "jpg", "gif", "webp", "bmp"}:
                continue
            filename = asset.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                continue
            source = image_root / filename
            if source.resolve().parent != image_root:
                continue
            try:
                payload = source.read_bytes()
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format not in raster_formats:
                        continue
                    if Image.MAX_IMAGE_PIXELS is not None and image.width * image.height > Image.MAX_IMAGE_PIXELS:
                        raise ValueError("预览图片超过像素上限")
                    image.verify()
                # 使用同一份已验证字节解码，重新编码以移除原文件元数据和附加内容。
                preview = staging / f"image-{index}.png"
                with Image.open(io.BytesIO(payload)) as image:
                    image.convert("RGBA").save(preview, format="PNG")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                logger.warning("跳过无法验证的预览图片：{}", filename)
                continue
            cached = gr.processing_utils.save_file_to_cache(preview, gr.utils.get_upload_folder())
            images.append((cached, " / ".join(asset.get("captions", []))))

        archive = staging / "parsed-artifacts.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            for name in ("raw.md", "document.md", "retrieval.md", "assets.json", "chunks.jsonl", "manifest.json"):
                source = root / name
                if source.is_file() and not source.is_symlink():
                    bundle.writestr(name, source.read_bytes())
        downloads = [gr.processing_utils.save_file_to_cache(archive, gr.utils.get_upload_folder())]
    return images, downloads


async def preview_artifacts(uploaded: Any):
    """只查看当前上传文件同名产物，不接受浏览器提供的任意本地路径。"""
    if not uploaded:
        return "请选择已处理的文档。", "", "", [], [], [], ""
    doc_name = Path(_uploaded_file_path(uploaded)).name
    paths = artifact_paths(hashlib.md5(doc_name.encode()).hexdigest()[:16])
    try:
        def read():
            manifest = read_manifest(paths)
            if not manifest or not manifest.get("complete") or manifest.get("doc_name") != doc_name:
                return "尚无解析产物，请先执行入库。", "", "", [], [], [], ""
            if manifest.get("source_hash") != source_hash(_uploaded_file_path(uploaded)):
                return "同名文档内容已变化，请先处理当前文件。", "", "", [], [], [], ""
            raw = (paths.root / "raw.md").read_text(encoding="utf-8")
            retrieval = (paths.root / "retrieval.md").read_text(encoding="utf-8")
            assets_file = paths.root / "assets.json"
            assets = json.loads(assets_file.read_text(encoding="utf-8")) if assets_file.exists() else []
            chunks_file = paths.root / "chunks.jsonl"
            chunks = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()] if chunks_file.exists() else []
            rows = [[item["chunk_index"], item["metadata"].get("section_path", ""), item["metadata"].get("chunk_type", ""), item["text"]] for item in chunks]
            images, downloads = _stage_artifact_preview(paths.root, assets)
            status = f"解析来源：{manifest['parser']} · 提取图片：{manifest['image_count']} 张 · {paths.root}"
            return status, raw[:30000], retrieval[:30000], rows[:200], images, downloads, json.dumps(manifest, ensure_ascii=False, indent=2)
        return await asyncio.to_thread(read)
    except Exception:
        return _backend_failure("读取解析产物失败"), "", "", [], [], [], ""


def _eval_dataset_choices() -> list[str]:
    return sorted(path.name for path in EVAL_DATA_DIR.glob("*.json")) if EVAL_DATA_DIR.is_dir() else []


@lru_cache(maxsize=4)
def _read_eval_dataset(name: str, mtime: float) -> tuple[dict[str, str], ...]:
    """统一两种评估集格式；mtime 参与缓存键，文件更新后自动重新读取。"""
    items = []
    for raw in json.loads((EVAL_DATA_DIR / name).read_text(encoding="utf-8")):
        if "reference_answer" in raw:
            # rag_eval_dataset.json：question / reference_answer / source / category
            items.append({
                "question": str(raw["question"]),
                "reference": str(raw.get("reference_answer") or ""),
                "category": str(raw.get("category") or "未分类"),
            })
        elif "choices" in raw:
            # medqa_zh.json：选择题，把选项拼进问题，参考答案为正确选项
            options = "；".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(raw["choices"]))
            items.append({
                "question": f"{raw['question']}\n选项：{options}",
                "reference": "、".join(raw.get("answer") or []),
                "category": "medqa",
            })
    return tuple(items)


def _load_eval_dataset(name: str) -> list[dict[str, str]]:
    # 下拉框只列出 data/eval 中的文件，这里再按文件名校验一次，避免读取目录外的路径
    if name not in _eval_dataset_choices():
        raise ValueError("评估集不存在")
    return list(_read_eval_dataset(name, (EVAL_DATA_DIR / name).stat().st_mtime))


def eval_dataset_categories(name: str):
    """切换评估集时刷新分类选项，并显示各分类题量。"""
    try:
        items = _load_eval_dataset(name)
    except Exception:
        return gr.update(choices=[EVAL_ALL_CATEGORIES], value=[EVAL_ALL_CATEGORIES]), _backend_failure("读取评估集失败")
    counts: dict[str, int] = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    choices = [EVAL_ALL_CATEGORIES, *[(f"{category}（{counts[category]} 题）", category)
                                    for category in sorted(counts, key=lambda c: -counts[c])]]
    return gr.update(choices=choices, value=[EVAL_ALL_CATEGORIES]), f"共 **{len(items)}** 题 · **{len(counts)}** 个分类，分类题量见下拉选项。"


def _eval_samples(dataset: str, categories: list[str], size: int, seed: int, custom: str,
                  sample_mode: str = "随机抽样") -> list[dict[str, str]]:
    """自定义问题优先（每行一题，无参考答案）；否则按分类过滤后用固定随机种子抽样，便于多次实验对比。"""
    questions = [line.strip() for line in (custom or "").splitlines() if line.strip()]
    if questions:
        return [{"question": q, "reference": "", "category": "自定义"} for q in questions]
    items = _load_eval_dataset(dataset)
    if categories and EVAL_ALL_CATEGORIES not in categories:
        items = [item for item in items if item["category"] in categories]
    if sample_mode == "全部题目":
        return items
    if sample_mode != "随机抽样" or int(size) < 1:
        raise ValueError("抽样方式无效或题数小于 1")
    return random.Random(int(seed)).sample(items, min(int(size), len(items)))


def _eval_sampling_spec(dataset, categories, size, seed, custom, sample_mode):
    return {
        "dataset": dataset, "categories": sorted(categories or []),
        "size": int(size), "seed": int(seed), "custom": custom or "",
        "mode": sample_mode,
    }


def _eval_sample_id(samples: list[dict[str, str]]) -> str:
    payload = json.dumps(samples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_eval_sample_batch(dataset, categories, size, seed, custom, sample_mode="随机抽样"):
    spec = _eval_sampling_spec(dataset, categories, size, seed, custom, sample_mode)
    samples = [dict(item) for item in _eval_samples(dataset, categories, size, seed, custom, sample_mode)]
    if not samples:
        raise ValueError("所选分类下没有题目，请调整抽样条件")
    return {"id": _eval_sample_id(samples), "spec": spec, "samples": samples}


def prepare_eval_samples(dataset, categories, size, seed, custom, sample_mode):
    """只生成会话内题单，不初始化模型或数据库。"""
    custom = custom or ""
    try:
        batch = _make_eval_sample_batch(dataset, categories, size, seed, custom, sample_mode)
    except ValueError as error:
        return {}, str(error), [], gr.update(interactive=False)
    except Exception:
        return {}, _backend_failure("生成固定题单失败"), [], gr.update(interactive=False)
    samples = batch["samples"]
    references = sum(bool(item["reference"]) for item in samples)
    source = "自定义问题" if custom.strip() else f"{dataset} · {sample_mode}"
    status = (f"已固定 **{len(samples)} 题** · 有参考答案 {references} 题 · {source}\n\n"
              f"题单编号：`{batch['id'][:12]}`。切换通道或优化策略会复用此题单和顺序；修改抽样条件后需重新生成。")
    preview = [[i, item["category"], item["question"], item["reference"]]
               for i, item in enumerate(samples, 1)]
    return batch, status, preview, gr.update(interactive=True)


def invalidate_eval_samples():
    return {}, "抽样条件已变化，请点击“生成并固定题单”。", [], gr.update(interactive=False)


@lru_cache(maxsize=1)
def _trulens_runtime() -> tuple[Any, Any]:
    """TruSession 是进程级单例；评估依赖按需导入，未安装 TruLens 时不影响问答和入库页。"""
    from src.agents.knowledge.evaluation.trulens_config import get_llm_provider, get_trulens_session

    return get_trulens_session(), get_llm_provider()


@lru_cache(maxsize=1)
def _eval_deps() -> Any:
    """评估各通道共用模型和客户端；角色在 TrackedKnowledge 上单独指定。"""
    return _build_deps(db_session=None)


@lru_cache(maxsize=1)
def _eval_store():
    from src.agents.knowledge.evaluation.store import EvaluationStore
    session, _ = _trulens_runtime()
    return EvaluationStore(session.connector.db.engine)


@lru_cache(maxsize=1)
def _eval_metrics(names: tuple[str, ...], golden_set_json: str) -> list[Any]:
    from src.agents.knowledge.evaluation.metrics import build_metrics

    _, provider = _trulens_runtime()
    return build_metrics(provider, list(names), json.loads(golden_set_json))


@dataclass
class _TrackedApps:
    identity: str
    deps: Any
    tracked: Any
    apps: dict[tuple[str, ...], Any] = field(default_factory=dict)
    app_version: str = ""

    def get_app(self, app_version: str, configuration: dict, metrics: list[Any], has_reference: bool) -> Any:
        from trulens.apps.app import TruApp

        applicable = [m for m in metrics if has_reference or m.name != "参考答案一致性"]
        key = tuple(m.name for m in applicable)
        if key not in self.apps:
            self.apps[key] = TruApp(
                self.tracked, app_name=EVAL_APP_NAME, app_version=app_version,
                feedbacks=applicable, start_evaluator=False,
                metadata={"evaluation_configuration": configuration},
            )
        return self.apps[key]

    def close(self) -> None:
        for app in self.apps.values():
            app.stop_evaluator()
        self.apps.clear()


# 每通道仅保留当前配置，最多两套指标（有/无参考答案），不随实验次数增长。
# TrackedKnowledge 含可变检索状态和异步 SQL 会话，复用期间必须串行访问。
_tracked_apps: dict[str, _TrackedApps] = {}
_tracked_apps_lock = asyncio.Lock()


def _clear_tracked_apps() -> None:
    for cached in _tracked_apps.values():
        cached.close()
    _tracked_apps.clear()
    _eval_metrics.cache_clear()


def _get_tracked_apps(channel: str, app_version: str, configuration: dict,
                      golden_set_json: str, deps: Any, doc_options: Any) -> _TrackedApps:
    from src.agents.knowledge.evaluation.tracked_knowledge import TrackedKnowledge

    # 每次运行的追踪版本独立；检索依赖只在实际配置改变时重建。
    stable_configuration = {
        **configuration,
        "evaluation_run": {key: value for key, value in configuration["evaluation_run"].items()
                           if key not in {"run_id", "started_at"}},
    }
    identity = json.dumps([stable_configuration, golden_set_json], sort_keys=True, ensure_ascii=False)
    cached = _tracked_apps.get(channel)
    if cached is not None and (cached.identity != identity or cached.deps is not deps):
        cached.close()
        del _tracked_apps[channel]
        cached = None
    if cached is None:
        tracked = TrackedKnowledge(
            channel=channel, llm=deps.llm, embedding_model=deps.embedding_model,
            milvus_client=deps.milvus_client, neo4j_driver=deps.neo4j_driver,
            role=configuration["role"],
            doc_type=configuration["doc_type"] if channel in {"doc_rag", "fusion"} else None,
            use_rewrite=configuration["use_rewrite"], doc_options=doc_options,
        )
        cached = _TrackedApps(identity, deps, tracked)
        _tracked_apps[channel] = cached
    if cached.app_version != app_version:
        cached.close()
        cached.app_version = app_version
    return cached


def _eval_summary(rows: list[dict[str, Any]], metric_names: list[str]) -> pd.DataFrame:
    from src.agents.knowledge.evaluation.results import summarize
    return summarize(rows, metric_names)


def _eval_detail(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """供执行检查和 CLI 使用的逐题状态；正文预览截断，完整内容存于 TruLens。"""
    frame = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in rows])
    for column in ("问题", "回答", "参考答案"):
        if column in frame:
            frame[column] = frame[column].map(lambda text: _short(text, 160))
    return frame


def doc_eval_mode_updates(mode: str):
    """基线锁定所有增强开关；切换策略不改变已固定的题单或片段预算。"""
    baseline = mode == DOC_EVAL_BASELINE
    return (
        gr.update(value=False, interactive=not baseline),  # Query 改写
        gr.update(value=False, interactive=not baseline),  # HyDE
        gr.update(value=False, interactive=not baseline),  # 重排
        gr.update(value=False, interactive=not baseline),  # 混合检索
        gr.update(value="original", interactive=not baseline),
        gr.update(value=False, interactive=not baseline),  # 父块扩展
    )


async def run_doc_rag_evaluation(
    dataset, categories, sample_size, seed, custom_questions,
    metric_names, role_label, doc_type_label, use_rewrite, experiment_label,
    doc_hyde, doc_rerank, doc_chunk_strategy, doc_parent,
    doc_top_k, doc_context_top_k, fixed_sample_batch, sample_mode, evaluation_mode, doc_hybrid=True,
):
    """文档策略实验；基线在执行端强制关闭增强，避免遗留控件值影响实验。"""
    if evaluation_mode not in {DOC_EVAL_BASELINE, DOC_EVAL_CUSTOM}:
        yield "文档评估模式无效。", "", pd.DataFrame()
        return
    if evaluation_mode == DOC_EVAL_BASELINE:
        use_rewrite = doc_hyde = doc_rerank = doc_parent = doc_hybrid = False
        doc_chunk_strategy = "original"
    async for update in run_rag_evaluation(
        dataset, categories, sample_size, seed, custom_questions, [MODE_DOCS],
        metric_names, role_label, doc_type_label, use_rewrite, experiment_label,
        doc_hyde, doc_rerank, doc_chunk_strategy, doc_parent,
        doc_top_k, doc_context_top_k, fixed_sample_batch, sample_mode, doc_hybrid=doc_hybrid,
    ):
        yield _eval_page_update(update, metric_names)


async def run_multichannel_rag_evaluation(
    dataset, categories, sample_size, seed, custom_questions, channel_labels,
    metric_names, role_label, doc_type_label, use_rewrite, experiment_label,
    doc_top_k, doc_context_top_k, fixed_sample_batch, sample_mode,
):
    """通道效果对比使用正式默认策略；保留页面显式设置的片段预算。"""
    from src.agents.knowledge.retrieval.options import DocRetrievalOptions
    defaults = DocRetrievalOptions()
    async for update in run_rag_evaluation(
        dataset, categories, sample_size, seed, custom_questions, channel_labels,
        metric_names, role_label, doc_type_label, use_rewrite, experiment_label,
        doc_hyde=defaults.use_hyde, doc_rerank=defaults.use_rerank,
        doc_chunk_strategy=defaults.chunk_strategy, doc_parent=defaults.use_parent,
        doc_hybrid=defaults.use_hybrid,
        doc_top_k=doc_top_k, doc_context_top_k=doc_context_top_k,
        fixed_sample_batch=fixed_sample_batch, sample_mode=sample_mode,
    ):
        yield _eval_page_update(update, metric_names)


async def run_rag_evaluation(
    dataset: str, categories: list[str], sample_size: float, seed: float, custom_questions: str | None,
    channel_labels: list[str], metric_names: list[str], role_label: str, doc_type_label: str,
    use_rewrite: bool, experiment_label: str,
    doc_hyde: bool = True, doc_rerank: bool = True, doc_chunk_strategy: str = "original",
    doc_parent: bool = True,
    doc_top_k: int = 20, doc_context_top_k: int = 5,
    fixed_sample_batch: dict | None = None, sample_mode: str = "随机抽样",
    run_source: str = "web",
    doc_hybrid: bool = True,
):
    """逐题运行各通道并同步计算 TruLens 指标；结果仅持久化到 PostgreSQL。"""
    # Gradio 未填写或清空的可选文本可能为 None，进入业务逻辑前统一为空字符串。
    custom_questions = custom_questions or ""
    lines: list[str] = []
    rows: list[dict[str, Any]] = []
    attempts = []
    empty = pd.DataFrame()

    def emit(status: str):
        return status, "\n".join(lines), _eval_summary(rows, metric_names), _eval_detail(rows)

    if not channel_labels or not metric_names:
        lines.append(_log_line("请至少选择一个检索通道和一个评估指标"))
        yield "请至少选择一个检索通道和一个评估指标。", "\n".join(lines), empty, empty
        return
    try:
        if fixed_sample_batch is None:
            # CLI 仍可按参数直接运行；页面必须先生成题单。
            batch = _make_eval_sample_batch(dataset, categories, sample_size, seed, custom_questions, sample_mode)
        else:
            spec = _eval_sampling_spec(dataset, categories, sample_size, seed, custom_questions, sample_mode)
            if not fixed_sample_batch or fixed_sample_batch.get("spec") != spec:
                raise ValueError("请先按当前抽样条件生成并固定题单")
            batch = fixed_sample_batch
        samples = [dict(item) for item in batch["samples"]]
        if not samples or batch["id"] != _eval_sample_id(samples):
            raise ValueError("固定题单无效，请重新生成")
    except ValueError as error:
        yield str(error), "\n".join(lines), empty, empty
        return
    except Exception:
        yield _backend_failure("读取评估题单失败"), "\n".join(lines), empty, empty
        return

    try:
        run = EvaluationRun.create(experiment_label, run_source)
    except ValueError as error:
        yield emit(f"实验命名错误：{error}")
        return
    label = run.label
    role = ROLE_OPTIONS.get(role_label, "patient")
    doc_type = DOC_TYPE_OPTIONS.get(doc_type_label, "")
    try:
        doc_options = DocRetrievalOptions(
            use_hyde=doc_hyde, use_rerank=doc_rerank, use_hybrid=doc_hybrid, chunk_strategy=doc_chunk_strategy,
            use_parent=doc_parent, top_k=int(doc_top_k),
            context_top_k=int(doc_context_top_k),
        )
    except (ValueError, TypeError) as error:
        yield emit(f"文档策略配置错误：{error}")
        return
    doc_index = {}
    total = len(samples) * len(channel_labels)
    started = time.perf_counter()
    lines.append(_log_line(f"运行 {run.source}-{run.owner}/{label}/{run.run_id}"))
    lines.append(_log_line(f"实验 {label}：{len(samples)} 题 × {len(channel_labels)} 个通道，共 {total} 次评估"))
    lines.append(_log_line(f"固定题单 {batch['id'][:12]}：各通道使用相同题目、参考答案和顺序"))
    lines.append(_log_line("初始化 TruLens 会话、评估模型和检索依赖"))
    yield emit("正在初始化…")

    try:
        from src.agents.knowledge.evaluation.execution import evaluate_question
        from src.agents.knowledge.evaluation.results import EvaluationResult

        deps = await asyncio.to_thread(_eval_deps)
        if any(MODE_TO_CHANNEL[name] in {"doc_rag", "fusion"} for name in channel_labels):
            if doc_options.use_hyde and deps.llm is None:
                yield emit("文档策略配置错误：启用 HyDE 时必须提供生成模型")
                return
            from src.agents.knowledge.retrieval.index_capabilities import validate_doc_index
            try:
                doc_index = await asyncio.to_thread(validate_doc_index, deps.milvus_client, doc_options)
            except ValueError as error:
                yield emit(f"文档索引预检失败：{error}")
                return
        await asyncio.to_thread(_trulens_runtime)
        store = await asyncio.to_thread(_eval_store)
        golden_set = [{"query": s["question"], "expected_response": s["reference"]} for s in samples if s["reference"]]
        golden_set_json = json.dumps(golden_set, sort_keys=True, ensure_ascii=False)
        metrics = await asyncio.to_thread(_eval_metrics, tuple(metric_names), golden_set_json)
    except Exception:
        status = _backend_failure("评估初始化失败（请确认已安装 TruLens 且 trulens_eval 数据库可连接）")
        lines.append(_log_line(status))
        yield emit(status)
        return
    skipped = set(metric_names) - {m.name for m in metrics}
    if skipped:
        lines.append(_log_line(f"所选题目没有参考答案，跳过指标：{'、'.join(skipped)}"))

    done = 0
    for channel_label in channel_labels:
        channel = MODE_TO_CHANNEL[channel_label]
        channel_doc_options = doc_options
        if channel == "prescription_review":
            from src.agents.knowledge.review.prescription_review import dosage_options
            channel_doc_options = dosage_options(doc_options)
        if channel in {"doc_rag", "fusion", "prescription_review"}:
            suffix = channel_doc_options.experiment_suffix(use_rewrite=use_rewrite if channel != "prescription_review" else False)
            app_version = run.app_version(channel, suffix)
        else:
            app_version = run.app_version(channel, f"rw{int(use_rewrite)}")
        configuration = {
            "evaluation_run": run.to_dict(),
            "dataset": "自定义问题" if custom_questions.strip() else dataset,
            "seed": int(seed), "sample_size": len(samples),
            "sample_batch_id": batch["id"], "sampling_mode": "自定义问题" if custom_questions.strip() else sample_mode,
            "categories": categories, "role": role, "doc_type": doc_type,
            "use_rewrite": use_rewrite, "metrics": list(metric_names),
            "generator_model": get_settings().DEEPSEEK_MODEL,
            "judge_model": get_settings().DEEPSEEK_MODEL,
            "embedding_model": get_settings().EMBEDDING_MODEL,
            "doc_options": channel_doc_options.to_dict() if channel in {"doc_rag", "fusion", "prescription_review"} else {},
            "doc_index": doc_index if channel in {"doc_rag", "fusion"} else {},
        }
        lines.append(_log_line(f"TruLens app_version：{app_version}"))
        lines.append(_log_line("策略配置：" + json.dumps(configuration, ensure_ascii=False)))
        async with _tracked_apps_lock, AsyncExitStack() as stack:
            db_session = None
            if channel == "nl2sql":
                # NL2SQL 需要真实的异步数据库会话，按通道开启、跑完即关闭
                from src.infra.database import AsyncSessionLocal

                db_session = await stack.enter_async_context(AsyncSessionLocal())
            cached = None
            for sample in samples:
                done += 1
                initialization_failed = False
                try:
                    if cached is None:
                        cached = _get_tracked_apps(channel, app_version, configuration, golden_set_json, deps, channel_doc_options)
                        tracked = cached.tracked
                        tracked.db_session = db_session
                        # 离开本通道（包括取消）时解除绑定，避免缓存持有已关闭的 SQL 会话。
                        stack.callback(setattr, tracked, "db_session", None)
                    tru_app = cached.get_app(app_version, configuration, metrics, bool(sample["reference"]))
                except Exception:
                    logger.exception("评估通道初始化失败")
                    initialization_failed = True
                result = EvaluationResult(
                    app_version, channel_label, sample["question"], sample["category"],
                    sample["reference"], list(metric_names), configuration=configuration,
                )
                attempts.append(result)
                if initialization_failed:
                    result.generation_status = "failed"
                    result.error_codes.append("app_initialization_failed")
                    result.finish("app_initialization_failed")
                    try:
                        await asyncio.to_thread(store.save, EVAL_APP_NAME, result)
                    except Exception:
                        result.persistence_status = "failed"
                        result.error_codes.append("status_persistence_failed")
                        logger.exception("评估初始化失败状态保存失败")
                    rows.append(result.display_row())
                    lines.append(_log_line(f"{channel_label} 初始化失败，详见后端日志"))
                    yield emit(f"已完成 {done}/{total} · runtime_failed")
                    continue
                row = result.display_row()
                rows.append(row)
                lines.append(_log_line(f"[{done}/{total}] {channel_label} · {_short(sample['question'], 40)}"))
                yield emit(f"正在评估 {done}/{total} · 已运行 {time.perf_counter() - started:.0f} 秒")
                async for message in evaluate_question(tracked, tru_app, result, store, EVAL_APP_NAME):
                    row.update(result.display_row())
                    lines.append(_log_line(message))
                    yield emit(f"正在评估 {done}/{total}")
                row.update(result.display_row())
                yield emit(f"已完成 {done}/{total} · {result.status}")

    try:
        session, _ = _trulens_runtime()
        await asyncio.to_thread(session.force_flush)
    except Exception:
        logger.exception("评估追踪写入失败")
        for result, row in zip(attempts, rows):
            result.error_codes.append("trace_persistence_failed")
            try:
                await asyncio.to_thread(store.save, EVAL_APP_NAME, result)
            except Exception:
                result.persistence_status = "failed"
            row.update(result.display_row())
    elapsed = time.perf_counter() - started
    incomplete = sum(row["评估状态"] != "complete" for row in rows)
    degraded = sum(row["降级"] for row in rows)
    lines.append(_log_line(f"实验 {label} 结束，不完整 {incomplete} 次，降级 {degraded} 次，耗时 {elapsed:.0f} 秒"))
    status = (f"**评估结束** · {label} · {len(samples)} 题 × {len(channel_labels)} 通道 · "
              f"不完整 {incomplete} 次 · 降级 {degraded} 次 · {elapsed:.0f} 秒")
    yield emit(status)


def _eval_page_update(update: tuple, metric_names: list[str]) -> tuple:
    """页面只传递当前状态、日志与简表，不向浏览器发送逐题明细。"""
    status, logs, summary, _ = update
    columns = ["通道", "回答完成", "评估完整", *metric_names, "降级次数", "运行失败", "耗时(秒)"]
    overview = (pd.DataFrame(columns=columns) if summary.empty
                else summary.loc[:, [name for name in columns if name in summary.columns]])
    return status, logs, overview


def eval_sampling_mode_updates(mode: str):
    random_sampling = mode == "随机抽样"
    return gr.update(interactive=random_sampling), gr.update(interactive=random_sampling)


def _build_evaluation_page(demo: gr.Blocks, *, doc_only: bool) -> None:
    """题单、实验配置、执行分区；完整结果与历史对比由 Dashboard 展示。"""
    description = ("固定题单，比较原始分块基线与 HyDE、重排序、父子分块和语义分块策略。"
                   if doc_only else "固定题单，比较文档、知识图谱、SQL、处方审核及融合通道。")
    gr.Markdown(description)
    dataset_choices = _eval_dataset_choices()
    with gr.Row(equal_height=False, elem_classes="eval-workspace"):
        with gr.Column(scale=5, min_width=340, elem_classes=["panel", "eval-card"]):
            gr.Markdown("### 1 · 固定评估题单")
            eval_sample_batch = gr.State({})
            with gr.Row():
                eval_dataset = gr.Dropdown(
                    choices=dataset_choices, label="评估集",
                    value="rag_eval_dataset.json" if "rag_eval_dataset.json" in dataset_choices
                    else (dataset_choices[0] if dataset_choices else None),
                )
                eval_categories = gr.Dropdown(choices=[EVAL_ALL_CATEGORIES], value=[EVAL_ALL_CATEGORIES],
                                              multiselect=True, label="题目分类")
            eval_dataset_info = gr.Markdown("")
            eval_sample_mode = gr.Radio(["随机抽样", "全部题目"], value="随机抽样", label="题目选择方式")
            with gr.Row():
                eval_size = gr.Slider(1, 100, value=10, step=1, label="抽样题数")
                eval_seed = gr.Number(value=42, precision=0, label="随机种子")
            with gr.Accordion("使用自定义问题", open=False):
                eval_custom = gr.Textbox(value="", label="每行一个问题", lines=3,
                                         placeholder="填写后使用这些问题，忽略评估集与抽样设置；无参考答案时跳过一致性评分")
            eval_prepare = gr.Button("生成并固定题单")
            eval_sample_status = gr.Markdown("先生成题单；切换通道与策略时会复用相同问题和顺序。")
            with gr.Accordion("题单预览", open=True):
                eval_sample_preview = gr.Dataframe(
                    headers=["序号", "分类", "问题", "参考答案"], datatype=["number", "str", "str", "str"],
                    value=[], interactive=False, wrap=True, label="固定题目与顺序", max_height=260,
                    column_widths=[55, 85, "40%", "40%"],
                )
        with gr.Column(scale=7, min_width=380, elem_classes=["panel", "eval-card"]):
            gr.Markdown("### 2 · 配置实验")
            eval_label = gr.Textbox(value="", label="实验标签", placeholder="例如 图谱验证、baseline；留空使用 manual-test",
                                    info="自动附加入口、系统用户、时间与运行编号；每次运行单独记录")
            with gr.Row():
                eval_role = gr.Dropdown(choices=list(ROLE_OPTIONS), value="医生", label="回答角色")
                eval_doc_type = gr.Dropdown(choices=list(DOC_TYPE_OPTIONS), value="不限制", label="文档类型",
                                            info="仅影响文档检索")
            if doc_only:
                eval_doc_mode = gr.Radio([DOC_EVAL_BASELINE, DOC_EVAL_CUSTOM], value=DOC_EVAL_BASELINE,
                                         label="文档评估模式")
            else:
                eval_channels = gr.CheckboxGroup(
                    choices=list(MODE_TO_CHANNEL), value=[MODE_DOCS, MODE_GRAPH, MODE_MULTI],
                    label="检索通道及组合", info="多路融合为文档 + 知识图谱；各通道共用固定题单",
                )
            eval_metrics = gr.CheckboxGroup(
                choices=["答案相关性", "上下文相关性", "有据性", "有害性", "参考答案一致性"],
                value=["答案相关性", "上下文相关性", "有据性", "参考答案一致性"], label="评估指标",
                info="评分 0–1；有害性越低越好，其他越高越好。每题每项指标会额外调用评估模型。",
            )
            gr.Markdown("基线：原始分块，关闭改写、HyDE、重排、混合检索和父块扩展。"
                        if doc_only else "文档与融合通道使用正式默认配置：原始分块、HyDE、重排序和混合检索；策略对比请使用文档评估页。")
            with gr.Accordion("检索策略与片段预算", open=False) as eval_strategy_panel:
                eval_rewrite = gr.Checkbox(value=not doc_only, interactive=not doc_only, label="启用 Query 改写")
                if doc_only:
                    eval_doc_strategy = gr.Dropdown(
                        choices=[("原始分块（固定分块）", "original"), ("父子分块", "parent_child"), ("语义分块", "semantic")],
                        value="original", label="文档分块策略", interactive=False,
                    )
                    with gr.Row():
                        eval_doc_hyde = gr.Checkbox(value=False, label="HyDE 假设回答检索", interactive=False)
                        eval_doc_rerank = gr.Checkbox(value=False, label="Rerank 重排", interactive=False)
                    eval_doc_hybrid = gr.Checkbox(value=False, label="混合检索（Dense + BM25）", interactive=False)
                    eval_doc_parent = gr.Checkbox(value=False, label="父块扩展（仅父子分块生效）", interactive=False)
                with gr.Row():
                    eval_doc_top_k = gr.Number(value=20, precision=0, minimum=1, maximum=200, label="文档粗召回数量 top_k")
                    eval_doc_context_top_k = gr.Number(value=5, precision=0, minimum=1, maximum=200,
                                                      label="文档最终片段数 rerank_top_k / context_top_k")
                gr.Markdown("最终片段数不超过粗召回数量；关闭重排序时也遵守此预算。"
                            + ("父子/语义分块需先入库对应策略的数据。" if doc_only else ""))
    with gr.Group(elem_classes=["panel", "eval-run"]):
        with gr.Row():
            gr.Markdown("### 3 · 执行评估")
            gr.Button("查看 TruLens Dashboard ↗", link="http://127.0.0.1:8501", link_target="_blank", size="sm", scale=0)
        with gr.Row():
            eval_button = gr.Button("使用固定题单开始评估", variant="primary", scale=3, interactive=False)
            eval_stop = gr.Button("停止", variant="stop", scale=1)
        eval_status = gr.Markdown("等待生成并固定题单。")
        eval_summary = gr.Dataframe(
            label="本次运行概览", headers=["通道", "回答完成", "评估完整", "答案相关性", "上下文相关性", "有据性"],
            value=[], interactive=False, wrap=True, max_height=250,
        )
        with gr.Accordion("运行日志与实验版本", open=False):
            eval_logs = gr.Textbox(label="评估运行日志", value="等待启动评估…", lines=10, max_lines=10,
                                   interactive=False, autoscroll=True, elem_classes="log-panel")
        gr.Markdown("完整回答、检索链路、逐题评分和历史对比统一在 TruLens Dashboard 查看。默认端口 8501，需单独启动。",
                    elem_classes="eval-note")
    eval_inputs = [eval_dataset, eval_categories, eval_size, eval_seed, eval_custom]
    if not doc_only:
        eval_inputs.append(eval_channels)
    eval_inputs += [eval_metrics, eval_role, eval_doc_type, eval_rewrite, eval_label]
    if doc_only:
        eval_inputs += [eval_doc_hyde, eval_doc_rerank, eval_doc_strategy, eval_doc_parent]
        eval_doc_mode.change(
            doc_eval_mode_updates, inputs=eval_doc_mode,
            outputs=[eval_rewrite, eval_doc_hyde, eval_doc_rerank, eval_doc_hybrid, eval_doc_strategy, eval_doc_parent],
            queue=False,
        )
        eval_doc_mode.change(lambda mode: gr.update(open=mode == DOC_EVAL_CUSTOM),
                             inputs=eval_doc_mode, outputs=eval_strategy_panel, queue=False)
    eval_inputs += [eval_doc_top_k, eval_doc_context_top_k, eval_sample_batch, eval_sample_mode]
    if doc_only:
        eval_inputs += [eval_doc_mode, eval_doc_hybrid]
    eval_event = eval_button.click(
        run_doc_rag_evaluation if doc_only else run_multichannel_rag_evaluation,
        inputs=eval_inputs, outputs=[eval_status, eval_logs, eval_summary],
        concurrency_id="knowledge_eval", concurrency_limit=1,
    )
    sample_outputs = [eval_sample_batch, eval_sample_status, eval_sample_preview, eval_button]
    eval_prepare.click(
        prepare_eval_samples,
        inputs=[eval_dataset, eval_categories, eval_size, eval_seed, eval_custom, eval_sample_mode],
        outputs=sample_outputs, concurrency_id="knowledge_eval", concurrency_limit=1,
    )
    for sampling_control in (eval_categories, eval_size, eval_seed, eval_custom, eval_sample_mode):
        sampling_control.input(invalidate_eval_samples, outputs=sample_outputs, queue=False)
    eval_sample_mode.change(eval_sampling_mode_updates, inputs=eval_sample_mode, outputs=[eval_size, eval_seed], queue=False)
    eval_stop.click(None, cancels=[eval_event])
    eval_dataset.change(eval_dataset_categories, inputs=eval_dataset, outputs=[eval_categories, eval_dataset_info]).then(
        invalidate_eval_samples, outputs=sample_outputs,
    )
    demo.load(eval_dataset_categories, inputs=eval_dataset, outputs=[eval_categories, eval_dataset_info])


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="天宫医疗 · RAG 测试工作台") as demo:
        gr.HTML("""<div class="hero"><div class="eyebrow">TIANGONG · KNOWLEDGE LAB</div>
            <h1>知识问答测试工作台</h1><p>导入知识文档，验证检索效果，用 TruLens 横向评估各检索通道。</p></div>""")
        with gr.Tabs():
            with gr.Tab("知识问答"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=340):
                        chatbot = gr.Chatbot(label="对话记录", height=470, buttons=["copy"],
                                             placeholder="输入问题，开始验证知识库中的回答。")
                        question = gr.Textbox(label="你的问题", lines=2,
                                              placeholder="例如：高血压指南推荐的一线用药有哪些？")
                        with gr.Row():
                            send = gr.Button("发送问题", variant="primary", scale=3)
                            clear = gr.Button("新建会话", scale=1)
                        gr.Examples(examples=[
                            ["高血压诊疗指南推荐的一线用药有哪些？"],
                            ["阿莫西林的禁忌症和常见不良反应是什么？"],
                            ["糖尿病常见并发症有哪些，需要做什么检查？"],
                        ], inputs=question, label="快速测试")
                    with gr.Column(scale=4, min_width=300):
                        with gr.Group(elem_classes="panel"):
                            gr.Markdown("### 检索设置")
                            mode = gr.Dropdown(choices=[MODE_AUTO, *MODE_TO_TOOL], value=MODE_AUTO,
                                               label="检索模式")
                            with gr.Row():
                                role = gr.Dropdown(choices=list(ROLE_OPTIONS), value="患者 / 普通用户", label="回答角色")
                                doc_type = gr.Dropdown(choices=list(DOC_TYPE_OPTIONS), value="不限制", label="文档类型",
                                                       info="仅在文档 RAG 模式生效")
                            with gr.Accordion("会话信息", open=False):
                                user_id = gr.Textbox(value="gradio-user", label="用户 ID")
                                session_id = gr.Textbox(value=f"gradio-{os.getpid()}", label="会话 ID")
                        debug = gr.Textbox(label="问答运行日志", value="等待提问…", lines=14, max_lines=14,
                                           interactive=False, autoscroll=True, elem_classes="log-panel")
            with gr.Tab("文档入库"):
                gr.Markdown("本地 MinerU 解析 → 文档分块 → 向量化 → Milvus 入库。向量化使用 DashScope 服务。")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes="panel"):
                        gr.Markdown("### 导入文档")
                        upload = gr.File(label="选择文档", file_count="single", type="filepath",
                                         file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".xlsx", ".jpg", ".jpeg", ".png"])
                        with gr.Row():
                            upload_doc_type = gr.Dropdown(choices=[k for k in DOC_TYPE_OPTIONS if k != "不限制"],
                                                          value="临床指南", label="文档类型")
                            upload_category = gr.Textbox(value="通用", label="所属分类")
                        upload_mode = gr.Radio(choices=["仅使用 MinerU（失败即停止）", "MinerU 优先，失败回退 LlamaIndex"],
                                               value="仅使用 MinerU（失败即停止）", label="解析策略")
                        upload_parent_child = gr.Checkbox(value=True, label="父子分块",
                                                          info="在结构块内再切 256 token 的子块，检索子块、返回整个结构块")
                        upload_semantic = gr.Checkbox(value=False, label="语义分块",
                                                      info="入库会明显变慢：要对每个结构块做句子级 embedding")
                        gr.Markdown("每次上传重新解析；同名文档更新原始分块，并按勾选重写父子/语义分块，"
                                    "未勾选的旧分块会被清除。解析产物保留供预览。")
                        ingest_button = gr.Button("解析并写入 Milvus", variant="primary")
                        mineru_button = gr.Button("检测 MinerU 服务")
                        mineru_status_box = gr.Markdown("服务状态尚未检测。")
                    with gr.Column(scale=7, min_width=340):
                        ingest_status = gr.Markdown("等待上传文档…")
                        ingest_logs = gr.Textbox(label="入库运行日志", value="等待启动任务…", lines=17, max_lines=17,
                                                 interactive=False, autoscroll=True, elem_classes="log-panel")
                        with gr.Group(elem_classes="panel"):
                            with gr.Row():
                                gr.Markdown("### 已入库文档")
                                refresh_documents = gr.Button("刷新文档列表", size="sm")
                            document_list = gr.Markdown("点击刷新查看知识库文档。", elem_classes="doc-list")
                with gr.Accordion("解析产物与分块预览", open=False):
                    preview_button = gr.Button("查看当前文档解析产物")
                    artifact_status = gr.Markdown("处理文档后可查看解析产物、图片和原始分块。")
                    with gr.Tabs():
                        with gr.Tab("原始 Markdown"):
                            raw_preview = gr.Textbox(label="原始解析正文（预览前 30,000 字符，可下载全文）", lines=15, interactive=False)
                        with gr.Tab("检索正文"):
                            retrieval_preview = gr.Textbox(label="实际用于向量化的正文（预览前 30,000 字符）", lines=15, interactive=False)
                        with gr.Tab("分块"):
                            chunk_preview = gr.Dataframe(headers=["序号", "章节", "类型", "正文"], datatype=["number", "str", "str", "str"], interactive=False)
                        with gr.Tab("图片"):
                            image_preview = gr.Gallery(label="提取图片", columns=3, height=350)
                        with gr.Tab("产物信息"):
                            manifest_preview = gr.Textbox(label="产物清单", lines=12, interactive=False)
                    artifact_downloads = gr.File(label="下载解析产物", file_count="multiple", interactive=False)
                    preview_button.click(preview_artifacts, inputs=upload, outputs=[artifact_status, raw_preview, retrieval_preview, chunk_preview, image_preview, artifact_downloads, manifest_preview])
            with gr.Tab("RAG 评估"):
                with gr.Tabs():
                    with gr.Tab("文档 RAG 评估"):
                        _build_evaluation_page(demo, doc_only=True)
                    with gr.Tab("多通道 RAG 评估"):
                        _build_evaluation_page(demo, doc_only=False)
        inputs = [question, chatbot, user_id, role, mode, doc_type, session_id]
        outputs = [chatbot, debug, session_id, question]
        send.click(answer_question, inputs=inputs, outputs=outputs, concurrency_id="knowledge_chat", concurrency_limit=1)
        question.submit(answer_question, inputs=inputs, outputs=outputs, concurrency_id="knowledge_chat", concurrency_limit=1)
        clear.click(reset_session, outputs=outputs, concurrency_id="knowledge_chat", concurrency_limit=1)
        ingest_button.click(ingest_uploaded_document,
                            inputs=[upload, upload_doc_type, upload_category, upload_mode,
                                    upload_parent_child, upload_semantic],
                            outputs=[ingest_status, document_list, ingest_logs],
                            concurrency_id="knowledge_ingestion", concurrency_limit=1)
        mineru_button.click(mineru_status, outputs=mineru_status_box)
        refresh_documents.click(_indexed_documents_markdown, outputs=document_list)
    return demo


if __name__ == "__main__":
    # 独立页面也使用项目日志配置，让异常出现在终端和日志文件中。
    from src.core.logger import setup_logger

    setup_logger()
    build_demo().queue().launch(
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=None, inbrowser=True,
    )
