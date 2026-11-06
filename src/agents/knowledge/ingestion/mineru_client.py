# src/agents/knowledge/ingestion/mineru_client.py
"""
MinerU 文档解析客户端：把 PDF / Word / 图片等文档转成 Markdown，供知识库分块入库。

MinerU 对版面、表格、公式的还原比通用解析器更精确，适合医学指南、药品说明书这类排版复杂的文档。
调用方见 doc_ingestion._parse_with_mineru，解析失败时由调用方回退到 LlamaIndex。

对接 MinerU 4.x 自部署 V1 API（mineru-kit api-server）：
    POST   /v1/tasks                  上传文件并提交解析任务，返回 task_id
    GET    /v1/tasks/{task_id}        查询任务状态
    GET    /v1/tasks/{task_id}/result 任务终态后获取内联 Markdown
    DELETE /v1/tasks/{task_id}        取消任务
    GET    /v1/health                 健康检查
4.0 已移除 3.x 根路径下的 /file_parse、/tasks 以及 backend 参数，改用 tier（flash/basic/standard/advanced）选择解析档位。
"""

from __future__ import annotations
import asyncio
import time
import httpx
from pathlib import Path
from loguru import logger

from src.core.config import get_settings

settings = get_settings()

# 任务终态：completed / partial 可取结果，failed / canceled 直接报错
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})
# 轮询间隔（秒）
POLL_INTERVAL = 2
# 单个 HTTP 请求的超时；与 MINERU_TIMEOUT（整个解析任务的总时长上限）区分开
REQUEST_TIMEOUT = 60


async def parse_document(
    file_path: str,
    file_name: str | None = None,
    tier: str | None = None,
) -> str:
    """
    调用 MinerU V1 API 解析文档，返回 Markdown 文本。
    支持 PDF、图片、Word、PPT、Excel 等格式；非 PDF/图片格式由服务端自动按 flash 档位原生解析。

    使用异步任务接口（POST /v1/tasks → 轮询 → 取结果），超大文件不会因单次请求超时而失败。

    Args:
        file_path: 本地文件路径
        file_name: 上传给 MinerU 的文件名，MinerU 依赖扩展名判断格式；默认取 file_path 的文件名
        tier: 解析档位，默认取配置中的 MINERU_TIER；为空时使用服务端 --tier 指定的默认档位

    Returns:
        Markdown 文本。

    Raises:
        RuntimeError: 未配置 MINERU_API_URL，或 MinerU 报告任务失败 / 被取消
        httpx.HTTPStatusError: 接口返回 4xx / 5xx（如档位不可用、文件过大、鉴权失败）
        TimeoutError: 超过 MINERU_TIMEOUT 仍未拿到解析结果
        ValueError: 结果中没有 Markdown 文本
    """
    base_url = settings.MINERU_API_URL.rstrip("/")
    if not base_url:
        raise RuntimeError("未配置 MINERU_API_URL")
    tier = tier or settings.MINERU_TIER
    timeout = settings.MINERU_TIMEOUT

    if not file_name:
        file_name = Path(file_path).name

    # 整个文件一次性读入内存；read_bytes 是同步 IO，放到线程中执行避免阻塞事件循环
    file_bytes = await asyncio.to_thread(Path(file_path).read_bytes)

    # 解析参数放查询串，文件放 multipart 的 files 字段；只要 Markdown，入库不需要图片等其他产物
    params = {"output_formats": "markdown"}
    if tier:
        params["tier"] = tier

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT,
    ) as client:
        # 1. 提交异步解析任务，成功返回 202 和 task_id
        resp = await client.post(
            "/v1/tasks",
            params=params,
            files={"files": (file_name, file_bytes)},
        )
        _raise_for_status(resp)
        task_id = resp.json()["task_id"]
        logger.debug(f"MinerU 任务已提交: {file_name} task_id={task_id} tier={tier or '服务端默认'}")

        # 2. 轮询到终态；timeout 是从提交后开始计算的总等待时长
        deadline = time.monotonic() + timeout
        while True:
            status_resp = await client.get(f"/v1/tasks/{task_id}")
            _raise_for_status(status_resp)
            task = status_resp.json()
            status = task.get("status", "")

            if status in TERMINAL_STATUSES:
                break
            # 其他状态（queued / running）继续等待
            if time.monotonic() >= deadline:
                # 服务端默认并发为 1，放弃的任务不取消会一直占住解析队列
                await _cancel_task(client, task_id)
                raise TimeoutError(f"MinerU 解析超时 ({timeout}s), task_id={task_id}")
            await asyncio.sleep(POLL_INTERVAL)

        if status in {"failed", "canceled"}:
            error = _file_error(task) or status
            logger.error(f"MinerU 解析失败: {file_name} task_id={task_id} {error}")
            raise RuntimeError(f"MinerU 解析失败: {error}")

        # 3. 任务完成后拉取结果，Markdown 内联在 files[].content.markdown 中
        result_resp = await client.get(f"/v1/tasks/{task_id}/result")
        _raise_for_status(result_resp)
        return _extract_markdown(result_resp.json())


def _extract_markdown(result: dict) -> str:
    """
    从 /v1/tasks/{task_id}/result 的响应中提取 Markdown 文本。

    响应格式：{"status": "completed", "files": [{"status": "completed", "content": {"markdown": "..."}, ...}]}
    每次只上传一个文件，因此只取第一个结果。

    Raises:
        RuntimeError: 文件解析失败
        ValueError: 响应中没有 Markdown 文本
    """
    files = result.get("files") or []
    if files and isinstance(files[0], dict):
        first = files[0]
        if first.get("status") == "failed":
            raise RuntimeError(f"MinerU 解析失败: {_error_message(first.get('error'))}")
        md = (first.get("content") or {}).get("markdown")
        if md:
            return md
    # 找不到 Markdown 时报错，由调用方回退到其他解析方式；不能把原始响应当正文返回，否则会混入知识库
    raise ValueError(f"MinerU 响应中未找到 Markdown，响应字段: {list(result.keys())}")


def _auth_headers() -> dict:
    """服务端以 --api-key 启动时需要 Bearer 鉴权；本地匿名访问不带该头。"""
    if settings.MINERU_API_KEY:
        return {"Authorization": f"Bearer {settings.MINERU_API_KEY}"}
    return {}


def _raise_for_status(resp: httpx.Response) -> None:
    """
    4xx / 5xx 时抛出 HTTPStatusError，并带上 V1 错误体中的 code 和 message。

    V1 错误格式：{"error": {"type": "...", "code": "...", "message": "...", "param": "..."}}
    """
    if not resp.is_error:
        return
    try:
        detail = _error_message(resp.json().get("error"))
    except ValueError:
        detail = resp.text[:200]
    raise httpx.HTTPStatusError(
        f"MinerU 接口返回 {resp.status_code}: {detail}",
        request=resp.request,
        response=resp,
    )


def _error_message(error: dict | None) -> str:
    """把 V1 错误对象格式化为 "code: message"。"""
    if not error:
        return "未知错误"
    code = error.get("code")
    message = error.get("message", "未知错误")
    return f"{code}: {message}" if code else message


def _file_error(task: dict) -> str | None:
    """取任务中第一个文件的错误信息；任务级失败（如被取消）时文件可能没有 error。"""
    for file in task.get("files") or []:
        if file.get("error"):
            return _error_message(file["error"])
    return None


async def _cancel_task(client: httpx.AsyncClient, task_id: str) -> None:
    """尽力取消任务；取消失败只记录告警，不覆盖调用方要抛出的超时异常。"""
    try:
        await client.delete(f"/v1/tasks/{task_id}")
    except httpx.HTTPError as e:
        logger.warning(f"MinerU 取消任务失败 task_id={task_id}: {e}")


async def check_mineru_health() -> dict:
    """
    检查 MinerU 服务健康状态。

    正常时返回 {"status": "ok", "version": "4.x.x", "features": {...}}；模型未就绪时服务端返回 503。
    不抛异常：服务不可达或返回非 2xx 时返回 {"status": "unavailable", "error": ...}，便于直接作为健康检查接口的响应。
    """
    try:
        # 健康检查用固定的短超时，不沿用解析任务的 MINERU_TIMEOUT
        async with httpx.AsyncClient(timeout=10, headers=_auth_headers()) as client:
            resp = await client.get(f"{settings.MINERU_API_URL.rstrip('/')}/v1/health")
            _raise_for_status(resp)
            return resp.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}
