"""复用现有 MinIO 接口保存解析产物；主入库流程每次重新解析，本地目录供预览。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.config import get_settings
from src.infra.minio_client import download_file, ensure_bucket_exists, object_exists, upload_file

PROCESSING_VERSION = "markdown-assets-v1"


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    doc_id: str

    @property
    def prefix(self) -> str:
        return f"knowledge/artifacts/{self.doc_id}"


def artifact_paths(doc_id: str) -> ArtifactPaths:
    root = Path(get_settings().PARSED_ARTIFACT_DIR).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[4] / root
    return ArtifactPaths(root / doc_id, doc_id)


def source_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, data: bytes) -> None:
    """原子写入，缓存读者不会读到写了一半的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(path: Path, payload: Any) -> None:
    write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def read_manifest(paths: ArtifactPaths) -> dict[str, Any] | None:
    try:
        manifest = json.loads((paths.root / "manifest.json").read_text(encoding="utf-8"))
        return manifest if isinstance(manifest, dict) else None
    except (OSError, ValueError):
        return None


def _valid_cache(manifest: dict, fingerprint: str, mineru_only: bool) -> bool:
    settings = get_settings()
    return (
        isinstance(manifest, dict)
        and manifest.get("parser") in {"mineru", "llamaindex"}
        and manifest.get("source_hash") == fingerprint
        and manifest.get("processing_version") == PROCESSING_VERSION
        and manifest.get("mineru_tier") == settings.MINERU_TIER
        and manifest.get("mineru_url") == settings.MINERU_API_URL
        and manifest.get("complete") is True
        and (not mineru_only or manifest.get("parser") == "mineru")
    )


def _documents_valid(documents: Any, manifest: dict) -> bool:
    if not isinstance(documents, list) or not documents:
        return False
    if not all(isinstance(doc, dict) and isinstance(doc.get("text"), str)
               and isinstance(doc.get("metadata"), dict) for doc in documents):
        return False
    checksum = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return checksum == manifest.get("documents_hash")


def load_cache(paths: ArtifactPaths, fingerprint: str, mineru_only: bool) -> dict | None:
    """保留现有 doc_id；用原文件哈希校验同名文件，严格模式不复用回退结果。"""
    manifest = read_manifest(paths)
    if manifest and _valid_cache(manifest, fingerprint, mineru_only):
        try:
            # 原始解析文档保留页码；每次重试重新清理和分块，避免复用半成品。
            documents = json.loads((paths.root / "documents.json").read_text(encoding="utf-8"))
            if _documents_valid(documents, manifest):
                return {"manifest": manifest, "documents": documents}
        except (OSError, ValueError, TypeError):
            pass
    # 本地缓存丢失时可从 MinIO 恢复解析正文，不把服务不可用当成有效缓存。
    try:
        key = f"{paths.prefix}/manifest.json"
        if not object_exists(key):
            return None
        manifest = json.loads(download_file(key))
        if not _valid_cache(manifest, fingerprint, mineru_only):
            return None
        documents = json.loads(download_file(f"{paths.prefix}/documents.json"))
        if not _documents_valid(documents, manifest):
            return None
        return {"manifest": manifest, "documents": documents}
    except Exception as exc:
        logger.warning("MinIO 缓存不可用，继续解析 doc_id={}: {}", paths.doc_id, type(exc).__name__)
        return None


def persist_artifacts(paths: ArtifactPaths, files: dict[str, bytes], *, reset: bool = False) -> dict[str, bool]:
    """写本地并复用 upload_file；磁盘故障直接按字节上传，不依赖临时目录。"""
    local, minio = True, True
    try:
        write_json(paths.root / "manifest.json", {"complete": False})
        if reset:
            for name in ("document.md", "retrieval.md", "assets.json", "chunks.jsonl"):
                (paths.root / name).unlink(missing_ok=True)
            images = paths.root / "images"
            if images.exists():
                for image in images.iterdir():
                    if image.is_file():
                        image.unlink()
        for name, data in files.items():
            if name != "manifest.json":
                write_bytes(paths.root / name, data)
        write_bytes(paths.root / "manifest.json", files["manifest.json"])
    except OSError:
        local = False
        logger.opt(exception=True).warning("本地产物保存失败，尝试直接上传 MinIO doc_id={}", paths.doc_id)
    try:
        ensure_bucket_exists()
        upload_file(f"{paths.prefix}/manifest.json", b'{"complete": false}', "application/json")
        for name, data in files.items():
            if name == "manifest.json":
                continue
            content_type = "text/markdown; charset=utf-8" if name.endswith(".md") else "application/json" if name.endswith(".json") else "application/octet-stream"
            upload_file(f"{paths.prefix}/{name}", data, content_type)
        upload_file(f"{paths.prefix}/manifest.json", files["manifest.json"], "application/json")
    except Exception:
        minio = False
        logger.opt(exception=True).warning("解析产物同步 MinIO 失败 doc_id={}", paths.doc_id)
    if not local and not minio:
        raise RuntimeError("本地和 MinIO 均未保存解析产物，停止向量化")
    return {"local": local, "minio": minio}
