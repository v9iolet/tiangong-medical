"""保存 Markdown/HTML/引用式内嵌图片；检索正文只保留图注。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.agents.knowledge.ingestion.artifact_store import ArtifactPaths

# 图片地址由原 Markdown 的语法边界限定，不用跨正文的贪婪 Base64 正则。
_IMAGE = re.compile(r'!\[(?P<alt>[^\]]*)\]\(\s*(?P<src><[^>]*>|[^)]+)\)', re.S)
_REFERENCE_IMAGE = re.compile(r'!\[(?P<alt>[^\]]*)\](?:\[(?P<ref>[^\]]*)\])?')
_DEFINITION = re.compile(r'^ {0,3}\[(?P<ref>[^\]]+)\]:\s*(?P<src>\S+)(?:[^\n]*)$', re.M)
_HTML_IMAGE = re.compile(r'<img\b[^>]*>', re.I | re.S)
_DATA = re.compile(r'^data:image/(?P<format>[\w.+-]+);base64,(?P<data>.*)$', re.I | re.S)
_DATA_MARKER = re.compile(r'data\s*:\s*image/[^,\n]*;\s*base64\s*,', re.I)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_EXTENSIONS = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "gif": "gif", "webp": "webp", "bmp": "bmp", "svg+xml": "svg"}


@dataclass(frozen=True)
class AssetResult:
    document_markdown: str
    retrieval_markdown: str
    assets: list[dict[str, Any]]
    image_files: dict[str, bytes]


def ensure_no_base64(text: str) -> None:
    if _DATA_MARKER.search(text):
        raise ValueError("检索正文仍含图片 Base64，停止向量化")


def _check_reference_payload_boundaries(markdown: str) -> None:
    # 非标准的跨行引用式 Data URI 没有明确闭合符号，不能仅删第一行后把编码
    # 残片当正文。停止本次处理，原始解析文件仍可用于人工纠正。
    for definition in _DEFINITION.finditer(markdown):
        if _DATA_MARKER.search(definition.group("src")):
            remainder = markdown[definition.end():]
            following = remainder.splitlines()
            if len(following) > 1 and re.fullmatch(r"[A-Za-z0-9+/=]+", following[1].strip()):
                raise ValueError("引用式图片 Base64 跨行边界不明确，停止向量化")


def _caption(alt: str) -> str:
    # 防止通过 alt 文本把编码再次带回向量输入。
    ensure_no_base64(alt)
    return f"[图片：{alt.strip() or '未命名图片'}]"


def extract_images_and_clean_markdown(markdown: str, paths: ArtifactPaths) -> AssetResult:
    _check_reference_payload_boundaries(markdown)
    assets: dict[str, dict[str, Any]] = {}
    image_files: dict[str, bytes] = {}
    definitions = {m.group("ref").strip().casefold(): m.group("src") for m in _DEFINITION.finditer(markdown)}

    def save_image(src: str, alt: str) -> str | None:
        match = _DATA.match(src.strip().strip("<>"))
        if not match:
            return None
        extension = _EXTENSIONS.get(match.group("format").lower())
        encoded = re.sub(r'\s+', '', match.group("data"))
        if extension is None or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
            logger.warning("内嵌图片格式或大小不支持，只保留图注")
            return None
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            logger.warning("内嵌图片编码无效，只保留图注")
            return None
        if not payload or len(payload) > MAX_IMAGE_BYTES:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"{digest}.{extension}"
        image_files[f"images/{filename}"] = payload
        entry = assets.setdefault(digest, {
            "asset_id": digest, "filename": filename, "size": len(payload),
            "format": extension, "minio_object": f"{paths.prefix}/images/{filename}", "captions": [],
        })
        if alt and alt not in entry["captions"]:
            entry["captions"].append(alt)
        return f"images/{filename}"

    def process(text: str, display: bool) -> str:
        def inline_image(match: re.Match) -> str:
            alt, src = match.group("alt"), match.group("src").strip()
            # Markdown 可选图片 title 不属于地址，Base64 中也不会出现引号。
            src = re.split(r'\s+[\'"]', src, maxsplit=1)[0]
            caption = _caption(alt)
            if display:
                if not _DATA_MARKER.search(src):
                    return match.group()
                image = save_image(src, alt)
                return f"![{alt}]({image})" if image else caption
            return caption

        def reference_image(match: re.Match) -> str:
            alt = match.group("alt")
            reference = (match.group("ref") or alt).strip().casefold()
            caption = _caption(alt)
            src = definitions.get(reference)
            if display and src:
                image = save_image(src, alt)
                return f"![{alt}]({image})" if image else caption
            return caption

        def html_image(match: re.Match) -> str:
            tag = BeautifulSoup(match.group(), "html.parser").find("img")
            alt = str(tag.get("alt", "")) if tag else ""
            src = str(tag.get("src", "")) if tag else ""
            caption = _caption(alt)
            if display:
                image = save_image(src, alt)
                return f"![{alt}]({image})" if image else caption
            return caption

        text = _HTML_IMAGE.sub(html_image, text)
        text = _IMAGE.sub(inline_image, text)
        # 不重复处理已改为 images/... 的展示链接。
        text = _REFERENCE_IMAGE.sub(reference_image, text) if not display else _REFERENCE_IMAGE.sub(
            lambda m: m.group() if text[m.end():m.end()+1] == "(" else reference_image(m), text,
        )
        text = _DEFINITION.sub(lambda m: "" if _DATA_MARKER.search(m.group("src")) else m.group(), text)
        ensure_no_base64(text)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    display = process(markdown, True)
    retrieval = process(markdown, False)
    return AssetResult(display, retrieval, list(assets.values()), image_files)


def conservative_clean(markdown: str) -> str:
    """结构提取失败时只去掉图片语法，不把含编码的原文退回分块器。"""
    _check_reference_payload_boundaries(markdown)
    cleaned = _HTML_IMAGE.sub("[图片]", markdown)
    cleaned = _IMAGE.sub(lambda m: _caption(m.group("alt")), cleaned)
    cleaned = _DEFINITION.sub(lambda m: "" if _DATA_MARKER.search(m.group("src")) else m.group(), cleaned)
    cleaned = _REFERENCE_IMAGE.sub(lambda m: _caption(m.group("alt")), cleaned)
    ensure_no_base64(cleaned)
    return cleaned.strip()
