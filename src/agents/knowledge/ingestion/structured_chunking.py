"""Markdown 结构分块：章节和正文切分用 LlamaIndex 组件，表格切分保留手写实现。

- 章节：MarkdownNodeParser 按标题切分，得到每节的完整标题路径
- 正文：SentenceSplitter.split_text_metadata_aware，切分时扣除“章节：…”前缀占用的 token
- 表格：LlamaIndex 只能整表成块（MarkdownElementNodeParser 还要逐表调用 LLM 生成摘要），
  这里按行打包、每块重复表头，超长单行按列切分；规则 HTML 表格先转成 Markdown
"""

from __future__ import annotations

import re
from typing import Any, Callable

from bs4 import BeautifulSoup, Tag
from markdown_it import MarkdownIt
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document, TextNode


MAX_CHUNK_TOKENS = 1024
# 标题路径分隔符，与 section_path 元数据和“章节：”前缀的格式一致
PATH_SEPARATOR = " > "
# 与 MarkdownNodeParser 识别标题的规则一致，用于取出每节第一行的当前标题
_HEADING = re.compile(r"^#+\s(.*)")


def _table_row(cells: list[str]) -> str:
    # 转义单元格内的竖线，不把正文误当成新的列。
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"


def _html_table_markdown(table: Tag) -> str | None:
    """规则矩形表格转 Markdown；无法可靠定位列时交给可读文本回退。"""
    if table.find("table") is not None:
        return None
    rows, header_count = [], 0
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if row.find_parent("tr") is not None or any(cell.find(["th", "td", "tr"]) for cell in cells):
            return None
        if not cells or any(
            str(cell.get(attribute, "1")) != "1"
            for cell in cells for attribute in ("rowspan", "colspan")
        ):
            return None
        if rows and len(cells) != len(rows[0]):
            return None
        if header_count == len(rows) and (
            row.find_parent("thead") is not None or all(cell.name == "th" for cell in cells)
        ):
            header_count += 1
        rows.append([cell.get_text(" ", strip=True) for cell in cells])
    if not rows:
        return None
    header_count = max(1, header_count)
    headers = [" / ".join(row[column] for row in rows[:header_count]) for column in range(len(rows[0]))]
    return "\n".join([
        _table_row(headers), _table_row(["---"] * len(headers)),
        *(_table_row(row) for row in rows[header_count:]),
    ])


def _html_blocks(html: str) -> list[tuple[str, str]]:
    """分离 HTML 表格与周围正文，保留多个表格的原有顺序。"""
    blocks = []

    def visit(element: Any) -> None:
        if isinstance(element, Tag):
            if element.name == "table":
                markdown = _html_table_markdown(element)
                if markdown:
                    caption = element.find("caption")
                    if caption:
                        blocks.append((caption.get_text(" ", strip=True), "paragraph"))
                    blocks.append((markdown, "table"))
                else:
                    blocks.append((element.get_text(" ", strip=True), "paragraph"))
            elif element.find("table") is not None:
                for child in element.children:
                    visit(child)
            else:
                blocks.append((element.get_text(" ", strip=True), "paragraph"))
        elif str(element).strip():
            blocks.append((str(element).strip(), "paragraph"))

    for element in BeautifulSoup(html, "html.parser").children:
        visit(element)
    return [(body, kind) for body, kind in blocks if body]


def _split_table_row(
    row: str, header: str, prefix: str, splitter: Any, parser: MarkdownIt, limit: int,
) -> list[str] | None:
    """预留章节、表头及列分隔符预算，逐列切分超长行。"""
    cells = []
    in_cell = False
    for token in parser.parse(header + "\n" + row):
        if token.type == "td_open":
            in_cell = True
        elif token.type == "td_close":
            in_cell = False
        elif in_cell and token.type == "inline":
            cells.append(token.content)
    if not cells:
        return None
    count = lambda text: len(splitter._tokenizer(text))
    overhead = count(prefix + header + "\n" + _table_row([""] * len(cells)))
    cell_budget = (limit - overhead) // len(cells)
    while cell_budget > 0:
        smaller = SentenceSplitter(
            chunk_size=cell_budget,
            chunk_overlap=min(splitter.chunk_overlap, cell_budget - 1),
            tokenizer=splitter._tokenizer,
            paragraph_separator=splitter.paragraph_separator,
        )
        pieces = [smaller.split_text(cell) or [""] for cell in cells]
        fragments = [
            _table_row([
                parts[0] if len(parts) == 1 else parts[index] if index < len(parts) else ""
                for parts in pieces
            ])
            for index in range(max(map(len, pieces)))
        ]
        # tokenizer 的合并边界不一定可加，检查最终文本后再收紧预算。
        excess = max(count(prefix + header + "\n" + fragment) - limit for fragment in fragments)
        if excess <= 0:
            return fragments
        cell_budget -= max(1, excess)
    return None


def _bounded_splitter(splitter: SentenceSplitter) -> SentenceSplitter:
    """正文块的总长度不超过 MAX_CHUNK_TOKENS；调用方配置更小时沿用原分块器。"""
    if splitter.chunk_size <= MAX_CHUNK_TOKENS:
        return splitter
    return SentenceSplitter(
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=min(splitter.chunk_overlap, MAX_CHUNK_TOKENS // 8),
        tokenizer=splitter._tokenizer,
        paragraph_separator=splitter.paragraph_separator,
    )


def _section_prefix(section: str, count: Callable[[str], int], limit: int) -> str:
    """“章节：A > B”前缀最多占块预算的一半，过深的路径从末尾截断。"""
    if not section:
        return ""
    while section and count(f"章节：{section}\n\n") >= limit // 2:
        section = section[:len(section) // 2]
    return f"章节：{section}\n\n" if section else ""


def _sections(document: Document) -> list[tuple[str, str]]:
    """MarkdownNodeParser 按标题切节，返回 (完整标题路径, 去掉标题行的正文)。"""
    parser = MarkdownNodeParser(header_path_separator=PATH_SEPARATOR, include_prev_next_rel=False)
    sections = []
    for node in parser.get_nodes_from_documents([Document(text=document.text)]):
        # header_path 形如 " > A > B > "，只含上级标题；当前标题在正文第一行
        parents = node.metadata.get("header_path", "")[len(PATH_SEPARATOR):-len(PATH_SEPARATOR)]
        titles = [parents] if parents else []
        first, _, rest = node.text.partition("\n")
        match = _HEADING.match(first)
        if match:
            titles.append(match.group(1).strip())
            body = rest
        else:
            body = node.text
        sections.append((PATH_SEPARATOR.join(titles), body.strip()))
    return sections


def _segments(body: str, parser: MarkdownIt) -> list[tuple[str, str]]:
    """按 markdown-it 顶层块把章节正文拆成 paragraph / table 片段，相邻正文合并，保持原有顺序。"""
    lines = body.splitlines()
    segments: list[tuple[str, str]] = []

    def add(text: str, kind: str) -> None:
        if kind == "paragraph" and segments and segments[-1][1] == "paragraph":
            segments[-1] = (segments[-1][0] + "\n\n" + text, "paragraph")
        else:
            segments.append((text, kind))

    depth = 0
    for token in parser.parse(body):
        if depth == 0 and token.map and token.type not in {"inline", "heading_close"}:
            start, end = token.map
            block = "\n".join(lines[start:end]).strip()
            if token.type == "table_open":
                add(block, "table")
            elif token.type == "html_block":
                for part, kind in _html_blocks(block):
                    add(part, kind)
            elif block:
                add(block, "paragraph")
        depth += token.nesting
    return segments


def _table_chunks(
    body: str, prefix: str, splitter: SentenceSplitter, parser: MarkdownIt,
) -> list[str] | None:
    """按行打包表格，每块重复章节前缀和表头；返回 None 表示无法按表格切分，交给正文分块。"""
    table_lines = body.splitlines()
    header = "\n".join(table_lines[:2])
    limit = min(splitter.chunk_size, MAX_CHUNK_TOKENS)
    count = lambda text: len(splitter._tokenizer(text))
    if count(prefix + header) >= limit:
        return None
    rows = []
    for row in table_lines[2:]:
        if count(prefix + header + "\n" + row) <= limit:
            rows.append(row)
        else:
            fragments = _split_table_row(row, header, prefix, splitter, parser, limit)
            if fragments is None:
                return None
            rows.extend(fragments)
    chunks, group = [], []
    for row in rows:
        if group and count(prefix + header + "\n" + "\n".join([*group, row])) > limit:
            chunks.append(prefix + header + "\n" + "\n".join(group))
            group = []
        group.append(row)
    chunks.append(prefix + header + ("\n" + "\n".join(group) if group else ""))
    return chunks


def _make_node(text: str, document: Document, section: str, chunk_type: str) -> TextNode:
    metadata = {**document.metadata, "section_path": section, "chunk_type": chunk_type}
    node = TextNode(text=text, metadata=metadata)
    # 章节路径已写进正文前缀，元数据不再重复参与向量化和 LLM 上下文
    node.excluded_embed_metadata_keys = list(metadata)
    node.excluded_llm_metadata_keys = list(metadata)
    return node


def chunk_documents(documents: list, splitter: SentenceSplitter) -> list[TextNode]:
    parser = MarkdownIt("commonmark").enable("table")
    text_splitter = _bounded_splitter(splitter)
    count = lambda text: len(splitter._tokenizer(text))
    nodes: list[TextNode] = []
    for document in documents:
        before = len(nodes)
        for section, body in _sections(document):
            prefix = _section_prefix(section, count, text_splitter.chunk_size)
            for segment, kind in _segments(body, parser):
                if kind == "table":
                    chunks = _table_chunks(segment, prefix, splitter, parser)
                    if chunks is not None:
                        nodes.extend(_make_node(chunk, document, section, "table") for chunk in chunks)
                        continue
                for piece in text_splitter.split_text_metadata_aware(segment, prefix):
                    if piece.strip():
                        nodes.append(_make_node(prefix + piece, document, section, "paragraph"))
        # 只有标题、没有正文的文档也保留一个分块，避免整篇文档检索不到
        if len(nodes) == before and document.text.strip():
            nodes.extend(
                _make_node(piece, document, "", "paragraph")
                for piece in text_splitter.split_text(document.text) if piece.strip()
            )
    return nodes
