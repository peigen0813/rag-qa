"""引用解析与来源格式化。"""

from __future__ import annotations

import re
from typing import Any, Mapping

from langchain_core.documents import Document

CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")


def parse_citations(text: str) -> list[int]:
    """提取回答中的 [n] 编号，按首次出现顺序去重。"""
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def source_label(metadata: Mapping[str, Any]) -> str:
    source = str(metadata.get("source") or "未知来源")
    page = metadata.get("page")
    if isinstance(page, int) and page > 0:
        return f"{source} · 第 {page} 页"
    return source


def preview_text(text: str, limit: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "……"


def build_sources(documents: list[Document]) -> list[dict]:
    """把文档列表转成 UI 可直接渲染的来源卡片。"""
    sources: list[dict] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        sources.append(
            {
                "index": index,
                "label": source_label(metadata),
                "source": str(metadata.get("source") or ""),
                "page": int(metadata.get("page") or 0),
                "score": float(metadata.get("score") or 0.0),
                "preview": preview_text(doc.page_content),
            }
        )
    return sources


def select_sources(
    answer: str,
    documents: list[Document],
    fallback_top: int = 3,
    max_sources: int = 8,
) -> list[dict]:
    """优先返回回答中真正引用到的来源；一个都没引用时返回前几条。"""
    sources = build_sources(documents)
    valid = {item["index"]: item for item in sources}
    cited = [valid[n] for n in parse_citations(answer) if n in valid]
    if cited:
        return cited[:max_sources]
    return sources[:fallback_top]
