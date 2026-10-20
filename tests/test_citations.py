from langchain_core.documents import Document

from rag.citations import (
    build_sources,
    parse_citations,
    select_sources,
    source_label,
)


def make_docs():
    return [
        Document(
            page_content="第一段资料",
            metadata={"source": "a.md", "page": 0, "score": 0.031},
        ),
        Document(
            page_content="第二段资料",
            metadata={"source": "manual.pdf", "page": 3, "score": 0.028},
        ),
        Document(
            page_content="第三段资料",
            metadata={"source": "b.html", "page": 0, "score": 0.02},
        ),
    ]


def test_parse_citations_dedupes_in_order():
    assert parse_citations("结论甲 [2]，结论乙 [1][2]，另见 [3]。") == [2, 1, 3]
    assert parse_citations("没有引用") == []
    assert parse_citations("") == []


def test_source_label_shows_pdf_page_only_when_positive():
    assert source_label({"source": "a.md", "page": 0}) == "a.md"
    assert source_label({"source": "manual.pdf", "page": 3}) == "manual.pdf · 第 3 页"


def test_build_sources_includes_index_and_preview():
    sources = build_sources(make_docs())
    assert [item["index"] for item in sources] == [1, 2, 3]
    assert sources[1]["label"] == "manual.pdf · 第 3 页"
    assert sources[0]["preview"] == "第一段资料"


def test_select_sources_prefers_cited_documents():
    selected = select_sources("根据资料 [2]，答案是……", make_docs())
    assert [item["index"] for item in selected] == [2]


def test_select_sources_falls_back_to_top_results():
    selected = select_sources("没有标注引用。", make_docs(), fallback_top=2)
    assert [item["index"] for item in selected] == [1, 2]


def test_select_sources_ignores_out_of_range_citations():
    selected = select_sources("见 [9]", make_docs(), fallback_top=1)
    assert [item["index"] for item in selected] == [1]
