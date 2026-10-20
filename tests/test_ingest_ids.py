from langchain_core.documents import Document

from rag.ingest import chunk_id, dedupe_chunks


def make_doc(content="同一段内容", source="a.md", page=0):
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_same_content_same_id():
    assert chunk_id(make_doc()) == chunk_id(make_doc())


def test_content_change_changes_id():
    assert chunk_id(make_doc(content="内容 A")) != chunk_id(make_doc(content="内容 B"))


def test_source_change_changes_id():
    assert chunk_id(make_doc(source="a.md")) != chunk_id(make_doc(source="b.md"))


def test_page_change_changes_id():
    assert chunk_id(make_doc(page=1)) != chunk_id(make_doc(page=2))


def test_id_is_short_hex():
    value = chunk_id(make_doc())
    assert len(value) == 32
    int(value, 16)  # 能按十六进制解析


def test_dedupe_chunks_keeps_first_occurrence():
    same = make_doc()
    first = Document(page_content=same.page_content, metadata={**same.metadata})
    first.metadata["chunk_id"] = chunk_id(first)
    second = Document(page_content=same.page_content, metadata={**same.metadata})
    second.metadata["chunk_id"] = chunk_id(second)
    other = make_doc(content="不同内容")

    unique = dedupe_chunks([first, second, other])

    assert [c.metadata["chunk_id"] for c in unique] == [
        first.metadata["chunk_id"],
        chunk_id(other),
    ]


def test_dedupe_chunks_fills_missing_id():
    chunks = dedupe_chunks([make_doc()])
    assert chunks[0].metadata["chunk_id"] == chunk_id(chunks[0])
