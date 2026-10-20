from langchain_core.documents import Document

from rag.ingest import make_splitter, split_documents

SAMPLE = (
    "FastAPI 是一个用于构建 API 的现代 Web 框架。"
    "它使用 Python 的类型提示来自动生成文档。"
    "路径参数用于从 URL 中提取数据。"
    "查询参数用于过滤和分页。"
) * 20


def _split(text: str = SAMPLE):
    doc = Document(
        page_content=text,
        metadata={"source": "guide.md", "page": 0, "title": "指南"},
    )
    return split_documents([doc], make_splitter(chunk_size=120, chunk_overlap=20))


def test_chunks_have_sequential_index_per_source():
    chunks = _split()
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "guide.md" for c in chunks)
    assert all(c.metadata["page"] == 0 for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    first = _split()
    second = _split()
    assert [c.metadata["chunk_id"] for c in first] == [
        c.metadata["chunk_id"] for c in second
    ]


def test_chunk_content_is_preserved_in_order():
    chunks = _split()
    joined = "".join(c.page_content for c in chunks)
    assert joined  # 非空
    assert "FastAPI" in joined
    assert "路径参数" in joined


def test_empty_input_returns_empty_list():
    assert split_documents([]) == []
