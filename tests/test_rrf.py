from langchain_core.documents import Document

from rag.retrieval import rrf_fuse, tokenize


def doc(key: str) -> Document:
    return Document(page_content=f"content-{key}", metadata={"chunk_id": key})


def test_document_in_both_lists_wins():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf_fuse([[a, b], [b, c]], k=60)

    assert fused[0][0].metadata["chunk_id"] == "b"
    assert fused[0][1] > fused[1][1] > 0


def test_duplicate_within_one_list_is_counted_once():
    a = doc("a")
    fused = rrf_fuse([[a, a, a]], k=60)
    assert len(fused) == 1
    assert fused[0][1] == 1 / 61


def test_order_is_deterministic_for_ties():
    a, b = doc("a"), doc("b")
    fused = rrf_fuse([[a], [b]], k=60)
    assert [d.metadata["chunk_id"] for d, _ in fused] == ["a", "b"]


def test_empty_lists():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []


def test_k_must_be_non_negative():
    try:
        rrf_fuse([[doc("a")]], k=-1)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("k=-1 应该报错")


def test_tokenize_handles_mixed_chinese_and_english():
    tokens = tokenize("FastAPI 的路径参数 path parameter 怎么用？")
    assert "fastapi" in tokens
    assert "path" in tokens
    assert any("参数" in token or "路径" in token for token in tokens)


def test_tokenize_drops_punctuation_only_input():
    assert tokenize("，。！？") == []
