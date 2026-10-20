from rag.ingest import load_documents


def test_sources_manifest_is_not_ingested(tmp_path):
    (tmp_path / "sources.txt").write_text(
        "https://example.com/doc\n", encoding="utf-8"
    )
    (tmp_path / "doc.txt").write_text("这是一篇正文。", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [d.metadata["source"] for d in documents] == ["doc.txt"]


def test_relative_source_and_title(tmp_path):
    (tmp_path / "guide.md").write_text("正文内容", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents[0].metadata["source"] == "guide.md"
    assert documents[0].metadata["title"] == "guide"
    assert documents[0].metadata["page"] == 0
