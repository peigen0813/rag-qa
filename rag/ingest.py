"""文档入库：加载 -> 切分 -> 写入 Chroma。

设计要点：
- 按后缀分派 loader（txt/md/html/pdf）；
- 中文友好切分（separators 包含中文标点）；
- chunk ID 由「来源 + 页码 + 内容」的 sha256 生成，重复入库不会产生重复块；
- `reset=True` 时删除旧索引重建。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import settings
from rag.embeddings import get_embeddings

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}

# 这是抓取脚本的 URL 清单，不是语料正文，入库时跳过。
MANIFEST_FILENAME = "sources.txt"

# 中文文档没有空格，默认分隔符会把整段当成一块；加入中文标点后切分更自然。
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def chroma_client_settings():
    """关闭 Chroma 遥测；导入失败时返回 None。"""
    try:
        from chromadb.config import Settings as ChromaSettings

        return ChromaSettings(anonymized_telemetry=False)
    except Exception:  # pragma: no cover - 仅旧版本或未安装时触发
        return None


def _html_to_text(html: str) -> tuple[str, str]:
    """从 HTML 中提取标题与正文。"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]
    ):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    text = node.get_text("\n", strip=True)
    return title, text


def _load_one(path: Path) -> list[Document]:
    """按后缀加载单个文件，返回带 source/page 的 Document。"""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": str(path)})]

    if suffix in {".html", ".htm"}:
        title, text = _html_to_text(
            path.read_text(encoding="utf-8", errors="replace")
        )
        return [
            Document(
                page_content=text,
                metadata={"source": str(path), "title": title},
            )
        ]

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        documents: list[Document] = []
        for index, page in enumerate(reader.pages):
            documents.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(path), "page": index},
                )
            )
        return documents

    return []


def load_documents(data_dir: Path | str) -> list[Document]:
    """加载目录下所有支持的文档，并把元数据规范化为相对路径。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{root}。请先运行 python scripts/fetch_docs.py"
        )

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name == MANIFEST_FILENAME:
            continue
        try:
            loaded = _load_one(path)
        except Exception as exc:  # 单个文件失败不影响整体
            print(f"[跳过] {path.name}: {exc}")
            continue

        relative = path.relative_to(root).as_posix()
        is_pdf = path.suffix.lower() == ".pdf"
        for doc in loaded:
            raw_page = doc.metadata.get("page", 0)
            try:
                page = int(raw_page) + 1 if is_pdf else 0
            except (TypeError, ValueError):
                page = 0
            doc.metadata.update(
                {
                    "source": relative,
                    "page": page,
                    "title": doc.metadata.get("title") or path.stem,
                }
            )
        documents.extend(loaded)
    return documents


def make_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=settings.chunk_overlap
        if chunk_overlap is None
        else chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )


def chunk_id(doc: Document) -> str:
    """稳定、与进程无关的块 ID。"""
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", 0))
    payload = f"{source}\x1f{page}\x1f{doc.page_content.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def dedupe_chunks(chunks: list[Document]) -> list[Document]:
    """按 chunk_id 去重，保留首次出现的块。

    同一文档里常有大段重复的样板文字（导航、提示块等），切分后会产生
    内容完全相同的块；它们的 chunk_id 相同，Chroma 会拒绝重复 ID。
    相同内容对检索没有额外价值，直接丢弃即可。
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        cid = str(chunk.metadata.get("chunk_id") or chunk_id(chunk))
        chunk.metadata["chunk_id"] = cid
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(chunk)
    return unique


def split_documents(
    documents: list[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """逐文档切分，并写入 source/page/chunk_index/chunk_id。"""
    splitter = splitter or make_splitter()
    chunks: list[Document] = []
    counters: dict[str, int] = {}

    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", 0)
        for piece in splitter.split_documents([doc]):
            piece.metadata["source"] = source
            piece.metadata["page"] = page
            piece.metadata.setdefault("title", doc.metadata.get("title", ""))
            group = f"{source}|{page}"
            index = counters.get(group, 0)
            counters[group] = index + 1
            piece.metadata["chunk_index"] = index
            piece.metadata["chunk_id"] = chunk_id(piece)
            chunks.append(piece)
    return chunks


def build_index(
    data_dir: Path | str | None = None,
    persist_dir: Path | str | None = None,
    *,
    reset: bool = False,
    collection_name: str | None = None,
) -> dict:
    """构建向量索引，返回统计信息。"""
    root = Path(data_dir or settings.data_dir)
    persist = Path(persist_dir or settings.chroma_dir)
    collection = collection_name or settings.collection_name

    if reset and persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)

    raw_documents = load_documents(root)
    split_chunks = split_documents(raw_documents)
    chunks = dedupe_chunks(split_chunks)
    duplicate_chunks = len(split_chunks) - len(chunks)
    if not chunks:
        raise RuntimeError(
            f"在 {root} 下没有找到可入库的文档。"
            "请先运行 python scripts/fetch_docs.py，或把 txt/md/pdf/html 放进 data/。"
        )

    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(persist),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=chroma_client_settings(),
    )

    # 幂等：只写入尚不存在的 chunk，重复运行不会产生重复向量。
    existing_ids = set(store.get(include=["metadatas"]).get("ids") or [])
    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    if new_chunks:
        store.add_documents(
            new_chunks, ids=[c.metadata["chunk_id"] for c in new_chunks]
        )

    stats = {
        "data_dir": str(root),
        "chroma_dir": str(persist),
        "collection": collection,
        "files": len({d.metadata.get("source") for d in raw_documents}),
        "raw_documents": len(raw_documents),
        "chunks": len(chunks),
        "duplicate_chunks": duplicate_chunks,
        "new_chunks": len(new_chunks),
        "skipped_existing": len(chunks) - len(new_chunks),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    (persist / "ingest_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
