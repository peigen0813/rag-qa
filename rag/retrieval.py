"""混合检索：向量召回 + BM25 关键词召回 + RRF 融合 + 可选 CrossEncoder 重排。

为什么这样做：
- 纯向量检索对「专有名词、错误码、函数名」不敏感；
- 纯 BM25 不懂同义改写；
- RRF（Reciprocal Rank Fusion）不依赖两路分数量纲，融合稳定；
- 重排模型在候选集上做精细打分，进一步提升 top-k 精度。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jieba
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import Settings, settings
from rag.embeddings import get_embeddings
from rag.ingest import chroma_client_settings

_TOKEN_PATTERN = re.compile(r"[一-鿿]+|[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中英文混合分词。

    中文用 jieba 搜索模式（切得更细，利于召回），
    英文/数字按连续串处理，其余符号直接丢弃。
    """
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer((text or "").lower()):
        piece = match.group(0)
        if piece[0].isascii():
            tokens.append(piece)
        else:
            tokens.extend(t for t in jieba.cut_for_search(piece) if t.strip())
    return tokens


def doc_key(doc: Document) -> str:
    """用于融合去重的稳定 key。"""
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", 0))
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"{source}|{page}|{digest}"


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion。

    score(d) = sum_i 1 / (k + rank_i(d))，rank 从 1 开始。
    同一列表内重复出现的文档只计一次。
    """
    if k < 0:
        raise ValueError("rrf k 必须 >= 0")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(documents[key], score) for key, score in ordered]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass
class RetrievalResult:
    documents: list[Document]
    top_similarity: float
    mode: str = "hybrid"


class HybridRetriever(BaseRetriever):
    """LangChain BaseRetriever 实现，可直接用于 LCEL / Agent 工具。"""

    vectorstore: Any
    bm25: Any = None
    corpus: list[Document] = Field(default_factory=list)
    k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    reranker: Any = None
    rerank_top_n: int = 20

    # ---------- 两路召回 ----------

    def _vector_hits(self, query: str, limit: int) -> list[tuple[Document, float]]:
        """返回 (文档, 余弦相似度)。集合使用 cosine 距离，similarity = 1 - distance。"""
        try:
            pairs = self.vectorstore.similarity_search_with_score(query, k=limit)
            return [
                (doc, max(0.0, min(1.0, 1.0 - float(distance))))
                for doc, distance in pairs
            ]
        except Exception:
            # 某些自定义向量库不返回分数，退化为纯向量召回。
            return [(doc, 0.0) for doc in self.vectorstore.similarity_search(query, k=limit)]

    def _bm25_hits(self, query: str, limit: int) -> list[Document]:
        if self.bm25 is None or not self.corpus:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Document] = []
        for index in ranked[:limit]:
            if float(scores[index]) <= 0:
                break
            hits.append(self.corpus[index])
        return hits

    # ---------- 融合 + 重排 ----------

    def retrieve(self, query: str) -> RetrievalResult:
        query = (query or "").strip()
        if not query:
            return RetrievalResult(documents=[], top_similarity=0.0)

        vector_pairs = self._vector_hits(query, self.candidate_k)
        vector_docs = [doc for doc, _ in vector_pairs]
        similarity_by_key = {doc_key(doc): score for doc, score in vector_pairs}
        bm25_docs = self._bm25_hits(query, self.candidate_k)

        fused = rrf_fuse([vector_docs, bm25_docs], k=self.rrf_k)
        candidates = [doc for doc, _ in fused][: self.rerank_top_n]
        mode = "hybrid"

        if self.reranker is not None and candidates:
            raw_scores = self.reranker.predict(
                [(query, doc.page_content) for doc in candidates]
            )
            rescored = [
                (doc, _sigmoid(float(score)))
                for doc, score in zip(candidates, raw_scores)
            ]
            rescored.sort(key=lambda item: item[1], reverse=True)
            candidates = [doc for doc, _ in rescored]
            scores = [score for _, score in rescored]
            top_similarity = scores[0] if scores else 0.0
            mode = "rerank"
        else:
            fused_scores = {doc_key(doc): score for doc, score in fused}
            scores = [fused_scores.get(doc_key(doc), 0.0) for doc in candidates]
            top_similarity = max(
                (similarity_by_key.get(doc_key(doc), 0.0) for doc in candidates[: self.k]),
                default=0.0,
            )

        documents: list[Document] = []
        for doc, score in zip(candidates[: self.k], scores[: self.k]):
            copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            copy.metadata["score"] = round(float(score), 6)
            copy.metadata["vector_similarity"] = round(
                float(similarity_by_key.get(doc_key(doc), 0.0)), 4
            )
            documents.append(copy)

        return RetrievalResult(
            documents=documents,
            top_similarity=float(top_similarity),
            mode=mode,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query).documents


# ---------- 构建 ----------


def build_bm25(documents: list[Document]):
    if not documents:
        return None
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(doc.page_content) for doc in documents])


def load_all_documents(vectorstore: Chroma) -> list[Document]:
    data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    for content, metadata in zip(
        data.get("documents") or [], data.get("metadatas") or []
    ):
        documents.append(
            Document(page_content=content or "", metadata=dict(metadata or {}))
        )
    return documents


def build_reranker(model_name: str | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name or settings.rerank_model, device="cpu")


def get_vectorstore(settings_obj: Settings | None = None) -> Chroma:
    cfg = settings_obj or settings
    persist = Path(cfg.chroma_dir)
    if not persist.exists():
        raise FileNotFoundError(
            f"向量库不存在：{persist}。请先运行 python scripts/build_index.py --reset"
        )
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=get_embeddings(cfg.embedding_model),
        persist_directory=str(persist),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=chroma_client_settings(),
    )


def get_retriever(
    vectorstore: Chroma | None = None,
    *,
    k: int | None = None,
    use_reranker: bool | None = None,
    reranker: Any = None,
    settings_obj: Settings | None = None,
) -> HybridRetriever:
    cfg = settings_obj or settings
    store = vectorstore or get_vectorstore(cfg)
    corpus = load_all_documents(store)

    want_rerank = cfg.use_reranker if use_reranker is None else use_reranker
    rerank_model = reranker
    if want_rerank and rerank_model is None:
        rerank_model = build_reranker(cfg.rerank_model)

    return HybridRetriever(
        vectorstore=store,
        bm25=build_bm25(corpus),
        corpus=corpus,
        k=cfg.top_k if k is None else k,
        candidate_k=cfg.candidate_k,
        rrf_k=cfg.rrf_k,
        reranker=rerank_model,
    )
