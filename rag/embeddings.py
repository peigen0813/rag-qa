"""本地开源嵌入模型封装（默认 BAAI/bge-m3）。

使用 sentence-transformers 在 CPU 上推理，不依赖任何外部向量 API。
模型只加载一次（lru_cache），避免 Streamlit 每次 rerun 都重新加载。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=4)
def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """加载并缓存嵌入模型。

    首次调用会从 Hugging Face 下载模型（约 2GB）。国内可先设置：
        set HF_ENDPOINT=https://hf-mirror.com
    """
    name = model_name or settings.embedding_model
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
