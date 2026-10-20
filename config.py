"""集中管理配置。

原则：
- 所有密钥只从环境变量 / `.env` 读取，绝不写进代码；
- 缺 key 时报错信息不回显 key；
- 所有可调参数都能用环境变量覆盖，方便实验和部署。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# 本地开发时从 .env 读取；已经存在的环境变量优先级更高（load_dotenv 默认不覆盖）。
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。"""

    # LLM（OpenAI 兼容接口：DeepSeek / OpenAI / 其他）
    api_key: str = ""
    base_url: str | None = None
    model: str = "deepseek-chat"
    temperature: float = 0.2
    request_timeout: int = 120

    # 本地向量模型与重排模型
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = False

    # 数据与索引
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "rag_docs"

    # 切分
    chunk_size: int = 500
    chunk_overlap: int = 80

    # 检索
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    # 实测：相关问题 top-1 相似度约 0.68–0.80，无关问题约 0.38–0.51。
    # 取 0.55 让明显无关的问题直接走兜底，避免多一次生成调用。
    relevance_threshold: float = 0.55

    def with_overrides(self, **kwargs) -> "Settings":
        """返回覆盖若干字段后的新配置，便于测试与实验。"""
        return replace(self, **kwargs)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """从映射（默认 os.environ）构造配置。

    传入 `env={...}` 时完全不读真实环境，便于离线测试。
    """
    source: Mapping[str, str] = os.environ if env is None else env

    def get(name: str, default: str = "") -> str:
        value = source.get(name, default)
        return "" if value is None else str(value).strip()

    base_url = get("OPENAI_BASE_URL") or None
    data_dir = get("RAG_DATA_DIR")
    chroma_dir = get("RAG_CHROMA_DIR")

    return Settings(
        api_key=get("OPENAI_API_KEY"),
        base_url=base_url,
        model=get("OPENAI_MODEL", "deepseek-chat") or "deepseek-chat",
        temperature=_as_float(get("OPENAI_TEMPERATURE"), 0.2),
        request_timeout=_as_int(get("RAG_REQUEST_TIMEOUT"), 120),
        embedding_model=get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3",
        rerank_model=get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
        or "BAAI/bge-reranker-base",
        use_reranker=_as_bool(get("RAG_USE_RERANKER"), False),
        data_dir=Path(data_dir) if data_dir else PROJECT_ROOT / "data",
        chroma_dir=Path(chroma_dir) if chroma_dir else PROJECT_ROOT / "chroma_db",
        collection_name=get("RAG_COLLECTION", "rag_docs") or "rag_docs",
        chunk_size=_as_int(get("RAG_CHUNK_SIZE"), 500),
        chunk_overlap=_as_int(get("RAG_CHUNK_OVERLAP"), 80),
        top_k=_as_int(get("RAG_TOP_K"), 5),
        candidate_k=_as_int(get("RAG_CANDIDATE_K"), 20),
        rrf_k=_as_int(get("RAG_RRF_K"), 60),
        relevance_threshold=_as_float(get("RAG_RELEVANCE_THRESHOLD"), 0.55),
    )


# 模块级默认配置：应用启动时直接 import 使用。
settings = load_settings()
