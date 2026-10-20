"""LLM 工厂：兼容 DeepSeek / OpenAI 等 OpenAI 协议服务。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import Settings, settings

_PLACEHOLDER_PREFIXES = ("sk-your", "your-", "changeme", "replace-me")


class MissingAPIKeyError(RuntimeError):
    """未配置有效 API Key。错误信息中绝不包含 key 本身。"""


def _is_placeholder(key: str) -> bool:
    lowered = key.lower()
    return not key or lowered.startswith(_PLACEHOLDER_PREFIXES)


def get_llm(
    settings_obj: Settings | None = None,
    *,
    streaming: bool = True,
    temperature: float | None = None,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。

    - `streaming=True` 是 LangGraph `stream_mode="messages"` 输出 token 的前提；
    - key 只从配置读取，不落日志、不进异常信息。
    """
    cfg = settings_obj or settings
    key = (cfg.api_key or "").strip()
    if _is_placeholder(key):
        raise MissingAPIKeyError(
            "未检测到有效的 OPENAI_API_KEY。请复制 .env.example 为 .env，"
            "填入你自己的 API Key（DeepSeek 或其他 OpenAI 兼容服务），"
            "然后重新运行。"
        )

    kwargs: dict = {
        "model": cfg.model,
        "api_key": key,
        "temperature": cfg.temperature if temperature is None else temperature,
        "streaming": streaming,
        "timeout": cfg.request_timeout,
        "max_retries": 2,
    }
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return ChatOpenAI(**kwargs)
