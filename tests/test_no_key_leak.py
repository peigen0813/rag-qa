import pytest

from config import Settings
from rag.llm import MissingAPIKeyError, get_llm


def test_placeholder_key_raises_without_echoing_it():
    secret = "sk-your-key-here"
    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_llm(Settings(api_key=secret))
    assert secret not in str(exc_info.value)


def test_empty_key_raises_helpful_message():
    with pytest.raises(MissingAPIKeyError) as exc_info:
        get_llm(Settings(api_key=""))
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_repr_of_client_masks_real_key():
    secret = "sk-real-secret-value-should-not-appear"
    llm = get_llm(
        Settings(
            api_key=secret,
            base_url="https://example.com/v1",
            model="deepseek-chat",
        )
    )
    assert secret not in repr(llm)
