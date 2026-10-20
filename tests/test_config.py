from config import load_settings


def test_defaults_are_sane():
    settings = load_settings(env={})
    assert settings.model == "deepseek-chat"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 80
    assert settings.top_k == 5
    assert settings.rrf_k == 60
    assert settings.api_key == ""
    assert settings.base_url is None


def test_env_overrides():
    settings = load_settings(
        env={
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-4o-mini",
            "OPENAI_BASE_URL": "https://example.com/v1",
            "RAG_TOP_K": "8",
            "RAG_CANDIDATE_K": "30",
            "RAG_USE_RERANKER": "true",
            "RAG_RELEVANCE_THRESHOLD": "0.5",
            "RAG_CHUNK_SIZE": "300",
        }
    )
    assert settings.api_key == "sk-test"
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "https://example.com/v1"
    assert settings.top_k == 8
    assert settings.candidate_k == 30
    assert settings.use_reranker is True
    assert settings.relevance_threshold == 0.5
    assert settings.chunk_size == 300


def test_invalid_numbers_fall_back_to_defaults():
    settings = load_settings(env={"RAG_TOP_K": "not-a-number", "RAG_RRF_K": ""})
    assert settings.top_k == 5
    assert settings.rrf_k == 60


def test_overrides_are_immutable_copies():
    original = load_settings(env={})
    changed = original.with_overrides(top_k=9)
    assert original.top_k == 5
    assert changed.top_k == 9
