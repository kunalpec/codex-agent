import pytest

from ollama_config import ollama_base_url, ollama_chat_model


def test_full_tags_url_resolves_api_root_and_chat_model(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_MODEL", "https://tunnel.example/api/tags"
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://stale.example")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

    assert ollama_base_url() == "https://tunnel.example"
    assert ollama_chat_model() == "qwen3:8b"


def test_legacy_model_name_uses_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://tunnel.example/")

    assert ollama_base_url() == "https://tunnel.example"
    assert ollama_chat_model() == "qwen3:8b"


def test_invalid_model_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "https://tunnel.example/models")

    with pytest.raises(ValueError, match="end in /api/tags"):
        ollama_base_url()
