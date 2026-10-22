"""Resolve Ollama endpoint and model settings from the environment."""

import os


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_base_url() -> str:
    """Return the Ollama API root, accepting a full `/api/tags` URL."""
    configured = os.environ.get("OLLAMA_MODEL", "").strip()
    if configured.startswith(("http://", "https://")):
        suffix = "/api/tags"
        if not configured.rstrip("/").endswith(suffix):
            raise ValueError("OLLAMA_MODEL URL must end in /api/tags")
        return configured.rstrip("/")[:-len(suffix)].rstrip("/")
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")


def ollama_chat_model() -> str:
    """Return the chat model name, separate from the model-list URL."""
    configured = os.environ.get("OLLAMA_MODEL", "").strip()
    if configured.startswith(("http://", "https://")):
        return os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:8b").strip()
    return configured or os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:8b").strip()
