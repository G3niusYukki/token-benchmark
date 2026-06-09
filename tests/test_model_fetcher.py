"""Tests for the model catalog and fetcher (no real network)."""
import sys
sys.path.insert(0, "/Users/peterzhang/token-benchmark")

from unittest.mock import patch, MagicMock

from benchmark.endpoint_detector import EndpointStyle
from benchmark.model_fetcher import (
    ModelInfo,
    fallback_models,
    fetch_anthropic_models,
    fetch_openai_models,
    _is_likely_chat_model,
)


# ----- chat-model filter ---------------------------------------------------

def test_chat_filter_strips_embeddings():
    assert _is_likely_chat_model("gpt-4o")
    assert _is_likely_chat_model("claude-sonnet-4-5-latest")
    assert not _is_likely_chat_model("text-embedding-3-small")
    assert not _is_likely_chat_model("text-embedding-ada-002")
    assert not _is_likely_chat_model("bge-large-en")
    assert not _is_likely_chat_model("whisper-1")
    assert not _is_likely_chat_model("dall-e-3")
    assert not _is_likely_chat_model("text-moderation-stable")


def test_chat_filter_empty_safe():
    assert not _is_likely_chat_model("")
    assert not _is_likely_chat_model(None)


# ----- fallback catalogs --------------------------------------------------

def test_fallback_openai_includes_main_models():
    models = fallback_models(EndpointStyle.OPENAI, "openai")
    ids = [m.id for m in models]
    assert "gpt-4o" in ids
    assert "gpt-4o-mini" in ids


def test_fallback_anthropic_includes_main_models():
    models = fallback_models(EndpointStyle.ANTHROPIC, "anthropic")
    ids = [m.id for m in models]
    assert "claude-sonnet-4-5-latest" in ids or "claude-3-5-sonnet-latest" in ids


def test_fallback_unknown_style_returns_empty():
    assert fallback_models(EndpointStyle.UNKNOWN) == []


def test_model_info_short_label():
    m = ModelInfo(id="gpt-4o", display_name="GPT-4o", owned_by="openai",
                  style=EndpointStyle.OPENAI)
    assert "gpt-4o" in m.short_label or "GPT-4o" in m.short_label


# ----- OpenAI fetcher (mocked SDK) ----------------------------------------

def test_fetch_openai_models_via_sdk():
    fake_model = MagicMock()
    fake_model.id = "gpt-4o"
    fake_model.owned_by = "openai"
    fake_page = MagicMock()
    fake_page.data = [fake_model]
    with patch("openai.OpenAI") as FakeClient:
        instance = FakeClient.return_value
        instance.models.list.return_value = fake_page
        models = fetch_openai_models("https://api.openai.com/v1", "sk-test")
    assert len(models) == 1
    assert models[0].id == "gpt-4o"


# ----- Anthropic fetcher (mocked httpx) -----------------------------------

def test_fetch_anthropic_models_via_httpx():
    with patch("httpx.Client.get") as fake_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {"id": "claude-sonnet-4-5-latest", "type": "model",
                 "display_name": "Claude Sonnet 4.5"},
            ]
        }
        resp.text = '{"data": [{"id": "claude-sonnet-4-5-latest", "type": "model"}]}'
        fake_get.return_value = resp
        models = fetch_anthropic_models("https://api.anthropic.com", "sk-test")
    assert len(models) == 1
    assert models[0].id == "claude-sonnet-4-5-latest"
    assert models[0].display_name == "Claude Sonnet 4.5"
    assert models[0].style == EndpointStyle.ANTHROPIC


def test_fetch_anthropic_models_handles_401():
    with patch("httpx.Client.get") as fake_get:
        resp = MagicMock()
        resp.status_code = 401
        fake_get.return_value = resp
        models = fetch_anthropic_models("https://api.anthropic.com", "sk-bad")
    assert models == []
