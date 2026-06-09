"""Tests for endpoint style auto-detection.

These exercise the URL normalizer, the host-name heuristic, and the body
shape matchers. We don't make real network calls; instead we patch
``httpx.Client`` so we can simulate a variety of endpoint dialects.
"""
import sys
sys.path.insert(0, "/Users/peterzhang/token-benchmark")

from unittest.mock import patch, MagicMock

import httpx

from benchmark.endpoint_detector import (
    EndpointStyle,
    _guess_from_host,
    _looks_like_anthropic_models,
    _looks_like_openai_models,
    detect_endpoint_style,
    normalize_base_url,
)


# ----- pure helpers ---------------------------------------------------------

def test_normalize_strips_trailing_slash():
    assert normalize_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert normalize_base_url("https://api.openai.com/v1///") == "https://api.openai.com/v1"


def test_normalize_adds_scheme():
    assert normalize_base_url("api.openai.com/v1").startswith("https://")


def test_normalize_empty_raises():
    try:
        normalize_base_url("   ")
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty input")


def test_host_guess_openai():
    assert _guess_from_host("https://api.openai.com/v1") == EndpointStyle.OPENAI
    assert _guess_from_host("https://api.deepseek.com/v1") == EndpointStyle.OPENAI
    assert _guess_from_host("http://localhost:11434/v1") == EndpointStyle.OPENAI
    assert _guess_from_host("http://127.0.0.1:1234/v1") == EndpointStyle.OPENAI


def test_host_guess_anthropic():
    assert _guess_from_host("https://api.anthropic.com") == EndpointStyle.ANTHROPIC
    assert _guess_from_host("https://api.anthropic.com/v1") == EndpointStyle.ANTHROPIC


def test_host_guess_unknown():
    assert _guess_from_host("https://api.example.com/v1") is None


# ----- body shape matchers --------------------------------------------------

def test_openai_body_shape():
    body = '{"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}'
    assert _looks_like_openai_models(body)
    assert not _looks_like_anthropic_models(body)


def test_anthropic_body_shape():
    body = '{"data": [{"id": "claude-sonnet-4-5-latest", "type": "model"}]}'
    assert _looks_like_anthropic_models(body)
    assert not _looks_like_openai_models(body)


def test_empty_body_safe():
    assert not _looks_like_openai_models("")
    assert not _looks_like_anthropic_models("")


def test_garbage_body_safe():
    assert not _looks_like_openai_models("not json at all")
    assert not _looks_like_anthropic_models("not json at all")


# ----- active probing (mocked) ---------------------------------------------

def _mock_get(url, headers=None, **kw):
    if "anthropic" in url:
        body = '{"data": [{"id": "claude-test", "type": "model"}]}'
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.text = body
        return r
    if "openai" in url or "/v1/models" in url:
        body = '{"object": "list", "data": [{"id": "gpt-test", "object": "model"}]}'
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.text = body
        return r
    r = MagicMock(spec=httpx.Response)
    r.status_code = 404
    r.text = "not found"
    return r


def test_detect_openai_when_body_matches():
    with patch("httpx.Client.get", side_effect=_mock_get):
        result = detect_endpoint_style("https://api.openai.com/v1", "sk-test")
    assert result.style == EndpointStyle.OPENAI
    assert result.status_code == 200


def test_detect_anthropic_when_body_matches():
    with patch("httpx.Client.get", side_effect=_mock_get):
        result = detect_endpoint_style("https://api.anthropic.com", "sk-test")
    assert result.style == EndpointStyle.ANTHROPIC


def test_detect_uses_host_heuristic_on_404():
    with patch("httpx.Client.get", return_value=MagicMock(status_code=404, text="")):
        result = detect_endpoint_style("https://api.openai.com/v1", "sk-test")
    assert result.style == EndpointStyle.OPENAI
    assert result.note  # some diagnostic


def test_detect_returns_unknown_on_total_failure():
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("nope")):
        result = detect_endpoint_style("https://unknown.example.com/v1", "sk-test")
    assert result.style == EndpointStyle.UNKNOWN
