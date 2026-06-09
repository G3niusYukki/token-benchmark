"""Tests for the UniversalProvider streaming path (fully mocked)."""
import sys
sys.path.insert(0, "/Users/peterzhang/token-benchmark")

from unittest.mock import patch, MagicMock

from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider, _count_tokens


def test_count_tokens_empty():
    assert _count_tokens("") == 0


def test_count_tokens_uses_tiktoken_when_available():
    n = _count_tokens("hello world")
    assert n >= 1


def test_count_tokens_falls_back_on_import_error():
    with patch("tiktoken.get_encoding", side_effect=Exception("boom")):
        # 20 chars / 4 = 5 tokens
        assert _count_tokens("x" * 20) == 5


def test_universal_provider_openai_construction():
    p = UniversalProvider(
        api_key="sk-test", model="gpt-4o",
        base_url="https://api.openai.com/v1",
        style=EndpointStyle.OPENAI, owner="openai", verbose=False,
    )
    assert p.style == EndpointStyle.OPENAI
    assert p.model == "gpt-4o"
    assert p.name == "openai"


def test_universal_provider_anthropic_construction():
    p = UniversalProvider(
        api_key="sk-test", model="claude-sonnet-4-5-latest",
        base_url="https://api.anthropic.com",
        style=EndpointStyle.ANTHROPIC, owner="anthropic", verbose=False,
    )
    assert p.style == EndpointStyle.ANTHROPIC
    assert p.name == "anthropic"


# ----- streaming run() with a fake SDK ------------------------------------

class _FakeChunk:
    def __init__(self, text):
        self.choices = [MagicMock()]
        self.choices[0].delta.content = text


def test_run_openai_streams_chunks():
    p = UniversalProvider(
        api_key="sk-test", model="gpt-4o",
        base_url="https://api.openai.com/v1",
        style=EndpointStyle.OPENAI, owner="openai", verbose=False,
    )
    chunks = [_FakeChunk("Hello"), _FakeChunk(", "), _FakeChunk("world!")]

    class _FakeStream:
        def __iter__(self):
            return iter(chunks)

    def _fake_create(**kwargs):
        return _FakeStream()

    p._openai_client = MagicMock()
    p._openai_client.chat.completions.create = _fake_create

    result = p.run("hi", timeout=10)
    assert result.success
    assert result.full_text == "Hello, world!"
    assert result.total_tokens > 0
    assert result.ttft_ms > 0
    assert result.tokens_per_second > 0
    assert result.provider == "openai"


def test_run_handles_exception_as_failure():
    p = UniversalProvider(
        api_key="sk-test", model="gpt-4o",
        base_url="https://api.openai.com/v1",
        style=EndpointStyle.OPENAI, owner="openai", verbose=False,
    )
    p._openai_client = MagicMock()
    p._openai_client.chat.completions.create.side_effect = RuntimeError("network down")

    result = p.run("hi", timeout=10)
    assert not result.success
    assert "network down" in result.error
    assert result.total_tokens == 0
    assert result.tokens_per_second == 0
