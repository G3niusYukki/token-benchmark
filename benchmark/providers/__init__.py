"""Built-in providers.

The hard-coded OpenAI/Anthropic/DeepSeek/Kimi classes are thin shims that
delegate to :class:`UniversalProvider` so the streaming and token-counting
logic lives in one place. New endpoints should use ``UniversalProvider``
directly.
"""
from __future__ import annotations

from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.base import BaseProvider
from benchmark.providers.openai import OpenAIProvider
from benchmark.providers.anthropic import AnthropicProvider
from benchmark.providers.deepseek import DeepSeekProvider
from benchmark.providers.kimi import KimiProvider
from benchmark.providers.universal import UniversalProvider, _count_tokens


# Backwards-compat mapping used by ``runner.py`` for legacy config entries.
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "kimi": KimiProvider,
}


# Default base URLs for the well-known providers.
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
}


def style_for_owner(owner: str) -> EndpointStyle:
    """Map a provider owner name to its expected endpoint style."""
    o = (owner or "").lower()
    if "anthropic" in o:
        return EndpointStyle.ANTHROPIC
    return EndpointStyle.OPENAI


__all__ = [
    "PROVIDERS",
    "UniversalProvider",
    "BaseProvider",
    "DEFAULT_BASE_URLS",
    "style_for_owner",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "KimiProvider",
]
