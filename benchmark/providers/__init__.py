from benchmark.providers.base import BaseProvider
from benchmark.providers.openai import OpenAIProvider
from benchmark.providers.anthropic import AnthropicProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

__all__ = ["PROVIDERS"]
