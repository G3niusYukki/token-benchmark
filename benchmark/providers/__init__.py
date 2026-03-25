from benchmark.providers.base import BaseProvider
from benchmark.providers.openai import OpenAIProvider
from benchmark.providers.anthropic import AnthropicProvider
from benchmark.providers.deepseek import DeepSeekProvider
from benchmark.providers.kimi import KimiProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "kimi": KimiProvider,
}

__all__ = ["PROVIDERS"]
