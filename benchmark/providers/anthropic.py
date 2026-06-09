"""Anthropic provider — thin shim over :class:`UniversalProvider`."""
from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider


class AnthropicProvider(UniversalProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, verbose: bool = False,
                 base_url: str | None = None, **kwargs):
        if not base_url:
            base_url = "https://api.anthropic.com"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            style=EndpointStyle.ANTHROPIC,
            owner="anthropic",
            verbose=verbose,
            **kwargs,
        )
