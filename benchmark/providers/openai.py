"""OpenAI provider — thin shim over :class:`UniversalProvider`."""
from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider


class OpenAIProvider(UniversalProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, verbose: bool = False,
                 base_url: str | None = None, **kwargs):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            style=EndpointStyle.OPENAI,
            owner="openai",
            verbose=verbose,
            **kwargs,
        )
