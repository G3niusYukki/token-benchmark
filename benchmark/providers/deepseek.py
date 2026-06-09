"""DeepSeek provider — thin shim over :class:`UniversalProvider`."""
from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider


class DeepSeekProvider(UniversalProvider):
    name = "deepseek"

    def __init__(self, api_key: str, model: str, verbose: bool = False,
                 base_url: str | None = None, **kwargs):
        if not base_url:
            base_url = "https://api.deepseek.com/v1"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            style=EndpointStyle.OPENAI,
            owner="deepseek",
            verbose=verbose,
            **kwargs,
        )
