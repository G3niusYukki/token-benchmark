"""Kimi/Moonshot provider — thin shim over :class:`UniversalProvider`."""
from benchmark.endpoint_detector import EndpointStyle
from benchmark.providers.universal import UniversalProvider


class KimiProvider(UniversalProvider):
    name = "kimi"

    def __init__(self, api_key: str, model: str, verbose: bool = False,
                 base_url: str | None = None, **kwargs):
        if not base_url:
            base_url = "https://api.moonshot.cn/v1"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            style=EndpointStyle.OPENAI,
            owner="kimi",
            verbose=verbose,
            **kwargs,
        )
