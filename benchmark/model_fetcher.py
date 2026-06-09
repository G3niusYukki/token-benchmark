"""Fetch and normalize the list of chat-capable models from an endpoint.

Wraps the OpenAI and Anthropic SDKs so the rest of the codebase only ever
deals with a uniform list of :class:`ModelInfo` objects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from benchmark.endpoint_detector import EndpointStyle, normalize_base_url


@dataclass
class ModelInfo:
    """A normalized model description ready for selection UI."""

    id: str                              # canonical id used in API calls
    display_name: str = ""               # human-friendly label
    owned_by: str = ""                   # provider/owner (e.g. "openai")
    style: EndpointStyle = EndpointStyle.OPENAI
    raw: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.id

    @property
    def short_label(self) -> str:
        owner = f" · {self.owned_by}" if self.owned_by else ""
        return f"{self.display_name}{owner}"

    def __str__(self) -> str:            # pragma: no cover — debug helper
        return f"ModelInfo(id={self.id!r}, style={self.style.value}, owner={self.owned_by!r})"


# ---------------------------------------------------------------------------
# OpenAI-style fetch
# ---------------------------------------------------------------------------

_EMBEDDING_HINTS = re.compile(r"(embedding|embed-|text-embed|bge-|e5-|gte-)", re.I)
_AUDIO_HINTS = re.compile(r"(audio|whisper|tts|realtime|voice)", re.I)
_IMAGE_HINTS = re.compile(r"(dall-e|dalle|stable-diffusion|sd-|imagen|image-)", re.I)


def _is_likely_chat_model(model_id: str, owner: str = "") -> bool:
    """Heuristic — strip out embeddings, image, audio models, keep chat ones."""
    if not model_id:
        return False
    mid = model_id.lower()
    if _EMBEDDING_HINTS.search(mid):
        return False
    if _AUDIO_HINTS.search(mid):
        return False
    if _IMAGE_HINTS.search(mid):
        return False
    # Older / non-chat completion endpoints
    if any(k in mid for k in ("moderation", "davinci", "babbage", "curie")):
        return False
    return True


def fetch_openai_models(base_url: str, api_key: str, timeout: float = 10.0) -> list[ModelInfo]:
    """Hit GET {base}/v1/models via the OpenAI SDK and normalize the response."""
    from openai import OpenAI
    base = normalize_base_url(base_url)
    client = OpenAI(api_key=api_key or "sk-no-key", base_url=base, timeout=timeout)
    try:
        page = client.models.list()
    except Exception:
        return _fetch_openai_models_httpx(base, api_key, timeout)

    models: list[ModelInfo] = []
    for m in page.data:
        mid = getattr(m, "id", None)
        if not mid or not _is_likely_chat_model(mid, getattr(m, "owned_by", "")):
            continue
        models.append(
            ModelInfo(
                id=mid,
                display_name=mid,
                owned_by=getattr(m, "owned_by", "") or "",
                style=EndpointStyle.OPENAI,
                raw=m.model_dump() if hasattr(m, "model_dump") else {"id": mid},
            )
        )
    return models


def _fetch_openai_models_httpx(base: str, api_key: str, timeout: float) -> list[ModelInfo]:
    """Direct REST fallback when the SDK raises (older Ollama, some proxies)."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base}/v1/models", headers=headers)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    out: list[ModelInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid or not _is_likely_chat_model(mid, item.get("owned_by", "")):
            continue
        out.append(
            ModelInfo(
                id=mid,
                display_name=mid,
                owned_by=item.get("owned_by", "") or "",
                style=EndpointStyle.OPENAI,
                raw=item,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Anthropic-style fetch
# ---------------------------------------------------------------------------

def fetch_anthropic_models(
    base_url: str,
    api_key: str,
    timeout: float = 10.0,
) -> list[ModelInfo]:
    """Hit GET {base}/v1/models via httpx and normalize the response.

    We use httpx directly because the Anthropic SDK doesn't ship a model
    listing helper as of 0.18+.
    """
    base = normalize_base_url(base_url)
    headers = {
        "x-api-key": api_key or "",
        "anthropic-version": "2023-06-01",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base}/v1/models", headers=headers)
    except httpx.HTTPError:
        return []

    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    models: list[ModelInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid:
            continue
        models.append(
            ModelInfo(
                id=mid,
                display_name=item.get("display_name", mid),
                owned_by="anthropic",
                style=EndpointStyle.ANTHROPIC,
                raw=item,
            )
        )
    return models


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def fetch_models(
    style: EndpointStyle,
    base_url: str,
    api_key: str,
    timeout: float = 10.0,
) -> list[ModelInfo]:
    """Fetch the model list using the right transport for the given style."""
    if style == EndpointStyle.ANTHROPIC:
        models = fetch_anthropic_models(base_url, api_key, timeout=timeout)
    else:
        models = fetch_openai_models(base_url, api_key, timeout=timeout)
    models.sort(key=lambda m: m.id.lower())
    return models


# Hard-coded fallback model catalogs for well-known providers — used when
# the endpoint doesn't expose /v1/models (older Ollama, some corporate proxies)
_FALLBACK_CATALOG: dict[str, list[str]] = {
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview", "o3-mini",
    ],
    "anthropic": [
        "claude-opus-4-1-latest",
        "claude-opus-4-latest",
        "claude-sonnet-4-5-latest",
        "claude-sonnet-4-latest",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ],
    "deepseek": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
    "kimi": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
}


def fallback_models(style: EndpointStyle, owner_hint: str = "") -> list[ModelInfo]:
    """Return a hard-coded list of well-known models for a provider."""
    if style == EndpointStyle.ANTHROPIC:
        return [
            ModelInfo(id=m, display_name=m, owned_by="anthropic",
                      style=EndpointStyle.ANTHROPIC)
            for m in _FALLBACK_CATALOG["anthropic"]
        ]
    if style == EndpointStyle.OPENAI:
        owner = (owner_hint or "openai").lower()
        ids = _FALLBACK_CATALOG.get(owner, _FALLBACK_CATALOG["openai"])
        return [
            ModelInfo(id=m, display_name=m, owned_by=owner or "openai",
                      style=EndpointStyle.OPENAI)
            for m in ids
        ]
    return []
