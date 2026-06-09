"""Endpoint style auto-detection.

Given a base URL and API key, probe it with lightweight requests and infer
whether the endpoint speaks the OpenAI Chat Completions dialect, the
Anthropic Messages dialect, or a custom compatible flavor.

Detection strategy (in order of cost / specificity):

1. Cheap URL heuristics — many providers ship obvious path hints
   (e.g. ``api.openai.com``, ``api.anthropic.com``, ``/v1``, ``/anthropic``).
2. Hit the OpenAI ``GET /v1/models`` endpoint and check for the
   ``{"object": "list", "data": [...]}`` shape.
3. Hit the Anthropic ``GET /v1/models`` endpoint and check for the
   ``{"data": [{"id": ...}]}`` shape (no ``object: "list"`` wrapper).
4. As a last resort, inspect path responses / status codes to make a
   best-effort guess.

The detector is intentionally tolerant: it never raises on a 401 (the key
might be wrong) or 404 (the path might be disabled). It just returns the
strongest signal it can find.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import httpx


class EndpointStyle(str, Enum):
    """Recognized API dialects."""

    OPENAI = "openai"          # OpenAI Chat Completions compatible
    ANTHROPIC = "anthropic"    # Anthropic Messages API compatible
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            EndpointStyle.OPENAI: "OpenAI-compatible",
            EndpointStyle.ANTHROPIC: "Anthropic-compatible",
            EndpointStyle.UNKNOWN: "Unknown",
        }[self]


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

_TRAILING_SLASH = re.compile(r"/+$")


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes and whitespace, add scheme if missing."""
    url = (url or "").strip()
    if not url:
        raise ValueError("base URL is empty")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _TRAILING_SLASH.sub("", url)


def candidate_model_paths(base_url: str) -> list[tuple[EndpointStyle, str]]:
    """Yield (style, full_url) pairs to probe, in priority order."""
    base = normalize_base_url(base_url)
    return [
        (EndpointStyle.OPENAI, f"{base}/v1/models"),
        (EndpointStyle.ANTHROPIC, f"{base}/v1/models"),
        (EndpointStyle.OPENAI, f"{base}/models"),
    ]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_HOST_HINTS: list[tuple[re.Pattern, EndpointStyle]] = [
    (re.compile(r"api\.openai\.com", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.anthropic\.com", re.I), EndpointStyle.ANTHROPIC),
    (re.compile(r"\.anthropic\.com", re.I), EndpointStyle.ANTHROPIC),
    (re.compile(r"api\.deepseek\.com", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.moonshot\.cn", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.siliconflow\.cn", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.groq\.com", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.together\.", re.I), EndpointStyle.OPENAI),
    (re.compile(r"api\.fireworks\.ai", re.I), EndpointStyle.OPENAI),
    (re.compile(r"openrouter\.ai", re.I), EndpointStyle.OPENAI),
    (re.compile(r"generativelanguage\.googleapis\.com", re.I), EndpointStyle.OPENAI),
    (re.compile(r"ollama", re.I), EndpointStyle.OPENAI),
    (re.compile(r":11434", re.I), EndpointStyle.OPENAI),       # Ollama default port
    (re.compile(r":8000", re.I), EndpointStyle.OPENAI),       # vLLM / TGI default port
    (re.compile(r":1234", re.I), EndpointStyle.OPENAI),       # LM Studio default port
]


def _guess_from_host(url: str) -> Optional[EndpointStyle]:
    host = urlparse(normalize_base_url(url)).netloc
    for pattern, style in _HOST_HINTS:
        if pattern.search(host):
            return style
    return None


# ---------------------------------------------------------------------------
# Active probing
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """What the detector learned about the endpoint."""

    style: EndpointStyle
    base_url: str
    model_url: str                         # the URL that succeeded, if any
    status_code: Optional[int] = None
    raw_excerpt: str = ""                  # first 200 chars of the body
    elapsed_ms: float = 0.0
    note: str = ""                         # human-readable diagnostic


def _looks_like_openai_models(body: str) -> bool:
    """OpenAI's /v1/models returns {"object": "list", "data": [{"id": ...}]}."""
    if not body:
        return False
    head = body.lstrip()[:512].lower()
    if '"object": "list"' in head or '"object":"list"' in head:
        return '"data"' in head
    # Anthropic's body also contains data+id (no "object" wrapper) — it
    # disambiguates by including a "type" field. If we see "type": "model"
    # we know it's Anthropic, so don't claim it for OpenAI.
    if '"type"' in head and '"model"' in head:
        return False
    # Some compatible providers omit "object" but still return {"data": [...]}
    if '"data"' in head and '"id"' in head:
        return True
    return False


def _looks_like_anthropic_models(body: str) -> bool:
    """Anthropic's /v1/models returns {"data": [{"id": ..., "type": "model"}]}.

    No ``object: "list"`` wrapper, and the type field is the tell.
    """
    if not body:
        return False
    head = body.lstrip()[:512].lower()
    if '"object": "list"' in head or '"object":"list"' in head:
        return False
    if '"data"' in head and '"id"' in head and '"type"' in head and '"model"' in head:
        return True
    return False


def _probe_url(url: str, headers: dict, timeout: float = 8.0):
    """GET a URL, return (response, elapsed_ms) or (None, elapsed_ms)."""
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
        return resp, (time.perf_counter() - start) * 1000
    except httpx.HTTPError:
        return None, (time.perf_counter() - start) * 1000


def detect_endpoint_style(
    base_url: str,
    api_key: str = "",
    *,
    timeout: float = 8.0,
    hint: Optional[EndpointStyle] = None,
) -> ProbeResult:
    """Return the strongest guess at the endpoint's dialect.

    The function never raises; on total failure it returns
    :pyattr:`EndpointStyle.UNKNOWN`.
    """
    base = normalize_base_url(base_url)
    host_guess = _guess_from_host(base)
    preferred = hint or host_guess

    headers = {
        "User-Agent": "token-benchmark/2.0",
        "Accept": "application/json",
    }
    if api_key:
        # Some providers reject empty bearer; some are fine. Send both flavors.
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"

    # Try each candidate path; pick the strongest signal.
    candidates = candidate_model_paths(base)
    if preferred == EndpointStyle.OPENAI:
        order = [EndpointStyle.OPENAI, EndpointStyle.ANTHROPIC]
    elif preferred == EndpointStyle.ANTHROPIC:
        order = [EndpointStyle.ANTHROPIC, EndpointStyle.OPENAI]
    else:
        order = [EndpointStyle.OPENAI, EndpointStyle.ANTHROPIC]
    candidates.sort(key=lambda c: order.index(c[0]) if c[0] in order else 99)

    best = ProbeResult(
        style=EndpointStyle.UNKNOWN,
        base_url=base,
        model_url="",
        note="no successful probe",
    )

    for style, url in candidates:
        resp, elapsed = _probe_url(url, headers, timeout=timeout)
        if resp is None:
            continue
        body = (resp.text or "")[:1024]
        excerpt = body[:200].replace("\n", " ")
        status = resp.status_code

        if status == 200:
            if style == EndpointStyle.OPENAI and _looks_like_openai_models(body):
                if not _looks_like_anthropic_models(body):
                    return ProbeResult(
                        style=EndpointStyle.OPENAI, base_url=base, model_url=url,
                        status_code=status, raw_excerpt=excerpt, elapsed_ms=elapsed,
                        note="GET /v1/models returned OpenAI list shape",
                    )
            if style == EndpointStyle.ANTHROPIC and _looks_like_anthropic_models(body):
                return ProbeResult(
                    style=EndpointStyle.ANTHROPIC, base_url=base, model_url=url,
                    status_code=status, raw_excerpt=excerpt, elapsed_ms=elapsed,
                    note="GET /v1/models returned Anthropic list shape",
                )
            # Open list shape but ambiguous; remember as a candidate
            if _looks_like_openai_models(body) and best.style == EndpointStyle.UNKNOWN:
                best = ProbeResult(
                    style=EndpointStyle.OPENAI, base_url=base, model_url=url,
                    status_code=status, raw_excerpt=excerpt, elapsed_ms=elapsed,
                    note="ambiguous list shape — assuming OpenAI",
                )
        elif status in (401, 403):
            # Auth-protected endpoint that exists; remember as a candidate
            if best.style == EndpointStyle.UNKNOWN and style == preferred:
                best = ProbeResult(
                    style=style, base_url=base, model_url=url,
                    status_code=status, raw_excerpt=excerpt, elapsed_ms=elapsed,
                    note=f"endpoint requires auth ({status}); inferring from host hint",
                )

    if best.style == EndpointStyle.UNKNOWN and host_guess:
        return ProbeResult(
            style=host_guess, base_url=base, model_url="",
            note="fell back to host-name heuristic",
        )

    if best.style == EndpointStyle.UNKNOWN:
        best.note = (
            f"could not determine style — checked {len(candidates)} paths "
            "without a successful response"
        )
    return best
