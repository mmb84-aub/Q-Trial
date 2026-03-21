from __future__ import annotations

import httpx

_client: httpx.Client | None = None


def get_http_client() -> httpx.Client:
    """Shared synchronous HTTP client for literature API calls."""
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=httpx.Timeout(8.0, connect=5.0),
            headers={"User-Agent": "Q-Trial/0.1 (clinical-trial-analyser)"},
        )
    return _client
