"""Shared HTTP client pool for external APIs."""
from __future__ import annotations

from typing import Any

import httpx

_http_clients: dict[str, httpx.AsyncClient] = {}


def get_http_client(name: str, timeout: float = 10.0, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    """Return a shared AsyncClient keyed by name."""
    if name not in _http_clients:
        _http_clients[name] = httpx.AsyncClient(
            timeout=timeout,
            headers=headers or {},
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _http_clients[name]


async def shutdown_http_clients() -> None:
    """Close all shared HTTP clients on shutdown."""
    for client in _http_clients.values():
        await client.aclose()
    _http_clients.clear()
