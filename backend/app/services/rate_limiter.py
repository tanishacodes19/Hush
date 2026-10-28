"""In-memory, bounded sliding-window rate limiter for Hush."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional
from starlette.requests import Request


def get_client_ip(
    request: Request,
    trust_proxy_headers: bool = False,
    trusted_proxy_hops: int = 1,
) -> str:
    """Extract client IP address respecting trusted proxy configuration."""
    socket_ip = request.client.host if request.client else "127.0.0.1"
    if not trust_proxy_headers:
        return socket_ip

    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return socket_ip

    entries = [entry.strip() for entry in xff.split(",") if entry.strip()]
    if len(entries) < trusted_proxy_hops:
        return socket_ip

    # Count from the right: hops=1 is entries[-1] (rightmost), hops=2 is entries[-2]
    return entries[-trusted_proxy_hops]


class RateLimiter:
    """In-memory sliding-window rate limiter with bounded storage."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        max_clients: int = 10000,
        cleanup_interval_seconds: float = 30.0,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._clients: OrderedDict[str, list[float]] = OrderedDict()
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_key: str, now: Optional[float] = None) -> bool:
        """Check if request is allowed under the sliding window rate limit.

        Returns True if allowed, False if limit exceeded.
        """
        if now is None:
            now = time.monotonic()

        self._maybe_cleanup(now)

        cutoff = now - self.window_seconds
        timestamps = self._clients.get(client_key)

        if timestamps is None:
            if len(self._clients) >= self.max_clients:
                self._evict_oldest(now)
            self._clients[client_key] = [now]
            self._clients.move_to_end(client_key)
            return True

        # Filter out expired timestamps
        valid_timestamps = [ts for ts in timestamps if ts > cutoff]

        if len(valid_timestamps) >= self.max_requests:
            self._clients[client_key] = valid_timestamps
            self._clients.move_to_end(client_key)
            return False

        valid_timestamps.append(now)
        self._clients[client_key] = valid_timestamps
        self._clients.move_to_end(client_key)
        return True

    def _maybe_cleanup(self, now: float) -> None:
        """Periodically clean up stale client entries."""
        if now - self._last_cleanup >= self.cleanup_interval_seconds:
            self.cleanup_stale_entries(now)
            self._last_cleanup = now

    def cleanup_stale_entries(self, now: Optional[float] = None) -> int:
        """Remove all entries whose timestamps are outside the active window."""
        if now is None:
            now = time.monotonic()
        cutoff = now - self.window_seconds
        stale_keys = []
        for key, timestamps in list(self._clients.items()):
            valid = [ts for ts in timestamps if ts > cutoff]
            if not valid:
                stale_keys.append(key)
            else:
                self._clients[key] = valid
        for key in stale_keys:
            del self._clients[key]
        return len(stale_keys)

    def _evict_oldest(self, now: float) -> None:
        """Evict oldest entries when max_clients is reached."""
        self.cleanup_stale_entries(now)
        while len(self._clients) >= self.max_clients:
            self._clients.popitem(last=False)

    def clear(self) -> None:
        """Clear all rate limit state (useful for tests)."""
        self._clients.clear()
        self._last_cleanup = time.monotonic()


rate_limiter = RateLimiter()
