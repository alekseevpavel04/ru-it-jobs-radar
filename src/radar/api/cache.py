"""JSON cache in Redis. A lost or unreachable cache costs recomputation only, so its errors are logged
and the request goes on without it; /health reports the outage."""

import hashlib
import json
import logging
from typing import Any, Protocol

import redis

log = logging.getLogger("radar.api.cache")

PREFIX = "radar:api:v1:"


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl_s: int) -> None: ...

    def ping(self) -> bool: ...


def key(kind: str, *parts: str) -> str:
    """A fixed-length key: the parts may hold a vacancy text, which is never stored as is."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{PREFIX}{kind}:{digest}"


class RedisCache:
    def __init__(self, url: str, timeout_s: float = 0.5) -> None:
        self.client = redis.Redis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)

    def get(self, key: str) -> Any | None:
        try:
            raw = self.client.get(key)
        except redis.RedisError as e:
            log.warning("cache get failed: %s", e)
            return None
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_s)
        except redis.RedisError as e:
            log.warning("cache set failed: %s", e)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    def close(self) -> None:
        self.client.close()
