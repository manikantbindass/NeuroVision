"""Factory for a durable API backed by PostgreSQL and Redis."""

import hashlib
import os
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from redis import Redis

from apps.api.contracts import APIError
from apps.api.durable import Database, SQLArtifactStore, SQLJobs
from apps.api.main import create_app as local_app


class RedisRateLimiter:
    def __init__(self, client: Redis, limit: int = 120, seconds: int = 60) -> None:
        if limit < 1 or seconds < 1:
            raise ValueError("Rate-limit settings must be positive")
        self.client, self.limit, self.seconds = client, limit, seconds

    def retry_after(self, owner: str) -> int | None:
        key = "neurovision:rate:" + hashlib.sha256(owner.encode()).hexdigest()
        script = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end
if count > tonumber(ARGV[1]) then return math.max(1, redis.call('TTL', KEYS[1])) end
return 0
"""
        try:
            remaining = int(
                cast(int, self.client.eval(script, 1, key, self.limit, self.seconds))
            )
        except Exception:
            raise APIError(
                503, "rate_service_unavailable", "Rate-limit service is unavailable."
            ) from None
        return remaining or None


def setting(name: str) -> str:
    path = os.environ.get(name + "_FILE")
    result = (
        Path(path).read_text(encoding="utf-8").strip()
        if path
        else os.environ.get(name, "")
    )
    if not result:
        raise ValueError(f"{name} or {name}_FILE is required")
    return result


def connection_settings() -> tuple[str, str]:
    return setting("NEUROVISION_DATABASE_URL"), setting("NEUROVISION_REDIS_URL")


def create_app() -> FastAPI:
    database_url, redis_url = connection_settings()
    database = Database(database_url)
    redis = Redis.from_url(redis_url, socket_timeout=2, socket_connect_timeout=2)
    token_file = os.environ.get("NEUROVISION_API_TOKEN_FILE")
    if token_file:
        from apps.api.adapters import BearerTokens

        auth = BearerTokens({Path(token_file).read_text().strip(): "local-research"})
    else:
        auth = None

    def ready() -> bool:
        try:
            return database.ready() and bool(redis.ping())
        except Exception:
            return False

    app = local_app(
        authenticator=auth,
        store=SQLArtifactStore(database),
        limiter=RedisRateLimiter(redis),
        queue_factory=lambda run, event: SQLJobs(database),
        readiness=ready,
    )
    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize the durable research schema."
    )
    parser.add_argument("operation", choices=["init-db"])
    parser.parse_args()
    database_url = setting("NEUROVISION_DATABASE_URL")
    Database(database_url).initialize()


if __name__ == "__main__":
    main()
