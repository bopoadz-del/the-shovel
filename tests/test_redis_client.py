import os
import pytest

from app.core import redis_client


@pytest.fixture(autouse=True)
def _reset():
    redis_client.reset_for_tests()
    old = os.environ.pop("REDIS_URL", None)
    yield
    redis_client.reset_for_tests()
    if old is not None:
        os.environ["REDIS_URL"] = old
    else:
        os.environ.pop("REDIS_URL", None)


@pytest.mark.asyncio
async def test_get_redis_client_returns_none_when_url_unset():
    assert await redis_client.get_redis_client() is None


@pytest.mark.asyncio
async def test_get_sync_redis_client_returns_none_when_url_unset():
    assert redis_client.get_sync_redis_client() is None


@pytest.mark.asyncio
async def test_redis_health_false_when_url_unset():
    health = await redis_client.redis_health()
    assert health == {"connected": False, "latency_ms": None}
