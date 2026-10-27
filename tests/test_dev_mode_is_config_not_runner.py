"""Dev-mode is decided by ENV config, not by detecting the pytest runner.

Pins A1–A3 (HARDWIRING_the-shovel_FINAL): the production paths that grant dev
privileges — open registration and the dev API key — must key on ENV/_DEV_ENVS
only, never on ``"pytest" in sys.modules``. pytest is ALWAYS imported while these
tests run, so if any runner-detection branch came back, the non-dev cases below
would flip to True and this module would go red. Red-first: on the pre-fix code
(the removed ``"pytest" in sys.modules`` branches) every ``test_non_dev_env_*``
assertion fails; on the fixed code they pass.
"""
import pytest

from app.core import auth as core_auth
from app.routers import users as users_router


@pytest.fixture(autouse=True)
def _no_open_registration_flag(monkeypatch):
    # The ALLOW_OPEN_REGISTRATION escape hatch must not mask the ENV decision.
    monkeypatch.delenv("ALLOW_OPEN_REGISTRATION", raising=False)
    yield


@pytest.mark.parametrize("env", ["production", "staging", None])
def test_non_dev_env_closes_every_dev_path(monkeypatch, env):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    if env is None:
        # Nothing set -> the code's own default ("production").
        monkeypatch.delenv("ENV", raising=False)
    else:
        monkeypatch.setenv("ENV", env)
    # pytest is in sys.modules right now; the old code returned True here.
    assert users_router._registration_allowed() is False
    assert users_router._is_dev_env() is False
    assert core_auth.is_dev_environment() is False
    # The dev API key must be refused in a non-dev env even when it is set.
    monkeypatch.setenv("CEREBRUM_DEV_KEY", "would-be-dev-key")
    assert core_auth.dev_key() is None


@pytest.mark.parametrize("env", ["dev", "test"])
def test_dev_env_opens_dev_paths(monkeypatch, env):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("ENV", env)
    assert users_router._registration_allowed() is True
    assert users_router._is_dev_env() is True
    assert core_auth.is_dev_environment() is True
