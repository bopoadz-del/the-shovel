"""Debug-router mount follows a single dest allow-list, including ENV=testing.

CI (and ``.github/workflows/test.yml``) sets ENV=testing. That value is an
explicit dest environment, so ``app.main`` mounts ``/v1/debug/env`` in those
jobs. A test that inspects the already-imported app and assumes ENV is unset
will fail on GitHub Actions even though production still does not mount the
router. Pin the allow-list and the import-time mount in a fresh interpreter.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.routers import debug as debug_mod

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEV = ("dev", "development", "local", "test", "testing")
_NOT_DEV = ("production", "PRODUCTION", "prod", "prod-eu", "staging")


@pytest.mark.parametrize("value", _DEV)
def test_listed_dest_environments_are_dev(monkeypatch, value):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("ENV", value)
    assert debug_mod.is_dev_environment() is True


@pytest.mark.parametrize("value", _NOT_DEV)
def test_unlisted_environments_are_not_dev(monkeypatch, value):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("ENV", value)
    assert debug_mod.is_dev_environment() is False


def test_unset_env_defaults_to_production(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert debug_mod.current_environment() == "production"
    assert debug_mod.is_dev_environment() is False


def test_environment_fallback_when_env_unset(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "testing")
    assert debug_mod.is_dev_environment() is True


def test_main_mounts_debug_via_the_shared_helper():
    """#469's contract: one helper, used by the import-time mount."""
    src = Path("app/main.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "if debug.is_dev_environment():" in line:
            following = [
                ln for ln in lines[i + 1 : i + 8]
                if ln.strip() and not ln.strip().startswith("#")
            ]
            assert following, "mount guard has no body"
            assert "include_router(debug.router)" in following[0]
            return
    pytest.fail("app.main no longer mounts the debug router via is_dev_environment()")


def _debug_paths_in_fresh_app(tmp_path: Path, env_value: str | None) -> list[str]:
    """Import app.main in a clean interpreter and report debug routes.

    Mount is decided at import. The pytest process already imported app.main
    under whatever ENV the suite has (CI: testing), so only a subprocess can
    prove the production-shaped process does not mount the router.
    """
    code = textwrap.dedent(
        """
        import json
        from app.main import app
        paths = sorted(
            getattr(r, "path", "")
            for r in app.routes
            if "debug" in getattr(r, "path", "")
        )
        print(json.dumps(paths))
        """
    )
    env = dict(os.environ)
    if env_value is None:
        env.pop("ENV", None)
    else:
        env["ENV"] = env_value
    env.pop("ENVIRONMENT", None)
    env["DATA_DIR"] = str(tmp_path)
    env["DATABASE_URL"] = f"sqlite:///{(tmp_path / 'boot.db').as_posix()}"
    env["RAG_EMBEDDING_MODEL"] = "fake"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"importing app.main with ENV={env_value!r} failed:\n"
            f"{proc.stderr[-3000:]}"
        )
    out = proc.stdout.strip()
    if not out:
        pytest.fail(f"subprocess produced no output.\nstderr:\n{proc.stderr[-3000:]}")
    return json.loads(out.splitlines()[-1])


def test_a_production_interpreter_does_not_mount_the_debug_router(tmp_path):
    paths = _debug_paths_in_fresh_app(tmp_path, "production")
    assert "/v1/debug/env" not in paths
    assert "/debug/env" not in paths


def test_a_testing_interpreter_does_mount_the_debug_router(tmp_path):
    """CI's ENV=testing is dest — the router MUST be present in that process."""
    paths = _debug_paths_in_fresh_app(tmp_path, "testing")
    assert "/v1/debug/env" in paths
    assert "/debug/env" in paths


def test_an_unset_env_interpreter_does_not_mount_the_debug_router(tmp_path):
    paths = _debug_paths_in_fresh_app(tmp_path, None)
    assert "/v1/debug/env" not in paths
    assert "/debug/env" not in paths


def test_debug_is_a_reserved_spa_prefix():
    from app.routers.static import _RESERVED_PREFIXES

    assert any(p == "debug" or p.startswith("debug") for p in _RESERVED_PREFIXES)


@pytest.mark.parametrize("path", ["debug/env", "debug", "v1/debug/env"])
def test_spa_fallback_404s_debug_paths_without_env_data(path):
    """Live leftover: /debug/env was 200 SPA HTML. Reserved prefix → 404."""
    import asyncio

    from app.routers.static import spa_fallback

    resp = asyncio.run(spa_fallback(path))
    assert resp.status_code == 404, (path, resp.status_code)
    body = (resp.body or b"").decode("utf-8", errors="replace")
    assert "data_dir" not in body
    assert "DATA_DIR" not in body
    assert "<!doctype html" not in body.lstrip().lower()
