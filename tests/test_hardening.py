"""Regression tests for the High/Medium production-hardening fixes."""

import pytest
from tests._dev_key import DEV_KEY
from tests._fixture_values import ENV_SECRET_05, ENV_SECRET_06, PW_06, STUB_TOKEN_16


# ── file_crypto: decrypt fails loud on a key mismatch ───────────────────────────

def test_decrypt_with_wrong_key_raises_loudly(monkeypatch):
    """A real Fernet token that cannot be decrypted (rotated/wrong key) must
    raise DecryptionError, not silently return the ciphertext."""
    from cryptography.fernet import Fernet
    from app.core import file_crypto

    key_a = Fernet.generate_key()
    key_b = Fernet.generate_key()

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", key_a.decode())
    token = file_crypto.encrypt_bytes(b"confidential contract")
    # Correct key still round-trips.
    assert file_crypto.decrypt_bytes(token) == b"confidential contract"

    # Wrong key: must fail loud rather than return ciphertext as "plaintext".
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", key_b.decode())
    with pytest.raises(file_crypto.DecryptionError):
        file_crypto.decrypt_bytes(token)


def test_legacy_plaintext_still_passes_through(monkeypatch):
    """A plaintext file is returned untouched even when a key is configured."""
    from cryptography.fernet import Fernet
    from app.core import file_crypto

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    assert file_crypto.decrypt_bytes(b"%PDF-1.4 plain bytes") == b"%PDF-1.4 plain bytes"


# ── startup: production requires SECRET_KEY ─────────────────────────────────────

def test_production_startup_requires_secret_key(monkeypatch):
    from app.main import _validate_startup_env

    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _validate_startup_env()

    # With SECRET_KEY and DATABASE_URL set, the next required prod secret is
    # DATA_ENCRYPTION_KEY — encryption at rest is not optional in production.
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_06)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/test")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DATA_ENCRYPTION_KEY"):
        _validate_startup_env()


def test_production_startup_requires_data_encryption_key(monkeypatch):
    """ENV=production must refuse to boot without DATA_ENCRYPTION_KEY."""
    from app.main import _validate_startup_env

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_06)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/test")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DATA_ENCRYPTION_KEY"):
        _validate_startup_env()


def _prod_env_complete(monkeypatch):
    """Minimum env for production startup validation to succeed."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_06)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/test")
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", STUB_TOKEN_16)
    monkeypatch.setenv("CEREBRUM_VIRGIN", "false")
    monkeypatch.setenv("CEREBRUM_DOMAIN_KITS", "construction")


def test_production_startup_requires_construction_kit_env(monkeypatch):
    """Production Fork pins this repo's construction kit — virgin/empty kits fail."""
    from app.main import _validate_startup_env

    _prod_env_complete(monkeypatch)
    monkeypatch.setenv("CEREBRUM_VIRGIN", "true")
    monkeypatch.setenv("CEREBRUM_DOMAIN_KITS", "")
    with pytest.raises(RuntimeError, match="construction"):
        _validate_startup_env()


def test_production_startup_fails_if_construction_container_missing(monkeypatch):
    """Even with env flags set, boot fails if the construction block did not load."""
    from app.main import _validate_startup_env
    import app.blocks as blocks_mod

    _prod_env_complete(monkeypatch)
    stripped = {k: v for k, v in blocks_mod.BLOCK_REGISTRY.items() if k != "construction"}
    monkeypatch.setattr(blocks_mod, "BLOCK_REGISTRY", stripped)
    with pytest.raises(RuntimeError, match="construction"):
        _validate_startup_env()


def test_production_startup_passes_with_required_secrets_and_kit(monkeypatch):
    from app.main import _validate_startup_env
    import app.blocks as blocks_mod

    _prod_env_complete(monkeypatch)
    patched = dict(blocks_mod.BLOCK_REGISTRY)
    patched["construction"] = object()
    monkeypatch.setattr(blocks_mod, "BLOCK_REGISTRY", patched)
    _validate_startup_env()


def test_non_production_skips_secret_key_check(monkeypatch):
    from app.main import _validate_startup_env

    monkeypatch.setenv("ENV", "testing")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    _validate_startup_env()  # must not raise outside production


# ── doc_types: registry mutation is admin-only ──────────────────────────────────

def test_doc_types_mutation_is_admin_only():
    """A non-admin user cannot add or delete entries in the global doc-type
    registry; reads stay open to any authenticated caller."""
    import uuid

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"dt-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post(
            "/v1/users/login", json={"email": email, "password": PW_06}
        ).json()["token"]
        user = {"Authorization": f"Bearer {token}"}

        assert c.post(
            "/v1/document-types", headers=user, json={"name": "Sneaky Type"}
        ).status_code == 403
        assert c.delete(
            "/v1/document-types/whatever", headers=user
        ).status_code == 403
        # Reads remain available to any authenticated user.
        assert c.get("/v1/document-types", headers=user).status_code == 200


# ── /execute: code-execution blocks are admin-only ──────────────────────────────

def test_code_blocks_are_admin_only_via_execute():
    """A non-admin user cannot run the arbitrary-code blocks through /execute,
    but can still run ordinary blocks."""
    import uuid

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"ex-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post(
            "/v1/users/login", json={"email": email, "password": PW_06}
        ).json()["token"]
        user = {"Authorization": f"Bearer {token}"}

        # `code` runs arbitrary Python — the registered code-execution block.
        r = c.post("/v1/execute", headers=user, json={"block": "code", "input": "x"})
        assert r.status_code == 403, (r.status_code, r.text)

        # An ordinary block is still runnable by the same non-admin user.
        ok = c.post(
            "/v1/execute", headers=user,
            json={"block": "translate", "input": "x",
                  "params": {"target": "en"}},
        )
        assert ok.status_code != 403, ok.text


def test_code_blocks_are_admin_only_via_chain():
    """JWT role=user POST /v1/chain with block: code must be 403, same as /v1/execute."""
    import uuid

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        email = f"ch-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post(
            "/v1/users/login", json={"email": email, "password": PW_06}
        ).json()["token"]
        user = {"Authorization": f"Bearer {token}"}

        r = c.post(
            "/v1/chain",
            headers=user,
            json={"steps": [{"block": "code", "params": {"operation": "execute"}}],
                  "initial_input": "print(1)"},
        )
        assert r.status_code == 403, (r.status_code, r.text)

        # Orchestrator must not launder the same privilege via /v1/execute.
        orch = c.post(
            "/v1/execute",
            headers=user,
            json={
                "block": "orchestrator",
                "params": {"steps": [{"block": "code", "params": {}}]},
            },
        )
        assert orch.status_code == 403, (orch.status_code, orch.text)


def test_non_admin_api_key_cannot_hit_privileged_blocks():
    """the dev key authenticates but must not inherit the system user's admin role."""
    from fastapi.testclient import TestClient

    from app.main import app

    key = {"Authorization": "Bearer " + DEV_KEY}
    with TestClient(app) as c:
        exe = c.post("/v1/execute", headers=key, json={"block": "code", "input": "x"})
        assert exe.status_code == 403, (exe.status_code, exe.text)

        chain = c.post(
            "/v1/chain",
            headers=key,
            json={"steps": [{"block": "sandbox", "params": {}}]},
        )
        assert chain.status_code == 403, (chain.status_code, chain.text)


def test_chain_500_does_not_leak_exception_text():
    """/v1/chain must return a generic 500 — never interpolate str(e)."""
    from fastapi.testclient import TestClient

    from app.dependencies import block_instances
    from app.main import app

    class Boom:
        async def execute(self, *_a, **_k):
            raise RuntimeError("secret internals xyz")

    previous = block_instances.get("orchestrator")
    block_instances["orchestrator"] = Boom()
    try:
        with TestClient(app) as c:
            r = c.post(
                "/v1/chain",
                headers={"Authorization": "Bearer " + DEV_KEY},
                json={"steps": [{"block": "translate", "params": {"target": "en"}}]},
            )
        assert r.status_code == 500, (r.status_code, r.text)
        body = r.text.lower()
        assert "secret internals" not in body
        assert "xyz" not in body
    finally:
        if previous is not None:
            block_instances["orchestrator"] = previous
        else:
            block_instances.pop("orchestrator", None)


def test_upload_json_omits_file_path():
    """Upload JSON must not leak the on-disk path; keep stored_as / document_id."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        r = c.post(
            "/v1/upload",
            headers={"Authorization": "Bearer " + DEV_KEY},
            files={"file": ("note.txt", b"hello upload", "text/plain")},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "file_path" not in data
        assert "stored_as" in data


def test_mcp_sse_requires_auth():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        assert c.get("/mcp/sse").status_code in (401, 403)


def test_mcp_sse_unavailable_stub_declares_auth():
    """The 503 SSE stub (mcp package missing) must still require a bearer token."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("app", "routers", "mcp.py").read_text(
        encoding="utf-8"
    )
    assert "async def mcp_sse_unavailable" in src
    assert "mcp_sse_unavailable" in src
    # Auth must be a FastAPI dependency on the stub, not an open 503.
    stub_idx = src.index("async def mcp_sse_unavailable")
    stub = src[stub_idx:stub_idx + 250]
    assert "require_api_key" in stub or "require_user" in stub


def test_mcp_call_tool_instantiates_via_registry_not_string():
    """_create_block_instance expects a class; MCP must not pass the block name string."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("app", "routers", "mcp.py").read_text(
        encoding="utf-8"
    )
    assert "_create_block_instance(name)" not in src
    assert "get_block_instance" in src


def test_privileged_blocks_helper_matches_execute_gate():
    from fastapi import HTTPException

    from app.core.privileges import raise_if_privileged_block

    with pytest.raises(HTTPException) as exc:
        raise_if_privileged_block("code", role="user")
    assert exc.value.status_code == 403

    with pytest.raises(HTTPException) as exc:
        raise_if_privileged_block("sandbox", role=None)
    assert exc.value.status_code == 403

    raise_if_privileged_block("code", role="admin")
    raise_if_privileged_block("translate", role="user")


# ── Drive: OAuth token is per-user, not process-global ──────────────────────────

def test_drive_token_is_per_user(tmp_path, monkeypatch):
    """One user connecting Google Drive must not expose it to another user."""
    import time
    import uuid

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", ENV_SECRET_05)

    from app.main import app
    from app.core import drive_auth

    with TestClient(app) as c:
        email = f"drv-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post(
            "/v1/users/login", json={"email": email, "password": PW_06}
        ).json()["token"]
        user_b = {"Authorization": f"Bearer {token}"}

        # The 'system' user (legacy the dev key) connects Drive.
        drive_auth.save_token("system", {
            "access_token": "AT", "refresh_token": "RT",
            "expiry": time.time() + 9999, "email": "system@example.com",
        })

        # System sees it connected; user B must not.
        sys_status = c.get(
            "/v1/drive/status", headers={"Authorization": "Bearer " + DEV_KEY}
        ).json()
        assert sys_status["connected"] is True

        b_status = c.get("/v1/drive/status", headers=user_b).json()
        assert b_status["connected"] is False

        # User B cannot list the system user's Drive files.
        assert c.get("/v1/drive/files", headers=user_b).status_code == 409


# ── agent_facts: durable facts are scoped per project ──────────────────────────

def test_agent_facts_are_project_scoped(tmp_path, monkeypatch):
    """A fact an agent remembers in one project must not surface in another."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from app.core import agent_memory

    agent_memory.init_db()
    agent_memory.set_agent_fact("pm", "site", "Project A address", project_id="proj-A")
    agent_memory.set_agent_fact("pm", "site", "Project B address", project_id="proj-B")

    a = agent_memory.list_agent_facts("pm", "proj-A")
    b = agent_memory.list_agent_facts("pm", "proj-B")
    assert [f["value"] for f in a] == ["Project A address"]
    assert [f["value"] for f in b] == ["Project B address"]

    # A different project sees nothing.
    assert agent_memory.list_agent_facts("pm", "proj-C") == []
