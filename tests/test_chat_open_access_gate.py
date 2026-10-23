"""Chat tenant gate follows the open-access rule (2026-07-26, third asymmetry).

#267 let admins/shared-users OPEN a project, #277 let them UPLOAD to it —
but the chat gates still did owner-only lookups and silently dropped the
project_id: zero injected RAG context and a no-op search tool on exactly the
projects the pilot shares (found live: the corpus project, owned by a seed
account, was invisible to every admin's chat). One rule now covers all
surfaces: owner -> admin-approved shared -> admin role. Strangers on private
projects stay dropped (fail-closed tenancy).

World-building goes through the HTTP register/login API + minimal ORM
promotion — the same pattern as test_authz_matrix_sweep, which is green on
both SQLite and the shared-Postgres CI job (store-direct fixtures were not).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from tests._fixture_values import PW_10, STUB_TOKEN_02


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, email):
    r = client.post("/v1/users/register", json={"email": email, "password": PW_10})
    assert r.status_code in (201, 409), r.text
    r = client.post("/v1/users/login", json={"email": email, "password": PW_10})
    return {"token": r.json()["token"], "id": r.json()["user"]["id"]}


def _promote_admin(uid):
    from app.core.db import SessionLocal
    from app.core.models import User
    with SessionLocal() as db:
        db.get(User, uid).role = "admin"
        db.commit()


def _mark_shared(pid):
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as db:
        p = db.get(Project, pid)
        p.origin = "admin_drive_approved"
        p.is_approved = True
        db.commit()


def _h(actor):
    return {"Authorization": f"Bearer {actor['token']}"}


@pytest.fixture
def world(client):
    owner = _register(client, "oag-owner@example.com")
    admin = _register(client, "oag-admin@example.com")
    stranger = _register(client, "oag-stranger@example.com")
    _promote_admin(admin["id"])

    private_pid = client.post(
        "/v1/projects", json={"name": "OAG private"}, headers=_h(owner)
    ).json()["id"]
    shared_pid = client.post(
        "/v1/projects", json={"name": "OAG shared"}, headers=_h(owner)
    ).json()["id"]
    _mark_shared(shared_pid)
    yield {
        "owner": owner, "admin": admin, "stranger": stranger,
        "private": private_pid, "shared": shared_pid,
    }
    for pid in (private_pid, shared_pid):
        store.archive_project(pid)


def test_owner_resolves_both(world):
    assert store.get_project_accessible(world["private"], world["owner"]["id"]) is not None
    assert store.get_project_accessible(world["shared"], world["owner"]["id"]) is not None


def test_admin_does_not_resolve_other_users_private_project(world):
    # SECURITY (legacy-admin tenancy fix): an admin — and therefore ANY
    # legacy/master-key holder, which require_user maps to the SYSTEM admin —
    # must NOT read another user's PRIVATE project through the chat/RAG data
    # path. Genuine admin cross-tenant work goes through /v1/admin/* instead.
    assert store.get_project_accessible(world["private"], world["admin"]["id"]) is None


def test_admin_resolves_admin_approved_shared_project(world):
    # The legitimate half of the original incident: an admin CAN chat over an
    # admin-approved shared platform project (same as any user — handled by the
    # scoped get_project call, not the admin fallthrough).
    proj = store.get_project_accessible(world["shared"], world["admin"]["id"])
    assert proj is not None and proj["id"] == world["shared"]


def test_admin_resolves_system_owned_platform_project(client, world):
    # The corpus-project half of the original incident: a system/seed-owned
    # platform corpus stays admin-readable via the admin fallthrough — that is
    # exactly what _is_platform_project preserves, without exposing private
    # user projects.
    from app.core.db import SessionLocal
    from app.core.models import Project
    from app.core.users import SYSTEM_USER_ID
    pid = client.post(
        "/v1/projects", json={"name": "OAG platform corpus"}, headers=_h(world["owner"])
    ).json()["id"]
    with SessionLocal() as db:
        db.get(Project, pid).user_id = SYSTEM_USER_ID  # seed/system-owned corpus
        db.commit()
    try:
        proj = store.get_project_accessible(pid, world["admin"]["id"])
        assert proj is not None and proj["id"] == pid
    finally:
        store.archive_project(pid)


def test_regular_user_cannot_read_a_system_owned_platform_project(client, world):
    """The mirror of the test above, and the one the role check actually buys.

    Found by a mutation probe: replacing

        if u and (u.get("role") or "").lower() == "admin":

    with

        if u:

    left every test in this file green. The inner _is_platform_project fence
    still held, so the admin fallthrough stayed scoped to platform projects —
    but ANY resolvable user reached it, and a system-owned corpus that is not
    admin-approved is exactly the row that fence does not cover. Nothing
    asserted that the role, not merely being a known user, is what opens it.

    A guard nothing tests is a guard nobody can safely change.
    """
    from app.core.db import SessionLocal
    from app.core.models import Project
    from app.core.users import SYSTEM_USER_ID

    pid = client.post(
        "/v1/projects", json={"name": "OAG platform corpus 2"}, headers=_h(world["owner"])
    ).json()["id"]
    with SessionLocal() as db:
        p = db.get(Project, pid)
        p.user_id = SYSTEM_USER_ID   # seed/system-owned corpus
        p.is_approved = False        # NOT admin-approved: not shared with users
        db.commit()
    try:
        # The admin reaches it (the behaviour the fallthrough exists for) …
        assert store.get_project_accessible(pid, world["admin"]["id"]) is not None
        # … and a regular, known, logged-in user does not.
        assert store.get_project_accessible(pid, world["stranger"]["id"]) is None
    finally:
        store.archive_project(pid)


def test_legacy_key_system_admin_cannot_read_private_project(world):
    # The exact blocker: a legacy API key resolves to SYSTEM_USER_ID (role
    # admin). Through the chat data-path helper it must NOT reach a real
    # user's private project.
    from app.core.users import SYSTEM_USER_ID
    assert store.get_project_accessible(world["private"], SYSTEM_USER_ID) is None


def test_regular_user_resolves_shared_but_not_private(world):
    assert store.get_project_accessible(world["shared"], world["stranger"]["id"]) is not None
    assert store.get_project_accessible(world["private"], world["stranger"]["id"]) is None


def test_anonymous_and_unknown_fail_closed(world):
    assert store.get_project_accessible(world["private"], None) is None
    assert store.get_project_accessible(world["private"], "no-such-user") is None
    assert store.get_project_accessible("nope_gate", world["admin"]["id"]) is None


def test_archived_projects_stay_invisible(client, world):
    pid = client.post(
        "/v1/projects", json={"name": "OAG archived"}, headers=_h(world["owner"])
    ).json()["id"]
    store.archive_project(pid)
    # Even the admin path must not resurface an archived project in chat.
    assert store.get_project_accessible(pid, world["admin"]["id"]) is None


def test_regular_user_can_open_system_seed_gk_conversation(client, world, monkeypatch):
    """Live tip f8d6e65: GET /v1/projects/curated_kb is 200 for role=user,
    but POST chat/stream with ws-curated_kb-<ms> was 404 Conversation not
    found. Conversation ACL must use the same include_admin_approved grant
    as project GET. Private user_create stays fail-closed.
    """
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists
    from app.agents.runtime import Agent

    async def _fake_llm(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": "gk-ok", "tool_calls": []}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _fake_llm)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True
    )
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)

    gk_id = f"oag_gkchat_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG General Knowledge chat",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="system_seed",
    )
    try:
        headers = _h(world["stranger"])
        cid = f"ws-{gk_id}-1740000000000"

        hist = client.get(
            f"/v1/agents/conversations/{cid}/messages", headers=headers
        )
        assert hist.status_code == 200, hist.text
        assert hist.json()["conversation_id"] == cid
        assert hist.json()["messages"] == []

        chat = client.post(
            "/v1/agents/project-assistant/chat/stream",
            headers=headers,
            json={
                "message": "hello general knowledge",
                "project_id": gk_id,
                "conversation_id": cid,
            },
        )
        assert chat.status_code != 404, chat.text
        assert "Conversation not found" not in chat.text

        # Alice's private project stays invisible to the stranger.
        private_cid = f"ws-{world['private']}-1740000000000"
        priv = client.get(
            f"/v1/agents/conversations/{private_cid}/messages",
            headers=headers,
        )
        assert priv.status_code == 404, priv.text
        assert "Conversation not found" in priv.text
        priv_chat = client.post(
            "/v1/agents/project-assistant/chat/stream",
            headers=headers,
            json={
                "message": "probe",
                "project_id": world["private"],
                "conversation_id": private_cid,
            },
        )
        assert priv_chat.status_code == 404, priv_chat.text
        assert "Conversation not found" in priv_chat.text

        # Physical master-corpus source id stays owner-only (UI-PHYS H1).
        source = store.MASTER_CORPUS_SOURCE_PROJECT_ID
        src = client.get(
            f"/v1/agents/conversations/ws-{source}-1740000000000/messages",
            headers=headers,
        )
        assert src.status_code == 404, src.text
    finally:
        store.delete_project(gk_id)


def test_non_ws_conversation_uses_shared_platform_grant(client, world, monkeypatch):
    """Non-ws ids fall back to the stored project_id (agents.py line ~110).

    That lookup must use include_admin_approved: a GK-bound ad-hoc
    conversation is readable by a role=user non-owner, a private-project
    binding stays 404. Virgin #602 failed diff-cover on this branch.
    """
    import uuid
    from app.core import agent_memory as am
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gknonws_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG GK non-ws",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="system_seed",
    )
    gk_cid = f"hr-gk-{uuid.uuid4().hex[:8]}"
    priv_cid = f"hr-priv-{uuid.uuid4().hex[:8]}"
    am.get_or_create_conversation(gk_cid, "project-assistant", project_id=gk_id)
    am.append_message(gk_cid, "user", "gk note")
    am.get_or_create_conversation(
        priv_cid, "project-assistant", project_id=world["private"]
    )
    am.append_message(priv_cid, "user", "secret")
    try:
        headers = _h(world["stranger"])
        gk = client.get(
            f"/v1/agents/conversations/{gk_cid}/messages", headers=headers
        )
        assert gk.status_code == 200, gk.text
        assert "gk note" in [m["content"] for m in gk.json()["messages"]]

        priv = client.get(
            f"/v1/agents/conversations/{priv_cid}/messages", headers=headers
        )
        assert priv.status_code == 404, priv.text
        assert "Conversation not found" in priv.text
    finally:
        store.delete_project(gk_id)


def test_stranger_http_surfaces_reach_system_seed_gk(client, world, monkeypatch):
    """Live QA 404s: GET project, documents, and rag/search on curated_kb."""
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gkhttp_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG GK HTTP",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="system_seed",
    )
    try:
        headers = _h(world["stranger"])
        detail = client.get(f"/v1/projects/{gk_id}", headers=headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["id"] == gk_id
        docs = client.get(f"/v1/projects/{gk_id}/documents", headers=headers)
        assert docs.status_code == 200, docs.text
        rag = client.post(
            "/v1/rag/search",
            headers=headers,
            json={"query": "general knowledge", "project_id": gk_id, "k": 3},
        )
        assert rag.status_code != 404, rag.text
        assert "Project" not in (rag.json().get("detail") or "")
        private = client.get(f"/v1/projects/{world['private']}", headers=headers)
        assert private.status_code == 404
    finally:
        store.delete_project(gk_id)


def test_regular_user_resolves_system_seed_gk_project(client, world, monkeypatch):
    """Live 404: curated_kb is SYSTEM-owned with origin=system_seed.

    Ordinary users (role=user) must reach it via include_admin_approved
    — not the admin fallthrough. Private user_create stays invisible.
    """
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gk_{uuid.uuid4().hex[:10]}"
    other_gk = f"oag_gk2_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", f"{gk_id},{other_gk}")
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG General Knowledge",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="system_seed",
    )
    try:
        stranger = world["stranger"]["id"]
        got = store.get_project(
            gk_id, user_id=stranger, include_admin_approved=True,
        )
        assert got is not None and got["id"] == gk_id
        assert store.get_project_accessible(gk_id, stranger) is not None
        listed = store.list_projects(
            user_id=stranger, include_admin_approved=True,
        )
        assert gk_id in {p["id"] for p in listed}
        # Without the shared-platform flag, GK stays owner-only.
        assert store.get_project(gk_id, user_id=stranger) is None
        assert gk_id not in {
            p["id"]
            for p in store.list_projects(user_id=stranger)
        }
        # Private user_create stays fail-closed.
        assert store.get_project(
            world["private"], user_id=stranger, include_admin_approved=True,
        ) is None
        assert store.get_project_accessible(world["private"], stranger) is None
    finally:
        store.delete_project(gk_id)


def test_regular_user_resolves_system_seed_origin_outside_gk_env(client, world):
    """origin=system_seed is shared even when the id is not in the env set."""
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    pid = f"oag_seed_{uuid.uuid4().hex[:10]}"
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG seeded corpus",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=pid,
        origin="system_seed",
    )
    try:
        stranger = world["stranger"]["id"]
        assert store.get_project(
            pid, user_id=stranger, include_admin_approved=True,
        ) is not None
        assert store.get_project_accessible(pid, stranger) is not None
        assert pid in {
            p["id"]
            for p in store.list_projects(
                user_id=stranger, include_admin_approved=True,
            )
        }
    finally:
        store.delete_project(pid)


def test_regular_user_resolves_gk_env_id_regardless_of_origin(client, world, monkeypatch):
    """A configured GK id is shared even if origin was left as user_create."""
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gkenv_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG GK env id",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="user_create",
    )
    try:
        stranger = world["stranger"]["id"]
        assert store.get_project(
            gk_id, user_id=stranger, include_admin_approved=True,
        ) is not None
        assert store.get_project_accessible(gk_id, stranger) is not None
        assert gk_id in {
            p["id"]
            for p in store.list_projects(
                user_id=stranger, include_admin_approved=True,
            )
        }
    finally:
        store.delete_project(gk_id)


def test_unapproved_gk_stays_owner_only(client, world, monkeypatch):
    """is_approved=False stays fail-closed even for a GK env id / system_seed."""
    import uuid
    from app.core.db import SessionLocal
    from app.core.models import Project
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gkunapp_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG GK pending",
        user_id=SYSTEM_USER_ID,
        is_approved=False,
        project_id=gk_id,
        origin="system_seed",
    )
    with SessionLocal() as db:
        row = db.get(Project, gk_id)
        row.is_approved = False
        db.commit()
    try:
        stranger = world["stranger"]["id"]
        assert store.get_project(
            gk_id, user_id=stranger, include_admin_approved=True,
        ) is None
        assert store.get_project_accessible(gk_id, stranger) is None
        assert gk_id not in {
            p["id"]
            for p in store.list_projects(
                user_id=stranger, include_admin_approved=True,
            )
        }
    finally:
        store.delete_project(gk_id)


def test_hidden_gk_is_readable_but_not_listed(client, world, monkeypatch):
    """hidden_from_sidebar drops GK from the picker, not from get_project."""
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk_id = f"oag_gkhide_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk_id)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG GK hidden",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=gk_id,
        origin="system_seed",
    )
    store.set_hidden_from_sidebar(gk_id, True)
    try:
        stranger = world["stranger"]["id"]
        assert store.get_project(
            gk_id, user_id=stranger, include_admin_approved=True,
        ) is not None
        assert gk_id not in {
            p["id"]
            for p in store.list_projects(
                user_id=stranger, include_admin_approved=True,
            )
        }
        assert gk_id in {
            p["id"]
            for p in store.list_projects(
                user_id=stranger, include_admin_approved=True,
                include_hidden=True,
            )
        }
    finally:
        store.delete_project(gk_id)


def test_master_corpus_source_id_is_same_membership_as_alias(client, world, monkeypatch):
    """S13: get_project_accessible(source) matches the alias for a member.

    get_project(source) stays None for a non-owner (UI-PHYS H1 picker 404).
    The data-path helper must not conflate that with 'not a member'.
    """
    import uuid
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    tag = uuid.uuid4().hex[:10]
    alias = f"oag_mc_{tag}"
    source = f"oag_src_{tag}"
    monkeypatch.setattr(store, "MASTER_CORPUS_PROJECT_ID", alias)
    monkeypatch.setattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", source)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="OAG source corpus",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=source,
        origin="user_create",
    )
    try:
        member = world["stranger"]["id"]
        assert store.get_project(alias, user_id=member, include_admin_approved=True) is not None
        assert store.get_project(source, user_id=member, include_admin_approved=True) is None
        assert store.get_project_accessible(alias, member) is not None
        src = store.get_project_accessible(source, member)
        assert src is not None
        assert src["id"] == source
        assert store.get_project_accessible(source, world["owner"]["id"]) is not None
        assert store.get_project_accessible(source, None) is None
    finally:
        store.delete_project(source)
