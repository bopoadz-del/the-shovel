"""Chat project-memory tenant isolation.

Regression for the cross-tenant leak where POSTing /chat with another user's
project_id injected that project's private facts into the caller's prompt
(`_with_project_memory` did no ownership check).
"""


def test_chat_project_memory_is_tenant_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from app.core import projects as projects_store
    from app.core import users as users_mod
    from app.routers.chat import _with_project_memory

    projects_store.init_db()  # create the schema in the relocated DATA_DIR
    users_mod.ensure_user_exists("user-A")
    users_mod.ensure_user_exists("user-B")
    proj = projects_store.create_project("Tenant A project", user_id="user-A")
    pid = proj["id"]
    projects_store.set_fact(pid, "site_address", "42 Secret Lane")

    prompt = "where is the site located"

    # The owner's own project facts ARE injected (feature still works).
    owner = _with_project_memory(prompt, pid, "user-A")
    assert "42 Secret Lane" in owner

    # A different tenant must NOT receive the project's facts — the prompt is
    # returned unchanged, with no fact content leaked.
    attacker = _with_project_memory(prompt, pid, "user-B")
    assert attacker == prompt
    assert "Secret Lane" not in attacker

    # An unknown project id is a safe no-op even for the owner.
    assert _with_project_memory(prompt, "nope1234", "user-A") == prompt
