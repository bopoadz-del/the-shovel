"""Parallel shard workers must converge on one project row per folder."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core import projects as projects_mod


def test_get_or_create_is_idempotent_same_id():
    name = "Example Infra Pack"
    pid = "example_infra_pack"
    a, created_a = projects_mod.get_or_create_project(name, project_id=pid)
    b, created_b = projects_mod.get_or_create_project(name, project_id=pid)
    assert a["id"] == b["id"] == pid
    assert created_a is True
    assert created_b is False


def test_get_or_create_stable_id_from_name_when_unspecified():
    name = "Shard Race Folder Alpha"
    a, _ = projects_mod.get_or_create_project(name)
    b, created = projects_mod.get_or_create_project(name)
    assert a["id"] == b["id"]
    assert a["id"].startswith("p_")
    assert created is False


def test_parallel_workers_share_one_project_id():
    """Simulate N shard workers racing on a fresh folder name."""
    name = "Parallel Shard Folder Beta"
    preferred = "parallel_shard_folder_beta"

    def _worker(_i: int):
        proj, _created = projects_mod.get_or_create_project(
            name, project_id=preferred,
        )
        return proj["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [fut.result() for fut in as_completed(
            [pool.submit(_worker, i) for i in range(8)]
        )]

    assert set(ids) == {preferred}
    # Only one active row with this name
    matches = [p for p in projects_mod.list_projects() if p.get("name") == name]
    assert len(matches) == 1
    assert matches[0]["id"] == preferred


def test_prefers_existing_name_match_over_new_stable_id():
    """Legacy rows created with a random id still win by name."""
    name = "Legacy Name Match Folder"
    legacy = projects_mod.create_project(name)  # random id
    got, created = projects_mod.get_or_create_project(
        name, project_id="should_not_be_used_when_name_exists",
    )
    assert created is False
    assert got["id"] == legacy["id"]
