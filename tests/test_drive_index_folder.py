"""Tests for the recursive Drive folder import engine.

PR P0B: approved project folders must import nested subfolders/files, not just
 the direct contents of the selected Drive folder.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from tests._fixture_values import STUB_TOKEN_07


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import agent_memory as _am, projects as _proj

    if hasattr(_am, "_initialized"):
        _am._initialized = False
    if hasattr(_proj, "_initialized"):
        _proj._initialized = False
    yield tmp_path


class _FakeResponse:
    def __init__(self, status_code: int, json_data: Dict[str, Any] | None = None, content: bytes = b""):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content

    def json(self) -> Dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeDriveClient:
    """In-memory Google Drive API stub.

    ``tree`` maps folder_id -> list of child metadata dicts. ``files`` maps
    file_id -> (mime_type, bytes). Download/metadata URLs are routed by id.
    """

    def __init__(self, tree: Dict[str, List[Dict[str, Any]]], files: Dict[str, tuple[str, bytes]]):
        self.tree = tree
        self.files = files
        self.calls: List[str] = []
        self.list_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url: str, **kwargs):
        self.calls.append(url)
        params = kwargs.get("params") or {}

        if "/files/" in url:
            # file metadata or download/export
            file_id = url.split("/files/")[-1].split("/")[0]
            if file_id not in self.files:
                return _FakeResponse(404)
            mime, content = self.files[file_id]
            if params.get("alt") == "media" or "/export" in url:
                ext = ".pdf"
                if "/export" in url:
                    export_mime = params.get("mimeType", "")
                    if "pdf" in export_mime:
                        ext = ".pdf"
                    elif "docx" in export_mime:
                        ext = ".docx"
                    elif "xlsx" in export_mime:
                        ext = ".xlsx"
                    else:
                        ext = ".pdf"
                return _FakeResponse(200, content=content + ext.encode())
            return _FakeResponse(
                200,
                {"mimeType": mime, "name": file_id, "shortcutDetails": None},
            )

        # files.list
        self.list_calls += 1
        q = params.get("q", "")
        folder_id = "root"
        if "'" in q and "in parents" in q:
            folder_id = q.split("'")[1]
        children = list(self.tree.get(folder_id, []))

        page_size = int(params.get("pageSize", 200))
        page_token = params.get("pageToken")
        start = int(page_token or 0)
        end = start + page_size
        page = children[start:end]
        next_token = str(end) if end < len(children) else None
        return _FakeResponse(
            200,
            {"files": page, "nextPageToken": next_token},
        )


def _drive_tree_factory():
    folder_mime = "application/vnd.google-apps.folder"
    return {
        "root": [
            {"id": "sub1", "name": "200-Project Controls", "mimeType": folder_mime},
            {"id": "sub2", "name": "600-Procurement", "mimeType": folder_mime},
        ],
        "sub1": [
            {"id": "deep1", "name": "2.3 Risk", "mimeType": folder_mime},
        ],
        "deep1": [
            {"id": "f1", "name": "risk_register.pdf", "mimeType": "application/pdf"},
        ],
        "sub2": [
            {"id": "f2", "name": "subcontract.docx", "mimeType": "application/msword"},
        ],
    }


def _drive_files_factory():
    return {
        "f1": ("application/pdf", b"pdf-bytes"),
        "f2": ("application/msword", b"doc-bytes"),
    }


@pytest.mark.asyncio
async def test_walk_drive_folder_into_project_recurses(isolated_data_dir, monkeypatch):
    from app.routers import drive as drive_router
    from app.core import projects as projects_mod

    tree = _drive_tree_factory()
    files = _drive_files_factory()
    fake_client = _FakeDriveClient(tree, files)
    monkeypatch.setattr(drive_router.httpx, "AsyncClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(drive_router.doc_index, "maybe_eager_index", lambda *a, **kw: None)

    proj = projects_mod.create_project(name="Recursion Test", user_id="system")
    result = await drive_router._walk_drive_folder_into_project(
        project_id=proj["id"],
        user_id="system",
        access_token=STUB_TOKEN_07,
        folder_id="root",
        max_files=100,
        max_depth=4,
        role="other",
    )

    imported_names = {item["name"] for item in result["imported"]}
    assert "risk_register.pdf" in imported_names, (
        f"Nested file not imported; got {imported_names}"
    )
    assert "subcontract.docx" in imported_names
    assert result["imported_count"] == 2


@pytest.mark.asyncio
async def test_include_extensions_narrows_but_never_widens_the_text_rule(isolated_data_dir, monkeypatch):
    """A caller-supplied include_extensions used to REPLACE the allow-list, so
    a request could pull drawings or photos into a project. It now narrows the
    one text-only declaration and can never widen it."""
    from app.routers import drive as drive_router
    from app.core import projects as projects_mod

    tree = _drive_tree_factory()
    tree["sub2"].append({"id": "f3", "name": "site_plan.dwg", "mimeType": "image/vnd.dwg"})
    files = {**_drive_files_factory(), "f3": ("image/vnd.dwg", b"dwg-bytes")}
    monkeypatch.setattr(drive_router.httpx, "AsyncClient", lambda *a, **kw: _FakeDriveClient(tree, files))
    monkeypatch.setattr(drive_router.doc_index, "maybe_eager_index", lambda *a, **kw: None)
    proj = projects_mod.create_project(name="Narrow Test", user_id="system")

    async def walk(include):
        return await drive_router._walk_drive_folder_into_project(
            project_id=proj["id"], user_id="system", access_token=STUB_TOKEN_07,
            folder_id="root", max_files=100, max_depth=4, role="other",
            include_extensions=include,
        )

    widened = await walk([".dwg", ".pdf"])
    assert {i["name"] for i in widened["imported"]} == {"risk_register.pdf"}
    only_dwg = await walk([".dwg"])
    assert only_dwg["imported_count"] == 0


@pytest.mark.asyncio
async def test_walk_drive_folder_into_project_paginates(isolated_data_dir, monkeypatch):
    from app.routers import drive as drive_router
    from app.core import projects as projects_mod

    folder_mime = "application/vnd.google-apps.folder"
    tree: Dict[str, List[Dict[str, Any]]] = {
        "root": [{"id": "sub", "name": "Sub", "mimeType": folder_mime}],
        "sub": [],
    }
    files: Dict[str, tuple[str, bytes]] = {}
    for i in range(250):
        fid = f"file_{i}"
        tree["sub"].append({"id": fid, "name": f"doc_{i}.pdf", "mimeType": "application/pdf"})
        files[fid] = ("application/pdf", f"x{i}".encode())

    fake_client = _FakeDriveClient(tree, files)
    monkeypatch.setattr(drive_router.httpx, "AsyncClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(drive_router.doc_index, "maybe_eager_index", lambda *a, **kw: None)

    proj = projects_mod.create_project(name="Pagination Test", user_id="system")
    result = await drive_router._walk_drive_folder_into_project(
        project_id=proj["id"],
        user_id="system",
        access_token=STUB_TOKEN_07,
        folder_id="root",
        max_files=1000,
        max_depth=4,
    )

    assert result["imported_count"] == 250, (
        f"Expected all 250 paginated files; got {result['imported_count']}"
    )
    assert fake_client.list_calls > 1, "Pagination should require more than one files.list call"


@pytest.mark.asyncio
async def test_run_drive_folder_import_uses_helper(isolated_data_dir, monkeypatch):
    from app.routers import admin as admin_mod
    from app.core import drive_auth, projects as projects_mod

    import app.routers.drive as drive_mod

    tree = _drive_tree_factory()
    files = _drive_files_factory()
    fake_client = _FakeDriveClient(tree, files)
    monkeypatch.setattr(drive_mod.httpx, "AsyncClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(drive_mod.doc_index, "maybe_eager_index", lambda *a, **kw: None)

    async def fake_token(_user_id):
        return STUB_TOKEN_07

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)

    proj = projects_mod.create_project(name="Worker Test", user_id="system")
    await admin_mod._run_drive_folder_import(
        project_id=proj["id"],
        user_id="system",
        folder_id="root",
        max_files=100,
        max_depth=4,
        role="other",
    )

    docs = projects_mod.list_documents(proj["id"])
    names = {d["original_name"] for d in docs}
    assert "risk_register.pdf" in names, f"Worker did not import nested files; got {names}"
    assert "subcontract.docx" in names


def test_index_folder_route_is_async_returns_queued(isolated_data_dir, monkeypatch):
    """The index-folder route must NOT walk Drive synchronously (a real folder
    exceeds the request timeout -> 502). It validates + queues a background
    worker and returns 202 'queued' immediately."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.dependencies import require_user
    from app.core import projects as projects_mod, users as users_mod
    from app.routers import drive as drive_router

    projects_mod.init_db()
    users_mod.ensure_user_exists("idx-user")
    projects_mod.create_project(
        name="Idx", user_id="idx-user", project_id="idx-proj-1", origin="user_create",
    )
    # Indexing a folder into a project is the admin path (privileges.py).
    app.dependency_overrides[require_user] = lambda: {"user_id": "idx-user", "role": "admin"}

    async def fake_token(uid):  # connection check passes
        return "tok"
    monkeypatch.setattr("app.core.drive_auth.get_access_token", fake_token)

    captured = {}
    async def fake_bg(**kwargs):  # capture that the walk was QUEUED, not run inline
        captured.update(kwargs)
    monkeypatch.setattr(drive_router, "_run_index_folder_bg", fake_bg)

    try:
        with TestClient(app) as c:
            r = c.post(
                "/v1/projects/idx-proj-1/drive/index-folder",
                json={"folder_id": "F1", "max_files": 7, "max_depth": 3},
            )
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "queued"
        assert captured.get("folder_id") == "F1"
        assert captured.get("max_files") == 7
        assert captured.get("project_id") == "idx-proj-1"
    finally:
        app.dependency_overrides.pop(require_user, None)
        projects_mod.delete_project("idx-proj-1")
