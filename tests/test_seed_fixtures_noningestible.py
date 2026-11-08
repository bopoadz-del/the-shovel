"""The fixture seeder treats a by-design 415 (non-ingestible type) as a skip,
not a crash (ORDERS_08NOV_0020Z M10b). The server is the single source of
truth for ingestibility; the seeder must not re-implement the extension list
and must not upload a .xer/.dxf into a 415 crash.
"""
from __future__ import annotations

from pathlib import Path

from scripts import seed_fixtures


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._p = payload or {}

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    """A live-API stand-in: GET /v1/projects finds the project, POST documents
    415s a non-ingestible extension and 200s a text file, and the documents
    list reflects only what was actually uploaded (200)."""

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self.project_id = "p1"
        self.uploaded: list[str] = []

    def get(self, path: str, params=None):
        if path == "/v1/projects":
            return _Resp(200, {"projects": [
                {"name": self.project_name, "id": self.project_id},
            ]})
        if path == f"/v1/projects/{self.project_id}/documents":
            docs = [{"id": f"doc{i}", "original_name": n, "chunk_count": 1}
                    for i, n in enumerate(self.uploaded)]
            return _Resp(200, {"documents": docs})
        return _Resp(404)

    def post(self, path: str, files=None, json=None):
        if path == f"/v1/projects/{self.project_id}/documents":
            fname = files["file"][0]
            if fname.lower().endswith((".dxf", ".xer")):
                return _Resp(415, {"detail": f"File type is not ingestible: {fname}"})
            self.uploaded.append(fname)
            return _Resp(200, {"document": {"id": f"doc{len(self.uploaded)}"}})
        return _Resp(404)


def test_upload_file_skips_noningestible_type(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("text is ingestible")
    dxf = tmp_path / "b.dxf"
    dxf.write_text("0\nSECTION")

    client = _FakeClient("proj")

    assert seed_fixtures._upload_file(client, client.project_id, txt) is not None
    assert seed_fixtures._upload_file(client, client.project_id, dxf) is None


def test_seed_from_dir_uploads_text_and_skips_cad(tmp_path):
    (tmp_path / "a.txt").write_text("text is ingestible")
    (tmp_path / "b.dxf").write_text("0\nSECTION")

    spec = {
        "name": "FIXTURE — Test",
        "required_files": ["a.txt", "b.dxf"],
        "required_any_files": [],
        "optional_files": [],
    }
    client = _FakeClient(spec["name"])

    result = seed_fixtures._seed_from_dir(client, spec, tmp_path, synthetic=False)

    assert client.uploaded == ["a.txt"]
    assert result["docs"] == 1
    assert result["chunks"] == 1
    assert result["zero_chunk_docs"] == []
