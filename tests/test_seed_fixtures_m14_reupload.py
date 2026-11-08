"""M14: the fixture seeder re-persists a fixture whose rows survived but whose
bytes are gone (zero-chunk / EXTRACT_FAILED), instead of skip-uploading it
forever. Re-upload is paired with the admin supersede endpoint, and the
zero-chunk tripwire ignores superseded rows.
"""
from __future__ import annotations

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
    """Stand-in live API with one pre-existing ZERO-chunk doc whose bytes are
    gone. Uploads mint a fresh indexed doc; supersede retires the old row."""

    def __init__(self, project_name: str, stale_name: str) -> None:
        self.project_name = project_name
        self.project_id = "p1"
        self.old_id = "old-1"
        self.docs = [{
            "id": self.old_id,
            "original_name": stale_name,
            "chunk_count": 0,
            "retrieval_visible": True,
        }]
        self.uploads: list[str] = []
        self.superseded: list[tuple[str, str]] = []

    def get(self, path: str, params=None):
        if path == "/v1/projects":
            return _Resp(200, {"projects": [
                {"name": self.project_name, "id": self.project_id},
            ]})
        if path == f"/v1/projects/{self.project_id}/documents":
            return _Resp(200, {"documents": list(self.docs)})
        return _Resp(404)

    def post(self, path: str, files=None, params=None, json=None):
        if path == f"/v1/projects/{self.project_id}/documents":
            fname = files["file"][0]
            did = f"doc-{len(self.uploads) + 1}"
            self.uploads.append(fname)
            self.docs.append({
                "id": did, "original_name": fname,
                "chunk_count": 1, "retrieval_visible": True,
            })
            return _Resp(200, {"document": {"id": did}})
        if path == f"/v1/admin/documents/{self.old_id}/supersede":
            new_id = (params or {}).get("new_id")
            self.superseded.append((self.old_id, new_id))
            for d in self.docs:
                if d["id"] == self.old_id:
                    d["retrieval_visible"] = False
                    d["superseded_by"] = new_id
            return _Resp(200, {"status": "superseded"})
        return _Resp(404)


def test_zero_chunk_fixture_is_reuploaded_and_superseded():
    client = _FakeClient("FIXTURE — Fresh Upload Eval", "case_a.txt")
    spec = {
        "name": "FIXTURE — Fresh Upload Eval",
        "cases": [
            ("case_a.txt", "the engineer is named Sandra Okafor", "engineer"),
            ("case_b.txt", "the retention is five percent", "retention"),
        ],
    }

    result = seed_fixtures._seed_fresh_upload_eval(client, spec)

    assert client.uploads == ["case_a.txt", "case_b.txt"]
    assert client.superseded == [("old-1", "doc-1")]
    assert result["docs"] == 2          # only the two visible, indexed docs
    assert result["chunks"] == 2
    assert result["zero_chunk_docs"] == []


def test_usable_doc_is_still_skipped():
    client = _FakeClient("FIXTURE — Fresh Upload Eval", "case_a.txt")
    # mark the stale doc indexed so it is NOT re-uploaded
    for d in client.docs:
        d["chunk_count"] = 3
    spec = {
        "name": "FIXTURE — Fresh Upload Eval",
        "cases": [("case_a.txt", "content", "q")],
    }

    result = seed_fixtures._seed_fresh_upload_eval(client, spec)

    assert client.uploads == []
    assert client.superseded == []
    assert result["zero_chunk_docs"] == []
