"""Drive v3 files.list rejects the v2-only field ``etag`` before it looks
at the folder. A synthetic transport returns HTTP 400
``Invalid field selection etag`` whenever ``etag`` appears in ``fields``.
"""
from __future__ import annotations

import json

import pytest

from app.core.gdrive_service import (
    find_file_id_by_exact_name,
    get_file_metadata,
    list_folder_files,
)
from app.core.ingest_reconcile import resume_source_changed, source_content_token
from tests._fixture_values import STUB_TOKEN_15

_FOLDER = "synthFolder0001"
_FILE_ID = "synthFileId01"
_NAME = "synthetic-note.txt"
_REJECT = {"error": {"message": "Invalid field selection etag"}}


class _Resp:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)
        self.content = self.text.encode()

    def json(self):
        return self._body


class _Transport:
    """Fake Drive v3. ``etag`` in any fields selection is a 400."""

    def __init__(self, calls: list):
        self.calls = calls
        self._pages_left = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None, params=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params})
        fields = str(params.get("fields") or "")
        if "etag" in fields.lower():
            return _Resp(400, _REJECT)
        if url.rstrip("/").endswith("/files"):
            if params.get("pageToken"):
                return _Resp(200, {
                    "files": [{
                        "id": "synthPageTwo01",
                        "name": "synthetic-page-two.txt",
                        "mimeType": "text/plain",
                        "md5Checksum": "md5-page-two",
                        "modifiedTime": "2024-02-02T00:00:00.000Z",
                    }],
                })
            body = {
                "files": [{
                    "id": _FILE_ID,
                    "name": _NAME,
                    "mimeType": "text/plain",
                    "size": "12",
                    "md5Checksum": "md5-page-one",
                    "modifiedTime": "2024-01-01T00:00:00.000Z",
                }],
            }
            if self._pages_left:
                body["nextPageToken"] = "page-2"
                self._pages_left -= 1
            return _Resp(200, body)
        return _Resp(200, {
            "id": _FILE_ID,
            "name": _NAME,
            "mimeType": "text/plain",
            "size": "12",
            "md5Checksum": "md5-get",
            "modifiedTime": "2024-01-01T00:00:00.000Z",
        })


@pytest.fixture
def drive(monkeypatch):
    calls: list = []

    def _client(*args, **kwargs):
        return _Transport(calls)

    monkeypatch.setattr(
        "app.core.gdrive_service._mint_access_token",
        lambda: STUB_TOKEN_15,
    )
    monkeypatch.setattr("httpx.Client", _client)
    return calls


def _list_calls(calls):
    return [c for c in calls if str(c["url"]).rstrip("/").endswith("/files")]


def test_drive_v3_rejects_etag_field_selection(drive):
    """Walk list, name lookup, and files.get must not ask for etag.

    The walk's files.list also sends both shared-drive flags, clamps
    pageSize, and sends pageToken only on a later page.
    """
    files, err = list_folder_files(_FOLDER, page_size=5000)
    assert err is None, err
    assert [f["id"] for f in files] == [_FILE_ID, "synthPageTwo01"]

    listed = _list_calls(drive)
    assert len(listed) == 2
    first, second = listed
    assert "etag" not in str(first["params"]["fields"]).lower()
    assert "etag" not in str(second["params"]["fields"]).lower()
    assert "md5Checksum" in first["params"]["fields"]
    assert first["params"]["supportsAllDrives"] == "true"
    assert first["params"]["includeItemsFromAllDrives"] == "true"
    assert first["params"]["pageSize"] == 1000
    assert "pageToken" not in first["params"]
    assert second["params"]["pageToken"] == "page-2"
    assert second["params"]["pageSize"] == 1000
    assert "corpora" not in first["params"]
    assert "driveId" not in first["params"]
    assert "corpora" not in second["params"]
    assert "driveId" not in second["params"]

    drive.clear()
    tiny, tiny_err = list_folder_files(_FOLDER, page_size=0)
    assert tiny_err is None, tiny_err
    assert tiny
    assert _list_calls(drive)[-1]["params"]["pageSize"] == 1

    drive.clear()
    negative, neg_err = list_folder_files(_FOLDER, page_size=-4)
    assert neg_err is None, neg_err
    assert negative
    assert _list_calls(drive)[-1]["params"]["pageSize"] == 1

    drive.clear()
    found, name_err = find_file_id_by_exact_name(_NAME)
    assert name_err is None, name_err
    assert found == _FILE_ID
    name_params = drive[-1]["params"]
    assert "etag" not in str(name_params["fields"]).lower()
    assert name_params["supportsAllDrives"] == "true"
    assert name_params["includeItemsFromAllDrives"] == "true"

    drive.clear()
    meta, meta_err = get_file_metadata(_FILE_ID)
    assert meta_err is None, meta_err
    assert meta["md5Checksum"] == "md5-get"
    assert "etag" not in meta
    get_params = drive[-1]["params"]
    assert "etag" not in str(get_params["fields"]).lower()
    assert get_params["supportsAllDrives"] == "true"

    for call in drive:
        assert "etag" not in str(call["params"].get("fields") or "").lower()


def test_source_token_falls_back_when_etag_is_absent():
    """A v3 file with no etag still has an identity, and never raises."""
    assert source_content_token({
        "md5Checksum": "abc",
        "version": "9",
        "modifiedTime": "2024-01-01T00:00:00.000Z",
    }) == "abc"
    assert source_content_token({"version": "9"}) == "9"
    assert source_content_token({
        "modifiedTime": "2024-03-03T00:00:00.000Z",
    }) == "2024-03-03T00:00:00.000Z"
    assert source_content_token({"etag": "legacy-etag"}) == "legacy-etag"
    assert source_content_token({"id": "only"}) is None
    assert source_content_token({}) is None
    assert source_content_token(None) is None
    assert resume_source_changed(
        {"drive_md5": "stored"}, {"version": "9"},
    ) is True
    assert resume_source_changed(
        {"drive_md5": "9"}, {"version": "9"},
    ) is False
    assert resume_source_changed(
        {"drive_md5": "stored"}, {"id": "only"},
    ) is False
