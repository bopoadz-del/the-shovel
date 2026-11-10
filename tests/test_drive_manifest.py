"""P2 Drive manifest — folder-id resolution, gate numbers, stdout transport.

The manifest script is the general mechanism behind SPEC_CORPUS_FINAL_BUILD_10NOV
P2: a read-only Drive walk whose output carries exactly one class per row.
These tests cover the pure pieces (no Drive network, no corpus writes): the
shared folder-id parser, the gate-number computation, and the gzip+base64
payload that ships the manifest out of the ECS task. Every decision is
exercised with at least two inputs, per the mutation gate.
"""
from __future__ import annotations

import base64
import gzip
import json

import pytest

from app.core import gdrive_service
from scripts import p1a_drive_manifest as p1a


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([], []),
        (None, []),
        (["a"], ["a"]),
        (["a,b", "c"], ["a", "b", "c"]),
        ([" a , b ", "a"], ["a", "b"]),
        (["", " , ", "x"], ["x"]),
    ],
)
def test_parse_folder_ids_flattens_dedupes_and_trims(raw, expected):
    assert gdrive_service.parse_folder_ids(raw) == expected


@pytest.mark.parametrize(
    "cli,env,expected",
    [
        (None, None, []),
        (None, "a,b", ["a", "b"]),
        (["a,b", "c"], "d, a", ["a", "b", "c", "d"]),
        (["  x , y "], None, ["x", "y"]),
        ([], "   ", []),
    ],
)
def test_folder_ids_from_merges_cli_and_env(cli, env, expected):
    assert p1a._folder_ids_from(cli, env) == expected


def _manifest(folders):
    return {"generated_at": "t", "service_account": "sa", "folders": folders}


@pytest.mark.parametrize(
    "folders,expected",
    [
        (
            [
                {
                    "total_files": 2,
                    "files": [
                        {"class": "TEXT"},
                        {"class": "SPREADSHEET"},
                    ],
                }
            ],
            {
                "rows": 2,
                "drive_file_count": 2,
                "classes": {
                    "NON_TEXT": 0,
                    "SCANNED_IMAGE": 0,
                    "SPREADSHEET": 1,
                    "TEXT": 1,
                },
                "not_exactly_one_class": 0,
            },
        ),
        (
            [
                {
                    "total_files": 2,
                    "files": [
                        {"class": "TEXT"},
                        {"class": "BOGUS"},
                        {"class": "SCANNED_IMAGE"},
                    ],
                }
            ],
            {
                "rows": 3,
                "drive_file_count": 2,
                "classes": {
                    "NON_TEXT": 0,
                    "SCANNED_IMAGE": 1,
                    "SPREADSHEET": 0,
                    "TEXT": 1,
                },
                "not_exactly_one_class": 1,
            },
        ),
        ([], {"rows": 0, "drive_file_count": 0, "classes": {"NON_TEXT": 0, "SCANNED_IMAGE": 0, "SPREADSHEET": 0, "TEXT": 0}, "not_exactly_one_class": 0}),
    ],
)
def test_manifest_gate_counts_rows_classes_and_mismatch(folders, expected):
    assert p1a._manifest_gate(_manifest(folders)) == expected


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest([]),
        _manifest(
            [
                {
                    "total_files": 1,
                    "files": [{"id": "i1", "name": "Hạ Long Xanh.pdf", "class": "TEXT"}],
                }
            ]
        ),
    ],
)
def test_manifest_payload_round_trips_utf8_through_gzip_base64(manifest):
    payload = p1a._manifest_payload_b64(manifest)
    decoded = json.loads(gzip.decompress(base64.b64decode(payload)).decode("utf-8"))
    assert decoded == manifest


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest([]),
        _manifest(
            [
                {
                    "total_files": 2,
                    "files": [
                        {"id": "i1", "name": "n.pdf", "class": "TEXT"},
                        {"id": "i2", "name": "s.xlsx", "class": "SPREADSHEET"},
                    ],
                }
            ]
        ),
    ],
)
def test_emit_stdout_carries_gate_and_reconstructable_payload(capsys, manifest):
    p1a._emit_stdout(manifest)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("MANIFEST_GATE: ")
    assert lines[1] == "MANIFEST_PAYLOAD_BEGIN"
    assert lines[-1] == "MANIFEST_PAYLOAD_END"
    gate = json.loads(lines[0].split(": ", 1)[1])
    assert gate["rows"] == sum(len(f["files"]) for f in manifest["folders"])
    assert gate["not_exactly_one_class"] == 0
    payload = "".join(lines[2:-1])
    decoded = json.loads(gzip.decompress(base64.b64decode(payload)).decode("utf-8"))
    assert decoded == manifest
