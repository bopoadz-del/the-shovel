"""The sharded local-folder ingest dry run, end to end on a synthetic folder.

Replaces the committed ``tests/fixtures/ingest_shard_sample`` tree and the
``manifests/ingest_shard_*_of_5.json`` dry-run outputs that were generated
from it: the folder is now built in ``tmp_path`` and the five shard manifests
are produced by running the real script, then checked against each other.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._synthetic_fixtures import build_ingest_shard_folder

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "p1b_ingest_local_folder.py"
TOTAL = 5


def _dry_run(tmp_path: Path, folder: str, shard: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("INGEST_")}
    env.update({"RAG_EMBEDDING_MODEL": "fake", "RAG_VECTOR_NAMESPACE": "v2"})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--folder", folder,
         "--drive-root", str(tmp_path / "drive"), "--dry-run",
         "--total-shards", str(TOTAL), "--shard-index", str(shard)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    manifest = tmp_path / "manifests" / f"ingest_shard_{shard}_of_{TOTAL}.json"
    assert manifest.is_file(), "the dry run did not write its shard manifest"
    return json.loads(manifest.read_text(encoding="utf-8"))


def test_five_shard_dry_run_partitions_the_folder(tmp_path):
    spec = build_ingest_shard_folder(tmp_path / "drive")
    reports = [_dry_run(tmp_path, spec["folder"], i) for i in range(TOTAL)]

    for i, r in enumerate(reports):
        assert r["dry_run"] is True and r["db_writes"] is False, r
        assert r["folder"] == spec["folder"]
        assert r["shard_index"] == i and r["total_shards"] == TOTAL
        assert r["total_files_discovered"] == spec["total"] == 60
        assert r["total_supported_files"] == spec["supported"] == 55
        assert r["total_skipped_unsupported"] == spec["unsupported"] == 5
        # Every worker sees the same global distribution ...
        assert r["per_shard_counts"] == reports[0]["per_shard_counts"]
        # ... and takes exactly its own slot of it.
        assert r["files_assigned_to_this_shard"] == r["per_shard_counts"][i]
        assert sum(r["extension_breakdown"].values()) == r["files_assigned_to_this_shard"]
        assert ".gdoc" not in r["extension_breakdown"], "unsupported files were sharded"
        assert r["batch_after_offset_limit"] == r["files_assigned_to_this_shard"]

    assert sum(reports[0]["per_shard_counts"]) == spec["supported"]
    # No file is assigned to two shards.
    seen: dict = {}
    for i, r in enumerate(reports):
        for ident in r["first_20_identities"]:
            assert ident.startswith(f"{spec['folder']}/"), ident
            assert ident not in seen, f"{ident} in shard {seen.get(ident)} and {i}"
            seen[ident] = i


def test_dry_run_is_deterministic(tmp_path):
    spec = build_ingest_shard_folder(tmp_path / "drive")
    first = _dry_run(tmp_path, spec["folder"], 3)
    second = _dry_run(tmp_path, spec["folder"], 3)
    for key in ("per_shard_counts", "first_20_identities", "extension_breakdown"):
        assert first[key] == second[key], key
