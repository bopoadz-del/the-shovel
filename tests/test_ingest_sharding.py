"""Unit tests for stable multi-worker ingest sharding."""
from __future__ import annotations

import hashlib

import pytest

from app.core import ingest_sharding as sharding


def test_same_identity_always_same_shard():
    ident = "1AbC_drive_file_id_xyz"
    a = sharding.stable_shard(ident, 7)
    b = sharding.stable_shard(ident, 7)
    assert a == b
    # Independent of Python's randomized hash()
    assert a == int(hashlib.sha256(ident.encode()).hexdigest(), 16) % 7


def test_all_files_assigned_to_exactly_one_shard():
    ids = [f"file-{i}" for i in range(200)]
    total = 5
    buckets = {i: [] for i in range(total)}
    for ident in ids:
        buckets[sharding.stable_shard(ident, total)].append(ident)
    assigned = [x for xs in buckets.values() for x in xs]
    assert sorted(assigned) == sorted(ids)
    assert sum(len(v) for v in buckets.values()) == len(ids)


def test_no_duplicate_assignment_across_shards():
    ids = [f"path/doc_{i}.pdf" for i in range(100)]
    total = 4
    seen = {}
    for ident in ids:
        for shard in range(total):
            if sharding.belongs_to_shard(ident, shard, total):
                assert ident not in seen
                seen[ident] = shard
    assert len(seen) == len(ids)


def test_invalid_shard_index_errors():
    with pytest.raises(ValueError):
        sharding.validate_shard_params(-1, 3)
    with pytest.raises(ValueError):
        sharding.validate_shard_params(3, 3)
    with pytest.raises(ValueError):
        sharding.validate_shard_params(0, 0)


def test_file_identity_preference():
    assert sharding.file_identity(drive_file_id="abc", source_path="/x") == "abc"
    assert sharding.file_identity(source_path=r"Foo\Bar.PDF") == "foo/bar.pdf"
    assert sharding.file_identity(document_id="doc-1") == "doc-1"
    with pytest.raises(ValueError):
        sharding.file_identity()


def test_filter_and_offset_limit_order():
    items = [{"id": f"id-{i}"} for i in range(50)]
    total, idx = 5, 2
    shard_items = sharding.filter_by_shard(
        items,
        identity_fn=lambda x: x["id"],
        shard_index=idx,
        total_shards=total,
    )
    # Every retained item belongs to shard 2
    for item in shard_items:
        assert sharding.belongs_to_shard(item["id"], idx, total)
    limited = sharding.apply_offset_limit(shard_items, offset=1, limit=3)
    assert limited == shard_items[1:4]


def test_total_shards_one_keeps_all():
    items = [{"id": "a"}, {"id": "b"}]
    out = sharding.filter_by_shard(
        items, identity_fn=lambda x: x["id"], shard_index=0, total_shards=1,
    )
    assert out == items


def test_dry_run_report_has_partition_counts():
    items = [f"f{i}" for i in range(30)]
    report = sharding.dry_run_report(
        total_discovered=40,
        supported_items=items,
        unsupported_count=10,
        shard_index=0,
        total_shards=5,
        identity_fn=lambda x: x,
        path_fn=lambda x: f"{x}.pdf",
    )
    assert report["db_writes"] is False
    assert report["dry_run"] is True
    assert sum(report["per_shard_counts"]) == 30
    assert report["files_assigned_to_this_shard"] == report["per_shard_counts"][0]


def test_resolve_shard_env_legacy_and_new(monkeypatch):
    monkeypatch.delenv("INGEST_TOTAL_SHARDS", raising=False)
    monkeypatch.delenv("INGEST_SHARD_INDEX", raising=False)
    monkeypatch.setenv("P1B_SHARD_COUNT", "3")
    monkeypatch.setenv("P1B_SHARD_INDEX", "1")
    assert sharding.resolve_shard_env() == (3, 1)
    monkeypatch.setenv("INGEST_TOTAL_SHARDS", "8")
    monkeypatch.setenv("INGEST_SHARD_INDEX", "2")
    assert sharding.resolve_shard_env() == (8, 2)


def test_dry_run_env(monkeypatch):
    monkeypatch.setenv("INGEST_DRY_RUN", "true")
    assert sharding.resolve_dry_run(False) is True
    monkeypatch.delenv("INGEST_DRY_RUN", raising=False)
    assert sharding.resolve_dry_run(False) is False
    assert sharding.resolve_dry_run(True) is True
