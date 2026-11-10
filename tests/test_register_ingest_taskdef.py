"""Pure-logic tests for the ingest task-definition repoint script.

The script moves WORKER_COMMAND from the ``secrets`` block to ``environment``
with the same value. The only runtime-critical invariant is the integrity
check: a SHA-256 of the container definition with the WORKER_COMMAND entry
removed must be byte-identical before and after the move -- proving ONLY that
one entry moved and nothing else changed. These tests exercise that invariant
and the entry-removal helper without any AWS call.
"""
from __future__ import annotations

import pytest

from scripts.register_ingest_taskdef import (
    WORKER_COMMAND,
    _container_without_worker,
    _sha,
    _without_entry,
)


def _container(other_env_name: str = "A", other_secret_name: str = "OTHER") -> dict:
    return {
        "name": "worker",
        "image": "the-fork:latest",
        "environment": [{"name": other_env_name, "value": "1"}],
        "secrets": [
            {"name": WORKER_COMMAND, "valueFrom": "arn:aws:secretsmanager:us-west-2:1:secret:worker"},
            {"name": other_secret_name, "valueFrom": "arn:aws:secretsmanager:us-west-2:1:secret:other"},
        ],
    }


@pytest.mark.parametrize("name", ["WORKER_COMMAND", "OTHER"])
def test_without_entry_removes_only_the_named_entry(name):
    items = [{"name": "A", "value": "1"}, {"name": name, "value": "2"}]
    assert [i["name"] for i in _without_entry(items, name)] == ["A"]


def test_container_without_worker_drops_it_from_both_blocks():
    stripped = _container_without_worker(_container())
    assert [e["name"] for e in stripped["environment"]] == ["A"]
    assert [s["name"] for s in stripped["secrets"]] == ["OTHER"]


@pytest.mark.parametrize("other_env_name", ["A", "B"])
def test_digest_unchanged_when_only_the_worker_entry_moves(other_env_name):
    before = _container(other_env_name=other_env_name)
    moved = dict(before)
    moved["secrets"] = _without_entry(before["secrets"], WORKER_COMMAND)
    moved["environment"] = before["environment"] + [{"name": WORKER_COMMAND, "value": "cmd"}]
    assert _sha(_container_without_worker(before)) == _sha(_container_without_worker(moved))


@pytest.mark.parametrize("value", ["python scripts/p1b_ingest_drive_server.py", "python -m p1b --tier 1"])
def test_moved_value_never_affects_the_integrity_digest(value):
    before = _container()
    moved = dict(before)
    moved["secrets"] = _without_entry(before["secrets"], WORKER_COMMAND)
    moved["environment"] = before["environment"] + [{"name": WORKER_COMMAND, "value": value}]
    # The digest intentionally excludes the WORKER_COMMAND value, so the exact
    # command string must not change the integrity comparison.
    assert _sha(_container_without_worker(before)) == _sha(_container_without_worker(moved))
