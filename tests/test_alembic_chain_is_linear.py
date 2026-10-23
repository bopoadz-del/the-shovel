"""The migration chain must have exactly one head.

Migrations run with `alembic upgrade head` on boot (see deploy docs). Two
heads, a duplicate revision id, or a down_revision pointing at nothing all
make that command fail -- which means the container starts, fails to migrate,
and the deploy is broken at the worst possible moment.

Written after nearly shipping exactly that: a new migration was numbered 0014
while `20260726_add_ingestion_jobs.py` already claimed 0014 off the same
parent. Nothing in the suite noticed, because no test looked at the chain at
all. CI would have been fully green and the deploy would have died on boot.

This asserts the SHAPE of the chain rather than any particular revision, so it
keeps working as migrations are added and never needs updating for a normal
one.
"""
from __future__ import annotations

import pathlib
import re

VERSIONS = pathlib.Path(__file__).resolve().parents[1] / "alembic" / "versions"

_REV = re.compile(r'^revision\s*=\s*["\']([^"\']+)["\']', re.M)
_DOWN = re.compile(r"^down_revision\s*=\s*(.+)$", re.M)


def _chain() -> list[tuple[str, str | None, str]]:
    """(revision, down_revision, filename) for every migration on disk."""
    out: list[tuple[str, str | None, str]] = []
    for path in sorted(VERSIONS.glob("*.py")):
        if path.name.startswith("__"):
            continue
        text = path.read_text(encoding="utf-8")
        rev = _REV.search(text)
        if not rev:
            continue
        down_raw = _DOWN.search(text)
        down = None
        if down_raw:
            value = down_raw.group(1).strip()
            down = None if value == "None" else value.strip("\"'")
        out.append((rev.group(1), down, path.name))
    return out


def test_revision_ids_are_unique():
    """Two files claiming the same id: alembic resolves one and drops the other."""
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for rev, _down, name in _chain():
        if rev in seen:
            dupes.append(f"{rev!r} in both {seen[rev]} and {name}")
        seen[rev] = name
    assert not dupes, "duplicate alembic revision ids: " + "; ".join(dupes)


def test_there_is_exactly_one_head():
    """More than one head makes `alembic upgrade head` fail outright."""
    chain = _chain()
    downs = {down for _rev, down, _name in chain if down}
    heads = [(rev, name) for rev, _down, name in chain if rev not in downs]
    assert len(heads) == 1, (
        "alembic must have exactly one head or `upgrade head` fails on boot; "
        f"found {len(heads)}: {heads}"
    )


def test_every_parent_exists():
    """A down_revision pointing at nothing breaks the walk from base."""
    chain = _chain()
    known = {rev for rev, _down, _name in chain}
    orphans = [
        f"{name} points at missing revision {down!r}"
        for _rev, down, name in chain
        if down and down not in known
    ]
    assert not orphans, "; ".join(orphans)


def test_chain_reaches_every_revision_from_base():
    """No detached islands: walking children from base must cover everything."""
    chain = _chain()
    assert chain, "no migrations found"
    children: dict[str | None, list[str]] = {}
    for rev, down, _name in chain:
        children.setdefault(down, []).append(rev)

    reachable: set[str] = set()
    stack = list(children.get(None, []))
    assert stack, "no base revision (down_revision = None)"
    while stack:
        rev = stack.pop()
        if rev in reachable:
            continue
        reachable.add(rev)
        stack.extend(children.get(rev, []))

    missing = {rev for rev, _d, _n in chain} - reachable
    assert not missing, f"revisions unreachable from base: {sorted(missing)}"
