#!/usr/bin/env python3
r"""Named mutation probes over decision code. Reject, never repair.

Every PR in this repo carries a RECEIPT with a "mutants run/survivors"
line, and every one of them has said UNPRODUCED, because nothing in the
tree could produce the number. A coverage percentage says a line ran; it
does not say a test would have noticed if the line were wrong. These
probes say that, for the handful of lines where being wrong is expensive.

Each probe replaces ONE exact substring in ONE file with a plausible wrong
version -- a guard turned off, a fence widened -- and runs the ONE test
that must catch it. A probe whose test stays green is a SURVIVOR and fails
this script: the guard is not actually guarded.

The targets are deliberately few and deliberately load-bearing. This is
not a coverage tool.

A killed run leaves ``.mutation_in_flight`` beside the mutated file's
original bytes, and the next start restores it before doing anything else.
Without that, a Ctrl-C in the middle leaves a sabotaged guard on disk that
looks like a clean working tree -- which is how a disabled monitor once
shipped in a sibling repo, twice, before anyone noticed.

    python scripts/mutation_probes.py                  # all
    python scripts/mutation_probes.py --only rag_search_gate
    python scripts/mutation_probes.py --list
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = ".mutation_in_flight"


@dataclass(frozen=True)
class Probe:
    name: str
    path: str
    old: str
    new: str
    test: str
    why: str


PROBES: tuple[Probe, ...] = (
    Probe(
        name="rag_search_gate",
        path="app/routers/rag.py",
        old=(
            "    if projects_store.get_project_accessible(project_id, "
            'auth.get("user_id")) is None:'
        ),
        new="    if False:",
        test=(
            "tests/test_rag_search_tenancy.py::"
            "test_non_member_gets_404_and_no_chunk_text"
        ),
        why="turn the tenancy gate off; the cross-tenant test must go red",
    ),
    Probe(
        name="master_corpus_source_acl",
        path="app/core/projects.py",
        old=(
            "    alias_id = ui_project_id(project_id)\n"
            "    if alias_id and alias_id != project_id:"
        ),
        new="    alias_id = None\n    if False:",
        test=(
            "tests/test_rag_search_tenancy.py::"
            "test_master_corpus_jwt_can_search_backing_source_id"
        ),
        why="drop the source→alias membership remap; WATCH-1 404 must come back",
    ),
    Probe(
        name="admin_fallthrough_platform_only",
        path="app/core/projects.py",
        old="            if candidate is not None and _is_platform_project(candidate):",
        new="            if candidate is not None:",
        test=(
            "tests/test_chat_open_access_gate.py::"
            "test_admin_does_not_resolve_other_users_private_project"
        ),
        why="let an admin read a private user project, not only platform ones",
    ),
    Probe(
        name="admin_fallthrough_needs_admin_role",
        path="app/core/projects.py",
        old='        if u and (u.get("role") or "").lower() == "admin":',
        new="        if u:",
        test=(
            "tests/test_chat_open_access_gate.py::"
            "test_regular_user_cannot_read_a_system_owned_platform_project"
        ),
        why="drop the role check so any resolvable user takes the admin path",
    ),
    Probe(
        name="gk_system_seed_acl",
        path="app/core/projects.py",
        old="            allowed = include_admin_approved and _is_shared_platform_grant(project)",
        new=(
            "            allowed = (\n"
            "                include_admin_approved\n"
            "                and getattr(project, \"origin\", \"user_create\") "
            "== \"admin_drive_approved\"\n"
            "                and bool(getattr(project, \"is_approved\", True))\n"
            "            )"
        ),
        test=(
            "tests/test_chat_open_access_gate.py::"
            "test_regular_user_resolves_system_seed_gk_project"
        ),
        why="revert GK/system_seed share to Drive-approved-only; QA 404 must return",
    ),
    Probe(
        name="gk_conversation_acl",
        path="app/routers/agents.py",
        old=(
            "            if store.can_access_project(\n"
            "                project_id, user_id=auth[\"user_id\"], "
            "include_admin_approved=True\n"
            "            ):"
        ),
        new=(
            "            include_admin = project_id == store.MASTER_CORPUS_PROJECT_ID\n"
            "            if store.can_access_project(\n"
            "                project_id, user_id=auth[\"user_id\"], "
            "include_admin_approved=include_admin\n"
            "            ):"
        ),
        test=(
            "tests/test_chat_open_access_gate.py::"
            "test_regular_user_can_open_system_seed_gk_conversation"
        ),
        why="revert conversation ACL to master-corpus-only; GK New chat 404s again",
    ),
    Probe(
        name="search_preamble_detector",
        path="app/agents/runtime.py",
        old=(
            "    if _SEARCH_PREAMBLE_RE.search(t) or "
            "_SEARCH_PROMISE_TAIL_RE.search(t):\n        return True"
        ),
        new=(
            "    if _SEARCH_PREAMBLE_RE.search(t) or "
            "_SEARCH_PROMISE_TAIL_RE.search(t):\n        return False"
        ),
        test=(
            "tests/test_dangling_search_preamble.py::"
            "test_live_dangling_preambles_are_detected"
        ),
        why="stop recognising a search promise, so a promise ships as an answer",
    ),
)


def _marker(marker_dir: Path) -> Path:
    return marker_dir / MARKER


def restore_probe(marker_dir: Path = ROOT) -> bool:
    """Undo an in-flight mutation if one was left behind. True if it did."""
    m = _marker(marker_dir)
    if not m.exists():
        return False
    rec = json.loads(m.read_text(encoding="utf-8"))
    Path(rec["path"]).write_bytes(bytes.fromhex(rec["original_hex"]))
    m.unlink()
    return True


def apply_probe(path: str, old: str, new: str, marker_dir: Path = ROOT) -> None:
    r"""Write the mutation, recording the original bytes first.

    Probe strings are written with LF. A checkout on Windows has CRLF on
    disk, so any multi-line target would never match there and every probe
    would report "target is not unique: 0 match(es)". A runner that only
    works on the CI platform is a runner nobody runs before pushing.

    So the probe is translated to the file's own line ending rather than
    the file being rewritten to the probe's: the only bytes that change
    are the mutation itself, and the restore is byte-exact either way.
    """
    p = Path(path)
    original = p.read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    old = old.replace("\n", newline)
    new = new.replace("\n", newline)
    if text.count(old) != 1:
        raise SystemExit(
            f"probe target is not unique in {path}: {text.count(old)} match(es). "
            "The code moved -- fix the probe, do not delete it."
        )
    # Marker BEFORE the write: a crash between the two must leave a
    # recoverable tree. An orphan marker is harmless; an orphan mutant is not.
    _marker(marker_dir).write_text(
        json.dumps({"path": str(p), "original_hex": original.hex()}),
        encoding="utf-8",
    )
    p.write_bytes(text.replace(old, new, 1).encode("utf-8"))


def _run_test(test: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", test, "-q", "-x", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="run one probe by name")
    ap.add_argument("--list", action="store_true", help="name every probe and exit")
    args = ap.parse_args()

    if args.list:
        for p in PROBES:
            print(f"{p.name:36} {p.path:28} {p.why}")
        return 0

    if restore_probe():
        print("recovered a mutant left on disk by an interrupted run")

    probes = [p for p in PROBES if not args.only or p.name == args.only]
    if not probes:
        print(f"no probe named {args.only!r}", file=sys.stderr)
        return 2

    survivors: list[str] = []
    for p in probes:
        apply_probe(p.path, p.old, p.new)
        try:
            rc = _run_test(p.test)
        finally:
            restore_probe()
        killed = rc != 0
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {p.name:36} {p.test}")
        if not killed:
            survivors.append(p.name)

    print(f"MUTANTS run={len(probes)} survivors={len(survivors)}")
    if survivors:
        print(
            "A survivor means the guard is not guarded: the code was made "
            "wrong and every test still passed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
