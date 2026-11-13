"""Restore drill: compare a restored database with live, counts only.

The owner (or the advisor) makes a Neon branch from a restore point (the
backup); its endpoint id (``ep-...``) is the one input. This script opens the
restore branch and the live database, each in a READ ONLY transaction, and
prints row counts per table plus documents per ingest status and chunks per
table, side by side. It never writes to either database and never prints a
connection string, a row or any row text.

Verdict:
  * ``unreadable`` -- a table could not be counted on the restore branch.
  * ``empty_on_restore`` -- a table holds rows on live but none on the
    restore branch (the backup lost it).
  * ``RESTORE_DRILL ok`` when neither list is non-empty.
A restore point is older than live, so live may hold more rows; the delta is
reported, not judged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PREFIX = "RESTORE_DRILL"

#: The tables whose loss would lose a customer's work. Counted on both sides.
TABLES = ("users", "projects", "documents", "conversations", "messages",
          "project_facts", "hydration_runs", "ingestion_jobs", "audit_events")


def _emit(**fields: object) -> None:
    print(f"{_PREFIX} " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


class Refused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def counts(session_factory: Callable, tables: Iterable[str], chunk_table: str) -> Dict[str, object]:
    """Row counts per table, documents per ingest status, chunk rows. A table
    that cannot be counted is recorded as ``None`` (unreadable), not skipped."""
    from sqlalchemy import text as sql

    out: Dict[str, object] = {}
    with session_factory() as s:
        if s.get_bind().dialect.name == "postgresql":
            s.execute(sql("SET TRANSACTION READ ONLY"))
        for name in list(tables) + [chunk_table]:
            try:
                with s.begin_nested():
                    out[name] = int(s.execute(sql(f'SELECT count(*) FROM "{name}"')).scalar_one())
            except Exception:  # noqa: BLE001 -- a missing table is the finding
                out[name] = None
        try:
            with s.begin_nested():
                rows = s.execute(sql(
                    'SELECT coalesce(ingest_status, \'\'), count(*) FROM "documents" GROUP BY 1')).all()
            out["documents_by_status"] = {str(k) or "unset": int(v) for k, v in rows}
        except Exception:  # noqa: BLE001
            out["documents_by_status"] = None
        s.rollback()
    return out


def compare(restore: Mapping[str, object], live: Mapping[str, object]) -> Dict[str, object]:
    """Side-by-side counts and the verdict lists."""
    unreadable: List[str] = []
    empty: List[str] = []
    rows: Dict[str, Dict[str, object]] = {}
    for name, live_n in live.items():
        if name == "documents_by_status":
            continue
        got = restore.get(name)
        rows[name] = {"restore": got, "live": live_n,
                      "delta": (live_n - got) if isinstance(got, int) and isinstance(live_n, int) else None}
        if got is None:
            unreadable.append(name)
        elif got == 0 and isinstance(live_n, int) and live_n > 0:
            empty.append(name)
    return {"tables": rows, "unreadable": unreadable, "empty_on_restore": empty,
            "ok": not unreadable and not empty}


def run(endpoint: str, *, restore_factory: Optional[Callable] = None,
        live_factory: Optional[Callable] = None, chunk_table: Optional[str] = None) -> Dict[str, object]:
    from scripts import scrub_verify as sv

    if restore_factory is None or live_factory is None:
        live_url = sv._live_url()
        restore_url = sv.restore_url(live_url, endpoint)  # refuses a malformed id
        if restore_url == live_url:
            raise Refused("restore_endpoint_is_live")
        restore_factory = sv._session_factory(restore_url)
        live_factory = sv._session_factory(live_url)
    table = chunk_table or sv.chunk_table()
    restore = counts(restore_factory, TABLES, table)
    live = counts(live_factory, TABLES, table)
    result = compare(restore, live)
    for name, r in result["tables"].items():
        _emit(table=name, restore=r["restore"], live=r["live"], live_minus_restore=r["delta"])
    for side, c in (("restore", restore), ("live", live)):
        for status, n in sorted((c.get("documents_by_status") or {}).items()):
            _emit(side=side, documents_status=status, n=n)
    _emit(verdict="ok" if result["ok"] else "FAIL",
          unreadable=",".join(result["unreadable"]) or "-",
          empty_on_restore=",".join(result["empty_on_restore"]) or "-")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Restore drill: restore branch vs live, counts only.")
    ap.add_argument("--endpoint", required=True, help="Neon endpoint id of the restore branch (ep-...)")
    a = ap.parse_args(argv)
    try:
        result = run(a.endpoint)
    except Exception as exc:  # noqa: BLE001 -- refusal codes only, never a URL
        code = getattr(exc, "code", type(exc).__name__)
        _emit(refused=code)
        return 2
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
