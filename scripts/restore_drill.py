"""Restore drill: compare a restored database with live, counts only.

Either ``--create`` makes a scratch Neon branch of live at a restore point
(read-only endpoint, ``NEON_API_KEY``), times it and deletes it after the
comparison, or ``--endpoint ep-...`` names an existing restore branch. The
comparison opens the restore branch and the live database, each in a READ ONLY transaction, and
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


# ── scratch restore branch (Neon) ────────────────────────────────────────────
NEON_API = "https://console.neon.tech/api/v2"


def neon_http(api_key: str) -> Callable:
    """``http(method, path, body) -> dict`` against the Neon API. The key is
    sent as a bearer header and never printed."""
    import json
    import urllib.request

    def call(method: str, path: str, body: Optional[dict] = None) -> dict:
        req = urllib.request.Request(
            NEON_API + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 -- fixed https host
                raw = resp.read()
        except Exception as exc:  # noqa: BLE001 -- status code and route only, never the body
            raise Refused(f"neon_http_{getattr(exc, 'code', 'error')}:"
                          f"{method}:{neon_route(path)}") from None
        return json.loads(raw or b"{}")
    return call


_ROUTE_COLLECTIONS = ("projects", "branches", "endpoints", "operations", "databases", "roles")


def neon_route(path: str) -> str:
    """The route of a Neon API path with its ids blanked and its query dropped
    (``/projects/p-1/branches?x=1`` -> ``/projects/*/branches``), so a refusal
    names the call that failed without printing an id."""
    parts = path.split("?", 1)[0].strip("/").split("/")
    out = []
    for i, seg in enumerate(parts):
        out.append("*" if i and parts[i - 1] in _ROUTE_COLLECTIONS else seg)
    return "/" + "/".join(out)


def live_endpoint_id(live_url: str) -> str:
    """The Neon endpoint id in the live host (``ep-...``; a ``-pooler``
    suffix is the same endpoint)."""
    from urllib.parse import urlsplit

    label = (urlsplit(live_url).hostname or "").split(".", 1)[0]
    if label.endswith("-pooler"):
        label = label[: -len("-pooler")]
    if not label.startswith("ep-"):
        raise Refused("live_host_not_neon")
    return label


def _locate(http: Callable, endpoint_id: str) -> tuple:
    """(project id, branch id) of the live endpoint."""
    for proj in http("GET", "/projects?limit=400").get("projects", []):
        for ep in http("GET", f"/projects/{proj['id']}/endpoints").get("endpoints", []):
            if ep.get("id") == endpoint_id:
                return proj["id"], ep["branch_id"]
    raise Refused("live_endpoint_not_found")


def scratch_drill(http: Callable, live_url: str, minutes_back: int,
                  compare_fn: Callable[[str], Dict[str, object]], *,
                  clock: Callable[[], float], sleep: Callable[[float], None],
                  now_utc: Callable, ready_timeout_s: float = 900.0,
                  poll_s: float = 5.0) -> Dict[str, object]:
    """Branch the live database at ``minutes_back`` minutes ago with a
    read-only endpoint, wait until it serves, compare counts, then delete
    the branch -- always, and only the branch made here. Times are seconds."""
    import datetime as dt

    if minutes_back < 1:
        raise Refused("minutes_back_below_1")
    live_ep = live_endpoint_id(live_url)
    project, parent = _locate(http, live_ep)
    point = (now_utc() - dt.timedelta(minutes=minutes_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = clock()
    made = http("POST", f"/projects/{project}/branches", {
        "branch": {"parent_id": parent, "parent_timestamp": point,
                   "name": f"restore-drill-{int(t0)}"},
        "endpoints": [{"type": "read_only"}]})
    branch = (made.get("branch") or {}).get("id")
    eps = [e.get("id") for e in made.get("endpoints") or []]
    if not branch or branch == parent:
        raise Refused("branch_not_created")
    times: Dict[str, object] = {"restore_point": point}
    try:
        if len(eps) != 1 or not eps[0] or eps[0] == live_ep:
            raise Refused("restore_endpoint_unusable")
        pending = [o["id"] for o in made.get("operations") or []]
        while pending:
            if clock() - t0 > ready_timeout_s:
                raise Refused("branch_not_ready")
            still = []
            for op in pending:
                status = http("GET", f"/projects/{project}/operations/{op}")["operation"]["status"]
                if status in ("failed", "error", "cancelled"):
                    raise Refused("branch_operation_failed")
                if status != "finished":
                    still.append(op)
            pending = still
            if pending:
                sleep(poll_s)
        times["ready_s"] = round(clock() - t0, 1)
        result = compare_fn(eps[0])
        times["compared_s"] = round(clock() - t0, 1)
    finally:
        http("DELETE", f"/projects/{project}/branches/{branch}")
        times["deleted_s"] = round(clock() - t0, 1)
        _emit(**{k: v for k, v in times.items()})
    return {**result, "times": times}


def main(argv: Optional[List[str]] = None) -> int:
    import datetime as dt
    import os
    import time

    ap = argparse.ArgumentParser(description="Restore drill: restore branch vs live, counts only.")
    ap.add_argument("--endpoint", default="", help="Neon endpoint id of an existing restore branch (ep-...)")
    ap.add_argument("--create", action="store_true",
                    help="make a scratch restore branch, compare, delete it (needs NEON_API_KEY)")
    ap.add_argument("--minutes-back", type=int, default=60, help="restore point, minutes before now")
    a = ap.parse_args(argv)
    try:
        if a.create:
            key = os.environ.get("NEON_API_KEY", "")
            if not key:
                raise Refused("neon_api_key_missing")
            from scripts import scrub_verify as sv
            result = scratch_drill(neon_http(key), sv._live_url(), a.minutes_back, run,
                                   clock=time.monotonic, sleep=time.sleep,
                                   now_utc=lambda: dt.datetime.now(dt.timezone.utc))
        else:
            result = run(a.endpoint)
    except Exception as exc:  # noqa: BLE001 -- refusal codes only, never a URL
        code = getattr(exc, "code", type(exc).__name__)
        _emit(refused=code)
        return 2
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
