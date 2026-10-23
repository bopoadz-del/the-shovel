#!/usr/bin/env python3
"""Verify the scrub-at-source clean did not cut into ordinary words; repair it if it did.

Counts only on stdout -- never a rule, a match or row text.

The rule list is the one the clean ran with: read inside the job from the
PREVIOUS version of the live task's shared secret (the key was removed from
the current version when the list was retired), held in memory, discarded at
exit. It never returns to live config, the repository or a log.

Every rule is matched WHOLE-WORD by default (whole_word): a match may not
start or end inside a longer run of letters and digits, so a short
identifier is never replaced inside an ordinary word.

Modes:
  measure  -- on the restore branch (read-only endpoint, the text as it was
              before the clean), shared layer only: occurrences as the clean
              ran and whole-word, and the rows / names holding a match that
              falls inside a word
  repair   -- requires confirm=REPAIR: for those rows and names only, the
              restore branch computes the whole-word clean of the ORIGINAL
              text (server-side, in its own query) and only that result is
              written to live; changed chunks are re-embedded
  dryrun   -- on live, shared layer: whole-word matches (must be 0) and
              matches inside words (0 once repaired)
  tsvector -- how live keeps the full-text column current (generated or not)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PREFIX = "SCRUB_VERIFY"
RULES_KEY = "RAG_SCRUB_RULES"
DEFAULT_REPLACEMENT = "the project"
_DOCUMENTS = "documents"


def _emit(**fields: object) -> None:
    print(f"{_PREFIX} " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


class Refused(Exception):
    def __init__(self, code: str, **fields: object) -> None:
        super().__init__(code)
        self.code, self.fields = code, fields


# ── rules ────────────────────────────────────────────────────────────────────
def parse_rules(raw: str) -> List[Tuple[str, str]]:
    """``<regex> => <replacement>`` per line, as the retired scrubber read
    them; longest pattern first, the order it applied them."""
    rules: List[Tuple[str, str]] = []
    for line in (raw or "").replace("\\n", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pat, _, repl = line.partition("=>")
        rules.append((pat.strip(), repl.strip() or DEFAULT_REPLACEMENT))
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


_PY_ONLY = re.compile(r"\(\?P[<=]|\(\?[aiLmsux-]+[):]|\\[zZG]|[*+?}]\+")


def pg_pattern(pattern: str) -> Optional[str]:
    """The rule as a PostgreSQL ARE (Python's \\b is \\y there), or None for
    Python-only syntax."""
    if _PY_ONLY.search(pattern):
        return None
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append({"b": r"\y", "B": r"\Y"}.get(pattern[i + 1], ch + pattern[i + 1]))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


#: Neither side of a match may continue a run of letters or digits.
_WORD = "[[:alnum:]_]"


def whole_word(pg: str) -> str:
    """``pg`` matched only where it is not part of a longer word."""
    return f"(?<!{_WORD})(?:{pg})(?!{_WORD})"


def whole_word_python(pattern: str) -> "re.Pattern[str]":
    """The same rule for Python's engine (tests and local runs)."""
    return re.compile(rf"(?<![^\W])(?:{pattern})(?![^\W])", re.IGNORECASE)


def translated(rules: Sequence[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """(as-run pattern, whole-word pattern, replacement) per rule; any rule
    PostgreSQL cannot run exactly stops the run, counted."""
    out, refused = [], 0
    for pat, repl in rules:
        p = pg_pattern(pat)
        if p is None or "\\" in repl:
            refused += 1
            continue
        out.append((p, whole_word(p), repl))
    if refused:
        raise Refused("rules_unsupported", rules_unsupported=refused, rules_loaded=len(rules))
    return out


def previous_rules() -> List[Tuple[str, str]]:
    """The rule list from the previous version of the live task's shared
    secret, through the AWS CLI (stdout captured, never forwarded). In
    memory only."""
    import subprocess

    def aws(*args: str) -> object:
        proc = subprocess.run(["aws", *args, "--output", "json"], check=False,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            code = re.search(r"\(([A-Za-z]+(?:Exception)?)\)", proc.stderr or "")
            raise Refused("aws_failed", **({"aws_code": code.group(1)} if code else {}))
        return json.loads(proc.stdout or "null")

    svc = aws("ecs", "describe-services", "--cluster", os.environ["ECS_CLUSTER"],
              "--services", os.environ["ECS_SERVICE"])["services"][0]
    td = aws("ecs", "describe-task-definition", "--task-definition", svc["taskDefinition"])["taskDefinition"]
    ids = {":".join(s["valueFrom"].split(":")[:7]) for c in td["containerDefinitions"]
           for s in c.get("secrets", []) if s["valueFrom"].split(":")[5:6] == ["secret"]}
    for sid in sorted(ids):
        got = aws("secretsmanager", "get-secret-value", "--secret-id", sid, "--version-stage", "AWSPREVIOUS")
        blob = json.loads(got["SecretString"])
        if RULES_KEY in blob:
            return parse_rules(blob[RULES_KEY])
    raise Refused("rules_not_found_in_previous_version")


# ── databases ────────────────────────────────────────────────────────────────
def restore_url(live_url: str, endpoint_id: str) -> str:
    """The live connection with its host's endpoint swapped for the restore
    branch's (same project, roles and database). Never printed."""
    if not re.fullmatch(r"ep-[a-z0-9-]+", endpoint_id or ""):
        raise Refused("bad_endpoint_id")
    parts = urlsplit(live_url)
    userinfo, _, hostport = parts.netloc.rpartition("@")
    host, sep, port = hostport.partition(":")
    domain = host.split(".", 1)[1] if "." in host else ""
    if not domain:
        raise Refused("live_host_unrecognised")
    new = f"{endpoint_id}.{domain}" + (sep + port if port else "")
    return urlunsplit((parts.scheme, f"{userinfo}@{new}" if userinfo else new, parts.path, parts.query, parts.fragment))


def _session_factory(url: str):
    from app.core.db import _session_factory_for_url
    return _session_factory_for_url(url)


def _live_url() -> str:
    from app.core.rag.vector_store import _database_url, _default_db_path
    return _database_url(_default_db_path())


def chunk_table() -> str:
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace
    return rag_chunk_table_name(_rag_vector_namespace())


_NOBODY = "scrub-verify-non-owner"


def shared_project_ids() -> List[str]:
    """Projects a user who owns nothing can read (the access check itself,
    aliases included) -- the layer the clean ran on."""
    from sqlalchemy import select

    from app.core import projects
    from app.core.models import Project

    ids = set(projects.general_knowledge_project_ids())
    projects._ensure_db()
    with projects.SessionLocal() as session:
        rows = list(session.scalars(select(Project.id)).all())
    for pid in rows + [projects.MASTER_CORPUS_PROJECT_ID]:
        if projects.can_access_project(pid, user_id=_NOBODY, include_admin_approved=True):
            ids.add(projects._master_corpus_source(pid) or pid)
    return sorted(ids)


# ── measure / dryrun ─────────────────────────────────────────────────────────
def _inside_expr(col: str, n: int) -> str:
    return " OR ".join(
        f"regexp_count({col}, :p{i}, 1, 'i') > regexp_count({col}, :w{i}, 1, 'i')" for i in range(n))


def occurrences(factory, table: str, col: str, rules, shared: Sequence[str]) -> Dict[str, int]:
    """Occurrences of every rule as run and whole-word, and rows holding at
    least one match inside a word -- counted in the database."""
    from sqlalchemy import text as sql

    params: Dict[str, object] = {"shared": list(shared)}
    as_run, whole = [], []
    for i, (p, w, _r) in enumerate(rules):
        params[f"p{i}"], params[f"w{i}"] = p, w
        as_run.append(f"COALESCE(SUM(regexp_count({col}, :p{i}, 1, 'i')), 0)")
        whole.append(f"COALESCE(SUM(regexp_count({col}, :w{i}, 1, 'i')), 0)")
    stmt = sql(
        f"SELECT {' + '.join(as_run)}, {' + '.join(whole)}, "
        f"COUNT(*) FILTER (WHERE {_inside_expr(col, len(rules))}) "
        f"FROM {table} WHERE project_id = ANY(:shared)"
    )
    with factory() as session:
        a, w, rows = session.execute(stmt, params).one()
    return {"as_run": int(a), "whole_word": int(w), "inside_word": int(a) - int(w), "rows_inside_word": int(rows)}


# ── repair ───────────────────────────────────────────────────────────────────
def _whole_word_clean_expr(col: str, n: int) -> str:
    expr = col
    for i in range(n):
        expr = f"regexp_replace({expr}, :w{i}, :r{i}, 'gi')"
    return expr


def repair(restore, live, table: str, key: str, col: str, rules, shared: Sequence[str],
           batch: int, queue: Optional[str] = None) -> int:
    """For rows whose ORIGINAL text holds a match inside a word: the restore
    branch computes the whole-word clean of that original text, and only the
    result for those rows is written to live (shared rows only). Keyset
    batches by key in byte order."""
    from sqlalchemy import text as sql

    params: Dict[str, object] = {"shared": list(shared), "n": batch}
    for i, (p, w, r) in enumerate(rules):
        params[f"p{i}"], params[f"w{i}"], params[f"r{i}"] = p, w, r
    select = sql(
        f"SELECT {key}, {_whole_word_clean_expr(col, len(rules))} AS cleaned FROM {table} "
        f"WHERE project_id = ANY(:shared) AND {key} COLLATE \"C\" > :last "
        f"AND ({_inside_expr(col, len(rules))}) ORDER BY {key} COLLATE \"C\" LIMIT :n"
    )
    update = sql(f"UPDATE {table} SET {col} = :v WHERE {key} = :k AND project_id = ANY(:shared)")
    done, last = 0, ""
    while True:
        with restore() as rs:
            rows = rs.execute(select, {**params, "last": last}).all()
        if not rows:
            return done
        with live() as ls:
            ls.execute(update, [{"v": r[1], "k": r[0], "shared": list(shared)} for r in rows])
            if queue:
                ls.execute(sql(f"CREATE TABLE IF NOT EXISTS {queue} (chunk_id TEXT PRIMARY KEY)"))
                ls.execute(sql(f"INSERT INTO {queue} (chunk_id) SELECT unnest(CAST(:ids AS text[])) "
                               "ON CONFLICT DO NOTHING"), {"ids": [r[0] for r in rows]})
            ls.commit()
        done += len(rows)
        last = max(str(r[0]) for r in rows)


_QUEUE = "scrub_reembed_queue"


def reembed(live, table: str, batch: int) -> int:
    """New vectors for the queued chunks (their text changed), one batch at a
    time; the queue is dropped when empty."""
    from sqlalchemy import text as sql

    from app.core.rag.embeddings import get_embedder

    with live() as s:
        if s.execute(sql("SELECT to_regclass(:t)"), {"t": _QUEUE}).scalar() is None:
            return 0
    emb = get_embedder()
    done = 0
    while True:
        with live() as s:
            rows = s.execute(sql(f"SELECT c.chunk_id, c.text FROM {_QUEUE} q JOIN {table} c "
                                 f"ON c.chunk_id = q.chunk_id ORDER BY q.chunk_id LIMIT :n"), {"n": batch}).all()
            if not rows:
                s.execute(sql(f"DROP TABLE IF EXISTS {_QUEUE}"))
                s.commit()
                return done
            vecs = emb.encode([r.text for r in rows])
            s.execute(sql(f"UPDATE {table} SET embedding = CAST(:e AS vector) WHERE chunk_id = :id"),
                      [{"id": r.chunk_id, "e": "[" + ",".join(f"{float(x):.7g}" for x in vecs[i]) + "]"}
                       for i, r in enumerate(rows)])
            s.execute(sql(f"DELETE FROM {_QUEUE} WHERE chunk_id = ANY(:ids)"),
                      {"ids": [r.chunk_id for r in rows]})
            s.commit()
            done += len(rows)


def tsvector_state(live, table: str) -> Dict[str, str]:
    """How the full-text column of the chunk table is kept current."""
    from sqlalchemy import text as sql
    with live() as s:
        row = s.execute(sql(
            "SELECT is_generated, generation_expression IS NOT NULL FROM information_schema.columns "
            "WHERE table_name = :t AND data_type = 'tsvector'"), {"t": table}).first()
    if not row:
        return {"tsvector_column": "absent"}
    return {"tsvector_column": "present", "generated": row[0], "has_expression": str(bool(row[1])).lower()}


# ── run ──────────────────────────────────────────────────────────────────────
def run(mode: str, *, endpoint: str = "", confirm: str = "", batch: int = 200,
        rules_raw: Optional[List[Tuple[str, str]]] = None) -> Dict[str, object]:
    if mode not in {"measure", "repair", "dryrun", "tsvector"}:
        raise Refused("bad_mode")
    table = chunk_table()
    live = _session_factory(_live_url())
    if mode == "tsvector":
        out = tsvector_state(live, table)
        _emit(mode=mode, **out)
        return out
    rules = translated(rules_raw if rules_raw is not None else previous_rules())
    shared = shared_project_ids()
    base = {"rules_loaded": len(rules), "shared_projects": len(shared)}
    if mode == "dryrun":
        c = occurrences(live, table, "text", rules, shared)
        n = occurrences(live, _DOCUMENTS, "original_name", rules, shared)
        out = {**base, "shared_matches": c["whole_word"], "shared_document_names": n["whole_word"],
               "inside_word_chunks": c["inside_word"], "inside_word_names": n["inside_word"]}
        _emit(mode=mode, **out)
        return out
    restore = _session_factory(restore_url(_live_url(), endpoint))
    if mode == "measure":
        c = occurrences(restore, table, "text", rules, shared)
        n = occurrences(restore, _DOCUMENTS, "original_name", rules, shared)
        out = {**base, **{f"chunks_{k}": v for k, v in c.items()}, **{f"names_{k}": v for k, v in n.items()}}
        _emit(mode=mode, **out)
        return out
    if confirm != "REPAIR":
        raise Refused("confirm_required")
    from app.core.rag.embeddings import get_embedder
    get_embedder()  # loads before any write: a repair never leaves stale vectors
    rows = repair(restore, live, table, "chunk_id", "text", rules, shared, batch, queue=_QUEUE)
    names = repair(restore, live, _DOCUMENTS, "id", "original_name", rules, shared, batch)
    out = {**base, "rows_restored_and_recleaned": rows, "names_restored_and_recleaned": names,
           "embeddings_updated": reembed(live, table, batch)}
    _emit(mode=mode, **out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify / repair the scrub-at-source clean (counts only).")
    ap.add_argument("--mode", required=True, choices=("measure", "repair", "dryrun", "tsvector"))
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--confirm", default="")
    ap.add_argument("--batch-size", type=int, default=200)
    a = ap.parse_args(argv)
    try:
        run(a.mode, endpoint=a.endpoint, confirm=a.confirm, batch=max(1, a.batch_size))
    except Refused as exc:
        _emit(error=exc.code, **exc.fields)
        return 2
    except Exception as exc:  # noqa: BLE001 -- the type only: a message could quote a row
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        _emit(error="failed", type=type(exc).__name__, **({"aws_code": code} if code else {}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
