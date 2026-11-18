"""Salted token digests of indexed chunks, read by id (read-only, measurement).

The answer-against-its-sources score (``scripts/answer_source_score.py``)
needs each cited chunk's text for two of its three figures: (a) numbers and
(c) overlap. That text lives in the live index and must not leave it. This
script reads it there, by chunk id, and prints for every chunk only the
keyed digests of the tokens the scorer itself compares:

* ``w``: one digest per content word (``answer_source_score.content_words``)
* ``n``: one digest per number (``answer_source_score.numbers_in``)

Digests are ``HMAC-SHA256(salt, kind + token)`` cut to 12 hex characters and
printed sorted, so no word order, no text and no length of text is printed.
The salt arrives as ``CHUNK_DIGEST_SALT`` (at least 16 characters), is never
printed, and is discarded by the operator after the run: without it a digest
cannot be turned back into a word.

Offline, ``surrogate_texts`` turns the digests back into a stand-in text per
chunk built ONLY from the scored answers' own tokens whose digest the chunk
carries. Both figures are set intersections against the answer's own tokens,
so ``score_row`` over the stand-in gives the same numbers_found and overlap as
over the real text (``tests/test_chunk_token_digest.py`` proves it on
invented rows). The scorer is not changed.

Live (one-off task, ``WORKER_COMMAND=python scripts/chunk_token_digest.py``):
    CHUNK_DIGEST_IDS   comma-separated chunk ids (at most ``_MAX_IDS``)
    CHUNK_DIGEST_SALT  the run's salt (never printed)
Offline:
    python scripts/chunk_token_digest.py --surrogate DIGESTS.jsonl ROWS.json \\
        --salt-env CHUNK_DIGEST_SALT > chunk_text.json
    python scripts/answer_source_score.py ROWS.json --chunk-text chunk_text.json
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from decimal import Decimal
from typing import Any, Iterable

# Invoked as `python scripts/chunk_token_digest.py` (entrypoint WORKER_COMMAND),
# which puts scripts/ on sys.path; make `app` importable as well.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import answer_source_score as scorer  # noqa: E402  (the scorer's own tokenizers)

_MAX_IDS = 200
_MIN_SALT = 16
_DIGEST_HEX = 12
_MARK = "chunk_digest"


def _digest(salt: bytes, kind: str, token: str) -> str:
    return hmac.new(salt, f"{kind}:{token}".encode("utf-8"), hashlib.sha256).hexdigest()[:_DIGEST_HEX]


def _num_key(value: float) -> str:
    return repr(float(value))


def _num_text(value: float) -> str:
    """A number written so the scorer's own number pattern reads it back as
    the same value (plain decimal, never exponent form)."""
    return format(Decimal(repr(float(value))), "f")


def chunk_digest(chunk_id: str, text: str, salt: bytes) -> dict[str, Any]:
    """Pure: the printed record of one chunk: its id and sorted token digests."""
    words = {_digest(salt, "w", w) for w in scorer.content_words(text)}
    nums = {_digest(salt, "n", _num_key(v)) for v in scorer.numbers_in(text)}
    return {"mark": _MARK, "chunk": str(chunk_id), "w": sorted(words), "n": sorted(nums)}


def parse_ids(raw: str | None) -> list[str]:
    """Distinct non-empty ids in input order; more than ``_MAX_IDS`` is refused."""
    seen: list[str] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if p and p not in seen:
            seen.append(p)
    if len(seen) > _MAX_IDS:
        raise ValueError(f"at most {_MAX_IDS} ids per run, got {len(seen)}")
    return seen


def read_salt(raw: str | None) -> bytes:
    if not raw or len(raw) < _MIN_SALT:
        raise ValueError(f"CHUNK_DIGEST_SALT must be at least {_MIN_SALT} characters")
    return raw.encode("utf-8")


def _chunk_text_sql(table: str) -> str:
    # The table name comes from the product's single validating chokepoint;
    # the ids are an expanding bind parameter. SELECT only.
    return f"SELECT chunk_id, text FROM {table} WHERE chunk_id IN :ids"


def digest_records(found: dict[str, str], ids: Iterable[str], salt: bytes) -> list[dict[str, Any]]:
    """Pure: one record per asked id, in asked order; an id the index does not
    hold prints ``missing`` and is never guessed."""
    out: list[dict[str, Any]] = []
    for cid in ids:
        if cid in found:
            out.append(chunk_digest(cid, found[cid], salt))
        else:
            out.append({"mark": _MARK, "chunk": cid, "missing": True})
    return out


def _read_live(ids: list[str]) -> dict[str, str]:  # pragma: no cover - live only
    from sqlalchemy import bindparam, text

    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    sql = text(_chunk_text_sql(rag_chunk_table_name(_rag_vector_namespace()))).bindparams(
        bindparam("ids", expanding=True))
    with SessionLocal() as s:
        rows = s.execute(sql, {"ids": ids}).all()
    return {str(r[0]): str(r[1] or "") for r in rows}


# --- offline: stand-in text from digests -------------------------------------

def _answer_vocab(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[float]]:
    words: set[str] = set()
    nums: set[float] = set()
    for row in rows:
        answer = str(row.get("answer") or "")
        words |= scorer.content_words(answer)
        nums |= set(scorer.numbers_in(answer))
    return words, nums


def surrogate_texts(records: Iterable[dict[str, Any]], rows: list[dict[str, Any]],
                    salt: bytes) -> dict[str, str]:
    """``{chunk_id: stand-in text}``: the rows' own answer tokens whose digest
    the chunk carries, one per line. A missing chunk gets no entry, so the
    scorer reads it as text-unavailable, exactly as before."""
    words, nums = _answer_vocab(rows)
    by_w = {_digest(salt, "w", w): w for w in words}
    by_n = {_digest(salt, "n", _num_key(v)): _num_text(v) for v in nums}
    out: dict[str, str] = {}
    for rec in records:
        if rec.get("missing") or rec.get("mark") != _MARK:
            continue
        toks = sorted(by_w[d] for d in rec.get("w") or [] if d in by_w)
        toks += sorted(by_n[d] for d in rec.get("n") or [] if d in by_n)
        out[str(rec.get("chunk"))] = "\n".join(toks)
    return out


def _read_records(path: str) -> list[dict[str, Any]]:
    """Digest records from a saved task log: any line holding one JSON record."""
    recs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            start = line.find("{")
            if start < 0:
                continue
            try:
                rec = json.loads(line[start:])
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("mark") == _MARK:
                recs.append(rec)
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--surrogate", nargs=2, metavar=("DIGESTS", "ROWS"),
                    help="offline: digest log + scorer rows -> {chunk_id: stand-in text} JSON")
    ap.add_argument("--salt-env", default="CHUNK_DIGEST_SALT",
                    help="name of the env var holding the salt (the salt is never an argument)")
    args = ap.parse_args(argv)
    try:
        salt = read_salt(os.getenv(args.salt_env))
    except ValueError as exc:
        print(f"{_MARK}: {exc}", file=sys.stderr)
        return 2

    if args.surrogate:
        digests, rows_path = args.surrogate
        with open(rows_path, encoding="utf-8") as fh:
            rows = json.load(fh)
        rows = rows if isinstance(rows, list) else [rows]
        print(json.dumps(surrogate_texts(_read_records(digests), rows, salt)))
        return 0

    try:
        ids = parse_ids(os.getenv("CHUNK_DIGEST_IDS"))
    except ValueError as exc:
        print(f"{_MARK}: {exc}", file=sys.stderr)
        return 2
    if not ids:
        print(f"{_MARK}: CHUNK_DIGEST_IDS is empty", file=sys.stderr)
        return 2
    try:
        found = _read_live(ids)
    except Exception as exc:  # name the failure by class, never its text
        print(f"{_MARK}: read failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    recs = digest_records(found, ids, salt)
    for rec in recs:
        print(json.dumps(rec, sort_keys=True))
    missing = sum(1 for r in recs if r.get("missing"))
    print(f"{_MARK}: asked={len(ids)} found={len(ids) - missing} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
