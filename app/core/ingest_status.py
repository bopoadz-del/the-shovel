"""One place that decides what an ingested document actually is.

The corpus was re-ingested repeatedly and every run reported success, because
"a documents row exists" and "text came out of it" were the same signal. This
module separates them, and it lives on its own so the rule cannot drift between
the three callers that need it: migration 0016's backfill, the offline stamper,
and the index-time gate in ``doc_index``.

Measured on the live corpus 2026-09-04 (see artifacts INGEST_PROOF / the gap
receipt) -- the numbers below are calibration evidence, not preference:

* 1,873 of 2,691 PDFs in one client corpus project held exactly ONE chunk while
  being indistinguishable from a fully-extracted specification.
* A chars-per-MB gate was tried and REJECTED: 935 one-chunk documents sit above
  any defensible cut, because chars/MB conflates image-heavy with text-poor.
* OCR is NOT the remedy for those sheets. Rendering them and running Tesseract
  returned 0.6-1.4x the characters the text layer already had, higher DPI was
  worse, and novel-token counts of 1-50 per sheet are consistent with noise on
  line-work. They are vector CAD exports whose content is geometry.

REVERSED 2026-09: .kmz/.kml used to sit in ``TEXT_BEARING_EXTS`` on the
strength of ".kmz averages 43 chunks/doc across 311 documents (it is zipped
KML, i.e. XML)". That evidence was real but the wrong test -- "produced
chunks" is not the same as "produced retrievable content". A RAG
data-quality incident traced back to exactly those chunks: 260 .kmz
documents produced 11,630 chunks, and auditing them found 146 GIS attribute
tables (key=value rows with coordinates), 39 pure CAD entity dumps ("Cameras
| Paths | Model | Polyline [9CA0] | Hatch [771B]"), 34 near-empty fragments,
and 41 bare name lists -- none of it prose a retriever could use. At ~45
chunks/doc, .kmz was silently one of the largest contributors to the index
while contributing nothing usable. The owner's ruling: NO kmz in the RAG,
full stop, even the ones that are technically retrievable. .kmz/.kml are now
an explicit denylist (``_GEOSPATIAL_EXTS``) checked in ``classify`` BEFORE
the chunk_count evidence check, so an already-chunked kmz cannot slip back
in as INDEXED the way the evidence-beats-policy ordering was designed to let
.dwg/.dxf slip in.
"""
from __future__ import annotations

import os
from typing import Any, Collection, Mapping, NamedTuple

from app.core import text_extractors as _text_extractors

# ── status vocabulary (mirrors the CHECK constraint in migration 0016) ────────

UNVERIFIED = "UNVERIFIED"
INDEXED = "INDEXED"
TEXT_SPARSE = "TEXT_SPARSE"
ZERO_CHUNK = "ZERO_CHUNK"
UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
EXTRACT_FAILED = "EXTRACT_FAILED"
TOMBSTONED = "TOMBSTONED"
QUARANTINED = "QUARANTINED"
# Reason token, not a CHECK status. OCR skipped above the size/memory
# ceiling must be visible on the ledger; adding a new ingest_status value
# would require an ALTER CONSTRAINT. Recoverable: force_ocr / raise ceiling.
OCR_DEGRADED = "OCR_DEGRADED"

# Statuses whose document holds chunks and is retrieved.
INDEXED_STATUSES = frozenset({INDEXED, TEXT_SPARSE})

ALL_STATUSES = frozenset(
    {
        UNVERIFIED,
        INDEXED,
        TEXT_SPARSE,
        ZERO_CHUNK,
        UNSUPPORTED_TYPE,
        EXTRACT_FAILED,
        TOMBSTONED,
        QUARANTINED,
    }
)

# Statuses that keep NO chunks: a failed, empty, unsupported, quarantined or
# tombstoned document. Writing one deletes the document's chunks in the same
# transaction (projects.delete_document_chunks), and retrieval ignores any
# chunk of such a document. UNVERIFIED (not yet classified) is neither: a
# document holds its chunks for a moment before the ledger classifies it.
NO_CHUNK_STATUSES = ALL_STATUSES - INDEXED_STATUSES - {UNVERIFIED}

# Extractor family stamped on INDEXED rows after the content-control fix.
# #550 / 8535199: Word SDT + deduped text boxes. Bump the suffix when the
# extract walk changes again; do not invent a hash that is not this family.
EXTRACTOR_VERSION = "8535199-sdt"
#: Reopen sentinel written by ``scripts/reopen_embed_failed_docx``. Not
#: ``EXTRACTOR_VERSION``, so ``docx_stale_extractor_open`` stays true.
EMBED_FAILED_SUFFIX = "/embed-failed"

#: Statuses that represent OPEN work. Everything else is a settled outcome.
#: ``TEXT_SPARSE`` is deliberately absent -- see ``is_open``.
OPEN_STATUSES = frozenset({UNVERIFIED, ZERO_CHUNK, EXTRACT_FAILED})

# Reason suffixes that qualify TEXT_SPARSE and UNSUPPORTED_TYPE.
TERMINAL = "terminal"        # nothing left to recover, anywhere
RECOVERABLE = "recoverable"  # a richer source for this document exists

# BOUNDED RETRIES. Every open outcome -- UNVERIFIED, ZERO_CHUNK,
# EXTRACT_FAILED, and the RECOVERABLE qualifiers -- is retried only while the
# document's attempt count is below ``max_attempts()``. Attempts are keyed to
# (source content, EXTRACTOR_VERSION): the same bytes through the same
# extractor give the same outcome, so a further retry can only repeat it.
# Live 2026-10-03 one drawing PDF OOM-killed every ingest pass at the same
# position and 711 settled-but-open rows were re-touched on every pass.
# Reaching the bound appends ``attempts_exhausted@<EXTRACTOR_VERSION>`` to
# the reason; the row is then closed for this extractor. A new
# EXTRACTOR_VERSION or new source bytes reopen it.
ATTEMPTS_MARK = "attempts_exhausted@"


def is_ingestible(name_or_ext: str | None) -> bool:
    """True when a file name (or bare extension) is a text format the RAG ingests."""
    import os

    raw = (name_or_ext or "").strip().lower()
    ext = os.path.splitext(raw)[1] or (raw if raw.startswith(".") else "")
    return ext in TEXT_BEARING_EXTS


def correct_empty_indexed(conn) -> int:
    """Re-stamp rows recorded INDEXED that hold no text: a file that yields no
    text is never recorded as indexed. Each gets what ``classify`` gives zero
    chunks (ZERO_CHUNK, or UNSUPPORTED_TYPE for a non-ingestible or empty
    file). A row counts as empty only when its ledger says 0 chunks AND no
    chunk table holds a chunk for it. Idempotent: a second run changes 0
    rows. Dialect neutral (``conn`` is a SQLAlchemy connection).
    """
    import sqlalchemy as sa

    insp = sa.inspect(conn)
    chunk_tables = [
        name for name in insp.get_table_names()
        if name.startswith("chunks") and "doc_id" in {c["name"] for c in insp.get_columns(name)}
    ]
    no_chunks = "".join(
        f" AND NOT EXISTS (SELECT 1 FROM {name} c WHERE c.doc_id = d.id)" for name in chunk_tables
    )
    rows = conn.execute(sa.text(
        "SELECT d.id, d.original_name, d.size FROM documents d "
        "WHERE d.ingest_status = :indexed AND COALESCE(d.chunk_count, 0) = 0" + no_chunks
    ), {"indexed": INDEXED}).all()
    for doc_id, name, size in rows:
        outcome = classify(
            chunk_count=0, extension=os.path.splitext(name or "")[1], size_bytes=size,
        )
        conn.execute(sa.text(
            "UPDATE documents SET ingest_status = :status, ingest_status_reason = :reason "
            "WHERE id = :id AND ingest_status = :indexed"
        ), {"status": outcome.status, "reason": outcome.reason, "id": doc_id, "indexed": INDEXED})
    return len(rows)


def enforce_chunk_invariants(conn) -> dict:
    """Make the ledger and the chunk tables agree. Idempotent; dialect neutral.

    1. A document of a format no extractor reads (compressed folders, files
       without an extension, images ...) is removed with its chunks and its
       index entries: it may not be a project document at all.
    2. A document in a no-chunk status keeps no chunks.
    3. A document recorded in an indexed status that holds no chunk, or a
       not-yet-classified one that holds chunks, is classified from what it
       actually holds.
    Returns the counts changed; a second run returns all zeros.
    """
    import sqlalchemy as sa

    insp = sa.inspect(conn)
    tables = set(insp.get_table_names())
    chunk_tables = sorted(
        name for name in tables
        if name.startswith("chunks") and "doc_id" in {c["name"] for c in insp.get_columns(name)}
    )
    out = {"removed_documents": 0, "removed_chunks": 0, "reclassified": 0}

    def _chunks(doc_id: str) -> int:
        return sum(int(conn.execute(sa.text(
            f"SELECT COUNT(*) FROM {t} WHERE doc_id = :d"), {"d": doc_id}).scalar() or 0)
            for t in chunk_tables)

    def _drop_chunks(doc_id: str) -> int:
        return sum(int(conn.execute(sa.text(
            f"DELETE FROM {t} WHERE doc_id = :d"), {"d": doc_id}).rowcount or 0)
            for t in chunk_tables)

    for doc_id, name in conn.execute(sa.text("SELECT id, original_name FROM documents")).all():
        if not is_ingestible(name or ""):
            out["removed_chunks"] += _drop_chunks(doc_id)
            if "doc_index_entries" in tables:
                conn.execute(sa.text("DELETE FROM doc_index_entries WHERE document_id = :d"), {"d": doc_id})
            conn.execute(sa.text("UPDATE documents SET superseded_by = NULL WHERE superseded_by = :d"),
                         {"d": doc_id})
            conn.execute(sa.text("DELETE FROM documents WHERE id = :d"), {"d": doc_id})
            out["removed_documents"] += 1

    barred = ", ".join(f"'{s}'" for s in sorted(NO_CHUNK_STATUSES))
    for (doc_id,) in conn.execute(sa.text(
            f"SELECT id FROM documents WHERE ingest_status IN ({barred})")).all():
        removed = _drop_chunks(doc_id)
        if removed:
            out["removed_chunks"] += removed
            conn.execute(sa.text("UPDATE documents SET chunk_count = 0 WHERE id = :d"), {"d": doc_id})

    held = ", ".join(f"'{s}'" for s in sorted(INDEXED_STATUSES | {UNVERIFIED}))
    for doc_id, name, size, status in conn.execute(sa.text(
            f"SELECT id, original_name, size, ingest_status FROM documents WHERE ingest_status IN ({held})")).all():
        n = _chunks(doc_id)
        if (status == UNVERIFIED and n == 0) or (status in INDEXED_STATUSES and n > 0):
            continue  # nothing to classify yet / already consistent
        outcome = classify(chunk_count=n, extension=os.path.splitext(name or "")[1],
                           size_bytes=size if n == 0 else None)
        conn.execute(sa.text(
            "UPDATE documents SET ingest_status = :s, ingest_status_reason = :r, chunk_count = :n "
            "WHERE id = :d"), {"s": outcome.status, "r": outcome.reason, "n": n, "d": doc_id})
        out["reclassified"] += 1
    return out


def max_attempts() -> int:
    """Configured retry bound (``INGEST_MAX_ATTEMPTS``, default 1, minimum 1)."""
    import os

    try:
        return max(1, int(os.getenv("INGEST_MAX_ATTEMPTS", "1")))
    except ValueError:
        return 1


def attempt_key(source_token: str | None) -> str:
    """What an attempt count is counted against: these bytes, this extractor."""
    return f"{source_token or ''}|{EXTRACTOR_VERSION}"


def exhausted_reason(reason: str | None, attempts: int) -> str:
    """``reason`` closed for the current extractor after ``attempts`` tries."""
    mark = f"{ATTEMPTS_MARK}{EXTRACTOR_VERSION}:{attempts}:{TERMINAL}"
    return f"{reason}|{mark}" if reason else mark


def attempts_exhausted(reason: str | None) -> bool:
    """True when ``reason`` carries the exhaustion mark for THIS extractor."""
    return f"{ATTEMPTS_MARK}{EXTRACTOR_VERSION}:" in (reason or "")


def recorded_attempts(doc: Mapping[str, Any]) -> int:
    """Attempts on ``doc``'s ledger row for its current bytes + this extractor.

    Read straight from ``metadata.ingest_attempts`` so resume can bound a row
    whose last attempt killed the process before any outcome was stamped.
    """
    meta = doc.get("metadata") or {}
    prior = meta.get("ingest_attempts") if isinstance(meta, Mapping) else None
    if not isinstance(prior, Mapping):
        return 0
    key = attempt_key(doc.get("content_sha256") or doc.get("drive_md5"))
    if prior.get("key") != key:
        return 0
    return attempt_count(prior)


def attempt_count(prior: Mapping[str, Any]) -> int:
    """The ``n`` of a stored ``ingest_attempts`` record; 0 if absent or malformed."""
    n = prior.get("n")
    if isinstance(n, bool) or not isinstance(n, (int, str)):
        return 0
    text = str(n).strip()
    return int(text) if text.isdigit() else 0

# Formats that can yield text at all. Everything else registers as
# UNSUPPORTED_TYPE rather than sitting at zero chunks looking like a failure.
#
# Consulted ONLY for documents that produced nothing -- see ``classify``. The
# archive entry is here on measured evidence, not intuition: .zip averages
# 119 chunks/doc across 5 documents, and .ifc is a STEP text format.
#
# .kmz/.kml are DELIBERATELY ABSENT -- see the reversal note above and the
# ``_GEOSPATIAL_EXTS`` guard in ``classify``. They used to sit here on the
# strength of "produces chunks"; that was the wrong test.
#: THE SINGLE DECLARATION OF WHAT IS INGESTIBLE. The RAG is text-only by
#: design (docs/RAG_GAPS_REVIEW_2026-09-12.md section E: CAD, images,
#: Google Earth, video, GIS internals and fonts are never ingested). Every
#: ingest entry point -- Drive ingest, the Drive walker and import, CDE,
#: project document upload, the indexer -- derives from this set; no other
#: list may decide ingestibility. Compressed files are never ingestible. See
#: ``is_ingestible`` and docs/INGEST_EXCLUSION_RULE.md.
# Built from the extractor registry (app/core/text_extractors.py): a format
# is ingestible only when a working text extractor is registered for it.
TEXT_BEARING_EXTS = _text_extractors.formats()

# .kmz/.kml are geospatial containers, not prose. They chunk (zipped/raw KML
# is XML, so extraction "succeeds"), but the RAG data-quality incident of
# 2026-09 showed what those chunks actually are: 260 .kmz documents produced
# 11,630 chunks, and a sample of them broke down as 146 GIS attribute tables
# (key=value rows with coordinates), 39 pure CAD entity dumps ("Polyline
# [9CA0]", "Hatch [771B]"), 34 near-empty fragments, and 41 bare name lists --
# none of it retrievable prose. At ~45 chunks/doc, .kmz was silently one of
# the largest contributors to the index while contributing zero usable
# content. The owner's ruling: NO kmz in the RAG, even the technically
# retrievable ones. So this is an EXPLICIT DENYLIST, not an absence from
# TEXT_BEARING_EXTS -- see the guard in ``classify`` that checks it before
# chunk_count, because evidence-beats-policy (the ordering that protected
# .dwg/.dxf) would otherwise let an already-chunked .kmz back in as INDEXED.
_GEOSPATIAL_EXTS = frozenset({".kmz", ".kml"})

# Formats whose text lives in a convertible source rather than the file we hold.
# ``.dwg`` carries real TEXT/MTEXT/ATTRIB entities; recovering them is the
# separate F-DWG campaign, so they are marked recoverable, not terminal.
RECOVERABLE_EXTS = frozenset({".dwg", ".dxf"})

# ── density gate ─────────────────────────────────────────────────────────────
# Calibrated on 10 known-sparse drawing sheets vs 10 known-rich text PDFs from
# the live corpus. The measurement overturned the obvious design:
#
#            chars/page                    bytes/page
#   sparse   median 1555, range 132-3594   median 520253, MIN 206016
#   rich     median 2370, range 1767-3042  median  34029, MAX 113462
#
# chars-per-page OVERLAPS almost completely -- a drawing sheet's title block
# carries as much text as a dense page of prose, so it cannot convict on its
# own and is not the primary signal. bytes-per-page separates cleanly, with an
# empty band between 113k and 206k; the threshold sits in that gap.
#
# chars-per-page survives only as a BACKSTOP, set well above both observed
# ranges: a physically large page that really is full of text must never be
# flagged, however heavy the file.
_SPARSE_BYTES_PER_PAGE = int(os.getenv("INGEST_SPARSE_BYTES_PER_PAGE", "150000"))
_SPARSE_CHARS_PER_PAGE = int(os.getenv("INGEST_SPARSE_CHARS_PER_PAGE", "6000"))


class Classification(NamedTuple):
    status: str
    reason: str | None
    extension: str | None = None
    extractor_version: str | None = None

    @property
    def is_open(self) -> bool:
        return is_open(
            self.status,
            self.reason,
            extension=self.extension,
            extractor_version=self.extractor_version,
        )


def _normalize_ext(extension: str | None) -> str:
    ext = (extension or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return ext


def _is_docx_ext(extension: str | None) -> bool:
    return _normalize_ext(extension) == ".docx"


def document_extension(doc: Mapping[str, Any]) -> str:
    """Best-effort file extension from a documents row (or resume dict)."""
    meta = doc.get("metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    for candidate in (
        doc.get("original_name"),
        doc.get("stored_as"),
        doc.get("file_path"),
        meta.get("drive_path"),
        meta.get("local_drive_path"),
    ):
        if not candidate:
            continue
        ext = _normalize_ext(os.path.splitext(str(candidate))[1])
        if ext:
            return ext
    return ""


#: Thin-but-chunked outcomes that a newer .docx extractor may still recover.
#: ZERO_CHUNK / UNVERIFIED / EXTRACT_FAILED are already in OPEN_STATUSES.
_THIN_STATUSES = frozenset({TEXT_SPARSE})


def embed_failed_sentinel(version: str | None = None) -> str:
    """``{EXTRACTOR_VERSION}/embed-failed`` — never null, never INDEXED."""
    return f"{version or EXTRACTOR_VERSION}{EMBED_FAILED_SUFFIX}"


def is_embed_failed_sentinel(extractor_version: str | None) -> bool:
    """True when ``extractor_version`` is the reopen sentinel."""
    return (extractor_version or "").endswith(EMBED_FAILED_SUFFIX)


def should_advance_extractor_version(
    *,
    ingest_status: str | None,
    rag_indexed: int = 0,
    rag_error: str | None = None,
    prior_extractor_version: str | None = None,
) -> bool:
    """True when embed landed and the row may stamp EXTRACTOR_VERSION.

    FK-REEXTRACT-2 / #573: ``rag_error`` or ``rag_indexed == 0`` must not
    stamp ``EXTRACTOR_VERSION`` — a failed embed keeps the row open for a
    later infra retry (the *only* legitimate reason a re-extracted row stays
    open).

    Once the embed lands (``rag_indexed > 0``, no ``rag_error``) the current
    extractor's text is final: rich (``INDEXED``) or genuinely thin
    (``TEXT_SPARSE``). Re-extracting the same bytes with the same extractor
    cannot change it, so the row advances to ``EXTRACTOR_VERSION`` and closes
    — **whether the prior stamp was the reopen sentinel or an older version.**

    This is the single terminating rule. #573 withheld the stamp on embed
    failure; #575 then refused *every* thin advance and #576 added back only
    the sentinel-prior thin case — leaving ``TEXT_SPARSE`` + successful embed
    + a plain old prior to loop ``docx_stale_extractor_open`` forever. The
    discriminator for "still work" is the embed outcome, never the prior
    version: a full embed (``rag_indexed>0``, no error) means thin is the
    document's true, settled state. ``prior_extractor_version`` is accepted
    for call-site compatibility and deliberately not consulted.
    """
    if rag_error:
        return False
    if int(rag_indexed or 0) <= 0:
        return False
    return ingest_status in (INDEXED, TEXT_SPARSE)


def docx_stale_extractor_open(
    status: str | None,
    *,
    extension: str | None = None,
    extractor_version: str | None = None,
) -> bool:
    """True when a .docx is still thin and was extracted by an older extractor.

    Scope is strictly ``.docx``. A missing extractor_version is stale. After
    the current ``EXTRACTOR_VERSION`` stamps the same sparse outcome, this
    returns False so resume cannot retry the row forever.
    """
    if not _is_docx_ext(extension):
        return False
    if (extractor_version or "") == EXTRACTOR_VERSION:
        return False
    if not status or status in _THIN_STATUSES:
        return True
    return False


def is_open(
    status: str,
    reason: str | None = None,
    *,
    extension: str | None = None,
    extractor_version: str | None = None,
) -> bool:
    """True when this document still represents work.

    ``TEXT_SPARSE`` is the subtle one. A vector drawing sheet with no
    convertible source is a SETTLED, honest outcome -- the sheet genuinely
    holds ~1.6-2.9k characters and no amount of OCR changes that. Treating it
    as open would build a gate that can never go green, which is the same trap
    as targeting a file count that includes formats which never chunk.

    Exception, ``.docx`` only: a thin (TEXT_SPARSE or status-missing) row
    whose ``extractor_version`` is missing or not ``EXTRACTOR_VERSION`` is
    still open. The same sparse outcome stamped with the current extractor
    is settled.
    """
    if attempts_exhausted(reason):
        return False
    if status in OPEN_STATUSES:
        return True
    if docx_stale_extractor_open(
        status, extension=extension, extractor_version=extractor_version,
    ):
        return True
    if status in (TEXT_SPARSE, UNSUPPORTED_TYPE):
        return reason is not None and reason.endswith(RECOVERABLE)
    return False


def resume_is_already_indexed(
    doc: Mapping[str, Any],
    chunk_count: int,
    *,
    parseable_exts: Collection[str] | None = None,
) -> bool:
    """Route B / p1b resume skip: chunks > 0 is not enough for stale .docx.

    Non-docx rows with chunks stay skipped (PDF / drawing / kmz rules
    unchanged). A .docx that ``docx_stale_extractor_open`` still flags is
    left assigned so the existing in-place ``index_document`` path can
    replace chunks on the same row.

    Zero chunks is not "work" by itself. A row whose ledger status is a
    closed outcome (``jpg:terminal``, ``empty_file:terminal`` ...) is done;
    re-running the indexer over it produces the same skip and, live
    2026-10-03, 673 such re-runs rewrote a project's index for nothing.
    A RECOVERABLE skip is work only when this build can parse the format
    (``parseable_exts``); without a parser the retry cannot recover anything.
    A row that was never classified (no ``ingest_status``) stays open.

    Fail closed: if this is a .docx and status / extractor_version cannot
    be read, do not count the row as already indexed.
    """
    if recorded_attempts(doc) >= max_attempts():
        # Bounded retries, every class and any chunk count: these bytes have
        # had their tries with this extractor -- including one that died
        # before stamping an outcome, and a stale-extractor .docx whose
        # re-extraction keeps landing thin.
        return True
    if chunk_count <= 0:
        status = doc.get("ingest_status")
        if not status:
            return False
        ext = document_extension(doc)
        reason = doc.get("ingest_status_reason")
        if not is_open(
            str(status), reason, extension=ext,
            extractor_version=doc.get("extractor_version"),
        ):
            return True
        if (
            (reason or "").endswith(RECOVERABLE)
            and parseable_exts is not None
            and ext not in parseable_exts
        ):
            return True
        return False
    ext = document_extension(doc)
    if not _is_docx_ext(ext):
        return True
    if "ingest_status" not in doc or "extractor_version" not in doc:
        return False
    return not docx_stale_extractor_open(
        doc.get("ingest_status"),
        extension=ext,
        extractor_version=doc.get("extractor_version"),
    )


def stale_extractor_run_complete(
    *,
    stale_extractor_open: int,
    retried: int,
    otherwise_complete: bool,
) -> bool:
    """RUN INCOMPLETE when stale .docx were found and none were retried."""
    if stale_extractor_open > 0 and retried == 0:
        return False
    return otherwise_complete


def classify(
    *,
    chunk_count: int,
    extension: str,
    chars: int | None = None,
    page_count: int | None = None,
    size_bytes: int | None = None,
    has_convertible_source: bool = False,
) -> Classification:
    """Decide a document's ingest status from what extraction actually produced.

    ``has_convertible_source`` means a richer original exists in the corpus --
    e.g. a ``.dwg`` sharing this sheet's name. It is what splits a sparse sheet
    into work (RECOVERABLE) versus a closed outcome (TERMINAL).
    """
    ext = (extension or "").lower()
    if not ext.startswith(".") and ext:
        ext = "." + ext

    # KMZ/KML ARE ALWAYS UNSUPPORTED, regardless of chunk_count. This guard
    # must run BEFORE the evidence-beats-policy check below: that check
    # trusts chunk_count > 0 as proof of usable text, and .kmz/.kml chunk
    # successfully (zipped/raw KML is XML) while producing GIS attribute
    # dumps and raw CAD entity handles, not retrievable prose -- the 2026-09
    # incident this module's docstring now records. Without this guard, an
    # already-chunked .kmz would fall straight through to INDEXED below and
    # silently re-admit exactly what the owner ordered purged.
    if ext in _GEOSPATIAL_EXTS:
        return Classification(UNSUPPORTED_TYPE, f"{ext.lstrip('.')}:{TERMINAL}", ext)

    # EVIDENCE BEATS POLICY. A document that produced chunks yielded text, and
    # no extension list may overrule that. Ordering this the other way round
    # was caught in a dry run about to demote 172 documents holding 14,000
    # real chunks -- 311 .kmz files averaging 43 chunks each (zipped KML is
    # XML), and .zip archives averaging 119. The list below was simply wrong
    # for .zip, which is the argument for consulting it last and only when
    # there is no evidence to consult instead. .kmz turned out to be the
    # counter-example: "produced chunks" was not "produced usable text" (see
    # the 2026-09 reversal above), which is exactly why it now has its own
    # guard ABOVE this check instead of relying on this ordering.
    if chunk_count <= 0:
        if ext in RECOVERABLE_EXTS:
            return Classification(UNSUPPORTED_TYPE, f"{ext.lstrip('.')}:{RECOVERABLE}", ext)
        if ext and ext not in TEXT_BEARING_EXTS:
            return Classification(UNSUPPORTED_TYPE, f"{ext.lstrip('.')}:{TERMINAL}", ext)
        # A zero-byte file is a settled outcome, not a failed extraction: the
        # first TIER-1 smoke upload was a 0-byte Drive placeholder that the
        # pipeline correctly reported as "0 chunks, error". 65 of the 231
        # reachable "missing" files were empties. Only trusted together with
        # zero chunks -- the bulk backfill routes recorded size=0 on 2,507
        # documents that hold real chunks, so size alone convicts nothing.
        if size_bytes == 0:
            return Classification(UNSUPPORTED_TYPE, f"empty_file:{TERMINAL}", ext)
        return Classification(ZERO_CHUNK, None, ext)

    qualifier = RECOVERABLE if has_convertible_source else TERMINAL

    # A single chunk means the whole extracted text fit one ~500-word window.
    # This needs no tuned threshold and is why the backfill can run in SQL.
    if chunk_count == 1:
        return Classification(TEXT_SPARSE, f"single_window:{qualifier}", ext)

    # Multi-chunk documents can still be sparse for their physical extent --
    # a 40-sheet drawing set produces one title-block chunk per sheet.
    if _is_sparse_for_extent(chars, page_count, size_bytes):
        return Classification(TEXT_SPARSE, f"low_density:{qualifier}", ext)

    return Classification(INDEXED, None, ext)


def with_ocr_degraded(reason: str | None) -> str:
    """Append OCR_DEGRADED to a classify reason without losing the original."""
    if not reason:
        return f"{OCR_DEGRADED}:{RECOVERABLE}"
    if OCR_DEGRADED in reason:
        return reason
    return f"{reason}+{OCR_DEGRADED}"


def _is_sparse_for_extent(
    chars: int | None, page_count: int | None, size_bytes: int | None
) -> bool:
    """Bytes-per-page convicts; chars-per-page can only acquit.

    Missing data never convicts. Callers that cannot supply page counts (the
    SQL backfill) do not reach this test -- they stop at ``chunk_count == 1``.
    """
    if not page_count or page_count <= 0:
        return False
    if size_bytes is None:
        return False
    if size_bytes / page_count <= _SPARSE_BYTES_PER_PAGE:
        return False
    # Backstop: a heavy page that is nonetheless full of text is a real
    # document (a large-format scan OCR'd successfully), not a drawing.
    return not (chars is not None and chars / page_count >= _SPARSE_CHARS_PER_PAGE)
