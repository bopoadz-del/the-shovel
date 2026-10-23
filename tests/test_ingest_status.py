"""The ledger must not be able to call an unusable document INDEXED.

Every prior ingest of this corpus reported success. It could, because "a
documents row exists" and "usable text came out" were the same signal. These
tests pin the separation, and the numbers in them are measurements against the
live corpus (2026-09-04), not invented fixtures.
"""
from __future__ import annotations

import importlib

import pytest

from app.core import ingest_status as ist

# Calibration measured on 10 known-sparse drawing sheets vs 10 known-rich text
# PDFs. The gap between these two numbers is empty in the sample; the threshold
# lives inside it. If either bound moves, the gate's justification moves too.
RICH_MAX_BYTES_PER_PAGE = 113_462
SPARSE_MIN_BYTES_PER_PAGE = 206_016


def test_one_chunk_is_never_indexed():
    """The defect this whole ledger exists for.

    1,873 of 2,691 PDFs in one project held exactly one chunk and were
    indistinguishable from a fully-extracted specification.
    """
    c = ist.classify(chunk_count=1, extension=".pdf")
    assert c.status == ist.TEXT_SPARSE
    assert c.status != ist.INDEXED


def test_multi_chunk_text_document_is_indexed():
    assert ist.classify(chunk_count=42, extension=".pdf").status == ist.INDEXED


def test_zero_chunks_is_zero_chunk_not_sparse():
    assert ist.classify(chunk_count=0, extension=".pdf").status == ist.ZERO_CHUNK


@pytest.mark.parametrize("ext", [".dwg", ".dxf"])
def test_cad_sources_are_unsupported_but_recoverable(ext):
    """.dwg carries real TEXT/MTEXT entities, so it is work, not a dead end."""
    c = ist.classify(chunk_count=0, extension=ext)
    assert c.status == ist.UNSUPPORTED_TYPE
    assert c.is_open, "a convertible CAD source must stay visible as open work"


@pytest.mark.parametrize("ext", [".jpg", ".ttf", ".gdbtable", ".shx"])
def test_formats_that_can_never_yield_text_are_closed(ext):
    c = ist.classify(chunk_count=0, extension=ext)
    assert c.status == ist.UNSUPPORTED_TYPE
    assert not c.is_open, "a format with no text cannot be a permanent red gate"


@pytest.mark.parametrize(
    "ext,chunks",
    [(".zip", 119), (".jpg", 1), (".ifc", 1), (".png", 1)],
)
def test_evidence_beats_the_extension_list(ext, chunks):
    """A document that produced chunks yielded text, whatever its extension.

    Checking policy before evidence was caught in a dry run about to demote
    172 documents holding ~14,000 real chunks to UNSUPPORTED_TYPE -- .zip
    archives averaging 119 chunks each. The extension list was wrong for
    .zip, which is exactly why it must be consulted last.

    .kmz is deliberately NOT in this table any more -- see
    ``test_kmz_is_never_indexed_even_with_chunks`` below. It used to be the
    poster child for this exact rule (311 files averaging 43 chunks) until a
    RAG data-quality audit showed those chunks were GIS attribute dumps and
    CAD entity handles, not text. .kmz is the one extension where evidence
    must NOT beat policy.
    """
    c = ist.classify(chunk_count=chunks, extension=ext)
    assert c.status != ist.UNSUPPORTED_TYPE
    assert c.status == (ist.TEXT_SPARSE if chunks == 1 else ist.INDEXED)


@pytest.mark.parametrize("ext", [".kmz", ".kml"])
def test_kmz_and_kml_with_no_chunks_are_unsupported_and_closed(ext):
    """RAG DATA-QUALITY INCIDENT (2026-09).

    260 .kmz documents in the corpus had produced 11,630 chunks. Auditing a
    sample found 146 GIS attribute tables (key=value rows with coordinates),
    39 pure CAD entity dumps ("Cameras | Paths | Model | Polyline [9CA0] |
    Hatch [771B]"), 34 near-empty fragments, and 41 bare name lists -- not
    one of them retrievable prose. .kmz averaged ~45 chunks/doc, so it was
    silently one of the largest contributors to the index. The owner's
    ruling: NO kmz in the RAG at all, even the technically retrievable ones.
    A zero-chunk .kmz/.kml must classify as UNSUPPORTED_TYPE(terminal), not
    ZERO_CHUNK (which would sit in OPEN_STATUSES and look like unfinished
    ingest work forever).
    """
    c = ist.classify(chunk_count=0, extension=ext)
    assert c.status == ist.UNSUPPORTED_TYPE
    assert c.reason == f"{ext.lstrip('.')}:terminal"
    assert not c.is_open


@pytest.mark.parametrize("ext", [".kmz", ".kml"])
def test_kmz_is_never_indexed_even_with_chunks(ext):
    """THE REGRESSION GUARD.

    ``classify`` checks chunk_count evidence BEFORE the extension list (see
    ``test_evidence_beats_the_extension_list``), which is exactly the trap
    here: .kmz DOES chunk successfully (zipped/raw KML is XML), so without an
    explicit denylist an already-chunked .kmz would fall straight through
    evidence-beats-policy and register as INDEXED -- silently re-admitting
    the 260 documents / 11,630 chunks the owner ordered purged. Pinning this
    at the real observed volume (45 chunks/doc) so the guard cannot be
    "fixed" back to the old evidence-beats-policy behavior without this test
    failing.
    """
    c = ist.classify(chunk_count=45, extension=ext)
    assert c.status == ist.UNSUPPORTED_TYPE
    assert c.reason == f"{ext.lstrip('.')}:terminal"
    assert not c.is_open


def test_zip_with_chunks_is_still_indexed_not_over_reached():
    """Proves the kmz/kml denylist did not over-reach onto ``.zip``.

    .zip stayed in TEXT_BEARING_EXTS and is not geospatial-container noise --
    it averaged 119 real chunks/doc in the same corpus and must still ride
    the evidence-beats-policy path to INDEXED."""
    c = ist.classify(chunk_count=119, extension=".zip")
    assert c.status == ist.INDEXED


def test_empty_file_with_no_chunks_is_settled_not_open():
    """First TIER-1 smoke upload: a 0-byte Drive placeholder, correctly
    reported by the pipeline as 0 chunks. That is not work."""
    c = ist.classify(chunk_count=0, extension=".pdf", size_bytes=0)
    assert c.status == ist.UNSUPPORTED_TYPE
    assert c.reason == "empty_file:terminal"
    assert not c.is_open


def test_size_zero_alone_convicts_nothing():
    """The bulk backfill routes wrote size=0 on 2,507 documents that hold real
    chunks. Evidence beats the size column too."""
    assert ist.classify(chunk_count=40, extension=".pdf", size_bytes=0).status == ist.INDEXED
    assert ist.classify(chunk_count=1, extension=".pdf", size_bytes=0).status == ist.TEXT_SPARSE


def test_extension_policy_still_applies_when_nothing_was_extracted():
    """The list is not dead -- it is just subordinate to evidence."""
    assert ist.classify(chunk_count=0, extension=".ttf").status == ist.UNSUPPORTED_TYPE


def test_terminal_sparse_is_settled_and_recoverable_sparse_is_work():
    """The distinction that stops the gate being permanently red.

    1,747 sparse sheets have no source to recover from; 126 have a .dwg.
    """
    terminal = ist.classify(chunk_count=1, extension=".pdf")
    recoverable = ist.classify(
        chunk_count=1, extension=".pdf", has_convertible_source=True
    )
    assert not terminal.is_open
    assert recoverable.is_open
    assert terminal.status == recoverable.status == ist.TEXT_SPARSE


# ── density gate ─────────────────────────────────────────────────────────────


def _dense(**kw):
    base = {"chunk_count": 40, "extension": ".pdf", "page_count": 40}
    base.update(kw)
    return ist.classify(**base)


def test_gate_does_not_flag_the_richest_measured_text_pdf():
    """False positives are worse than misses here: wrongly demoting a real
    specification hides it from the very report meant to surface gaps."""
    c = _dense(size_bytes=RICH_MAX_BYTES_PER_PAGE * 40, chars=2370 * 40)
    assert c.status == ist.INDEXED


def test_gate_flags_the_lightest_measured_drawing_sheet():
    c = _dense(size_bytes=SPARSE_MIN_BYTES_PER_PAGE * 40, chars=1555 * 40)
    assert c.status == ist.TEXT_SPARSE
    assert c.reason.startswith("low_density")


def test_chars_per_page_can_acquit_but_never_convict():
    """chars/page OVERLAPS between the two populations (sparse 132-3594, rich
    1767-3042), so it is a backstop only. A heavy page full of text is a real
    document -- a large-format scan that OCR'd well."""
    heavy_but_wordy = _dense(
        size_bytes=SPARSE_MIN_BYTES_PER_PAGE * 40, chars=20_000 * 40
    )
    assert heavy_but_wordy.status == ist.INDEXED


def test_missing_extent_data_never_convicts():
    """The SQL backfill supplies no page counts. It must fall through to the
    chunk_count rule rather than guessing."""
    assert _dense(page_count=None, size_bytes=None, chars=None).status == ist.INDEXED
    assert _dense(size_bytes=None, chars=None).status == ist.INDEXED
    assert _dense(page_count=0, size_bytes=10**9, chars=0).status == ist.INDEXED


def test_thresholds_are_env_tunable_without_a_deploy():
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv("INGEST_SPARSE_BYTES_PER_PAGE", "999999999")
        importlib.reload(ist)
        # With the bar above anything real, nothing is convicted by density.
        assert (
            ist.classify(
                chunk_count=40, extension=".pdf", page_count=1,
                size_bytes=10_000_000, chars=10,
            ).status
            == ist.INDEXED
        )
    finally:
        monkey.undo()
        importlib.reload(ist)


def test_status_vocabulary_matches_the_migration_check_constraint():
    """0016 pins these in SQL. A status added here and not there is written to
    production as a constraint violation, at ingest time, on a live run."""
    from pathlib import Path

    src = Path("alembic/versions/0016_document_ingest_status.py").read_text(
        encoding="utf-8"
    )
    for status in ist.ALL_STATUSES:
        assert f'"{status}"' in src, f"{status} missing from migration 0016"


def test_migration_backfill_only_ever_touches_unverified_rows():
    """0016 was applied to production once out of band and stamped back, so
    the entrypoint re-runs it on deploy. Every UPDATE it makes must be scoped
    to UNVERIFIED rows, or a release silently regresses the stamper's
    refinements -- caught in review, would have shipped."""
    import re
    from pathlib import Path

    src = Path("alembic/versions/0016_document_ingest_status.py").read_text(
        encoding="utf-8"
    )
    updates = re.findall(r"UPDATE documents[^;]*?\"\s*\)\)", src, flags=re.DOTALL)
    assert updates, "expected backfill UPDATE statements in migration 0016"
    for stmt in updates:
        assert "ingest_status = 'UNVERIFIED'" in stmt, (
            "backfill statement not scoped to UNVERIFIED rows:\n" + stmt
        )


def test_mutation_probe_treating_any_chunk_as_success_reopens_the_defect():
    """MUTATION PROBE.

    The historical rule was "chunks > 0 means indexed". Expressed directly so
    the probe fails loudly if anyone restores it: under that rule the 1,873
    one-chunk sheets are INDEXED again and the ledger resumes lying.
    """
    def old_rule(chunk_count: int) -> str:
        return ist.INDEXED if chunk_count > 0 else ist.ZERO_CHUNK

    assert old_rule(1) == ist.INDEXED, "the old rule was already safe -- then " \
        "there was no defect to fix"
    assert ist.classify(chunk_count=1, extension=".pdf").status != old_rule(1)
    # ...and the two rules must still agree where the old one was right.
    assert ist.classify(chunk_count=40, extension=".pdf").status == old_rule(40)


# ── stale-extractor .docx (SDT re-extract) ────────────────────────────────


def test_sparse_docx_old_extractor_is_open():
    """TEXT_SPARSE .docx from an older extractor is still work."""
    assert ist.is_open(
        ist.TEXT_SPARSE,
        "single_window:terminal",
        extension=".docx",
        extractor_version="pre-sdt",
    )
    c = ist.classify(chunk_count=1, extension=".docx")._replace(
        extractor_version="pre-sdt",
    )
    assert c.is_open


def test_sparse_docx_current_extractor_is_closed():
    """After the current extractor stamps the same sparse outcome, settle."""
    assert not ist.is_open(
        ist.TEXT_SPARSE,
        "single_window:terminal",
        extension=".docx",
        extractor_version=ist.EXTRACTOR_VERSION,
    )
    c = ist.classify(chunk_count=1, extension=".docx")._replace(
        extractor_version=ist.EXTRACTOR_VERSION,
    )
    assert not c.is_open


def test_sparse_docx_missing_extractor_is_open():
    assert ist.is_open(
        ist.TEXT_SPARSE,
        "single_window:terminal",
        extension=".docx",
        extractor_version=None,
    )
    # classify() carries extension; a missing stamp is stale.
    assert ist.classify(chunk_count=1, extension=".docx").is_open


def test_non_docx_sparse_ignores_extractor_version():
    """PDF / drawing / kmz rules must not change."""
    assert not ist.is_open(
        ist.TEXT_SPARSE,
        "single_window:terminal",
        extension=".pdf",
        extractor_version="pre-sdt",
    )
    assert not ist.classify(chunk_count=1, extension=".pdf").is_open
    terminal = ist.classify(chunk_count=1, extension=".pdf")._replace(
        extractor_version="pre-sdt",
    )
    assert not terminal.is_open


def test_indexed_docx_current_version_is_closed():
    assert not ist.is_open(
        ist.INDEXED,
        None,
        extension=".docx",
        extractor_version=ist.EXTRACTOR_VERSION,
    )
    c = ist.classify(chunk_count=12, extension=".docx")._replace(
        extractor_version=ist.EXTRACTOR_VERSION,
    )
    assert c.status == ist.INDEXED
    assert not c.is_open


def test_resume_filter_stale_docx_with_chunks_is_not_already_indexed():
    """chunks > 0 used to hide these in already_indexed forever."""
    stale = {
        "original_name": "spec.docx",
        "ingest_status": ist.TEXT_SPARSE,
        "ingest_status_reason": "single_window:terminal",
        "extractor_version": "old-extractor",
    }
    assert not ist.resume_is_already_indexed(stale, 4)
    stamped = {**stale, "extractor_version": ist.EXTRACTOR_VERSION}
    assert ist.resume_is_already_indexed(stamped, 4)


def test_resume_skips_terminal_status_with_zero_chunks():
    """A closed ledger outcome is done at zero chunks; re-running it only
    rewrites the index (live 2026-10-03: 673 such re-runs in one pass)."""
    terminal = {
        "original_name": "photo.jpg",
        "ingest_status": ist.UNSUPPORTED_TYPE,
        "ingest_status_reason": "jpg:terminal",
    }
    assert ist.resume_is_already_indexed(terminal, 0)
    empty = {
        "original_name": "placeholder.pdf",
        "ingest_status": ist.UNSUPPORTED_TYPE,
        "ingest_status_reason": "empty_file:terminal",
    }
    assert ist.resume_is_already_indexed(empty, 0)


def test_resume_zero_chunk_open_rows_stay_open():
    never_classified = {"original_name": "a.pdf"}
    assert not ist.resume_is_already_indexed(never_classified, 0)
    zero_chunk = {"original_name": "a.pdf", "ingest_status": ist.ZERO_CHUNK}
    assert not ist.resume_is_already_indexed(zero_chunk, 0)
    failed = {"original_name": "a.pdf", "ingest_status": ist.EXTRACT_FAILED}
    assert not ist.resume_is_already_indexed(failed, 0)


def test_resume_retries_recoverable_status():
    dwg = {
        "original_name": "sheet.dwg",
        "ingest_status": ist.UNSUPPORTED_TYPE,
        "ingest_status_reason": "dwg:recoverable",
    }
    # No parser list given: historical behaviour, stays open.
    assert not ist.resume_is_already_indexed(dwg, 0)
    # This build has no .dwg parser: nothing to recover, settled.
    assert ist.resume_is_already_indexed(dwg, 0, parseable_exts={".pdf", ".txt"})
    # A build that parses .dwg reopens it.
    assert not ist.resume_is_already_indexed(dwg, 0, parseable_exts={".pdf", ".dwg"})


def test_resume_filter_fail_closed_when_docx_fields_unreadable():
    bare_docx = {"original_name": "spec.docx"}
    assert not ist.resume_is_already_indexed(bare_docx, 4)
    pdf = {"original_name": "sheet.pdf"}
    assert ist.resume_is_already_indexed(pdf, 4)


def test_should_advance_extractor_version_requires_indexed_and_embed():
    """FK-REEXTRACT-2: TEXT_SPARSE / rag miss must not stamp EXTRACTOR_VERSION.

    Live 3867aff advanced ``8535199-sdt/embed-failed`` → ``8535199-sdt``
    after classify() returned TEXT_SPARSE on a 1-chunk embed that looked
    like success (no rag_error, rag_indexed==1). That closed stale-open.

    index_document now sets rag_error on ANY embed shortfall, so "no
    rag_error" means a complete embed — a clean thin embed is the
    document's settled state and now advances.
    """
    assert ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE, rag_indexed=1, rag_error=None,
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE, rag_indexed=0, rag_error=None,
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.INDEXED, rag_indexed=0, rag_error=None,
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.INDEXED, rag_indexed=4, rag_error="embedded 0 of 4",
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.ZERO_CHUNK, rag_indexed=0, rag_error=None,
    )
    assert ist.should_advance_extractor_version(
        ingest_status=ist.INDEXED, rag_indexed=4, rag_error=None,
    )


def test_should_advance_terminal_closes_sentinel_thin_embed():
    """Follow-on to #575: successful thin embed on the reopen sentinel closes.

    #575 refused every TEXT_SPARSE advance so ``{EXTRACTOR_VERSION}/embed-failed``
    could not false-close. After a genuine thin / single_window reextract
    with ``rag_indexed>0`` and no ``rag_error``, that sentinel must
    terminal-close or ``docx_stale_extractor_open`` loops forever.

    The rule is now: embed success always closes, regardless of prior
    version. Still never advance on rag_error / rag_indexed==0.
    """
    sentinel = f"{ist.EXTRACTOR_VERSION}/embed-failed"
    assert ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE,
        rag_indexed=1,
        rag_error=None,
        prior_extractor_version=sentinel,
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE,
        rag_indexed=1,
        rag_error="RuntimeError: 403 Forbidden",
        prior_extractor_version=sentinel,
    )
    assert not ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE,
        rag_indexed=0,
        rag_error=None,
        prior_extractor_version=sentinel,
    )
    assert ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE,
        rag_indexed=1,
        rag_error=None,
        prior_extractor_version=ist.EXTRACTOR_VERSION,
    )
    assert ist.should_advance_extractor_version(
        ingest_status=ist.TEXT_SPARSE,
        rag_indexed=1,
        rag_error=None,
        prior_extractor_version="pre-sdt",
    )
    assert ist.should_advance_extractor_version(
        ingest_status=ist.INDEXED,
        rag_indexed=4,
        rag_error=None,
        prior_extractor_version=sentinel,
    )


def test_reextract_terminates_embed_success_always_closes():
    """The terminating invariant that the 8-PR loop lacked: once the embed
    lands (rag_indexed>0, no rag_error) a re-extracted .docx closes for EVERY
    prior version; only a real embed failure keeps it open for infra retry."""
    sentinel = ist.embed_failed_sentinel()
    for prior in (None, "pre-sdt", sentinel, ist.EXTRACTOR_VERSION):
        for status in (ist.INDEXED, ist.TEXT_SPARSE):
            # embed success -> advance -> closed
            assert ist.should_advance_extractor_version(
                ingest_status=status, rag_indexed=3, rag_error=None,
                prior_extractor_version=prior,
            ), (status, prior)
            final = ist.EXTRACTOR_VERSION  # advanced
            assert not ist.docx_stale_extractor_open(
                status, extension=".docx", extractor_version=final,
            ), (status, prior)
        # embed FAILURE -> stays open for retry
        assert not ist.should_advance_extractor_version(
            ingest_status=ist.TEXT_SPARSE, rag_indexed=0, rag_error=None,
            prior_extractor_version=prior,
        )
        assert not ist.should_advance_extractor_version(
            ingest_status=ist.TEXT_SPARSE, rag_indexed=1, rag_error="HF offline",
            prior_extractor_version=prior,
        )


def test_stale_extractor_open_without_retry_is_incomplete():
    """The gate that would have caught runs 1–7."""
    assert ist.stale_extractor_run_complete(
        stale_extractor_open=54, retried=0, otherwise_complete=True,
    ) is False
    assert ist.stale_extractor_run_complete(
        stale_extractor_open=54, retried=12, otherwise_complete=True,
    ) is True
    assert ist.stale_extractor_run_complete(
        stale_extractor_open=0, retried=0, otherwise_complete=True,
    ) is True
    assert ist.stale_extractor_run_complete(
        stale_extractor_open=0, retried=0, otherwise_complete=False,
    ) is False
