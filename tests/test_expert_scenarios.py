"""Tests for scripts.generate_expert_scenarios — deterministic Q&A pairs
derived from app/prompts/construction_expert.txt.

The key property: every response is a substring of the source file (no
LLM paraphrasing), tags trace back to the section the answer came from,
and the output is reproducible byte-for-byte across runs.
"""
from __future__ import annotations

import re

from scripts.generate_expert_scenarios import (
    SOURCE_FILE,
    _all_kinds_in_source,
    gen_critical_rules,
    gen_document_numbering,
    gen_formulas_and_timeframes,
    gen_prc_procedures,
    gen_raci_matrix,
    generate_all,
    load_source_text,
)


# ── Schema invariant ──────────────────────────────────────────────────────


def test_every_row_has_required_keys():
    rows = generate_all()
    assert rows, "generator produced no rows"
    for r in rows:
        assert set(r.keys()) == {"instruction", "response", "source"}, r
        assert r["instruction"].strip(), f"empty instruction: {r}"
        assert r["response"].strip(), f"empty response: {r}"
        assert r["source"].startswith("construction_expert.txt:"), r


# ── Total in target band ──────────────────────────────────────────────────


def test_total_meets_target():
    rows = generate_all()
    assert 350 <= len(rows) <= 600, f"got {len(rows)} rows, target band 350-600"


# ── Determinism ───────────────────────────────────────────────────────────


def test_generator_is_deterministic():
    a = generate_all()
    b = generate_all()
    assert a == b, "generator output is not deterministic — answers must be reproducible"


# ── Critical-rule sentences quote the source verbatim ─────────────────────


def test_critical_rules_quote_source_verbatim():
    source = load_source_text()
    # Pick a load-bearing single-line rule from the file.
    target = (
        'NEVER use the word "approved" for design documents. '
        'The correct terms are "accepted" or "for comment".'
    )
    assert target in source, "sanity: chosen rule sentence is not actually in the source"
    rows = list(gen_critical_rules())
    assert any(target in r["response"] for r in rows), (
        "expected the verbatim NEVER-use-approved sentence to appear in at least one "
        "gen_critical_rules row's response"
    )
    # Spot-check a second rule from the explicit CRITICAL RULES block.
    target2 = "An RFM is NOT a Variation Order."
    assert target2 in source
    assert any(target2 in r["response"] for r in rows), (
        f"expected verbatim '{target2}' in some critical-rules response"
    )


# ── Procedure coverage: every procedure-kind section appears in some tag ──


def test_prc_procedures_cover_all_mentions():
    """Every procedure section of the prompt (tagged by catalogue kind --
    the prompt ships no procedure codes) has rows tagged with its kind."""
    prcs = _all_kinds_in_source()
    from app.core.procedure_catalogue import procedures

    assert set(prcs) == set(procedures()), f"every catalogue kind needs its section: {prcs}"
    rows = generate_all()
    missing = []
    for prc in prcs:
        # Match the kind as a token in the source tag (anchor on ":" or end).
        pattern = re.compile(rf"(?:^|:){re.escape(prc)}(?::|$)")
        if not any(pattern.search(r["source"]) for r in rows):
            missing.append(prc)
    assert not missing, f"procedure kinds with no source-tag coverage: {missing}"


# ── Per-generator floor checks ────────────────────────────────────────────


def test_prc_procedures_floor():
    rows = list(gen_prc_procedures())
    assert len(rows) >= 150, f"prc_procedures produced {len(rows)} rows, expected >=150"


def test_critical_rules_floor():
    rows = list(gen_critical_rules())
    assert len(rows) >= 60, f"critical_rules produced {len(rows)} rows, expected >=60"


def test_document_numbering_floor():
    rows = list(gen_document_numbering())
    assert len(rows) >= 40, f"document_numbering produced {len(rows)} rows, expected >=40"


def test_raci_matrix_floor():
    rows = list(gen_raci_matrix())
    assert len(rows) >= 80, f"raci_matrix produced {len(rows)} rows, expected >=80"


def test_formulas_and_timeframes_floor():
    rows = list(gen_formulas_and_timeframes())
    assert len(rows) >= 40, f"formulas_and_timeframes produced {len(rows)} rows, expected >=40"


# ── Document numbering: every documented doc type is covered ──────────────


def test_document_numbering_covers_each_doc_type():
    rows = list(gen_document_numbering())
    # Responses are the raw bullet lines, which always contain the
    # short code (e.g. "VO-015", "DD-023") even when the bullet uses
    # the long-form name ("Variation Order", "Design Directive").
    text = " ".join(r["response"] for r in rows)
    for code in ("RFI", "NCR", "IR", "VO", "RFM", "JR", "PDN", "DD"):
        assert code in text, f"doc type {code} missing from gen_document_numbering responses"


# ── Spot-check: timeframes include the "7 calendar days" rule ─────────────


def test_timeframes_include_seven_calendar_days():
    rows = list(gen_formulas_and_timeframes())
    text = " ".join(r["response"] for r in rows)
    assert "7 calendar days" in text, (
        "expected the design-review minimum-distribution timeframe to appear verbatim "
        "in some formulas_and_timeframes response"
    )


# ── Responses are substrings of the source file (modulo wrapped lines) ────
#
# For bullets that are NOT wrapped, the response should be a literal
# substring of the source. We check this loosely — at least 80% of
# critical-rule responses must appear verbatim in the source. Wrapped
# bullets (joined with a single space) won't match raw file lines but
# their constituent words will, so the verbatim claim still holds at the
# sentence level for non-wrapped ones.


def test_majority_of_responses_are_verbatim_in_source():
    source = load_source_text()
    rows = list(gen_critical_rules())
    verbatim = sum(1 for r in rows if r["response"] in source)
    assert verbatim >= int(0.8 * len(rows)), (
        f"only {verbatim}/{len(rows)} critical-rule responses are verbatim substrings "
        "of the source; this is the no-paraphrase guarantee — investigate"
    )
