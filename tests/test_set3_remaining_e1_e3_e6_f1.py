"""Set3 remaining live misses on tip 4ab5561 (after #643/#649/#651).

Live standing-order (uploads/set3_fails_4ab5561.json):

    E1  "Calculate the Advance Payment in SAR."
        shipped the internal tool-truncation notice; expect ~175,450,445.6
        (10% × excl-VAT ACA 1,754,504,456.25). A7 already states 10%.
    E3  "If Milestones 3 and 4 are each 20 days late, combined milestone
        delay damages in SAR?"
        used whole-of-Works 0.1% → 70,180,178.25; expect ~10,527,026/027
        (Contract Data milestone rate 0.015% × ACA × 20 × 2).
    F1  "Among the Northern Community milestones, which have the longest
        Time for Completion and by how much do they exceed the shortest?"
        answered Milestones 1–5 only; expect days 547 and 150.
    E6  combined Part Summary of d/3/1 + d/3/2 + d/3/3
        summed 34,645,529 + 1,852,848 + 17,496,857 as 54,995,234;
        the printed page totals add to 53,995,234.

Class, not one-row special cases. Do not touch parked D1. Do not revert
#649 plastering/Y16 or #651 empty-graft paths — extend them.
"""
from __future__ import annotations

import json
import re

import pytest

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _graft_composed_delay_damages_over_period,
    _graft_composed_percentage_of_aca,
    _postprocess_answer,
    _text_needs_tool_recovery,
)
from app.core.rag.vector_store import Chunk
from app.lib.construction_formulas_commercial import (
    compose_delay_damages_over_period_from_excerpts,
    compose_percentage_of_aca_from_excerpts,
    parse_milestone_delay_rate_percent,
    query_asks_delay_damages_over_a_period,
    query_asks_percentage_particular_in_money,
)


# ── Live asks ──────────────────────────────────────────────────────────────

E1 = "Calculate the Advance Payment in SAR."
E3 = (
    "If Milestones 3 and 4 are each 20 days late, what are the "
    "combined milestone delay damages in SAR?"
)
F1 = (
    "Among the Northern Community milestones, which have the longest "
    "Time for Completion and by how much do they exceed the shortest?"
)
E6 = (
    "What is the combined Part Summary total of pages d/3/1, d/3/2 and d/3/3?"
)
A7 = "What is the Advance Payment amount under the contract?"
A2_PLASTER = (
    "How much will the remaining 3,400 m2 of plastering cost if "
    "the gang does 42 m2 per gang-day at SAR 1,950 per gang-day?"
)
A2_Y16 = "What is the weight of 12 tonnes of Y16 bars in metres run?"

ACA_EXCL = "SAR 1,754,504,456.25"
LABEL = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
)
CD_ADVANCE = LABEL + (
    "14.2.1: | | Advance Payment: 10% of the Accepted Contract Amount | |\n"
)
CD_ACA = LABEL + (
    "1.1.1: | | Accepted Contract Amount: "
    f"{ACA_EXCL} excluding VAT |\n"
)
CD_MILESTONE_RATES = LABEL + (
    "8.8.1: | | Delay Damages (if applicable per Milestone): "
    "Milestone | Delay Damages\n"
    "|: | | Milestone 1 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 2 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 3 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 4 | 0.015% of the Contract Price per calendar day\n"
)
CD_WHOLE_WORKS_RATE = LABEL + (
    "8.8.1: | | Delay Damages (for the whole of the Works): "
    "0.1% of the Contract Price per calendar day |\n"
)
# Live E3 shape: the 0.1% whole-of-Works cell sits in the same packed
# "per Milestone" particulars row. First-percent parse elects 0.1%.
CD_PACKED_WHOLE_WORKS_AS_MILESTONE = LABEL + (
    "8.8.1: | | Delay Damages (if applicable per Milestone) "
    "Milestone 3 Milestone 4 | Delay Damages (for the whole of the Works): "
    "0.1% of the Contract Price per calendar day |\n"
)
LIVE_TRUNCATION = (
    "Tool result exceeded 4000 characters and was truncated. Call this tool "
    "again with the SAME file_path and char_offset=3005 to read the next "
    "window, and repeat until chars_remaining is 0. Do that BEFORE telling "
    "the user you could not find something."
)
LIVE_E3_WRONG = (
    "Combined delay damages for Milestone 3 and Milestone 4 (20 days each) "
    "are SAR 70,180,178.25 (0.1%, 0.1% of Accepted Contract Amount "
    "SAR 1,754,504,456.25 × 20 days × 2)."
)
LIVE_E6_WRONG = (
    "The combined Part Summary total of pages d/3/1, d/3/2 and d/3/3 is "
    "**54,995,234.00 SAR**.\n\n"
    "That is the sum of the three page subtotals carried to the Part Summary:\n"
    "- Page d/3/1: 34,645,529.00 SAR\n"
    "- Page d/3/2: 1,852,848.00 SAR\n"
    "- Page d/3/3: 17,496,857.00 SAR\n\n"
    "Total = 34,645,529.00 + 1,852,848.00 + 17,496,857.00 = **54,995,234.00 SAR**"
)
COLLECTION = (
    "PART SUMMARY From Page Nr. d/3/1 34,645,529.00 From Page Nr. d/3/2 "
    "1,852,848.00 From Page Nr. d/3/3 17,496,857.00 From Page Nr. d/3/4 "
    "8,240,875.00 Carried to Grand Summary 99,000,000.00"
)

CD_DOC = "cdreal"
CD_NAME = "XX-2099-001_Vol 1.0_Cond of Contract (complete)_Contract Data.pdf"
ACTIVE = "p_master"

CD_LIST = LABEL + (
    "1.1.50: | | Milestones (if applicable) | Milestone 1 | Southern Community 1a Media\n"
    "|: | | Milestone 5 | Boulevard Community 1d Boulevard South-East\n"
    "|: | | Milestone 7 | Northern Community 1b Civic Quarter\n"
    "|: | | Milestone 8 | Northern Community 1d Northern West\n"
    "|: | | Milestone 9 | Northern Community 1e Northern Central\n"
    "|: | | Milestone 10 | Northern Community 1e Northern East\n"
)
CD_TIMES_FIRST = LABEL + (
    "1.1.75: | | Time for Completion (for the whole of the Works): "
    "852 days from the Commencement Date |\n"
    "1.1.75: | | Time for Completion (by Milestone, if applicable): "
    "Milestone 1 | 397 days from the date the Contractor is given right "
    "of access to Southern Community 1a Media.\n"
    "|: | | Milestone 2 | 549 days from the date the Contractor is given "
    "right of access to Southern Community 1e Southern.\n"
    "|: | | Milestone 3 | 487 days from the date the Contractor is given "
    "right of access to Boulevard Community 1c Boulevard South-West.\n"
    "|: | | Milestone 4 | 487 days from the date the Contractor is given "
    "right of access to Boulevard Community 1c Boulevard North-West.\n"
    "|: | | Milestone 5 | 731 days from the date the Contractor is given "
    "right of access to Boulevard Community 1d Boulevard South-East.\n"
)
# Live continuation: a long repeated header pushed Milestone 6 past the
# 400-char opening window, so enumeration rescue never saw 547 / 397.
_CONT_HEADER = (
    "amended): | Description | Data | |\n"
    "|: | Clause (as amended) | Description | Data |\n"
    "Classification - Public  Page 12 of 46  RFP No. XX-2099-001  "
    "Time for Completion table continued from previous page. "
    "Do not invent durations. "
) * 3
CD_TIMES_CONT = LABEL + _CONT_HEADER + (
    "|: | Milestone 6 | | 640 days from the date the Contractor is given "
    "right of access to East Quarter 3b.\n"
    "|: | Milestone 7 | | 397 days from the date the Contractor is given "
    "right of access to Northern Community 1b Civic Quarter.\n"
    "|: | Milestone 8 | | 547 days from the date the Contractor is given "
    "right of access to Northern Community 1d Northern West.\n"
    "|: | Milestone 9 | | 520 days from the date the Contractor is given "
    "right of access to Northern Community 1e Northern Central.\n"
    "|: | Milestone 10 | | 400 days from the date the Contractor is given "
    "right of access to Northern Community 1e Northern East.\n"
)
LIVE_F1_REFUSE = (
    "I can't answer that from the retrieved excerpts — the Contract Data "
    "table gives durations for Milestones 1–5 only, and none of those are "
    "Northern Community milestones."
)


def _sys(*texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=cd{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def _msgs(ask: str, *extra: dict) -> list[dict]:
    return [{"role": "user", "content": ask}, *extra]


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(
        chunk_id=cid, project_id=ACTIVE, doc_id=doc_id,
        chunk_index=index, text=text, score=score,
    )


# ── E1: never ship truncation; compose 10% × ACA ───────────────────────────


def test_live_truncation_notice_needs_tool_recovery():
    from app.agents.runtime import _looks_like_tool_truncation_notice

    assert _looks_like_tool_truncation_notice(LIVE_TRUNCATION)
    assert _text_needs_tool_recovery(LIVE_TRUNCATION)
    assert not _looks_like_tool_truncation_notice(
        "The Advance Payment is 10% of the Accepted Contract Amount."
    )


def test_e1_truncation_notice_with_both_operands_composes_sar():
    assert query_asks_percentage_particular_in_money(E1)
    rag = _sys(CD_ADVANCE, CD_ACA)
    composed = compose_percentage_of_aca_from_excerpts(E1, rag["content"])
    assert composed is not None
    assert "175,450,445.6" in f"{composed['amount']:,.2f}"
    out = _postprocess_answer(LIVE_TRUNCATION, rag, _msgs(E1))
    assert "175,450,445.6" in out
    assert "Tool result exceeded" not in out
    assert "char_offset=" not in out
    assert "chars_remaining" not in out


def test_e1_truncation_notice_last_chance_scans_loaded_cd(monkeypatch):
    """Live E1: top-k had neither 10% nor ACA; the rows sit later in the volume."""
    volume = "\n\n".join((CD_ADVANCE, CD_ACA))
    monkeypatch.setattr(
        "app.core.rag.retriever.percentage_of_aca_excerpts_from_loaded_cd_volume",
        lambda *a, **k: volume,
    )
    rag = _sys(CD_WHOLE_WORKS_RATE)
    assert compose_percentage_of_aca_from_excerpts(E1, rag["content"]) is None
    out = _postprocess_answer(LIVE_TRUNCATION, rag, _msgs(E1), project_id=ACTIVE)
    assert "175,450,445.6" in out
    assert "Tool result exceeded" not in out


def test_a7_still_states_10_percent_not_the_sar_product():
    rag = _sys(CD_ADVANCE, CD_ACA)
    out = _postprocess_answer("", rag, _msgs(A7))
    assert re.search(r"\b10\s*%", out)
    assert "175,450,445" not in out


def test_e1_graft_does_not_steal_plastering_or_y16():
    rag = _sys(CD_ADVANCE, CD_ACA)
    plaster = _graft_composed_percentage_of_aca(
        "SAR 157,857.14 for 80.95 gang-days.", rag, _msgs(A2_PLASTER),
    )
    assert "157,857.14" in plaster
    assert "175,450,445" not in plaster
    y16 = _graft_composed_percentage_of_aca(
        "Y16 unit mass is 1.578 kg/m; 12 t is 7,602.94 m.", rag, _msgs(A2_Y16),
    )
    assert "7,602.94" in y16
    assert "175,450,445" not in y16


# ── E3: milestone 0.015%, never whole-of-Works 0.1% ────────────────────────


def test_packed_whole_works_row_is_not_a_milestone_rate():
    """Live 4ab5561: 0.1% whole-of-Works packed under 'per Milestone'."""
    blob = CD_PACKED_WHOLE_WORKS_AS_MILESTONE + "\n" + CD_ACA
    assert parse_milestone_delay_rate_percent(blob, 3) is None
    assert parse_milestone_delay_rate_percent(blob, 4) is None
    assert compose_delay_damages_over_period_from_excerpts(E3, blob) is None


def test_e3_prefers_milestone_015_over_whole_works_01():
    rag = _sys(CD_WHOLE_WORKS_RATE, CD_MILESTONE_RATES, CD_ACA)
    composed = compose_delay_damages_over_period_from_excerpts(E3, rag["content"])
    assert composed is not None
    assert abs(composed["amount"] - 10_527_026.74) < 0.5
    assert all(abs(r - 0.015) < 1e-9 for r in composed["rates"])
    out = _postprocess_answer(LIVE_E3_WRONG, rag, _msgs(E3))
    assert re.search(r"10,527,02[67]", out)
    assert "70,180,178" not in out


def test_e3_last_chance_scans_milestone_rate_from_loaded_cd(monkeypatch):
    volume = "\n\n".join((CD_MILESTONE_RATES, CD_ACA))
    monkeypatch.setattr(
        "app.core.rag.retriever.milestone_period_excerpts_from_loaded_cd_volume",
        lambda *a, **k: volume,
    )
    rag = _sys(CD_WHOLE_WORKS_RATE, CD_ACA)
    assert compose_delay_damages_over_period_from_excerpts(E3, rag["content"]) is None
    out = _postprocess_answer(LIVE_E3_WRONG, rag, _msgs(E3), project_id=ACTIVE)
    assert re.search(r"10,527,02[67]", out)
    assert "70,180,178" not in out


def test_e3_does_not_steal_whole_works_rate_lookup():
    from app.lib.construction_formulas_commercial import (
        compose_delay_damages_over_period_from_excerpts as compose,
    )
    a5 = "What are the Delay Damages for the whole of the Works?"
    rag = _sys(CD_WHOLE_WORKS_RATE, CD_ACA)
    assert not query_asks_delay_damages_over_a_period(a5)
    assert compose(a5, rag["content"]) is None
    grafted = _graft_composed_delay_damages_over_period(
        "Delay damages for the whole of the Works are 0.1% of the Contract Price.",
        rag, _msgs(a5),
    )
    assert "10,527,02" not in grafted
    assert "70,180,178" not in grafted


# ── F1: Northern Community 547 / 150 ───────────────────────────────────────


def _f1_sheet(monkeypatch):
    chunks = [
        _chunk("list", CD_DOC, 0.0, CD_LIST, 0),
        _chunk("times", CD_DOC, 0.0, CD_TIMES_FIRST, 1),
        _chunk("cont", CD_DOC, 0.0, CD_TIMES_CONT, 2),
        _chunk("other", CD_DOC, 0.0, CD_ADVANCE, 3),
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: chunks,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.search",
        lambda self, pid, qvec, k, query_text=None: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search",
        lambda self, pid, identifiers, k=20: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        lambda self, pid, needles, k=20, **_kw: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count",
        lambda self, pid=None: 11,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(
        "app.core.rag.retriever._doc_name_for_id",
        lambda did: CD_NAME, raising=False,
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: (
            [{"id": CD_DOC, "original_name": CD_NAME, "file_path": ""}]
            if "contract data" in (phrase or "").lower() else []
        ),
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return chunks


def test_f1_retrieves_northern_community_547_and_397(monkeypatch):
    from app.core.rag import retriever as ret

    _f1_sheet(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(F1, ACTIVE, k=5)
    blob = "\n".join(c.text for c in chunks)
    assert "547 days" in blob
    assert "Northern Community" in blob
    assert re.search(r"\b397\s*days", blob)


def test_f1_composes_longest_547_exceeds_shortest_by_150():
    from app.core.rag.retriever import compose_named_community_tfc_span

    excerpts = "\n\n".join((CD_LIST, CD_TIMES_FIRST, CD_TIMES_CONT))
    composed = compose_named_community_tfc_span(F1, excerpts)
    assert composed is not None
    assert composed["longest_days"] == 547
    assert composed["shortest_days"] == 397
    assert composed["delta"] == 150
    assert 8 in composed["longest_milestones"]


def test_f1_last_chance_scans_northern_days_from_loaded_cd(monkeypatch):
    """Live 4ab5561: top-k stopped at Milestone 5; 547 sits later in the volume."""
    from app.core.rag.retriever import compose_named_community_tfc_span

    volume = "\n\n".join((CD_LIST, CD_TIMES_FIRST, CD_TIMES_CONT))
    monkeypatch.setattr(
        "app.core.rag.retriever.community_tfc_span_excerpts_from_loaded_cd_volume",
        lambda *a, **k: volume,
    )
    rag = _sys(CD_LIST, CD_TIMES_FIRST)
    assert compose_named_community_tfc_span(F1, rag["content"]) is None
    out = _postprocess_answer(LIVE_F1_REFUSE, rag, _msgs(F1), project_id=ACTIVE)
    assert re.search(r"\b547\b", out)
    assert re.search(r"\b150\b", out)
    assert "Milestones 1–5 only" not in out


def test_f1_empty_or_refuse_grafts_547_and_150():
    from app.core.rag.retriever import format_named_community_tfc_span_line

    rag = _sys(CD_LIST, CD_TIMES_FIRST, CD_TIMES_CONT)
    out = _postprocess_answer(LIVE_F1_REFUSE, rag, _msgs(F1))
    assert re.search(r"\b547\b", out)
    assert re.search(r"\b150\b", out)
    assert "Milestones 1–5 only" not in out
    line = format_named_community_tfc_span_line(
        {
            "community": "Northern Community",
            "longest_days": 547,
            "shortest_days": 397,
            "delta": 150,
            "longest_milestones": (8,),
            "shortest_milestones": (7,),
        }
    )
    assert "547" in line and "150" in line


# ── E6: printed page totals add to 53,995,234 ──────────────────────────────


def test_collection_page_figures_add_to_53995234_not_54995234():
    """Corpus truth: the three printed totals sum to 53,995,234.

    Live added 1,000,000. 34,645,529 + 1,852,848 + 17,496,857 = 53,995,234.
    """
    assert 34_645_529 + 1_852_848 + 17_496_857 == 53_995_234
    assert 34_645_529 + 1_852_848 + 17_496_857 != 54_995_234


def test_e6_combined_compose_sums_printed_page_totals():
    from app.core.rag.retriever import (
        compose_combined_part_summary_total,
        query_asks_combined_part_summary,
    )

    assert query_asks_combined_part_summary(E6)
    parsed = compose_combined_part_summary_total(E6, COLLECTION)
    assert parsed is not None
    assert parsed["amount"] == 53_995_234.0
    assert sorted(parsed["pages"]) == ["d/3/1", "d/3/2", "d/3/3"]


def test_e6_replaces_the_million_off_sum():
    rag = {
        "role": "system",
        "content": "Reference context:\n[doc_id=boq chunk=6 score=0.80] " + COLLECTION,
    }
    out = _postprocess_answer(LIVE_E6_WRONG, rag, _msgs(E6))
    assert "53,995,234" in out
    assert "54,995,234" not in out


def test_e6_does_not_steal_one_page_b3():
    from app.core.rag.retriever import (
        compose_combined_part_summary_total,
        compose_part_summary_total,
        query_asks_combined_part_summary,
        query_asks_for_part_summary_total,
    )

    b3 = (
        "What is the Part Summary total for page d/3/1 of the "
        "Demolition and Site Clearance bill?"
    )
    assert query_asks_for_part_summary_total(b3)
    assert not query_asks_combined_part_summary(b3)
    assert compose_combined_part_summary_total(b3, COLLECTION) is None
    one = compose_part_summary_total(b3, COLLECTION)
    assert one and one["amount"] == 34_645_529.0
