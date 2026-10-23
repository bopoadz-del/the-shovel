"""Leftover F1: refuse + AB-2022 CoC cite is FAIL, not a soft PASS.

Live Master Corpus (theshovel.ai c922f78), signed-in pack ask::

    Answer only from the client project documents. Generate a high-level
    WBS for the demolition and site clearance scope in this project's BOQ.

Observed FAIL: the assistant refuses to produce the WBS and cites
AB-2022-202 Conditions of Contract narrative (hard flag
``ab2022_wrong_contract``). Soft ``must_any: clearance|trees|pavement``
marks PASS when the refuse merely mentions demolition.

These tests pin the election, the glass, and the judge (no live LLM).
B4/B5/E1/G4 stay off this path. Kill-switch ``F1_BOQ_WBS_COMPOSE=0``
restores the LLM hop after a BOQ-derived predispatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.runtime import (
    _apply_rag_context,
    _compose_boq_scope_wbs_answer,
    _forced_specific_tool,
    _graft_boq_scope_wbs_if_wrong_contract,
    _inject_predispatch,
    _message_wants_first_run_wbs,
    _postprocess_answer,
    _predispatch_remaining_deliverables,
)
from app.containers.construction import ConstructionContainer
from tests.conftest import requires_construction_kit
from app.core.predefined_reasoning import message_wants_boq_scope_wbs
from app.lib.boq_schedule import (
    boq_wbs_answer_fails_wrong_contract,
    boq_wbs_answer_is_grounded,
    retrieve_boq_scope_items,
)
from tests.test_leftover_f1_boq_wbs import (
    DD22_COC_NAME,
    DD23_BOQ_NAME,
    F1_ASK,
    LIVE_PREFIX,
    S2_BOQ,
    _BOQ_ITEMS,
    _CONTRACT_PROSE,
    _install_boq_pool,
)

REFUSE_DD22 = (
    "I cannot produce a demolition and site clearance WBS from the "
    "retrieved excerpts. Volume 1 of AB-2022-202 (Conditions of Contract) "
    "describes the Contractor's obligation to execute the demolition and "
    "site clearance scope in the Bill of Quantities, but those excerpts "
    "do not contain a work breakdown structure."
)
MENTION_ONLY = (
    "The demolition and site clearance scope is described in the "
    "contract. I don't have a structured WBS to show."
)
GROUNDED = (
    "High-level WBS\n"
    "### 1 Demolition and Site Clearance\n"
    "1.1 D110 General site clearance\n"
    "1.2 D290.1 Removal of trees in existing sidewalks\n"
    "1.3 D599.5 Breaking out existing carriageway\n"
)


def test_soft_mention_of_demolition_is_not_a_pass():
    """The live soft judge (must_any: clearance) would mark this PASS."""
    assert "clearance" in REFUSE_DD22.lower()
    assert "demolition" in REFUSE_DD22.lower()
    assert boq_wbs_answer_fails_wrong_contract(REFUSE_DD22)
    assert not boq_wbs_answer_is_grounded(REFUSE_DD22)
    assert boq_wbs_answer_fails_wrong_contract(MENTION_ONLY)
    assert not boq_wbs_answer_is_grounded(MENTION_ONLY)
    assert "clearance" in MENTION_ONLY.lower()


def test_ab2022_cite_without_refuse_is_still_wrong_contract():
    cite = (
        "Per AB-2022-202 the Contractor shall execute the demolition "
        "and site clearance scope described in the Bill of Quantities."
    )
    assert boq_wbs_answer_fails_wrong_contract(cite)
    assert not boq_wbs_answer_is_grounded(cite)


def test_grounded_boq_wbs_is_pass_and_not_wrong_contract():
    assert boq_wbs_answer_is_grounded(GROUNDED)
    assert not boq_wbs_answer_fails_wrong_contract(GROUNDED)
    assert "AB-2022" not in GROUNDED
    assert "template scaffold" not in GROUNDED.lower()


def test_building_template_is_not_grounded():
    scaffold = (
        "Template scaffold, project_type inferred: building — not "
        "derived from this project's BOQ.\n"
        "### 1 Site Preparation\n"
        "### 2 Superstructure\n"
        "Schedule built: 204 activities"
    )
    assert not boq_wbs_answer_is_grounded(scaffold)


def test_f1_ask_elects_first_run_wbs_and_forces_generate_wbs():
    assert message_wants_boq_scope_wbs(F1_ASK)
    assert _message_wants_first_run_wbs(F1_ASK)
    assert _forced_specific_tool(
        [{"role": "user", "content": F1_ASK}],
        {"generate_wbs", "construction_calc"},
    ) == "generate_wbs"


def test_b4_b5_e1_g4_do_not_elect_f1_force():
    from tests.test_b4_priced_boq_compose import LIVE_B4, LIVE_B5, LIVE_G4
    from tests.test_e1_delay_damages_daily_compose import LIVE_E1

    for ask in (LIVE_B4, LIVE_B5, LIVE_E1, LIVE_G4):
        assert not message_wants_boq_scope_wbs(ask), ask
        assert not _message_wants_first_run_wbs(ask), ask
        assert _forced_specific_tool(
            [{"role": "user", "content": ask}],
            {"generate_wbs", "construction_calc"},
        ) != "generate_wbs"


def test_f1_rag_directive_forbids_refuse_and_ab2022():
    msgs = [{"role": "user", "content": F1_ASK}]
    rag = {
        "content": (
            "REFERENCE CONTEXT\n"
            f"{DD22_COC_NAME}\n{_CONTRACT_PROSE}\n"
        ),
    }
    assert _apply_rag_context(msgs, rag) is True
    folded = msgs[-1]["content"]
    assert "Do NOT refuse" in folded
    assert "another contract year's Conditions of Contract" in folded
    assert "using ONLY the reference context" not in folded
    assert F1_ASK in folded


def test_graft_replaces_refuse_ab2022_with_predispatched_wbs():
    from app.agents.runtime import _format_wbs_result

    rendered = _format_wbs_result({
        "actual_count": 4,
        "project_type": "demolition_site_clearance",
        "summary": {"activity_count": 4},
        "wbs_tree": {
            "1": "Demolition and Site Clearance",
            "1.1": "D110 General site clearance",
            "1.2": "D290.1 Removal of trees",
        },
        "scaffold": {
            "derived_from_boq": True,
            "declaration": (
                "High-level WBS derived from this project's BOQ "
                "demolition and site clearance items."
            ),
        },
    })
    msgs = [
        {"role": "user", "content": F1_ASK},
    ]
    _inject_predispatch(
        msgs, "generate_wbs", rendered,
        "Present this WBS in full. Do not refuse.",
    )
    out = _graft_boq_scope_wbs_if_wrong_contract(REFUSE_DD22, msgs)
    assert boq_wbs_answer_is_grounded(out)
    assert not boq_wbs_answer_fails_wrong_contract(out)
    assert "AB-2022" not in out
    assert "D110" in out
    post = _postprocess_answer(REFUSE_DD22, {"content": _CONTRACT_PROSE}, msgs)
    assert boq_wbs_answer_is_grounded(post)
    assert "AB-2022" not in post


def test_compose_short_circuit_requires_boq_derived_predispatch():
    pre = {
        "name": "generate_wbs",
        "result": {
            "status": "success",
            "actual_count": 2,
            "project_type": "demolition_site_clearance",
            "summary": {"activity_count": 2},
            "wbs_tree": {
                "1": "Demolition and Site Clearance",
                "1.1": "D110 General site clearance",
            },
            "scaffold": {
                "derived_from_boq": True,
                "declaration": "High-level WBS derived from this project's BOQ.",
            },
        },
    }
    out = _compose_boq_scope_wbs_answer(pre, F1_ASK)
    assert boq_wbs_answer_is_grounded(out)
    assert "D110" in out
    template = {
        "name": "generate_wbs",
        "result": {
            "status": "success",
            "scaffold": {"derived_from_boq": False, "declaration": "Template scaffold"},
            "wbs_tree": {"1": "Site Preparation"},
        },
    }
    assert _compose_boq_scope_wbs_answer(template, F1_ASK) == ""
    assert _compose_boq_scope_wbs_answer(pre, "produce the schedule") == ""


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_passes_project_id_and_elects_boq(monkeypatch):
    seen: dict = {}

    async def fake_generate_wbs(self, input_data, params):
        seen["params"] = dict(params or {})
        seen["data"] = dict(input_data or {})
        return {
            "status": "success",
            "actual_count": 2,
            "activities": [
                {"id": "1.1", "name": "D110 General site clearance",
                 "duration_days": 1, "predecessors": []},
            ],
            "wbs_tree": {
                "1": "Demolition and Site Clearance",
                "1.1": "D110 General site clearance",
            },
            "summary": {"activity_count": 2},
            "scaffold": {
                "source": "boq",
                "derived_from_boq": True,
                "declaration": (
                    "High-level WBS derived from this project's BOQ "
                    "demolition and site clearance items."
                ),
            },
            "brief": F1_ASK,
        }

    monkeypatch.setattr(ConstructionContainer, "generate_wbs", fake_generate_wbs)

    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    msgs = [{"role": "user", "content": F1_ASK}]
    out = await _predispatch_remaining_deliverables(
        _A(), msgs, "master_corpus", operator_text=F1_ASK,
        conversation_id="ws-master_corpus-1",
    )
    assert out is not None
    assert out["name"] == "generate_wbs"
    assert seen["params"].get("project_id") == "master_corpus"
    assert seen["data"].get("project_id") == "master_corpus"
    draft = msgs[-1]["content"]
    assert "D110" in draft
    assert "Template scaffold" not in draft
    assert boq_wbs_answer_is_grounded(_compose_boq_scope_wbs_answer(out, F1_ASK))


@requires_construction_kit
@pytest.mark.asyncio
async def test_predispatch_does_not_inject_building_template_for_f1(monkeypatch):
    async def fake_generate_wbs(self, input_data, params):
        return {
            "status": "success",
            "actual_count": 204,
            "activities": [{"id": "A1", "name": "Hall A", "duration_days": 5}],
            "wbs_tree": {"1": "Site Preparation", "2": "Superstructure"},
            "scaffold": {
                "source": "template",
                "derived_from_boq": False,
                "declaration": (
                    "Template scaffold, project_type inferred: building "
                    "— not derived from this project's BOQ"
                ),
            },
            "brief": F1_ASK,
        }

    monkeypatch.setattr(ConstructionContainer, "generate_wbs", fake_generate_wbs)

    class _A:
        allowed_blocks = ["construction"]
        name = "construction-pm"

    msgs = [{"role": "user", "content": F1_ASK}]
    out = await _predispatch_remaining_deliverables(
        _A(), msgs, "master_corpus", operator_text=F1_ASK,
    )
    assert out is None
    assert len(msgs) == 1


def test_newer_year_boq_wins_over_ab2022_bill(monkeypatch):
    """Both years' demolition bills in the pool — unnamed F1 takes the later one."""
    from app.core.rag.vector_store import Chunk
    from app.lib import boq_schedule as bs

    dd22_rows = (
        "Demolition and Site Clearance. Page d/3/1.\n"
        "D110 General site clearance 12 ha\n"
        "D199 Fake earlier-year only item 1 Nr\n"
    )
    texts = {"boq23": [S2_BOQ], "boq22": [dd22_rows]}
    docs = [
        {
            "id": "boq22",
            "original_name": "AB-2022-202 - Demolition and Site Clearance BOQ.pdf",
            "file_path": "/docs/AB-2022-202 - Demolition and Site Clearance BOQ.pdf",
        },
        {
            "id": "boq23",
            "original_name": DD23_BOQ_NAME,
            "file_path": f"/docs/{DD23_BOQ_NAME}",
        },
    ]

    def fake_title(pid, phrase, limit=8):
        needle = (phrase or "").lower()
        return [
            doc for doc in docs
            if needle and needle in (
                f"{doc['original_name']} {doc['file_path']}".lower()
            )
        ][:limit]

    def fake_terms(pid, terms, min_terms=2, require_letter=False, limit=8):
        return list(docs)[:limit]

    class _Store:
        def doc_chunk_texts(self, project_id, doc_ids):
            return {did: list(texts.get(did) or []) for did in (doc_ids or [])}

        def chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
            return []

    coc = Chunk(
        chunk_id="coc1",
        project_id="drive_archive",
        doc_id="coc1",
        chunk_index=0,
        text=_CONTRACT_PROSE,
    )
    coc.source_name = DD22_COC_NAME

    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase", fake_title,
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms", fake_terms,
    )
    monkeypatch.setattr(bs, "_resolve_rag_project_id", lambda pid: pid)
    monkeypatch.setattr(
        "app.core.rag.vector_store.get_store", lambda **_k: _Store(),
    )
    monkeypatch.setattr(
        "app.core.rag.retriever.retrieve_with_filter",
        lambda *a, **k: ([coc], 0),
    )
    monkeypatch.setattr(
        "app.core.rag.retriever._doc_name_for_id",
        lambda did: {
            "boq23": DD23_BOQ_NAME,
            "boq22": "AB-2022-202 - Demolition and Site Clearance BOQ.pdf",
        }.get(did, DD22_COC_NAME),
    )
    items = retrieve_boq_scope_items(F1_ASK, "master_corpus")
    codes = {i.get("item_key") for i in items}
    sources = " ".join(str(i.get("source") or "") for i in items)
    assert "D110" in codes
    assert "D290.1" in codes
    assert "D199" not in codes
    assert "AB-2022" not in sources
    assert "AB-2023-101" in sources


@pytest.mark.asyncio
async def test_generate_wbs_f1_answer_has_no_ab2022_cite(monkeypatch):
    _install_boq_pool(monkeypatch)
    r = await ConstructionContainer().generate_wbs(
        {"brief": F1_ASK},
        {"project_id": "master_corpus", "target_count": 200},
    )
    from app.agents.runtime import _format_wbs_result
    out = _format_wbs_result(r)
    assert r["scaffold"]["derived_from_boq"] is True
    assert boq_wbs_answer_is_grounded(out)
    assert not boq_wbs_answer_fails_wrong_contract(out)
    assert "AB-2022" not in out
    assert "AB2022" not in out
    assert "Template scaffold" not in out
    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
        .read_text(encoding="utf-8")
    )
    for banned in catalog["cases"]["F1"].get("must_not") or []:
        assert banned.lower() not in out.lower(), banned
    assert any(
        tok.lower() in out.lower()
        for tok in catalog["cases"]["F1"]["must_any"]
    )
