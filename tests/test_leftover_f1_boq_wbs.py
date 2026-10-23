"""Leftover F1 beyond #521: WBS from BOQ demolition / site-clearance rows.

Live leftover on Master Corpus (f785682 / #521):

    "Answer only from the client project documents. Generate a high-level
     WBS for the demolition and site clearance scope in this project's BOQ."

    Actual FAIL: "Template scaffold, project_type inferred: building —
    not derived from this project's BOQ…" then Site Preparation /
    Substructure / Superstructure. No citations. Hierarchy from #521
    printed, but the tree was still the building template.

These tests pin the election and the glass (no live LLM). A2/A3/A5/A9/
C1/E1 stay off this path. Kill-switch ``BOQ_SCOPE_WBS=0`` restores the
template scaffold.

#524 still FAILED live (592199c): generate_wbs only parsed
``retrieve_with_filter`` top-k, which on Master Corpus is Conditions of
Contract prose. Measured D-rows sat in the document pool and were never
read. The pins below fail if that happens again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.runtime import _format_wbs_result
from app.containers.construction import ConstructionContainer
from app.core.predefined_reasoning import (
    deliver_schedule,
    format_wbs_outline,
    message_wants_boq_scope_wbs,
    message_wants_wbs_outline,
    run_workflow,
)
from app.lib.boq_schedule import (
    activities_from_boq_scope_outline,
    filter_demolition_site_clearance_items,
    parse_boq_measured_rows,
    retrieve_boq_scope_items,
    wbs_tree_from_boq_items,
)
from app.schemas.project_session import ProjectSession

CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
LIVE_PREFIX = "Answer only from the client project documents. "
F1_ASK = LIVE_PREFIX + CATALOG["cases"]["F1"]["ask"]
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
A9_ASK = CATALOG["cases"]["A9"]["ask"]
C1_ASK = CATALOG["cases"]["C1"]["ask"]
E1_ASK = CATALOG["cases"]["E1"]["ask"]
SCHEDULE_ONLY = "produce the schedule"

S2_BOQ = (Path(__file__).parent / "fixtures" / "ui_phys" / "S2_demolition_boq.md").read_text(
    encoding="utf-8"
)

_BOQ_ITEMS = [
    {"item_key": "D110", "description": "General site clearance", "source": "S2_demolition_boq.md"},
    {"item_key": "D290.1", "description": "Removal of trees in existing sidewalks"},
    {"item_key": "D549.2", "description": "Removal of existing chain link fence"},
    {"item_key": "D599.5", "description": "Breaking out existing carriageway including road markings"},
]


def test_f1_elects_boq_scope_wbs_and_plain_schedule_does_not():
    assert message_wants_wbs_outline(F1_ASK)
    assert message_wants_boq_scope_wbs(F1_ASK)
    assert message_wants_boq_scope_wbs(CATALOG["cases"]["F1"]["ask"])
    assert not message_wants_boq_scope_wbs(SCHEDULE_ONLY)
    assert not message_wants_boq_scope_wbs("Generate a WBS for a 10-storey tower")
    assert not message_wants_boq_scope_wbs(
        "Generate a high-level WBS for the demolition and site clearance scope"
    )


def test_a2_a3_a5_a9_c1_e1_do_not_elect_boq_scope_wbs():
    """Contract Data / spec-precedence / delay-damages asks stay on RAG."""
    from app.core.rag.retriever import (
        query_asks_for_accepted_contract_amount,
        query_asks_for_delay_damages_rate,
        query_asks_for_time_for_completion,
        query_needs_a_monetary_base,
    )

    for ask in (A2_ASK, A3_ASK, A5_ASK, A9_ASK, C1_ASK, E1_ASK, LIVE_PREFIX + E1_ASK):
        assert not message_wants_boq_scope_wbs(ask), ask
        assert not message_wants_wbs_outline(ask), ask

    assert query_asks_for_accepted_contract_amount(A2_ASK)
    assert query_asks_for_time_for_completion(A3_ASK)
    assert query_asks_for_delay_damages_rate(A5_ASK)
    assert query_needs_a_monetary_base(LIVE_PREFIX + E1_ASK) or query_needs_a_monetary_base(
        "Calculate the delay damages per calendar day in SAR for the whole of the Works."
    )


def test_parse_s2_fixture_rows_does_not_invent():
    rows = parse_boq_measured_rows(S2_BOQ)
    codes = [r.get("item_key") for r in rows]
    assert "D110" in codes
    assert "D290.1" in codes
    assert "D549.2" in codes
    assert "D599.5" in codes
    assert codes.count("D529.3") == 1
    descs = " ".join(r.get("description") or "" for r in rows).lower()
    assert "site clearance" in descs
    assert "trees" in descs
    assert "superstructure" not in descs
    assert "hall a" not in descs
    assert "no quantity is priced" not in descs
    assert not any("part summary" in (r.get("description") or "").lower() for r in rows)


def test_filter_drops_non_demolition_rows():
    mixed = _BOQ_ITEMS + [
        {"item_key": "C20", "description": "Concrete to columns"},
        {"description": "Hall A slab"},
    ]
    scoped = filter_demolition_site_clearance_items(mixed)
    codes = {r.get("item_key") for r in scoped}
    assert codes == {"D110", "D290.1", "D549.2", "D599.5"}
    tree = wbs_tree_from_boq_items(mixed)
    blob = " ".join(tree.values()).lower()
    assert "d110" in blob and "site clearance" in blob
    assert "superstructure" not in blob
    assert "hall a" not in blob
    assert "concrete to columns" not in blob


def test_tree_and_activities_are_only_retrieved_rows():
    tree = wbs_tree_from_boq_items(_BOQ_ITEMS)
    assert tree["1"] == "Demolition and Site Clearance"
    assert "D110" in tree["1.1"]
    acts = activities_from_boq_scope_outline(_BOQ_ITEMS)
    assert [a["boq"]["item_key"] for a in acts] == ["D110", "D290.1", "D549.2", "D599.5"]
    assert wbs_tree_from_boq_items([]) == {}
    assert activities_from_boq_scope_outline([]) == []


@pytest.mark.asyncio
async def test_generate_wbs_f1_ask_uses_boq_items_not_building_template():
    r = await ConstructionContainer().generate_wbs(
        {"brief": F1_ASK, "boq_items": _BOQ_ITEMS},
        {"target_count": 60},
    )
    assert r["status"] == "success"
    assert r["scaffold"]["derived_from_boq"] is True
    assert r["scaffold"]["source"] == "boq"
    assert "template scaffold" not in r["scaffold"]["declaration"].lower()
    blob = " ".join(r["wbs_tree"].values()).lower()
    assert "site clearance" in blob
    assert "d110" in blob
    assert "trees" in blob
    assert "site preparation" not in blob
    assert "superstructure" not in blob
    names = " ".join(a["name"] for a in r["activities"]).lower()
    assert "d110" in names and "hall a" not in names
    out = _format_wbs_result(r)
    assert "Template scaffold" not in out
    assert "Template activities:" not in out
    assert "BOQ-derived items:" in out
    assert "D110" in out


@pytest.mark.asyncio
async def test_generate_wbs_office_brief_still_ignores_silent_boq():
    """F1b boundary for a generic brief: a leftover boq key is not an election."""
    r = await ConstructionContainer().generate_wbs(
        {
            "brief": "office building, 12 storeys",
            "boq": [{"item_key": "D110", "description": "General site clearance"}],
            "boq_items": [{"item_key": "D110", "description": "General site clearance"}],
        },
        {"target_count": 40},
    )
    assert r["scaffold"]["derived_from_boq"] is False
    assert "template scaffold" in r["scaffold"]["declaration"].lower()
    assert "D110" not in r["wbs_tree"].values()


@pytest.mark.asyncio
async def test_generate_wbs_f1_without_rows_keeps_honest_scaffold():
    r = await ConstructionContainer().generate_wbs(
        {"brief": F1_ASK},
        {"target_count": 40, "project_type": "building"},
    )
    assert r["scaffold"]["derived_from_boq"] is False
    assert "template scaffold" in r["scaffold"]["declaration"].lower()
    blob = " ".join(r["wbs_tree"].values()).lower()
    assert "d110" not in blob


def test_deliver_f1_boq_tree_has_no_template_scaffold():
    s = ProjectSession.new("f1-boq")
    tree = wbs_tree_from_boq_items(_BOQ_ITEMS)
    s.data["schedule_summary"] = {
        "activities": 4,
        "duration_days": 4,
        "critical_count": 4,
    }
    s.data["wbs"] = {
        "wbs_tree": tree,
        "scaffold": {
            "derived_from_boq": True,
            "declaration": (
                "High-level WBS derived from this project's BOQ "
                "demolition and site clearance items."
            ),
        },
    }
    text = deliver_schedule(s, True, F1_ASK)
    assert "Template scaffold" not in text
    assert "D110" in text and "site clearance" in text.lower()
    assert "High-level WBS" in text
    assert "Site Preparation" not in text
    assert format_wbs_outline(tree)


def test_deliver_schedule_only_still_skips_tree():
    s = ProjectSession.new("sched")
    s.data["schedule_summary"] = {
        "activities": 204,
        "duration_days": 688,
        "critical_count": 44,
    }
    s.data["wbs"] = {
        "wbs_tree": wbs_tree_from_boq_items(_BOQ_ITEMS),
        "scaffold": {"derived_from_boq": True, "declaration": "from BOQ"},
    }
    text = deliver_schedule(s, True, SCHEDULE_ONLY)
    assert "Schedule built: 204 activities" in text
    assert "High-level WBS" not in text
    assert "D110" not in text


DD23_BOQ_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Demolition and Site Clearance BOQ.pdf"
)
DD22_COC_NAME = (
    "AB-2022-202_Vol 1 - Conditions of Contract.pdf"
)
_CONTRACT_PROSE = (
    "Volume 1 - Conditions of Contract. Sub-Clause 4.1. The Contractor "
    "shall execute the demolition and site clearance scope described in the "
    "Bill of Quantities and is responsible for the scope of the Works."
)
_OCR_D_ROWS = (
    "Demolition and Site Clearance. Page d/3/1.\n"
    "D110 General site clearance 12 ha 2500.00 30000.00\n"
    "D290.1 Removal of trees in existing sidewalks 48 Nr\n"
    "D549.2 Removal of existing chain link fence 80 m\n"
    "D599.5 Breaking out existing carriageway including road markings 1200 m2\n"
)


def _pool_docs():
    return [{
        "id": "boq1",
        "original_name": DD23_BOQ_NAME,
        "file_path": f"/docs/{DD23_BOQ_NAME}",
    }]


def _install_boq_pool(monkeypatch, *, texts=None, alias_source=None):
    """D-rows live in the document pool; semantic top-k returns only CoC prose."""
    from app.core.rag.vector_store import Chunk
    from app.lib import boq_schedule as bs

    texts = texts if texts is not None else { "boq1": [S2_BOQ] }
    seen_store: dict = {}

    def fake_title(pid, phrase, limit=8):
        needle = (phrase or "").lower()
        return [
            doc for doc in _pool_docs()
            if needle and needle in (
                f"{doc['original_name']} {doc['file_path']}".lower()
            )
        ][:limit]

    def fake_terms(pid, terms, min_terms=2, require_letter=False, limit=8):
        blob = f"{DD23_BOQ_NAME}".lower()
        hits = sum(1 for t in (terms or []) if t and t.lower() in blob)
        if hits < (min_terms or 1):
            return []
        return _pool_docs()[:limit]

    class _Store:
        def doc_chunk_texts(self, project_id, doc_ids):
            seen_store["rag_pid"] = project_id
            seen_store["doc_ids"] = list(doc_ids or [])
            return {
                did: list(texts.get(did) or [])
                for did in (doc_ids or [])
            }

        def chunks_for_docs(self, project_id, doc_ids, k_per_doc=12):
            out = []
            for did in doc_ids or []:
                for i, text in enumerate(texts.get(did) or []):
                    out.append(Chunk(
                        chunk_id=f"{did}-{i}",
                        project_id=project_id,
                        doc_id=did,
                        chunk_index=i,
                        text=text,
                    ))
            return out[: max(1, int(k_per_doc or 12))]

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
    monkeypatch.setattr(bs, "_resolve_rag_project_id", lambda pid: (
        alias_source if alias_source and pid == "master_corpus" else pid
    ))
    monkeypatch.setattr(
        "app.core.rag.vector_store.get_store", lambda **_k: _Store(),
    )
    monkeypatch.setattr(
        "app.core.rag.retriever.retrieve_with_filter",
        lambda *a, **k: ([coc], 0),
    )
    monkeypatch.setattr(
        "app.core.rag.retriever._doc_name_for_id",
        lambda did: DD23_BOQ_NAME if did == "boq1" else DD22_COC_NAME,
    )
    return seen_store


def test_retrieve_boq_scope_items_parses_named_bill(monkeypatch):
    from app.core.rag.vector_store import Chunk
    from app.lib import boq_schedule as bs

    chunk = Chunk(
        chunk_id="c1",
        project_id="p1",
        doc_id="boq1",
        chunk_index=0,
        text=S2_BOQ,
    )
    chunk.source_name = (
        "AB-2023-101_the client project II Infrastructure Package 1_"
        "Demolition and Site Clearance BOQ.pdf"
    )

    monkeypatch.setattr(
        "app.core.rag.retriever.retrieve_with_filter",
        lambda *a, **k: ([chunk], 0),
    )
    monkeypatch.setattr(
        "app.core.rag.retriever._doc_name_for_id",
        lambda _d: chunk.source_name,
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    items = bs.retrieve_boq_scope_items(F1_ASK, "p1")
    codes = {i.get("item_key") for i in items}
    assert "D110" in codes and "D290.1" in codes
    assert all(i.get("source") for i in items)


@pytest.mark.asyncio
async def test_generate_wbs_retrieves_when_project_has_boq(monkeypatch):
    from app.lib import boq_schedule as bs

    monkeypatch.setattr(bs, "retrieve_boq_scope_items", lambda *_a, **_k: list(_BOQ_ITEMS))
    r = await ConstructionContainer().generate_wbs(
        {"brief": F1_ASK},
        {"project_id": "master_corpus", "target_count": 40},
    )
    assert r["scaffold"]["derived_from_boq"] is True
    assert "D110" in " ".join(r["wbs_tree"].values())


def test_kill_switch_restores_template_election(monkeypatch):
    monkeypatch.setenv("BOQ_SCOPE_WBS", "0")
    assert not message_wants_boq_scope_wbs(F1_ASK)
    assert message_wants_wbs_outline(F1_ASK)


@pytest.mark.asyncio
async def test_f1_run_workflow_with_boq_items_is_not_building_template():
    ctx = {
        "message": F1_ASK,
        "project_name": "Master Corpus",
        "project_id": "f1-boq-wf",
        "params": {
            "target_count": 40,
            "start_date": "2026-01-05",
            "boq_items": _BOQ_ITEMS,
        },
    }
    out = await run_workflow("generate_wbs", ctx, ProjectSession.new("f1-boq-wf"))
    assert out["handled"] and out["status"] == "success", out
    answer = out["answer"]
    assert "Template scaffold" not in answer
    assert "D110" in answer
    assert "site clearance" in answer.lower()
    assert "Site Preparation" not in answer
    assert "Superstructure" not in answer
    assert "Hall A" not in answer
    assert "High-level WBS" in answer
    assert "Schedule built:" in answer


def test_ocr_d_rows_count_as_measured_without_pipes_or_filename():
    from app.lib.boq_schedule import _chunk_looks_like_measured_boq

    assert _chunk_looks_like_measured_boq(_OCR_D_ROWS, "")
    assert not _chunk_looks_like_measured_boq(_CONTRACT_PROSE, DD22_COC_NAME)
    codes = {r.get("item_key") for r in parse_boq_measured_rows(_OCR_D_ROWS)}
    assert "D110" in codes and "D599.5" in codes


def test_pool_d_rows_win_when_semantic_topk_is_only_contract_prose(monkeypatch):
    """Live leftover after #524: D-rows are in the corpus, cosine returns CoC.

    The building-template scaffold must not win when measured D-rows are
    already in the project document pool.
    """
    seen = _install_boq_pool(monkeypatch)
    items = retrieve_boq_scope_items(F1_ASK, "master_corpus")
    codes = {i.get("item_key") for i in items}
    assert "D110" in codes
    assert "D290.1" in codes
    assert "D549.2" in codes
    assert "D599.5" in codes
    assert seen.get("doc_ids") == ["boq1"]
    blob = " ".join(i.get("description") or "" for i in items).lower()
    assert "site clearance" in blob
    assert "superstructure" not in blob
    assert "hall a" not in blob


def test_pool_ocr_d_rows_without_pipes_still_elect(monkeypatch):
    """Scanned demolition BOQ pages have no markdown pipes."""
    _install_boq_pool(monkeypatch, texts={"boq1": [_OCR_D_ROWS]})
    items = retrieve_boq_scope_items(F1_ASK, "p1")
    codes = {i.get("item_key") for i in items}
    assert "D110" in codes and "D599.5" in codes


def test_master_corpus_alias_reads_backing_store(monkeypatch):
    seen = _install_boq_pool(monkeypatch, alias_source="drive_archive")
    items = retrieve_boq_scope_items(F1_ASK, "master_corpus")
    assert {i.get("item_key") for i in items} >= {"D110", "D290.1"}
    assert seen.get("rag_pid") == "drive_archive"


@pytest.mark.asyncio
async def test_generate_wbs_pool_d_rows_are_not_building_template(monkeypatch):
    """Pin: F1 ask + D-rows in docs/pool must not render the building scaffold."""
    _install_boq_pool(monkeypatch)
    r = await ConstructionContainer().generate_wbs(
        {"brief": F1_ASK},
        {"project_id": "master_corpus", "target_count": 200},
    )
    assert r["scaffold"]["derived_from_boq"] is True
    assert r["scaffold"]["source"] == "boq"
    assert "template scaffold" not in r["scaffold"]["declaration"].lower()
    blob = " ".join(r["wbs_tree"].values()).lower()
    assert "d110" in blob and "site clearance" in blob
    assert "site preparation" not in blob
    assert "superstructure" not in blob
    assert "mep" not in blob
    out = _format_wbs_result(r)
    assert "Template scaffold" not in out
    assert "Template activities:" not in out
    assert "D110" in out


@pytest.mark.asyncio
async def test_f1_run_workflow_pool_is_not_building_template(monkeypatch):
    _install_boq_pool(monkeypatch)
    ctx = {
        "message": F1_ASK,
        "project_name": "Master Corpus",
        "project_id": "master_corpus",
        "params": {"target_count": 200, "start_date": "2026-01-05"},
    }
    out = await run_workflow("generate_wbs", ctx, ProjectSession.new("f1-pool"))
    assert out["handled"] and out["status"] == "success", out
    answer = out["answer"]
    assert "Template scaffold" not in answer
    assert "project_type inferred: building" not in answer
    assert "D110" in answer
    assert "site clearance" in answer.lower()
    assert "Site Preparation" not in answer
    assert "Superstructure" not in answer
    assert "Hall A" not in answer
    assert "High-level WBS" in answer
    assert "no quantity is priced" not in answer.lower()
    packages = [ln for ln in answer.splitlines() if ln.startswith("1.")]
    assert len([p for p in packages if "D529.3" in p]) <= 1
