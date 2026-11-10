"""Tests for the dataset-logging mechanism (item 2b)."""
from __future__ import annotations

import json

from app.core.rag.dataset_log import (
    DatasetRecord,
    append_dataset_record,
    dataset_log_path,
    reduced_citation,
    verdict_from_sources,
)


def _record(**overrides) -> DatasetRecord:
    base = dict(
        question="what is the raft volume?",
        answer="900 m3",
        verdict="project_document",
        evidence_plan=("Bill No.4 raft row",),
        retrieved_ids=("doc-1", "doc-2"),
        citations=(
            {"doc_id": "doc-1", "name": "BOQ.xlsx", "layer": "own"},
            {"doc_id": "doc-2", "name": "Contract Data.pdf", "layer": "master_corpus"},
        ),
    )
    base.update(overrides)
    return DatasetRecord(**base)


def test_record_as_jsonable_round_trips_through_json():
    rec = _record().with_timestamp()
    loaded = json.loads(json.dumps(rec.as_jsonable()))
    assert loaded["question"] == "what is the raft volume?"
    assert loaded["answer"] == "900 m3"
    assert loaded["verdict"] == "project_document"
    assert loaded["evidence_plan"] == ["Bill No.4 raft row"]
    assert loaded["retrieved_ids"] == ["doc-1", "doc-2"]
    assert isinstance(loaded["logged_at"], str) and loaded["logged_at"]


def test_with_timestamp_does_not_mutate_and_sets_utc_stamp():
    rec = _record()
    stamped = rec.with_timestamp()
    assert rec.logged_at == ""
    assert stamped.logged_at.endswith("+00:00")


def test_append_writes_one_json_line_per_record(tmp_path):
    path = dataset_log_path("p-abc", base=tmp_path)
    append_dataset_record("p-abc", _record(), base=tmp_path)
    append_dataset_record(
        "p-abc",
        _record(question="what is the blinding thickness?", answer="75 mm"),
        base=tmp_path,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["question"] == "what is the raft volume?"
    assert second["answer"] == "75 mm"
    # Two different questions must land on two distinct records, not overwrite.
    assert first["question"] != second["question"]


def test_append_reuses_callers_timestamp_when_present(tmp_path):
    rec = _record().with_timestamp()
    path = append_dataset_record("p-abc", rec, base=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["logged_at"] == rec.logged_at


def test_reduced_citation_drops_chunk_text_keeps_identity():
    row = {
        "doc_id": "doc-9",
        "name": "BOQ.xlsx",
        "layer": "own",
        "page": 3,
        "text": "1,200 m3 bulk excavation",
        "score": 0.9,
    }
    reduced = reduced_citation(row)
    assert reduced["doc_id"] == "doc-9"
    assert reduced["name"] == "BOQ.xlsx"
    assert reduced["layer"] == "own"
    assert reduced["page"] == 3
    assert "text" not in reduced
    assert "score" not in reduced


def test_verdict_refused_when_no_sources():
    assert verdict_from_sources([]) == "refused"
    assert verdict_from_sources(None) == "refused"


def test_verdict_project_document_wins_over_general():
    mixed = [
        {"doc_id": "d1", "layer": "general_knowledge"},
        {"doc_id": "d2", "layer": "own"},
    ]
    assert verdict_from_sources(mixed) == "project_document"


def test_verdict_general_and_other():
    assert verdict_from_sources([{"layer": "general_knowledge"}]) == "general_knowledge"
    assert verdict_from_sources([{"layer": "unexpected"}]) == "other"


def test_log_dataset_turn_writes_one_record_and_clears(monkeypatch, tmp_path):
    from app.agents.runtime import _log_dataset_turn

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    agent = type("Agent", (), {})()
    agent._dataset_turn = {
        "question": "what is the lift wall volume?",
        "evidence_plan": ("Bill No.4 lift walls row",),
    }
    sources = [
        {"doc_id": "d1", "name": "BOQ.xlsx", "layer": "own", "text": "ignored"},
    ]
    _log_dataset_turn(agent, "p-x", "42 m3", sources)

    path = tmp_path / "dataset" / "p-x" / "qa.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["question"] == "what is the lift wall volume?"
    assert rec["answer"] == "42 m3"
    assert rec["evidence_plan"] == ["Bill No.4 lift walls row"]
    assert rec["retrieved_ids"] == ["d1"]
    assert rec["verdict"] == "project_document"
    assert rec["citations"][0]["doc_id"] == "d1"
    assert "text" not in rec["citations"][0]
    assert agent._dataset_turn is None


def test_log_dataset_turn_is_noop_without_a_capture(monkeypatch, tmp_path):
    from app.agents.runtime import _log_dataset_turn

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    agent = type("Agent", (), {})()
    _log_dataset_turn(agent, "p-y", "no answer", [{"doc_id": "d2", "layer": "own"}])
    assert not (tmp_path / "dataset" / "p-y" / "qa.jsonl").exists()
