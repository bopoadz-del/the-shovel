"""Tests for the dataset-logging mechanism (item 2b)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

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


@pytest.mark.parametrize("question,answer", [
    ("what is the raft volume?", "900 m3"),
    ("what is the blinding thickness?", "75 mm"),
])
def test_record_as_jsonable_round_trips_through_json(question, answer):
    rec = _record(question=question, answer=answer).with_timestamp()
    loaded = json.loads(json.dumps(rec.as_jsonable()))
    assert loaded["question"] == question
    assert loaded["answer"] == answer
    assert loaded["verdict"] == "project_document"
    assert loaded["evidence_plan"] == ["Bill No.4 raft row"]
    assert loaded["retrieved_ids"] == ["doc-1", "doc-2"]
    assert isinstance(loaded["logged_at"], str) and loaded["logged_at"]


@pytest.mark.parametrize("now", [
    datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    datetime(2026, 6, 7, 8, 9, 10, tzinfo=timezone.utc),
])
def test_with_timestamp_does_not_mutate_and_sets_utc_stamp(now):
    rec = _record()
    stamped = rec.with_timestamp(now)
    assert rec.logged_at == ""
    assert stamped.logged_at == now.isoformat()


@pytest.mark.parametrize("first,second", [
    (("what is the raft volume?", "900 m3"), ("what is the blinding thickness?", "75 mm")),
    (("what is the lift wall volume?", "42 m3"), ("what is the dune sand density?", "1.6 t/m3")),
])
def test_append_writes_one_json_line_per_record(tmp_path, first, second):
    path = dataset_log_path("p-abc", base=tmp_path)
    append_dataset_record("p-abc", _record(question=first[0], answer=first[1]), base=tmp_path)
    append_dataset_record("p-abc", _record(question=second[0], answer=second[1]), base=tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_line = json.loads(lines[0])
    second_line = json.loads(lines[1])
    assert first_line["question"] == first[0]
    assert first_line["answer"] == first[1]
    assert second_line["question"] == second[0]
    assert second_line["answer"] == second[1]
    # Two different questions must land on two distinct records, not overwrite.
    assert first_line["question"] != second_line["question"]


@pytest.mark.parametrize("now", [
    datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    datetime(2026, 6, 7, 8, 9, 10, tzinfo=timezone.utc),
])
def test_append_reuses_callers_timestamp_when_present(tmp_path, now):
    rec = _record().with_timestamp(now)
    path = append_dataset_record("p-abc", rec, base=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["logged_at"] == rec.logged_at


@pytest.mark.parametrize("row", [
    {
        "doc_id": "doc-9", "name": "BOQ.xlsx", "layer": "own",
        "page": 3, "text": "1,200 m3 bulk excavation", "score": 0.9,
    },
    {
        "document_id": "doc-11", "title": "Contract Data.pdf", "layer": "master_corpus",
        "page": 7, "text": "delay damages clause", "score": 0.4,
    },
])
def test_reduced_citation_drops_chunk_text_keeps_identity(row):
    reduced = reduced_citation(row)
    assert "text" not in reduced
    assert "score" not in reduced
    for key in ("doc_id", "document_id", "name", "title", "layer", "page"):
        if key in row:
            assert reduced[key] == row[key]


@pytest.mark.parametrize("sources", [
    [],
    None,
])
def test_verdict_refused_when_no_sources(sources):
    assert verdict_from_sources(sources) == "refused"


@pytest.mark.parametrize("mixed", [
    [
        {"doc_id": "d1", "layer": "general_knowledge"},
        {"doc_id": "d2", "layer": "own"},
    ],
    [
        {"doc_id": "d3", "layer": "master_corpus"},
        {"doc_id": "d4", "layer": "general_knowledge"},
    ],
])
def test_verdict_project_document_wins_over_general(mixed):
    assert verdict_from_sources(mixed) == "project_document"


@pytest.mark.parametrize("sources,expected", [
    ([{"layer": "general_knowledge"}], "general_knowledge"),
    ([{"layer": "unexpected"}], "other"),
])
def test_verdict_general_and_other(sources, expected):
    assert verdict_from_sources(sources) == expected


@pytest.mark.parametrize("question,answer,evidence_plan", [
    ("what is the lift wall volume?", "42 m3", ("Bill No.4 lift walls row",)),
    ("what is the dune sand density?", "1.6 t/m3", ("Dune sand specification row",)),
])
def test_log_dataset_turn_writes_one_record_and_clears(monkeypatch, tmp_path, question, answer, evidence_plan):
    from app.agents.runtime import _log_dataset_turn

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    agent = type("Agent", (), {})()
    agent._dataset_turn = {
        "question": question,
        "evidence_plan": evidence_plan,
    }
    sources = [
        {"doc_id": "d1", "name": "BOQ.xlsx", "layer": "own", "text": "ignored"},
    ]
    _log_dataset_turn(agent, "p-x", answer, sources)

    path = tmp_path / "dataset" / "p-x" / "qa.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["question"] == question
    assert rec["answer"] == answer
    assert rec["evidence_plan"] == list(evidence_plan)
    assert rec["retrieved_ids"] == ["d1"]
    assert rec["verdict"] == "project_document"
    assert rec["citations"][0]["doc_id"] == "d1"
    assert "text" not in rec["citations"][0]
    assert agent._dataset_turn is None


@pytest.mark.parametrize("capture,project_id", [
    (None, "p-y"),
    ({"question": "what is the cover?", "evidence_plan": ("cover row",)}, ""),
])
def test_log_dataset_turn_is_noop_without_capture_or_project_id(monkeypatch, tmp_path, capture, project_id):
    from app.agents.runtime import _log_dataset_turn

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    agent = type("Agent", (), {})()
    agent._dataset_turn = capture
    _log_dataset_turn(agent, project_id, "no answer", [{"doc_id": "d2", "layer": "own"}])
    assert not (tmp_path / "dataset" / "qa.jsonl").exists()
    assert not (tmp_path / "dataset" / "p-y" / "qa.jsonl").exists()
