"""F: the void-coverage footer names the document the turn READ, whether or
not the answer repeats its name. Every input here is invented in the test."""

from __future__ import annotations

import json

import pytest

from app.agents.runtime import _documents_read_this_turn, _postprocess_answer

VOID = (0, 3)


def _fetch_pair(call_id: str, doc_id: str, name: str, *, ok: bool = True,
                body: str = "Item 1 Excavation 40 m3") -> list[dict]:
    args = json.dumps({"document_id": doc_id})
    result = (
        {"name": "fetch_document", "ok": True,
         "result": {"document_id": doc_id, "filename": name, "content": body}}
        if ok else
        {"name": "fetch_document", "ok": False,
         "result": {"status": "error", "error": "not found"}}
    )
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "fetch_document", "arguments": args}}]},
        {"role": "tool", "name": "fetch_document", "tool_call_id": call_id,
         "content": json.dumps(result)},
    ]


@pytest.mark.parametrize("doc_id,name,answer", [
    ("doc-a1", "trial_pit_log_rev2.pdf", "The deepest pit reached 2.4 m."),
    ("doc-b7", "kerb_schedule.xlsx", "There are 18 kerb runs, 312 m in total."),
])
def test_a_document_read_this_turn_is_the_attached_tier_without_naming_it(
        doc_id, name, answer):
    messages = [{"role": "user", "content": f"[attached: {name}] summarise it"},
                *_fetch_pair("c1", doc_id, name)]
    assert name not in answer
    out = _postprocess_answer(
        answer, None, messages, coverage=VOID,
        attached_documents=[{"id": doc_id, "original_name": name}],
    )
    assert f"attached file {name}" in out
    assert "general reference material" not in out
    assert "0 of 3 project documents indexed" in out


def test_an_answer_that_cites_the_document_is_still_the_attached_tier():
    name = "pour_record_block_c.md"
    out = _postprocess_answer(
        f"Per {name}, the pour took 6 hours.", None,
        [{"role": "user", "content": "how long was the pour"}], coverage=VOID,
        attached_documents=[{"id": "d9", "original_name": name}],
    )
    assert f"attached file {name}" in out


def test_a_failed_read_keeps_the_general_wording():
    name = "scaffold_inspection_week3.pdf"
    messages = [{"role": "user", "content": f"[attached: {name}] any defects"},
                *_fetch_pair("c2", "d3", name, ok=False)]
    out = _postprocess_answer(
        "I could not open it.", None, messages, coverage=VOID,
        attached_documents=[{"id": "d3", "original_name": name}],
    )
    assert "general reference material" in out
    assert "attached file" not in out


def test_a_read_from_an_earlier_turn_does_not_count():
    name = "drainage_layout_notes.txt"
    messages = [{"role": "user", "content": "read the drainage notes"},
                *_fetch_pair("c3", "d4", name),
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "what is the boiling point of water"}]
    out = _postprocess_answer(
        "100 degrees Celsius at sea level.", None, messages, coverage=VOID,
        attached_documents=None,
    )
    assert "general reference material" in out
    assert "attached file" not in out


def test_a_marker_carrying_its_own_content_is_a_read():
    name = "site_board.jpg"
    op = f"[attached: {name} | text read from image:\nGATE 4 DELIVERIES 07:00] when?"
    out = _postprocess_answer(
        "Deliveries start at 07:00.", None,
        [{"role": "user", "content": op}], coverage=VOID,
    )
    assert f"attached file {name}" in out


@pytest.mark.parametrize("attached", [True, False])
def test_a_result_cut_at_the_cap_still_counts_as_read(attached):
    name = "long_spec_section_05.docx"
    full = json.dumps({"name": "fetch_document", "ok": True,
                       "result": {"document_id": "d5", "filename": name,
                                  "content": "x" * 500}})
    messages = [{"role": "user", "content": "what grade of steel"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "c5", "type": "function",
                    "function": {"name": "fetch_document",
                                 "arguments": json.dumps({"document_id": "d5"})}}]},
                {"role": "tool", "name": "fetch_document", "tool_call_id": "c5",
                 "content": full[:200]}]
    assert not full[:200].endswith("}")
    names = _documents_read_this_turn(
        "Grade S355.", messages,
        [{"id": "d5", "original_name": name}] if attached else None)
    assert names == [name]


@pytest.mark.parametrize("tool,name,rows", [
    ("boq_processor", "bill_section_2.pdf", 12),
    ("spec_analyzer", "spec_div_03.docx", 4),
])
def test_a_block_that_consumed_the_attached_file_counts_as_read(tool, name, rows):
    messages = [
        {"role": "user", "content": f"[attached: {name}] read it"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c6", "type": "function",
            "function": {"name": tool,
                         "arguments": json.dumps({"file_path": name})}}]},
        {"role": "tool", "name": tool, "tool_call_id": "c6",
         "content": json.dumps({"name": tool, "ok": True,
                                "result": {"status": "success", "rows": rows}})},
    ]
    names = _documents_read_this_turn(
        f"{rows} rows were read.", messages, [{"id": "d6", "original_name": name}])
    assert names == [name]


@pytest.mark.parametrize("name", ["method_statement_piling.pdf", "lift_plan_crane_2.md"])
def test_a_document_read_by_name_without_an_attachment_counts(name):
    messages = [{"role": "user", "content": f"open {name}"},
                *_fetch_pair("c7", "d7", name)]
    assert _documents_read_this_turn("Rig set-up first.", messages, None) == [name]


@pytest.mark.parametrize("first,second", [
    ("alpha_notes.md", "beta_notes.md"),
    ("gamma_log.pdf", "delta_log.pdf"),
])
def test_the_cited_document_is_named_first(first, second):
    messages = [{"role": "user", "content": "compare them"},
                *_fetch_pair("c8", "d8", first), *_fetch_pair("c9", "d9", second)]
    names = _documents_read_this_turn(f"See {second}.", messages, None)
    assert names == [second, first]


@pytest.mark.parametrize("content", ["[]", '"done"'])
def test_a_tool_result_that_is_not_an_object_is_not_a_read(content):
    name = "pile_cap_schedule.pdf"
    messages = [{"role": "user", "content": f"[attached: {name}] check it"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "c10", "type": "function",
                    "function": {"name": "fetch_document",
                                 "arguments": json.dumps({"filename": name})}}]},
                {"role": "tool", "name": "fetch_document", "tool_call_id": "c10",
                 "content": content}]
    out = _postprocess_answer(
        "Nothing came back.", None, messages, coverage=VOID,
        attached_documents=[{"id": "d10", "original_name": name}],
    )
    assert "general reference material" in out
    assert "attached file" not in out
