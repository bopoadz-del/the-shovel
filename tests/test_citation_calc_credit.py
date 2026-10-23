"""Calculator figures are credited to the calculation, not a retrieved chunk.

Live Dewatering-p2 (gate G8): the uplift figure was right and came from
``construction_calc`` / ``dewatering_uplift_check``, but the answer credited
"Contractor's Proposal.pdf chunk 40". ``gate`` kept that ``Source:`` line
because the filename was any retrieved record, and an inline "chunk N" that
is not a whole ``Source:`` line never entered ``_SOURCE_LINE_RE``.

Gate G7: the original phrasing, three rewordings, and one sibling calculator
from ``app/lib/construction_formulas.py``. Documents are synthetic
``FIXTURE-d-20260928-...`` only. A purely retrieval-based answer keeps its
Source line.
"""

from __future__ import annotations

import json
import re

from app.agents.citation_provenance import UNVERIFIED_NOTE, gate
from app.lib.construction_formulas import run_calculation
from app.lib.source_labels import calculator_label

_PROPOSAL = "FIXTURE-d-20260928-proposal.pdf"
_BOREHOLE = "FIXTURE-d-20260928-borehole.pdf"
_METHOD = "FIXTURE-d-20260928-method.pdf"
_CONTRACT = "FIXTURE-d-20260928-contract.pdf"

_DEWATER_PARAMS = {"water_depth": 23, "raft_thickness": 2.0, "floor_count": 5}
_DEWATER_ENV = run_calculation("dewatering_uplift_check", dict(_DEWATER_PARAMS))
_FORMWORK_PARAMS = {"concrete_strength_7h": 2.0}
_FORMWORK_ENV = run_calculation("formwork_striking_time", dict(_FORMWORK_PARAMS))

_PROPOSAL_TEXT = (
    "FIXTURE-d-20260928 method statement for excavation support and traffic management."
)
_METHOD_TEXT = "FIXTURE-d-20260928 site method statement for vertical formwork panels."
# Supplies raft_thickness and does not contain the committed FOS / uplift figure.
_BOREHOLE_TEXT = (
    "The FIXTURE-d-20260928 borehole log records a raft thickness of 2.0 m."
)


def _marked(*rows: tuple[str, int, str]) -> dict:
    parts = ["AUTHORITATIVE REFERENCE CONTEXT"]
    for i, (src, chunk, text) in enumerate(rows):
        parts.append(
            f"[doc_id=FIXTURE-d-20260928-{i} chunk={chunk} score=0.420 src={src}] {text}"
        )
    return {"role": "system", "content": "\n".join(parts)}


def _calc_turn(calculation: str, params: dict, envelope: dict, ask: str) -> list[dict]:
    args = json.dumps({"calculation": calculation, "params": params})
    return [
        {"role": "user", "content": ask},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_FIXTURE_d_20260928",
                "function": {"name": "construction_calc", "arguments": args},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_FIXTURE_d_20260928",
            "name": "construction_calc",
            "content": json.dumps(envelope),
        },
    ]


def _dewater_messages() -> list[dict]:
    return _calc_turn(
        "dewatering_uplift_check",
        _DEWATER_PARAMS,
        _DEWATER_ENV,
        "FIXTURE-d-20260928 dewatering uplift check",
    )


def _dewater_rag() -> dict:
    return _marked((_PROPOSAL, 40, _PROPOSAL_TEXT))


def _assert_no_chunk(out: str, filename: str, chunk_no: int) -> None:
    assert filename not in out
    assert re.search(rf"\bchunks?\s*{chunk_no}\b", out, re.IGNORECASE) is None


def _assert_plain_credit(out: str, calculation: str, inputs: dict) -> None:
    """The credit names the formula as a user reads it, with its inputs; the
    tool and function names are internal and never shown."""
    from app.lib.source_labels import calculator_label

    assert "Source: " + calculator_label(calculation, inputs) in out
    assert "construction_calc" not in out
    assert calculation not in out


def _assert_notes_not_on_source_line(out: str, notes: list) -> None:
    """The Source line is the registry label and the user's inputs, the same
    entry the Sources panel shows; the result notes are not its source."""
    assert notes, notes
    for line in out.splitlines():
        if line.startswith("Source: "):
            for note in notes:
                assert note not in line, line


def _assert_dewater_credit(out: str) -> None:
    _assert_plain_credit(out, "dewatering_uplift_check",
                         {"water_depth": 23, "raft_thickness": 2, "floor_count": 5})
    _assert_notes_not_on_source_line(out, _DEWATER_ENV["result"]["notes"])
    assert "0.380" in out
    assert "The factor of safety is 0.380" in out


def _assert_formwork_credit(out: str) -> None:
    _assert_plain_credit(out, "formwork_striking_time", {"concrete_strength_7h": 2})
    _assert_notes_not_on_source_line(out, _FORMWORK_ENV["result"]["notes"])
    hours = _FORMWORK_ENV["result"]["recommended_hours"]
    assert str(hours) in out


# --------------------------------------------------------------------------
# Dewatering-p2 — original phrasing plus three rewordings
# --------------------------------------------------------------------------

def test_dewatering_p2_source_line_credits_calculation_not_chunk():
    """Original: a whole Source line naming the retrieved proposal chunk."""
    answer = (
        "Dewatering cannot stop. The factor of safety is 0.380.\n"
        f"Source: {_PROPOSAL} chunk 40\n"
    )
    out = gate(answer, _dewater_rag(), _dewater_messages())
    _assert_no_chunk(out, _PROPOSAL, 40)
    _assert_dewater_credit(out)
    assert UNVERIFIED_NOTE.strip() in out


def test_dewatering_p2_reword_bold_source_line():
    answer = (
        "Dewatering cannot stop. The factor of safety is 0.380.\n"
        f"**Source:** {_PROPOSAL}, chunk 40\n"
    )
    out = gate(answer, _dewater_rag(), _dewater_messages())
    _assert_no_chunk(out, _PROPOSAL, 40)
    _assert_dewater_credit(out)
    assert UNVERIFIED_NOTE.strip() in out


def test_dewatering_p2_reword_inline_parenthetical():
    answer = (
        f"Dewatering cannot stop. The factor of safety is 0.380 "
        f"({_PROPOSAL} chunk 40).\n"
    )
    out = gate(answer, _dewater_rag(), _dewater_messages())
    _assert_no_chunk(out, _PROPOSAL, 40)
    _assert_dewater_credit(out)
    assert UNVERIFIED_NOTE.strip() in out


def test_dewatering_p2_reword_inline_see_chunk():
    """Not a Source line, so ``_SOURCE_LINE_RE`` never sees it."""
    answer = "Dewatering cannot stop. The factor of safety is 0.380; see chunk 40.\n"
    out = gate(answer, _dewater_rag(), _dewater_messages())
    _assert_no_chunk(out, _PROPOSAL, 40)
    _assert_dewater_credit(out)
    assert UNVERIFIED_NOTE.strip() in out


def test_dewatering_figure_is_credited_even_without_a_chunk_citation():
    """The tool produced the figure. The answer must still name the calculation."""
    answer = "Dewatering cannot stop. The factor of safety is 0.380.\n"
    out = gate(answer, _dewater_rag(), _dewater_messages())
    _assert_dewater_credit(out)
    assert UNVERIFIED_NOTE.strip() not in out


def test_retrieved_input_citation_stays_beside_the_calculation_credit():
    """A chunk that supplied a governing input may stay. The figure's chunk may not."""
    rag = _marked(
        (_PROPOSAL, 40, _PROPOSAL_TEXT),
        (_BOREHOLE, 3, _BOREHOLE_TEXT),
    )
    answer = (
        "Dewatering cannot stop. The factor of safety is 0.380. "
        "The raft thickness input is the borehole record.\n"
        f"Source: {_PROPOSAL} chunk 40\n"
        f"Source: {_BOREHOLE} chunk 3\n"
    )
    out = gate(answer, rag, _dewater_messages())
    _assert_no_chunk(out, _PROPOSAL, 40)
    assert _BOREHOLE in out
    assert re.search(r"\bchunks?\s*3\b", out, re.IGNORECASE)
    _assert_dewater_credit(out)


# --------------------------------------------------------------------------
# Sibling calculator in the same module
# --------------------------------------------------------------------------

def test_sibling_formwork_striking_credits_calculation_not_chunk():
    hours = _FORMWORK_ENV["result"]["recommended_hours"]
    answer = (
        f"Formwork may be struck after {hours} hours.\n"
        f"Source: {_METHOD} chunk 7\n"
    )
    rag = _marked((_METHOD, 7, _METHOD_TEXT))
    messages = _calc_turn(
        "formwork_striking_time",
        _FORMWORK_PARAMS,
        _FORMWORK_ENV,
        "FIXTURE-d-20260928 formwork striking time",
    )
    out = gate(answer, rag, messages)
    _assert_no_chunk(out, _METHOD, 7)
    _assert_formwork_credit(out)
    assert f"Formwork may be struck after {hours} hours." in out
    assert UNVERIFIED_NOTE.strip() in out


# --------------------------------------------------------------------------
# Control — retrieval only, no calculator
# --------------------------------------------------------------------------

def test_retrieval_only_answer_keeps_its_source_line():
    rag = _marked((
        _CONTRACT,
        1,
        "The FIXTURE-d-20260928 accepted contract amount is 1754504.25.",
    ))
    answer = (
        "The accepted contract amount is 1754504.25.\n"
        f"Source: {_CONTRACT}\n"
    )
    assert gate(answer, rag, [{"role": "user", "content": "FIXTURE-d-20260928 amount"}]) is answer


def test_malformed_calculator_payload_is_not_a_credit():
    """A tool body that is not JSON is not a calculation envelope."""
    answer = "The factor of safety is 0.380.\n"
    messages = [{
        "role": "tool",
        "name": "construction_calc",
        "content": "{not-json",
    }]
    assert gate(answer, None, messages) is answer


# The credit line as the platform writes it: inputs by their declared labels.
_DEWATER_CREDIT = "Source: " + calculator_label(
    "dewatering_uplift_check", {"water_depth": 23, "raft_thickness": 2, "floor_count": 5})


def test_answer_that_already_credits_the_calculation_is_unchanged():
    answer = (
        "Dewatering cannot stop. The factor of safety is 0.380.\n"
        f"{_DEWATER_CREDIT}\n"
    )
    assert gate(answer, _dewater_rag(), _dewater_messages()) is answer


def test_credit_line_carrying_result_notes_is_reduced_to_the_label():
    notes = "; ".join(_DEWATER_ENV["result"]["notes"])
    label = _DEWATER_CREDIT
    answer = (
        "Dewatering cannot stop. The factor of safety is 0.380.\n"
        f"{label} — {notes}\n"
    )
    out = gate(answer, _dewater_rag(), _dewater_messages())
    assert out.count("Source: ") == 1
    assert label in out.splitlines()
    assert UNVERIFIED_NOTE.strip() not in out
