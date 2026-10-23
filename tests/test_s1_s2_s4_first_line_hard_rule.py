"""First line must carry the figure and the document that states it.

Live SET5 close-out on 209bc83 (Agent E, 6x), synthetic stand-ins only:

* S1 cover — SPLIT. The first line often had 75 mm and no document, or a
  document and no figure.
* S2 compaction — FAIL 0/6. The first line named a specification volume and
  "properly compacted" and left out the numeric figure that the retrieved
  context already held.
* S4 governing contract — SPLIT. The first line sometimes never named a
  contract that was visible in the context. Two contracts means ask which.

The Hard-rule sentences already told the model to open with the figure and
the document. These tests pin the behaviour the operator scores: the first
line of the answer that leaves ``_postprocess_answer``, not the prompt.
"""
import re
from pathlib import Path

import pytest

from app.agents.runtime import _postprocess_answer

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]

S1_ASK = (
    "Per the project specification, what is the minimum concrete cover "
    "to reinforcement for foundations?"
)
S2_ASK = (
    "Per the project specification, what compaction is required under "
    "road pavement?"
)
S4_ASK = "Which contract governs this project?"

SPEC_COVER = "SYN-SPEC-001 Particular Specification Concrete.pdf"
DRAWING_COVER = "SYN-DWG-010 Foundation General Notes.pdf"
SPEC_QUAL = "SYN-SPEC-002 Vol 2 Specification.pdf"
VOL5 = "SYN-OD-005 Vol 5 Other Documents Geotech.pdf"
SPEC_98 = "SYN-SPEC-003 Particular Specification Earthworks.pdf"
OTHER_95 = "SYN-OD-006 Other Documents Mobilization Note.pdf"
CONTRACT_A = "SYN-2024-014 Infrastructure Works Contract.pdf"
CONTRACT_B = "SYN-2024-022 Enabling Works Contract.pdf"
TEMPLATE = "SYN-2014-001 Minor Works Contract TEMPLATE.pdf"

COVER_TEXT = (
    "Minimum concrete cover to reinforcement for foundations is 75 mm "
    "where the foundation is cast against soil."
)
VOL5_TEXT = (
    "At least twenty cm of material placed to form the sub-grade layer "
    "under the road pavement, when compacted to ninety five percent (95%) "
    "of maximum dry density to get a minimum CBR value of 25."
)
QUAL_TEXT = (
    "All soils shall be properly compacted, and where required a compaction "
    "test shall be carried out in accordance with BS 1377."
)
SPEC_98_TEXT = (
    "Compaction of structural backfill under foundations to minimum 98% of "
    "maximum dry density of the modified proctor test."
)
OTHER_95_TEXT = (
    "Site office platform compacted to 95% of maximum dry density."
)


def _chunk(doc_id: str, name: str, text: str, score: str = "0.800") -> str:
    return (
        f"[doc_id={doc_id} chunk=0 score={score} class=project_corpus "
        f"src={name}] {text}"
    )


def _rag(*chunks: str) -> dict:
    return {"role": "system", "content": "\n\n".join(chunks)}


def _msgs(ask: str) -> list:
    return [{"role": "user", "content": ask}]


def _first(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _has_95(line: str) -> bool:
    # "%" is not a word character, so a trailing \\b never matches "95% of".
    return bool(re.search(r"(?i)\b95\s*(?:%|percent\b)", line))


def _has_75_mm(line: str) -> bool:
    return bool(re.search(r"(?i)\b75\s*mm\b", line))


# ── prompt: the sentence the model is given ───────────────────────────────

def _config(agent: str) -> str:
    return (CONFIGS / f"{agent}.md").read_text(encoding="utf-8")


def _hard_section(agent: str) -> str:
    body = _config(agent)
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", body, re.MULTILINE):
        nxt = body.find("\n## ", m.end())
        sections.append(body[m.start(): nxt if nxt != -1 else len(body)])
    return "\n".join(sections)


@pytest.mark.parametrize("agent", AGENTS)
def test_governing_source_rule_requires_figure_and_document_together(agent):
    section = _hard_section(agent)
    assert "When the question names the governing source" in section
    rule = section[section.index("When the question names the governing source"):]
    rule = rule[: rule.index("\n- ")] if "\n- " in rule else rule
    low = rule.lower()
    assert "both the numeric figure" in low
    assert "first line" in low
    assert "properly compacted" in low, (
        "a qualitative clause must be named as not the figure"
    )
    assert "95%" in rule or "95 %" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_which_contract_governs_is_a_first_line_duty(agent):
    section = _hard_section(agent).lower()
    assert "which contract governs this project" in section
    assert "ask which is meant" in section


# ── S1: cover figure and the document that carries it ─────────────────────

def test_s1_figure_without_document_is_repaired_on_the_first_line():
    """The live miss: 75 mm in the first line, the document only later."""
    bad = (
        "Minimum concrete cover to reinforcement for foundations: 75 mm "
        "where the foundation is cast against soil."
    )
    assert _has_75_mm(_first(bad))
    assert "SYN-SPEC-001" not in _first(bad)
    out = _postprocess_answer(
        bad, _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)), _msgs(S1_ASK),
    )
    first = _first(out)
    assert _has_75_mm(first), first
    assert "SYN-SPEC-001" in first, first


def test_s1_document_without_figure_is_repaired_on_the_first_line():
    bad = (
        f"Per {SPEC_COVER}, the specification addresses concrete cover "
        "for foundations cast against soil.\n\n"
        "The cover to reinforcement is 75 mm."
    )
    assert "SYN-SPEC-001" in _first(bad)
    assert not _has_75_mm(_first(bad))
    out = _postprocess_answer(
        bad, _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)), _msgs(S1_ASK),
    )
    first = _first(out)
    assert _has_75_mm(first), first
    assert "SYN-SPEC-001" in first, first


def test_s1_narrative_opener_does_not_keep_the_figure_off_the_first_line():
    bad = (
        "I have the answer now — it comes from the structural notes, "
        "so I'll flag that clearly.\n\n"
        f"The minimum cover is 75 mm per {SPEC_COVER}."
    )
    out = _postprocess_answer(
        bad, _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)), _msgs(S1_ASK),
    )
    first = _first(out)
    assert _has_75_mm(first), first
    assert "SYN-SPEC-001" in first, first
    assert not first.lower().startswith("i have the answer"), first


def test_s1_drawing_figure_is_named_as_the_drawing_not_the_specification():
    """The number is not in the named source. Say which document it is."""
    bad = "Minimum concrete cover is 75 mm."
    drawing = (
        "Clear cover to reinforcement shall not be less than 75 mm "
        "for concrete in contact with soil."
    )
    out = _postprocess_answer(
        bad, _rag(_chunk("d1", DRAWING_COVER, drawing)), _msgs(S1_ASK),
    )
    first = _first(out)
    assert _has_75_mm(first), first
    assert "SYN-DWG-010" in first, first
    assert "not the specification" in first.lower(), first


def test_s1_first_line_already_carrying_both_is_left_alone():
    good = f"75 mm is the concrete-cover figure in {SPEC_COVER}."
    out = _postprocess_answer(
        good, _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)), _msgs(S1_ASK),
    )
    assert _first(out) == good


# ── S2: numeric compaction figure, not "properly compacted" ───────────────

def test_s2_properly_compacted_plus_spec_name_fails_the_first_line():
    """The live 0/6 shape. Qualitative spec wording is not the figure."""
    bad = (
        f"Per the project specification ({SPEC_QUAL}), the retrieved clause "
        "states that all soils shall be properly compacted.\n\n"
        "The sub-grade figure is 95% of maximum dry density."
    )
    first_bad = _first(bad)
    assert "properly compacted" in first_bad.lower()
    assert "SYN-SPEC-002" in first_bad
    assert not _has_95(first_bad)
    out = _postprocess_answer(
        bad,
        _rag(
            _chunk("dspec", SPEC_QUAL, QUAL_TEXT, "0.910"),
            _chunk("dvol", VOL5, VOL5_TEXT, "0.740"),
        ),
        _msgs(S2_ASK),
    )
    first = _first(out)
    assert _has_95(first), first
    assert "SYN-OD-005" in first, first
    assert "properly compacted" not in first.lower(), first


def test_s2_does_not_invent_a_percent_the_context_does_not_hold():
    bad = (
        f"Per {SPEC_QUAL}, all soils shall be properly compacted."
    )
    out = _postprocess_answer(
        bad, _rag(_chunk("dspec", SPEC_QUAL, QUAL_TEXT)), _msgs(S2_ASK),
    )
    assert not _has_95(_first(out))
    assert "95" not in _first(out)


def test_s2_specification_percent_beats_another_documents_percent():
    """The figure the body states first is the one the first line carries.

    The model commits to 98% before it mentions the other document's 95%.
    """
    bad = (
        "Compaction for this pavement is 98% of maximum dry density.\n\n"
        f"A separate note in {OTHER_95} states 95% of maximum dry density."
    )
    out = _postprocess_answer(
        bad,
        _rag(
            _chunk("dother", OTHER_95, OTHER_95_TEXT, "0.880"),
            _chunk("dspec", SPEC_98, SPEC_98_TEXT, "0.770"),
        ),
        _msgs(S2_ASK),
    )
    first = _first(out)
    assert re.search(r"(?i)\b98\s*%", first), first
    assert "SYN-SPEC-003" in first, first
    assert not re.search(r"(?i)\b95\s*%", first), first


# ── S4: name the contract, or ask which ───────────────────────────────────

def test_s4_first_line_names_the_one_visible_contract():
    bad = (
        "The retrieved excerpts point to two different governing-law "
        "statements, and they come from different documents."
    )
    assert "SYN-2024-014" not in _first(bad)
    body = "This agreement is the contract for the infrastructure works."
    out = _postprocess_answer(
        bad, _rag(_chunk("c1", CONTRACT_A, body)), _msgs(S4_ASK),
    )
    first = _first(out)
    assert "SYN-2024-014" in first, first
    assert "which contract is meant" not in first.lower(), first


def test_s4_two_contracts_ask_which_instead_of_picking():
    bad = f"{CONTRACT_A} governs this project."
    body = "Executed form of agreement."
    out = _postprocess_answer(
        bad,
        _rag(
            _chunk("c1", CONTRACT_A, body),
            _chunk("c2", CONTRACT_B, body, "0.790"),
        ),
        _msgs(S4_ASK),
    )
    first = _first(out)
    assert "which contract is meant" in first.lower(), first
    assert "SYN-2024-014" in first and "SYN-2024-022" in first, first


def test_s4_a_template_does_not_count_as_a_second_contract():
    bad = "Governing law is stated in the excerpts."
    out = _postprocess_answer(
        bad,
        _rag(
            _chunk("c1", CONTRACT_A, "Executed form of agreement."),
            _chunk("t1", TEMPLATE, "This template contract is governed by the laws of the sample emirate."),
        ),
        _msgs(S4_ASK),
    )
    first = _first(out)
    assert "SYN-2024-014" in first, first
    assert "which contract is meant" not in first.lower(), first
    assert "TEMPLATE" not in first


def test_s4_first_line_that_already_names_the_contract_is_left_alone():
    good = f"The contract in the retrieved context is {CONTRACT_A}."
    out = _postprocess_answer(
        good,
        _rag(_chunk("c1", CONTRACT_A, "Executed form of agreement.")),
        _msgs(S4_ASK),
    )
    assert _first(out) == good


# ── stay out of questions this rule does not govern ───────────────────────

def test_a_plain_question_is_not_rewritten():
    ask = "What is the Defects Notification Period under this contract?"
    answer = "The Defects Notification Period is 365 days."
    out = _postprocess_answer(
        answer,
        _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)),
        _msgs(ask),
    )
    assert _first(out) == answer


def test_kill_switch_leaves_the_live_miss_in_place(monkeypatch):
    monkeypatch.setenv("FIRST_LINE_HARD_RULE", "0")
    bad = "Minimum concrete cover to reinforcement for foundations: 75 mm."
    out = _postprocess_answer(
        bad, _rag(_chunk("d1", SPEC_COVER, COVER_TEXT)), _msgs(S1_ASK),
    )
    assert _first(out) == bad
