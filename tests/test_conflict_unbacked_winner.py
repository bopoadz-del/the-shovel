"""A2: an answer that states both differing values but names a winner no
document ranks.

An answer that already states every differing value is left as written: it
has told the user the documents differ. If the same answer also says which
one governs (prevails, takes precedence) and no ranking statement in the
evidence backs that, it has told the user something no document says. The
conflict line then leads with "No document states which of them governs."
and the unbacked winner sentence goes.

Decisions, two inputs each:
  1. unbacked winner about the thing -> conflict line, winner sentence gone;
  2. winner the evidence backs -> left as written;
  3. a sentence that asks which governs, or denies a winner -> left as written;
  4. a winner sentence about another thing -> left as written;
  5. a winner a passage about the thing names itself ("shall be as per X")
     -> left as written, in either voice; the same name pointed at another
     thing backs nothing.

Every document, value and question here is invented for this file; the
expected values are written by hand.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.evidence_conflicts import asserts_precedence, detect

_LINE = "The documents differ on this:"
_NONE = "No document states which of them governs."


def _rows(*pairs: tuple[str, str]) -> list[dict]:
    return [{"doc_id": f"d{i}", "doc_name": n, "snippet": t} for i, (n, t) in enumerate(pairs)]


_CASES = [
    pytest.param(
        "What blinding is required under the chamber base slab?",
        ("Harlow Chamber Section Drawing.pdf", "Blinding under the chamber base slab is concrete grade C25."),
        ("Harlow Priced Bill.pdf", "Blinding under the chamber base slab, concrete grade C20."),
        "The chamber base slab sits on grade C25 blinding [1]. The bill carries the blinding as grade C20 [2].",
        "Drawing", id="blinding-grade"),
    pytest.param(
        "What is the cover to the retaining wall reinforcement?",
        ("Wrenfield Retaining Wall Drawing.pdf", "Cover to the retaining wall reinforcement is 50 mm."),
        ("Wrenfield Concrete Specification.pdf", "Cover to the retaining wall reinforcement shall be 40 mm."),
        "The retaining wall reinforcement has 50 mm cover [1]. The specification gives 40 mm cover [2].",
        "Specification", id="wall-cover"),
]


def _apply(question, a, b, answer, extra=()):
    rows = _rows(a, b, *extra)
    return answer_contract.apply(answer, sources=rows, question=question), rows


# ── 1. an unbacked winner ───────────────────────────────────────────────────

@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("winner", [
    "The {kind} governs the {thing}.",
    "For the {thing}, the {kind} takes precedence.",
])
def test_an_unbacked_winner_brings_the_conflict_line_and_goes(question, a, b, told, kind, winner):
    thing = question.rstrip("?").split(" the ", 1)[1].split(" under ")[0].split(" to ")[0]
    claim = winner.format(kind=kind.lower(), thing=thing)
    out, rows = _apply(question, a, b, f"{told} {claim}")
    assert detect(f"{told} {claim}", rows, question)
    assert out.text.startswith(_LINE)
    assert _NONE in out.text
    assert claim not in out.text
    assert out.conflicts and out.conflicts[0]["governs"] is None


@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
def test_control_the_told_answer_alone_is_left_as_written(question, a, b, told, kind):
    out, rows = _apply(question, a, b, told)
    assert detect(told, rows, question) == []
    assert _LINE not in out.text and not out.conflicts


# ── 2. a winner the evidence backs ─────────────────────────────────────────

@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("statement", [
    "Where the {other} and the {kind} disagree, the {kind} shall prevail.",
    "Should the {kind} and the {other} conflict, the {kind} shall govern.",
])
def test_a_backed_winner_with_its_ranking_is_left_as_written(question, a, b, told, kind, statement):
    names = [a[0], b[0]]
    winner_doc = next(n for n in names if kind.lower() in n.lower())
    loser_doc = next(n for n in names if n != winner_doc)
    other = next(w for w in ("Bill", "Drawing", "Specification") if w in loser_doc)
    ranking = ("Quayside General Conditions.pdf", statement.format(kind=kind, other=other))
    claim = f"The {kind.lower()} governs here."
    out, rows = _apply(question, a, b, f"{told} {claim}", extra=(ranking,))
    assert detect(f"{told} {claim}", rows, question) == []
    assert claim in out.text and _LINE not in out.text


# ── 3. asked or denied, not a winner ────────────────────────────────────────

@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("sentence", [
    "Check which document governs the {thing} before ordering.",
    "Confirm with the designer whether the {kind} prevails for the {thing}.",
    "No document states that the {kind} governs the {thing}.",
    "Neither document takes precedence for the {thing}.",
])
def test_asking_or_denying_a_winner_is_left_as_written(question, a, b, told, kind, sentence):
    thing = question.rstrip("?").split(" the ", 1)[1].split(" under ")[0].split(" to ")[0]
    text = sentence.format(kind=kind.lower(), thing=thing)
    out, rows = _apply(question, a, b, f"{told} {text}")
    assert detect(f"{told} {text}", rows, question) == []
    assert text in out.text and _LINE not in out.text


# ── 4. a winner about another thing ────────────────────────────────────────

@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("other", [
    "The fire strategy governs the door ratings on the upper floors.",
    "For the roof drainage outfall, the civil layout takes precedence.",
])
def test_a_winner_about_another_thing_is_left_as_written(question, a, b, told, kind, other):
    out, rows = _apply(question, a, b, f"{told} {other}")
    assert detect(f"{told} {other}", rows, question) == []
    assert other in out.text and _LINE not in out.text


# ── 5. a winner a passage about the thing names itself ─────────────────────
#
# A document can point the thing at a named standard or document ("shall be
# as per", "shall be governed by"). An answer repeating that winner took it
# from the evidence: it stays, in either voice. The same name pointed at
# ANOTHER thing does not back it.

_STANDARDS = [
    pytest.param("Kestrel Code KC-41", "{thing} shall be as per Kestrel Code KC-41.", id="as-per"),
    pytest.param("Annex T-7", "{thing} shall be governed by Annex T-7 throughout.", id="governed-by"),
]
_VOICES = ["{std} governs the {thing} [1].", "The {thing} is governed by {std} [1]."]


def _thing_of(question):
    return question.rstrip("?").split(" the ", 1)[1].split(" under ")[0].split(" to ")[0]


@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("std, ref", _STANDARDS)
@pytest.mark.parametrize("voice", _VOICES)
def test_a_winner_a_passage_about_the_thing_names_is_left_as_written(question, a, b, told, kind, std, ref, voice):
    thing = _thing_of(question)
    a = (a[0], a[1] + " " + ref.format(thing=thing.capitalize()))
    claim = voice.format(std=std, thing=thing)
    out, rows = _apply(question, a, b, f"{told} {claim}")
    assert detect(f"{told} {claim}", rows, question) == []
    assert claim in out.text and _LINE not in out.text


@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
@pytest.mark.parametrize("std, ref", _STANDARDS)
@pytest.mark.parametrize("voice", _VOICES)
def test_control_the_same_name_pointed_at_another_thing_backs_nothing(question, a, b, told, kind, std, ref, voice):
    thing = _thing_of(question)
    a = (a[0], a[1] + " " + ref.format(thing="Pipe bedding"))
    claim = voice.format(std=std, thing=thing)
    out, rows = _apply(question, a, b, f"{told} {claim}")
    assert detect(f"{told} {claim}", rows, question)
    assert out.text.startswith(_LINE) and _NONE in out.text
    assert claim not in out.text


@pytest.mark.parametrize("question, a, b, told, kind", _CASES)
def test_a_passive_unbacked_winner_brings_the_conflict_line(question, a, b, told, kind):
    claim = f"The {_thing_of(question)} is governed by the {kind.lower()}."
    out, rows = _apply(question, a, b, f"{told} {claim}")
    assert detect(f"{told} {claim}", rows, question)
    assert _NONE in out.text and claim not in out.text


# ── the reading itself ──────────────────────────────────────────────────────

@pytest.mark.parametrize("sentence, names", [
    ("The drawing governs the grade.", True),
    ("The specification shall prevail for cover.", True),
    ("Here the schedule takes precedence.", True),
    ("The bill has priority on quantities.", True),
    ("The cover is governed by the specification.", True),
    ("Check which governs your chamber type.", False),
    ("Ask whether the drawing prevails.", False),
    ("No document states which of them governs.", False),
    ("The drawing does not govern the grade.", False),
    ("Neither document takes precedence.", False),
    ("The bill gives C20.", False),
])
def test_the_winner_reading(sentence, names):
    assert asserts_precedence(sentence) is names
