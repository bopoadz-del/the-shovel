"""P4b: a project figure must not wear a named code that was not retrieved.

Live tip 209bc83, six runs: "Per NFPA 51B (2019), how long must a fire
watch be maintained after hot work?" answered **30 minutes** and named
that figure as NFPA's. The 30 minutes are the project's hot-work permit.
NFPA 51B is not a retrieved document. Same shape: a project lux value
named as Dubai Municipality's (P3b), and a project 32°C named as ACI
305's (P6b).

Acceptable answers: the code's own retrieved figure, or an explicit
"not in the retrieved excerpts" with any project figure labelled
project-only. Never the project figure as the code's.

Fixture wording only. No live client names.
"""
from __future__ import annotations

import re

from app.agents.runtime import _postprocess_answer
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


P4B_ASK = (
    "Per NFPA 51B (2019), how long must a fire watch be maintained "
    "after hot work?"
)
P3B_ASK = (
    "Per Dubai Municipality, what maintained illuminance is required "
    "for circulation lighting?"
)
P6B_ASK = (
    "Per ACI 305, what is the maximum temperature of fresh concrete?"
)
PROJECT_ASK = (
    "How long must the fire watch be maintained on this project's "
    "hot-work permit?"
)

PTW_TEXT = (
    "Project hot-work permit. A fire watch shall be maintained for "
    "30 minutes after hot work is complete."
)
PTW_CITES_NFPA = (
    "Project hot-work permit. In accordance with NFPA 51B, a fire "
    "watch shall be maintained for 30 minutes after hot work is complete."
)
LIGHT_TEXT = (
    "Project lighting specification. Maintained average illuminance "
    "in circulation areas is 200 lux."
)
CONC_TEXT = (
    "Project concrete specification. The maximum temperature of fresh "
    "concrete at the point of delivery is 32°C."
)
NFPA_TEXT = (
    "NFPA 51B standard for fire prevention during hot work. A fire "
    "watch shall be maintained for at least 60 minutes after completion "
    "of hot work."
)

P4B_BAD = (
    "Per NFPA 51B (2019), a fire watch must be maintained for "
    "30 minutes after hot work is complete."
)
P3B_BAD = "Dubai Municipality requires a maintained illuminance of 200 lux."
P6B_BAD = (
    "ACI 305 limits the maximum temperature of fresh concrete to 32°C."
)


def _chunk(cid, doc_id, text, *, source_name="", layer="own", score=0.81):
    return Chunk(
        chunk_id=cid,
        project_id="p_fixture",
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
        source_name=source_name,
        layer=layer,
    )


def _rag(chunks, query):
    return format_chunks_as_system_message(chunks, len(chunks), query=query)


def _msgs(ask):
    return [{"role": "user", "content": ask}]


def _presents_number_as_standard(text: str, standard: str, number: str) -> bool:
    """True when one sentence gives ``number`` as ``standard``'s requirement."""
    std = re.compile(re.escape(standard), re.IGNORECASE)
    num = re.compile(rf"\b{re.escape(number)}\b")
    disclaimer = re.compile(
        r"project[-\s]?only|project requirement",
        re.IGNORECASE,
    )
    not_the_code = re.compile(
        rf"\bnot\b.{{0,80}}{re.escape(standard)}",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not std.search(sentence) or not num.search(sentence):
            continue
        if disclaimer.search(sentence) or not_the_code.search(sentence):
            continue
        return True
    return False


def test_p4b_detector_flags_the_live_misattribution():
    """The helper must see the bug the live answer actually wrote."""
    assert _presents_number_as_standard(P4B_BAD, "NFPA 51B", "30")


def test_p4b_project_30_minutes_is_not_attributed_to_nfpa():
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_TEXT,
            source_name="project-hot-work-permit.txt",
        ),
    ]
    out = _postprocess_answer(P4B_BAD, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    assert not _presents_number_as_standard(out, "NFPA 51B", "30"), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert re.search(r"\b30\b", out), out
    assert "NAMED STANDARD ABSENT" not in out


def test_p4b_a_project_citation_of_nfpa_does_not_back_the_code():
    """The permit may name NFPA 51B. That still is not the code document."""
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_CITES_NFPA,
            source_name="project-hot-work-permit.txt",
            layer="master_corpus",
        ),
    ]
    out = _postprocess_answer(P4B_BAD, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    assert not _presents_number_as_standard(out, "NFPA 51B", "30"), out
    assert re.search(r"project-only", out, re.IGNORECASE), out


def test_p4b_nfpa_file_keeps_its_own_figure():
    chunks = [
        _chunk(
            "nfpa", "nfpa1", NFPA_TEXT,
            source_name="NFPA-51B-2019.txt",
            layer="master_corpus",
        ),
    ]
    answer = "NFPA 51B requires a fire watch of 60 minutes after hot work."
    out = _postprocess_answer(answer, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    assert out == answer
    assert "60" in out
    assert not re.search(r"not in the retrieved excerpts", out, re.IGNORECASE)


def test_p4b_knowledge_base_excerpt_keeps_the_code_figure():
    chunks = [
        _chunk(
            "kb", "kb1", NFPA_TEXT,
            source_name="hot-work-notes.md",
            layer="general_knowledge",
        ),
    ]
    answer = "NFPA 51B requires a fire watch of 60 minutes after hot work."
    out = _postprocess_answer(answer, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    assert out == answer


def test_project_fire_watch_question_keeps_30_minutes():
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_TEXT,
            source_name="project-hot-work-permit.txt",
        ),
    ]
    answer = "The project hot-work permit requires a fire watch of 30 minutes."
    out = _postprocess_answer(answer, _rag(chunks, PROJECT_ASK), _msgs(PROJECT_ASK))
    assert out == answer
    assert _presents_number_as_standard(out, "NFPA 51B", "30") is False


def test_p3b_project_lux_is_not_dubai_municipality():
    chunks = [
        _chunk(
            "lux", "lux1", LIGHT_TEXT,
            source_name="project-lighting-spec.txt",
        ),
    ]
    out = _postprocess_answer(P3B_BAD, _rag(chunks, P3B_ASK), _msgs(P3B_ASK))
    assert not _presents_number_as_standard(out, "Dubai Municipality", "200"), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert "200" in out


def test_p6b_project_32c_is_not_aci_305():
    chunks = [
        _chunk(
            "conc", "conc1", CONC_TEXT,
            source_name="project-concrete-spec.txt",
        ),
    ]
    out = _postprocess_answer(P6B_BAD, _rag(chunks, P6B_ASK), _msgs(P6B_ASK))
    assert not _presents_number_as_standard(out, "ACI 305", "32"), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert "32" in out


def test_already_honest_answer_is_left_in_place():
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_TEXT,
            source_name="project-hot-work-permit.txt",
        ),
    ]
    honest = (
        "NFPA 51B is not in the retrieved excerpts, so this answer "
        "cannot state what NFPA 51B requires.\n\n"
        'The project document "project-hot-work-permit.txt" states '
        "30 minutes. That figure is project-only and is not NFPA 51B's "
        "requirement."
    )
    out = _postprocess_answer(honest, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    assert out == honest


def test_inject_warns_only_when_the_named_document_is_absent():
    permit = _chunk(
        "ptw", "ptw1", PTW_TEXT, source_name="project-hot-work-permit.txt",
    )
    code = _chunk(
        "nfpa", "nfpa1", NFPA_TEXT, source_name="NFPA-51B-2019.txt",
    )
    absent = _rag([permit], P4B_ASK)["content"]
    present = _rag([code], P4B_ASK)["content"]
    project = _rag([permit], PROJECT_ASK)["content"]
    assert "NAMED STANDARD ABSENT" in absent
    assert "NFPA 51B" in absent
    assert "NAMED STANDARD ABSENT" not in present
    assert "NAMED STANDARD ABSENT" not in project


P6B_LIVE_ASK = (
    "Per ACI 305 (hot weather concreting), what maximum fresh concrete "
    "temperature is recommended?"
)
# Same file states the placing limit and three unrelated temperatures.
# Live P6b on 7c0b255 listed 32/70/20/25 °C and dropped clause 3.1.23.1.
CONC_CLAUSE_TEXT = (
    "3.1.23 Hot weather concreting.\n"
    "3.1.23.1 The temperature of fresh concrete at the time of placing "
    "shall not exceed 32°C.\n"
    "Mixing water shall not be heated above 70°C.\n"
    "When the ambient temperature is below 20°C, cold-weather concreting "
    "applies.\n"
    "Do not place concrete when the shade temperature exceeds 25°C "
    "without the Engineer's consent.\n"
)
CONC_SOURCE = "Vol 2 – Specification (4 of 11).pdf"
P6B_BARE = (
    "ACI 305 is not in the retrieved excerpts, so this answer cannot "
    "state what ACI 305 requires.\n\n"
    f'The project document "{CONC_SOURCE}" states 32°C. That figure is '
    "project-only and is not ACI 305's requirement.\n\n"
    f'The project document "{CONC_SOURCE}" states 70°C. That figure is '
    "project-only and is not ACI 305's requirement."
)
P4B_HONEST_LIVE = (
    "NFPA 51B (2019) is not in the retrieved excerpts — no excerpt is "
    "that document — so I cannot state what it requires.\n\n"
    "What the retrieved project documents do say, as a project "
    "requirement only (not an NFPA 51B citation): a fire watch shall "
    "remain for no less than 30 minutes after the hot work is finished."
)
P3B_LIVE_ASK = (
    "Per Dubai Municipality requirements, what minimum lighting level "
    "applies to general construction site areas?"
)
P3B_HONEST_LIVE = (
    "I could not confirm a Dubai Municipality minimum lighting level in "
    "the retrieved excerpts — the named Dubai Municipality document is "
    "not among them, so I cannot state what it requires.\n\n"
    "On the project's own table, general work areas are 323 lux "
    "minimum. This is a project requirement only — it is not "
    "attributable to Dubai Municipality.\n\n"
    "The location, design and style of light fittings are subject to "
    "the Engineer's approval."
)


def test_p6b_placing_limit_keeps_clause_and_drops_other_temperatures():
    """32 °C is the placing limit at clause 3.1.23.1, still not ACI's."""
    paragraph = " ".join(CONC_CLAUSE_TEXT.splitlines())
    for text in (CONC_CLAUSE_TEXT, paragraph):
        chunks = [_chunk("conc", "conc1", text, source_name=CONC_SOURCE)]
        _assert_p6b_clause(_postprocess_answer(
            P6B_BAD, _rag(chunks, P6B_LIVE_ASK), _msgs(P6B_LIVE_ASK),
        ))


def _assert_p6b_clause(out: str) -> None:
    assert not _presents_number_as_standard(out, "ACI 305", "32"), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert "3.1.23.1" in out
    assert re.search(r"32\s*°C", out), out
    assert "fresh-concrete placing temperature limit" in out
    assert CONC_SOURCE in out
    assert not re.search(r"\b70\b", out), out
    assert not re.search(r"\b20\b", out), out
    assert not re.search(r"\b25\b", out), out
    assert out.count("fresh-concrete placing temperature limit") == 1
    assert "ACI 305 limits" not in out


# Verbatim spec clause 3.1.23.1. 32 °C is the fresh-concrete placing
# limit. 70 °C is peak hydration. 20 °C (or 25 °C) is the core-to-surface
# differential. They share one sentence, so a whole-sentence "placing"
# label tags 70 °C as a placing limit.
CLAUSE_31231 = (
    "3.1.23.1 Temperature Control of Concrete Regardless of the ambient "
    "temperature, the temperature of fresh concrete during placing shall "
    "not exceed 32°C, the peak hydration temperature anywhere in the "
    "concrete element shall not exceed 70°C and the temperature difference "
    "between any point within the element and the nearest surface shall "
    "not exceed 20°C (or 25°C where the thermal expansion coefficient of "
    "the concrete is controlled to less than 10x10-6/°C)."
)
_P6B_SENT = re.compile(r"[^.\n]+")
_P6B_NONPLACING = re.compile(r"\b(70|20|25)\s*°\s*C\b", re.IGNORECASE)
_P6B_PLACING_WORD = re.compile(
    r"fresh[- ]concrete|placing|placement|as placed|during placing",
    re.IGNORECASE,
)
_P6B_ROLE_WORD = re.compile(
    r"peak|hydration|core|differential|difference|nearest surface|thermal",
    re.IGNORECASE,
)


def _p6b_false_placing_sentence(text: str) -> str:
    """Owner checker: 70/20/25 °C called a placing limit, with no other role."""
    for sentence in _P6B_SENT.findall(text or ""):
        if (
            _P6B_NONPLACING.search(sentence)
            and _P6B_PLACING_WORD.search(sentence)
            and not _P6B_ROLE_WORD.search(sentence)
        ):
            return sentence.strip()
    return ""


def test_p6b_verbatim_clause_does_not_label_70c_as_placing():
    """32 °C is the placing limit. 70 °C in the same sentence is not."""
    chunks = [
        _chunk("conc", "conc1", CLAUSE_31231, source_name=CONC_SOURCE),
    ]
    out = _postprocess_answer(
        P6B_BAD, _rag(chunks, P6B_LIVE_ASK), _msgs(P6B_LIVE_ASK),
    )
    false_placing = _p6b_false_placing_sentence(out)
    assert not false_placing, (
        f"presents a non-placing figure as the placing limit: {false_placing}"
    )
    assert "3.1.23.1" in out
    assert re.search(r"32\s*°\s*C", out), out
    assert "fresh-concrete placing temperature limit" in out
    assert out.count("fresh-concrete placing temperature limit") == 1
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert re.search(r"not in the retrieved excerpts", out, re.IGNORECASE), out
    for number in ("32", "70", "20", "25"):
        assert not _presents_number_as_standard(out, "ACI 305", number), out


def test_p6b_templated_70c_placing_line_is_removed():
    """The live template that calls 70 °C a placing limit must not be kept."""
    templated = (
        "ACI 305 is not in the retrieved excerpts, so this answer cannot "
        "state what ACI 305 requires.\n\n"
        f'The project document "{CONC_SOURCE}" states, at clause 3.1.23.1, '
        "that 32 °C is the fresh-concrete placing temperature limit. "
        "That figure is project-only and is not ACI 305's requirement.\n\n"
        f'The project document "{CONC_SOURCE}" states, at clause 3.1.23.1, '
        "that 70 °C is the fresh-concrete placing temperature limit. "
        "That figure is project-only and is not ACI 305's requirement."
    )
    chunks = [
        _chunk("conc", "conc1", CLAUSE_31231, source_name=CONC_SOURCE),
    ]
    out = _postprocess_answer(
        templated, _rag(chunks, P6B_LIVE_ASK), _msgs(P6B_LIVE_ASK),
    )
    false_placing = _p6b_false_placing_sentence(out)
    assert not false_placing, (
        f"presents a non-placing figure as the placing limit: {false_placing}"
    )
    assert re.search(r"32\s*°\s*C", out), out
    assert "fresh-concrete placing temperature limit" in out
    assert not _presents_number_as_standard(out, "ACI 305", "32"), out
    assert not _presents_number_as_standard(out, "ACI 305", "70"), out


def test_p6b_already_labelled_temperatures_still_gain_the_clause():
    """An honest bare list is not a reason to drop clause 3.1.23.1."""
    chunks = [
        _chunk("conc", "conc1", CONC_CLAUSE_TEXT, source_name=CONC_SOURCE),
    ]
    out = _postprocess_answer(
        P6B_BARE, _rag(chunks, P6B_LIVE_ASK), _msgs(P6B_LIVE_ASK),
    )
    assert "3.1.23.1" in out
    assert "fresh-concrete placing temperature limit" in out
    assert not _presents_number_as_standard(out, "ACI 305", "32"), out
    assert re.search(r"project-only", out, re.IGNORECASE), out
    assert not re.search(r"\b70\b", out), out
    assert not re.search(r"\b20\b", out), out
    assert not re.search(r"\b25\b", out), out
    assert out.count("fresh-concrete placing temperature limit") == 1


def test_p4b_honest_fire_watch_stays_unattributed_to_nfpa():
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_TEXT,
            source_name="project-hot-work-permit.txt",
        ),
    ]
    out = _postprocess_answer(
        P4B_HONEST_LIVE, _rag(chunks, P4B_ASK), _msgs(P4B_ASK),
    )
    assert out == P4B_HONEST_LIVE
    assert not _presents_number_as_standard(out, "NFPA 51B", "30"), out
    assert re.search(r"\b30\b", out)


def test_p3b_honest_lux_stays_unattributed_to_dubai_municipality():
    chunks = [
        _chunk(
            "lux", "lux1",
            "General work areas: minimum illumination 323 lux.",
            source_name="project-lighting-spec.txt",
        ),
    ]
    out = _postprocess_answer(
        P3B_HONEST_LIVE, _rag(chunks, P3B_LIVE_ASK), _msgs(P3B_LIVE_ASK),
    )
    assert "323" in out
    assert not _presents_number_as_standard(
        out, "Dubai Municipality", "323",
    ), out
    assert re.search(r"project requirement", out, re.IGNORECASE), out
    assert "Standards note" not in out
    assert "design review and acceptance procedure" not in out


def test_kill_switch_leaves_the_misattribution():
    chunks = [
        _chunk(
            "ptw", "ptw1", PTW_TEXT,
            source_name="project-hot-work-permit.txt",
        ),
    ]
    import os
    os.environ["NAMED_STANDARD_ATTRIBUTION_GATE"] = "0"
    try:
        out = _postprocess_answer(P4B_BAD, _rag(chunks, P4B_ASK), _msgs(P4B_ASK))
    finally:
        os.environ.pop("NAMED_STANDARD_ATTRIBUTION_GATE", None)
    assert _presents_number_as_standard(out, "NFPA 51B", "30"), out
