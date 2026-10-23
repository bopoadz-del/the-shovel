"""AN EXCERPT MUST SAY WHAT KIND OF SOURCE IT IS (owner's numbered item 2).

Two failures in ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``,
one cause. Both are quoted here verbatim from that pack:

    G1 | FAIL -- asserted Schedule 10 "sets out any applicable Works
    Guarantees" and quoted CONTRACT TEMPLATE wording. Expected: "Schedule
    10: Not Used", must not invent contents. It invented contents by
    importing a template.

    F-KB-1 -- generic knowledge displacing project retrieval (A5). The
    answer reproduces docs/knowledge/fidic_2017_administration.md almost
    verbatim ... instead of the project's own 0.1% at 8.8.1, which A7/E2
    show is retrievable. The knowledge base answered a question the corpus
    could answer better.

The battery questions are quoted from the instrument itself
(``UI-PHYS_QP2_results.xlsx``, column "Question (ask exactly)"), by way of
the sanitized catalog at ``tests/fixtures/ui_phys/questions.json`` which
carries the same strings. No paraphrase: a PASS obtained on a paraphrase is
not evidence about the battery question, which is F-PHRASE-1's whole point.

What this file can and cannot prove is worth stating, because the difference
is where the honest limit sits. It proves that the class is DERIVED
correctly, that it REACHES the model in the marker, that the precedence rule
is STATED when it can matter, and that the citation guard ENFORCES it. It
does not prove the model then obeys the instruction -- only the re-battery
on the deployed SHA can, and that is the next-but-one item in the order.
"""

import json
from pathlib import Path

import pytest

from app.agents.citation_provenance import (
    UNVERIFIED_NOTE,
    build_evidence,
    gate,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.source_class import (
    DEFAULT_CLASS,
    SOURCE_CLASSES,
    classify,
    classify_chunk,
    name_marks_template,
    placeholder_kinds,
    text_marks_template,
)
from app.core.rag.vector_store import Chunk

CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json").read_text(
        encoding="utf-8"
    )
)
G1_ASK = CATALOG["cases"]["G1"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]


def _chunk(idx, layer, name, text, score=0.7):
    c = Chunk(
        chunk_id=f"c{idx}",
        project_id="p",
        doc_id=f"d{idx}",
        chunk_index=idx,
        text=text,
        score=score,
    )
    c.layer = layer
    c.source_name = name
    return c


# -- the questions are the sheet's, not mine ------------------------------


def test_the_battery_questions_used_here_are_the_instrument_s_own():
    """Mutation killed: quietly rewording a battery question so a fence
    passes. F-PHRASE-1 measured that the sheet's phrasing is the one that
    fails, so a fence written against a paraphrase proves nothing."""
    assert G1_ASK == "What does Schedule 10 of the contract contain?"
    assert A5_ASK == "What are the Delay Damages for the whole of the Works?"


# -- derivation -----------------------------------------------------------


def test_the_active_project_is_the_project_corpus():
    assert classify("own", "S1_contract_data.md", "Schedule 10: Not Used") == (
        "project_corpus"
    )


def test_curated_cross_project_reference_is_the_knowledge_base():
    """A5's displacing source, by its real filename."""
    assert classify(
        "general_knowledge",
        "fidic_2017_administration.md",
        "Pre-agreed rate; capped in Contract Data",
    ) == "knowledge_base"


def test_the_disclosed_fallback_corpus_is_named_rather_than_folded_in():
    """Mutation killed: mapping master_corpus onto knowledge_base, which
    would strip the attribution off every answer on the fallback path, or
    onto project_corpus, which would license another project's documents as
    this one's own words."""
    assert classify("master_corpus", "Another Project Vol1.pdf", "x") == (
        "master_corpus"
    )
    assert set(SOURCE_CLASSES) == {
        "project_corpus", "knowledge_base", "template", "master_corpus",
    }


def test_an_unknown_layer_defaults_to_the_citable_class():
    """A chunk built by a path that never set ``layer`` must not be silently
    demoted -- an unclassified excerpt losing its citation would be a
    regression caused by the fix.

    Mutation killed: defaulting to "knowledge_base"."""
    assert classify(None, "x.pdf", "body") == DEFAULT_CLASS
    assert classify("", "x.pdf", "body") == "project_corpus"
    assert classify("something_new", "x.pdf", "body") == "project_corpus"


@pytest.mark.parametrize("name", [
    "Schedule 10 template.pdf",
    "Contract TEMPLATE - works guarantees.docx",
    "Pro Forma Bank Guarantee.pdf",
    "pro-forma warranty.pdf",
    "Specimen Performance Bond.pdf",
    "Blank Variation Order.pdf",
    "Model Form of Agreement.pdf",
    "Standard Form of Contract.pdf",
    "Unexecuted subcontract.pdf",
])
def test_a_document_that_names_itself_a_blank_form_is_a_template(name):
    assert name_marks_template(name), name
    assert classify("own", name, "some wording") == "template"


@pytest.mark.parametrize("name", [
    "Schedule 8 form of Parent Company Guarantee.pdf",
    "Form of Tender.pdf",
    "Contemporary records.pdf",
    "Templeton Road drainage.pdf",
    "Blanket wayleave agreement.pdf",
])
def test_a_real_contract_document_is_not_demoted_by_a_near_miss(name):
    """The filename still must not be demoted by the bare word "form".
    G3 honesty (Contract Data 4.3.7 "not required" over the Schedule 8
    blank %) is a dedicated retriever/compose gate, not this classifier.
    A rule that fired on "form" -- or on "Templeton", or "Blanket" --
    would reclassify live contract documents as blank ones.

    Mutation killed: dropping the word boundaries from the name pattern.
    """
    assert not name_marks_template(name), name
    assert classify("own", name, "10% of the Accepted Contract Amount") == (
        "project_corpus"
    )


def test_the_template_class_beats_the_layer():
    """G1 exactly: the template that answered "what does Schedule 10
    contain" was reachable as the project's own. A blank form in the
    project's own folder is still a blank form.

    Mutation killed: deciding layer first, which returns project_corpus and
    licenses the template's wording as the contract's.
    """
    assert classify(
        "own",
        "AB-2023-101 Contract Template Vol 4.pdf",
        "Schedule 10 sets out any applicable Works Guarantees",
    ) == "template"


@pytest.mark.parametrize("layer", ["own", "general_knowledge", "master_corpus", None])
def test_the_template_class_beats_every_layer_not_just_the_project_s(layer):
    """Mutation killed: checking the layer first and only falling back to the
    template test when the layer said project_corpus. A blank form reached
    through the knowledge base is still a blank form, and the header's
    "do not describe what such a section contains in a standard form" is
    written against class=template, not against class=knowledge_base.
    """
    assert classify(
        layer,
        "Standard Form of Contract Vol 4.pdf",
        "Schedule 10 sets out any applicable Works Guarantees",
    ) == "template"


@pytest.mark.parametrize("name", [
    "BOQtemplate.xlsx",
    "DWGTEMPLATE-A.pdf",
    "Vol4_PROFORMA.pdf",
])
def test_a_marker_glued_to_a_discipline_code_still_reads_as_a_template(name):
    """Engineering filenames run words together. Mutation killed: a leading
    word boundary on the name pattern, which protects nothing on the
    near-miss list above and misses the shape these names actually take."""
    assert name_marks_template(name), name


def test_one_placeholder_shape_is_not_evidence_of_anything():
    """A scanned table of dotted rules is one shape and is not a template.

    Mutation killed: _MIN_PLACEHOLDER_KINDS = 1, which would reclassify
    ordinary scanned contract pages and silently strip their citations.
    """
    one_kind = "Signed ____________ on ____________ by ____________"
    assert placeholder_kinds(one_kind) == 1
    assert not text_marks_template(one_kind)
    assert classify("own", "Signature page.pdf", one_kind) == "project_corpus"


def test_an_excerpt_that_is_mostly_placeholders_is_a_template():
    body = "THIS AGREEMENT is made between [INSERT NAME] of <<ADDRESS>> on ____________."
    assert placeholder_kinds(body) >= 2
    assert text_marks_template(body)
    assert classify("own", "Volume 4.pdf", body) == "template"


def test_classify_chunk_survives_a_partially_built_object():
    """The classifier runs on the answer path. An object without the
    attributes must produce a class, not an AttributeError."""
    class Bare:
        pass

    assert classify_chunk(Bare()) == DEFAULT_CLASS
    assert classify_chunk(_chunk(1, "general_knowledge", "kb.md", "x")) == (
        "knowledge_base"
    )


def test_a_classifier_failure_falls_back_to_the_citable_class():
    """The classifier runs on the answer path. If it throws, the excerpt
    must default to the class that KEEPS its citation -- an unclassifiable
    chunk silently becoming knowledge_base would strip correct attributions
    off answers as a side effect of an unrelated bug.

    Mutation killed: returning "knowledge_base" from the except branch.
    """
    from app.core.rag import inject as _inject

    class Exploding:
        layer = "own"

        @property
        def source_name(self):
            raise RuntimeError("boom")

    assert _inject._source_class(Exploding()) == DEFAULT_CLASS
    assert DEFAULT_CLASS == "project_corpus"


# -- the class reaches the model ------------------------------------------


def test_every_excerpt_carries_its_class_in_the_marker():
    msg = format_chunks_as_system_message(
        [
            _chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used"),
            _chunk(2, "general_knowledge", "fidic_2017_administration.md", "generic"),
        ],
        12,
    )
    lines = [l for l in msg["content"].splitlines() if l.startswith("[doc_id=")]
    assert len(lines) == 2
    assert "class=project_corpus" in lines[0]
    assert "class=knowledge_base" in lines[1]


def test_the_class_attribute_does_not_switch_off_the_internal_leak_guard():
    """THE regression this ordering exists to prevent.

    ``_RETRIEVAL_MARKER_RE`` and ``_EXCERPT_SPLIT_RE`` both anchor on
    "chunk=N score=" being ADJACENT. Inserting class= between them would
    leave every test in this file green while silently disabling the guard
    shipped in #457/#463 -- an internal-telemetry leak into a client answer.

    Mutation killed: emitting class= before score=.
    """
    from app.agents.runtime import _EXCERPT_SPLIT_RE, _RETRIEVAL_MARKER_RE

    msg = format_chunks_as_system_message(
        [_chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used")], 3
    )
    marker = next(l for l in msg["content"].splitlines() if l.startswith("[doc_id="))
    assert _RETRIEVAL_MARKER_RE.search(marker), marker
    assert len(_EXCERPT_SPLIT_RE.split("head " + marker)) == 2


def test_the_source_name_still_parses_with_a_class_present():
    """src= is read by a lookahead that ends it at the next ``word=``. The
    class attribute must not eat the filename or be eaten by it."""
    msg = format_chunks_as_system_message(
        [_chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used")], 3
    )
    records = build_evidence(msg, None)
    assert [r.source_name for r in records.records] == ["S1_contract_data.md"]


def test_the_marker_the_injector_writes_is_the_marker_the_guard_reads():
    """The round trip, end to end, rather than two halves that agree with
    their own fixtures. Mutation killed: renaming the attribute on one side.
    """
    msg = format_chunks_as_system_message(
        [
            _chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used"),
            _chunk(2, "general_knowledge", "fidic_2017_administration.md", "generic"),
            _chunk(3, "master_corpus", "Other Project.pdf", "elsewhere"),
            _chunk(4, "own", "Contract Template Vol 4.pdf", "standard wording"),
        ],
        40,
    )
    got = [r.source_class for r in build_evidence(msg, None).records]
    assert got == ["project_corpus", "knowledge_base", "master_corpus", "template"]


# -- the precedence rule is stated ----------------------------------------


def test_the_precedence_rule_is_stated_when_the_classes_are_mixed():
    msg = format_chunks_as_system_message(
        [
            _chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used"),
            _chunk(2, "general_knowledge", "fidic_2017_administration.md", "generic"),
        ],
        12,
    )
    text = msg["content"]
    assert "SOURCE CLASS" in text
    assert "PRECEDENCE" in text
    # A5: the project's own document wins when both can answer.
    assert "project_corpus excerpt IS the answer" in text
    # G1: "Not Used" is an answer, not a gap to fill from a standard form.
    assert "Not Used" in text
    assert "do not describe what such a section contains in a standard form" in text


def test_the_rule_is_not_repeated_when_it_cannot_change_anything():
    """Mutation killed: emitting the block unconditionally. On a
    single-class set it cannot change any answer, and an instruction that
    never applies costs context and dilutes the ones that do."""
    msg = format_chunks_as_system_message(
        [
            _chunk(1, "own", "S1_contract_data.md", "Schedule 10: Not Used"),
            _chunk(2, "own", "S1_contract_data.md", "Schedule 9: H&S KPIs"),
        ],
        12,
    )
    assert "SOURCE CLASS" not in msg["content"]


# -- the guard enforces it ------------------------------------------------


def _answer_citing(doc_id: str) -> str:
    return (
        "Schedule 10 sets out any applicable Works Guarantees.\n\n"
        f"Source: {doc_id}\n"
    )


def test_a_template_cannot_lend_its_id_to_a_claim_about_the_contract(monkeypatch):
    """G1, at the attribution surface. The template is a real document and
    was really read; what it may not do is back "per AB-2023-101"."""
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "1")
    rag = {
        "role": "system",
        "content": (
            "AUTHORITATIVE REFERENCE CONTEXT\n"
            "[doc_id=d1 chunk=1 score=0.700 class=template "
            "src=AB-2023-101 Contract Template Vol 4.pdf] "
            "Schedule 10 sets out any applicable Works Guarantees\n"
        ),
    }
    out = gate(
        _answer_citing("AB-2023-101"),
        rag,
        [{"role": "user", "content": G1_ASK}],
    )
    assert "AB-2023-101" not in out
    assert UNVERIFIED_NOTE.strip() in out


def test_the_knowledge_base_cannot_lend_its_id_either(monkeypatch):
    """A5's shape: the reference note is background, not the contract."""
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "1")
    rag = {
        "role": "system",
        "content": (
            "AUTHORITATIVE REFERENCE CONTEXT\n"
            "[doc_id=d1 chunk=1 score=0.700 class=knowledge_base "
            "src=fidic_2017_administration.md] "
            "Pre-agreed rate; capped in Contract Data (AB-2019-002)\n"
        ),
    }
    answer = (
        "Delay damages are a pre-agreed rate capped in the Contract Data.\n\n"
        "Source: AB-2019-002\n"
    )
    out = gate(answer, rag, [{"role": "user", "content": A5_ASK}])
    assert "AB-2019-002" not in out
    assert UNVERIFIED_NOTE.strip() in out


def test_a_reference_note_may_still_be_named_as_the_source_it_is(monkeypatch):
    """The class removes a licence, not a document. Mutation killed:
    stripping every attribution from a non-project class, which would delete
    correct citations of the knowledge base."""
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "1")
    rag = {
        "role": "system",
        "content": (
            "AUTHORITATIVE REFERENCE CONTEXT\n"
            "[doc_id=d1 chunk=1 score=0.700 class=knowledge_base "
            "src=fidic_2017_administration.md] "
            "Delay damages under FIDIC 2017 are a pre-agreed rate\n"
        ),
    }
    answer = (
        "Under FIDIC 2017 delay damages are a pre-agreed rate.\n\n"
        "Source: fidic_2017_administration.md\n"
    )
    assert gate(answer, rag, [{"role": "user", "content": A5_ASK}]) == answer


def test_the_disclosed_fallback_path_keeps_its_attributions(monkeypatch):
    """When the master corpus is in play it is the only corpus there is and
    the runtime already labels the answer a fallback. Refusing its
    identifiers would strip the attribution off every answer on that path.

    Mutation killed: dropping master_corpus from CITABLE_CLASSES.
    """
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "1")
    rag = {
        "role": "system",
        "content": (
            "AUTHORITATIVE REFERENCE CONTEXT\n"
            "[doc_id=d1 chunk=1 score=0.700 class=master_corpus "
            "src=AB-2021-044 Vol1.pdf] Delay damages are 0.1% per calendar day\n"
        ),
    }
    answer = "Delay damages are 0.1% per calendar day.\n\nSource: AB-2021-044\n"
    assert gate(answer, rag, [{"role": "user", "content": A5_ASK}]) == answer


def test_the_project_s_own_record_is_untouched(monkeypatch):
    """The gate removes invention, not correct attribution."""
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "1")
    rag = {
        "role": "system",
        "content": (
            "AUTHORITATIVE REFERENCE CONTEXT\n"
            "[doc_id=d1 chunk=1 score=0.700 class=project_corpus "
            "src=AB-2023-101_Vol1.pdf] Schedule 10: Not Used\n"
        ),
    }
    answer = "Schedule 10 is Not Used.\n\nSource: AB-2023-101\n"
    assert gate(answer, rag, [{"role": "user", "content": G1_ASK}]) == answer
