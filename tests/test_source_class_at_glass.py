"""Source class must be visible at the glass — Sources panel AND docx footer.

Consumes #468 ``classify`` / ``source_class``. Does not retag.

G1 / A5 questions are the instrument's own strings (no paraphrase).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from app.agents.runtime import _build_sources_from_audit
from app.core.rag.source_class import SOURCE_CLASS_LABELS, classify, label_for, visible_label
from app.routers.exports import _render_message_docx

CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json").read_text(
        encoding="utf-8"
    )
)
G1_ASK = CATALOG["cases"]["G1"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]

FRONTEND_LABELS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "chat" / "sourceClassLabels.ts"
SOURCES_LIST = Path(__file__).resolve().parents[1] / "frontend" / "src" / "chat" / "SourcesList.tsx"


def test_the_battery_questions_used_here_are_the_instrument_s_own():
    assert G1_ASK == "What does Schedule 10 of the contract contain?"
    assert A5_ASK == "What are the Delay Damages for the whole of the Works?"


def test_visible_labels_are_the_four_the_owner_named():
    assert SOURCE_CLASS_LABELS["project_corpus"] == "this contract"
    assert SOURCE_CLASS_LABELS["master_corpus"] == "master corpus"
    assert SOURCE_CLASS_LABELS["knowledge_base"] == "knowledge base"
    assert SOURCE_CLASS_LABELS["template"] == "template"


def test_frontend_label_map_matches_the_backend():
    ts = FRONTEND_LABELS.read_text(encoding="utf-8")
    for cls, label in SOURCE_CLASS_LABELS.items():
        assert f"{cls}:" in ts or f"{cls} :" in ts
        assert label in ts


def test_sources_list_renders_class_in_the_dom_contract():
    """The machine id stays on a data attribute. The visible row is the label."""
    src = SOURCES_LIST.read_text(encoding="utf-8")
    assert "data-source-class" in src
    assert "data-testid=\"source-class\"" in src
    assert "class=${sourceClass}" not in src
    assert "sources-list__class-code" not in src
    assert "data-source-class={sourceClass}" in src
    assert "source_class_label" in src
    assert "visibleSourceClassLabel" in src


def _audit(chunks):
    return {"project_id": "p", "chunks": chunks, "user_message_preview": ""}


def _g1_chunk():
    name = "AB-2023-101 Contract Template Vol 4.pdf"
    text = "Schedule 10 sets out any applicable Works Guarantees"
    return {
        "doc_id": "d-g1",
        "chunk_index": 1,
        "chunk_id": "c-g1",
        "project_id": "p",
        "score": 0.81,
        "layer": "own",
        "source_name": name,
        "source_class": classify("own", name, text),
        "text": text,
    }


def _a5_chunk():
    name = "fidic_2017_administration.md"
    text = "Pre-agreed rate; capped in Contract Data"
    return {
        "doc_id": "d-a5",
        "chunk_index": 2,
        "chunk_id": "c-a5",
        "project_id": "p",
        "score": 0.77,
        "layer": "general_knowledge",
        "source_name": name,
        "source_class": classify("general_knowledge", name, text),
        "text": text,
    }


def _visible(html: str) -> str:
    """Text a person reads. Attribute values are not text."""
    import re

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _sources_dom(sources: list[dict]) -> str:
    """The Sources row markup the React component emits (string-equivalent)."""
    parts = ['<div class="sources-list" data-testid="sources-block">']
    for s in sources:
        cls = s.get("source_class") or ""
        label = visible_label(cls, s.get("source_class_label"))
        parts.append(
            f'<div class="sources-list__class" data-source-class="{cls}" '
            f'data-testid="source-class">'
            f'<span class="sources-list__class-label">{label}</span>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def test_g1_shaped_turn_renders_class_template_in_sources_dom(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "AB-2023-101 Contract Template Vol 4.pdf"},
    )
    sources = _build_sources_from_audit(_audit([_g1_chunk()]), final_text="")
    assert sources, G1_ASK
    assert sources[0]["source_class"] == "template"
    assert sources[0]["source_class_label"] == "template"
    dom = _sources_dom(sources)
    visible = _visible(dom)
    assert 'data-source-class="template"' in dom
    assert ">template<" in dom
    assert "class=template" not in visible
    assert "class=" not in visible


def test_a5_shaped_turn_renders_class_knowledge_base_in_sources_dom(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "fidic_2017_administration.md"},
    )
    sources = _build_sources_from_audit(_audit([_a5_chunk()]), final_text="")
    assert sources, A5_ASK
    assert sources[0]["source_class"] == "knowledge_base"
    assert sources[0]["source_class_label"] == "knowledge base"
    dom = _sources_dom(sources)
    visible = _visible(dom)
    assert 'data-source-class="knowledge_base"' in dom
    assert "class=knowledge_base" not in visible
    assert "knowledge_base" not in visible
    assert "knowledge base" in visible


def test_g1_and_a5_class_strings_reach_the_docx_footer_xml(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "fixture.pdf"},
    )
    g1 = _build_sources_from_audit(_audit([_g1_chunk()]), final_text="")
    a5 = _build_sources_from_audit(_audit([_a5_chunk()]), final_text="")
    path = _render_message_docx(
        "Fixture Project",
        "Schedule 10 is Not Used.\n",
        "conv-glass-1",
        -1,
        "https://theshovel.ai",
        sources=g1 + a5,
    )
    with zipfile.ZipFile(path) as z:
        part = next(n for n in z.namelist() if n.startswith("word/footer") and n.endswith(".xml"))
        footer = z.read(part).decode()
        body = z.read("word/document.xml").decode()
    assert "class=template" not in footer
    assert "template" in footer
    assert "class=knowledge_base" not in footer
    assert "knowledge_base" not in footer
    assert "knowledge base" in footer
    assert "class=template" not in body
    assert "class=knowledge_base" not in body
    assert "knowledge_base" not in body
    assert "knowledge base" in body


def test_glass_consumes_classify_it_does_not_retarget_a_class():
    """A project_corpus excerpt stays this contract at the glass."""
    assert classify("own", "S1_contract_data.md", "Schedule 10: Not Used") == "project_corpus"
    assert label_for("project_corpus") == "this contract"
    assert label_for("master_corpus") == "master corpus"


def test_an_unseen_class_is_words_not_an_id():
    assert label_for("alpha_beta") == "alpha beta"
    assert visible_label("alpha_beta", "alpha_beta") == "alpha beta"
    assert visible_label("project_corpus", "project_corpus") == "this contract"
    assert "class=" not in label_for("alpha_beta")
