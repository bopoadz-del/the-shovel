"""Driver answer hygiene (lane 4 CORE faults), synthetic data only."""
from __future__ import annotations

import json

import pytest

from app.agents.driver import loop, sources
from app.lib import formula_registry as fr
from app.lib.source_labels import calculator_label


@pytest.fixture
def synthetic_formula():
    def syn_core_check(core_temp_c: float, udl_w: float = 1.0):
        return {"ok": True}
    fr._REGISTRY.pop("syn_core_check", None)
    fr.formula(owner="base", description="Synthetic check.", display_name="Synthetic core check",
               inputs={"core_temp_c": fr.Param(unit="degC", lo=-50, hi=120, label="core temperature"),
                       "udl_w": fr.Param(unit="kN/m", lo=0, hi=1000)},
               outputs={"ok": "-"})(syn_core_check)
    yield
    fr._REGISTRY.pop("syn_core_check", None)


def test_the_credit_line_names_inputs_by_their_declared_labels(synthetic_formula):
    label = calculator_label("syn_core_check", {"core_temp_c": 68, "udl_w": 20})
    assert label.startswith("Synthetic core check — platform calculator (core temperature 68")
    assert "core temp c" not in label


def _trail(hits):
    return [{"role": "tool", "name": "search_general_knowledge",
             "content": json.dumps({"ok": True, "result": {"results": hits}})}]


def test_a_readable_source_label_is_shown_instead_of_the_file_id():
    out = sources.panel(_trail([{"document_id": "g1", "filename": "syn_wind_design_values.md",
                                 "source_label": "Wind design values", "score": 0.9,
                                 "origin": "general_knowledge", "snippet": "x"}]), [])
    assert out[0]["doc_name"] == "Wind design values"


def test_a_page_that_is_not_a_page_is_never_shown():
    out = sources.panel(_trail([{"document_id": "g1", "filename": "Spec.pdf", "page": -1, "score": 0.9,
                                 "origin": "own", "snippet": "x"}]), [])
    assert out[0]["page_or_section"] == ""
    cleaned = loop._without_missing_pages({"result": {"results": [{"page": -1, "snippet": "s"}, {"page": 4}]}})
    assert cleaned == {"result": {"results": [{"snippet": "s"}, {"page": 4}]}}


def test_a_calculator_answer_lists_only_passages_it_used():
    hits = [{"document_id": "g1", "filename": "Unrelated section.md", "score": 0.95,
             "origin": "general_knowledge", "snippet": "Sea defences are designed for surge heights."},
            {"document_id": "g2", "filename": "Formwork notes.md", "score": 0.6, "origin": "general_knowledge",
             "snippet": "Strike soffit formwork only when the concrete has reached the stated strength."}]
    answer = ("The striking time is 7 days. As the notes put it: strike soffit formwork only when the "
              "concrete has reached the stated strength.")
    out = sources.panel(_trail(hits), [], answer, calculated=True)
    assert [s["doc_name"] for s in out] == ["Formwork notes.md"]
    assert sources.panel(_trail(hits[:1]), [], "The striking time is 7 days.", calculated=True) == []


def test_brackets_emptied_by_the_exit_check_are_removed():
    assert loop._tidy("The volume is 6 m3 (). Rate ( ; ) applies [ ].") == "The volume is 6 m3. Rate applies."
    assert loop._tidy("Total (incl. VAT) is due.") == "Total (incl. VAT) is due."
