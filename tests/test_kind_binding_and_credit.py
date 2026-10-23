"""Kind-typed binding and calculator credits.

A paraphrase that names a formula and writes each required figure as its
declared kind — a bare number, a percent, a length — selects that formula
without repeating parameter names. Two figures of one kind are not guessed
apart. When a calculator is credited, a retrieval sentence that names no
evidence document does not survive, and the Sources title is the registry
label of the inputs the user stated.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from app.agents.citation_provenance import gate
from app.agents.runtime import (
    _build_sources_from_audit,
    _platform_calculator_source,
    _postprocess_answer,
)
from app.lib.construction_formulas import (
    describe_calculation_params,
    formula_completed_by_user_text,
    run_calculation,
)
from app.lib.formula_registry import all_specs
from app.lib.source_labels import calculator_label

_BARE = {"", "-", "currency", "ratio", "count", "no", "nr"}

DELAY_PARAPHRASE = (
    "Work out the delay damages per day for a contract of 12,000,000 "
    "at 0.05% per day."
)
DELAY_AMBIGUOUS = (
    "Work out the delay damages per day for 12,000,000 and 99 at 0.05% per day."
)
SLAB_ASK = "Calculate the concrete volume for a slab 15 m by 8 m, 0.2 m thick."
MODEL_SOURCE = (
    "Source: Platform calculator → Concrete volume (geometry). "
    "The 5% allowance is the calculator's default"
)


def _kind(unit: str) -> str:
    text = str(unit or "").strip().lower().replace("²", "2").replace("³", "3")
    text = text.replace(" ", "").replace(".", "")
    return "" if text in _BARE else text


def _scalar(param: inspect.Parameter) -> bool:
    ann = getattr(param.annotation, "__name__", "") or str(param.annotation)
    low = ann.lower()
    if any(tok in low for tok in ("str", "list", "dict", "bool", "tuple")):
        return False
    if param.default is not inspect.Parameter.empty and isinstance(
        param.default, (str, list, dict, tuple, bool)
    ):
        return False
    return True


def _figure(unit: str, index: int) -> tuple[str, float]:
    kind = _kind(unit)
    if kind == "":
        value = 12_000_000 + index * 1_000
        return f"{value:,}", float(value)
    if kind == "%":
        value = round(0.05 + index * 0.01, 2)
        return f"{value:g}%", float(value)
    value = 12 + index * 3
    shown = (unit or "").strip() or kind
    return f"{value:g} {shown}", float(value)


def _cases() -> tuple[list, list]:
    """Unique-kind formulas, and formulas whose required inputs share a kind."""
    positive: list = []
    ambiguous: list = []
    for spec in all_specs():
        rows = describe_calculation_params(spec.fn, name=spec.name)
        sig = inspect.signature(spec.fn)
        required = [row for row in rows if row.get("required")]
        if not required:
            continue
        if any(not _scalar(sig.parameters[row["name"]]) for row in required):
            continue
        words = re.findall(r"[a-z0-9]+", (spec.display_name or "").lower())
        if not [w for w in words if len(w) >= 3]:
            continue
        kinds = [_kind((spec.inputs or {}).get(row["name"]) or row.get("unit") or "")
                 for row in required]
        phrases = []
        expected: dict[str, float] = {}
        blocked = False
        for index, row in enumerate(required):
            unit = (spec.inputs or {}).get(row["name"]) or row.get("unit") or ""
            phrase, value = _figure(unit, index)
            lowered = phrase.lower()
            if row["name"].lower() in lowered or row["name"].replace("_", " ") in lowered:
                blocked = True
                break
            phrases.append(phrase)
            expected[row["name"]] = value
        if blocked or not phrases:
            continue
        ask = f"Work out the {spec.display_name} for " + ", ".join(phrases) + "."
        item = (spec.name, ask, expected)
        if len(kinds) == len(set(kinds)):
            positive.append(item)
        else:
            ambiguous.append((spec.name, ask))
    return positive, ambiguous


POSITIVE, AMBIGUOUS = _cases()


def test_kind_paraphrases_cover_the_registry():
    assert len(POSITIVE) >= 15
    names = {name for name, _ask, _expected in POSITIVE}
    assert "delay_damages_daily" in names


@pytest.mark.parametrize(
    "name,ask,expected",
    POSITIVE,
    ids=[name for name, _ask, _expected in POSITIVE],
)
def test_kind_paraphrase_selects_the_formula_without_parameter_names(name, ask, expected):
    figure = ask.split(" for ", 1)[1]
    for key in expected:
        assert key not in figure.lower()
        assert key.replace("_", " ") not in figure.lower()
    got = formula_completed_by_user_text(ask)
    assert got is not None, ask
    assert got[0] == name
    for key, value in expected.items():
        assert got[1][key] == pytest.approx(value)


@pytest.mark.parametrize(
    "name,ask",
    AMBIGUOUS,
    ids=[name for name, _ask in AMBIGUOUS],
)
def test_shared_kind_is_not_guessed(name, ask):
    got = formula_completed_by_user_text(ask)
    assert got is None or got[0] != name


def test_delay_paraphrase_binds_bare_amount_and_percent():
    got = formula_completed_by_user_text(DELAY_PARAPHRASE)
    assert got is not None
    assert got[0] == "delay_damages_daily"
    assert got[1]["contract_amount"] == pytest.approx(12_000_000)
    assert got[1]["rate_percent"] == pytest.approx(0.05)


def test_two_bare_numbers_are_not_assigned_to_the_one_amount():
    assert formula_completed_by_user_text(DELAY_AMBIGUOUS) is None


def _pp(ask: str, answer: str, messages: list | None = None, retrieval: str = "") -> str:
    turned = list(messages or [{"role": "user", "content": ask}])
    return _postprocess_answer(
        answer,
        {"content": retrieval},
        turned,
        project_id=None,
        audit_rec={"user_message_preview": ask, "chunks": []},
    )


def test_unnamed_retrieval_citation_does_not_survive_beside_the_calculator(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "curing-notes.pdf"},
    )
    answer = (
        "Daily delay damages are 6,000. "
        "The retrieved FIDIC commentary indicates a cap on the total is normally agreed."
    )
    out = _pp(DELAY_PARAPHRASE, answer)
    assert "6,000" in out
    assert "commentary" not in out.lower()
    assert "platform calculator" in out
    audit = {
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "score": 0.91, "chunk_id": "c0",
             "layer": "general_knowledge"},
        ],
        "user_message_preview": DELAY_PARAPHRASE,
    }
    sources = _build_sources_from_audit(audit, out)
    names = " ".join(s.get("doc_name") or "" for s in sources)
    assert "platform calculator" in names
    assert "curing-notes.pdf" not in names
    assert "commentary" not in names.lower()


def test_named_evidence_stays_in_the_answer_and_in_sources(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "curing-notes.pdf"},
    )
    answer = (
        "Daily delay damages are 6,000. "
        "The retrieved commentary in curing-notes.pdf indicates a cap on the total."
    )
    out = _pp(
        DELAY_PARAPHRASE,
        answer,
        retrieval="[doc_id=d1 chunk=0 src=curing-notes.pdf] Keep the surface damp.",
    )
    assert "curing-notes.pdf" in out
    audit = {
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "score": 0.91, "chunk_id": "c0",
             "layer": "own"},
        ],
        "user_message_preview": DELAY_PARAPHRASE,
    }
    sources = _build_sources_from_audit(audit, out)
    names = [s.get("doc_name") or "" for s in sources]
    assert any("platform calculator" in name for name in names)
    assert any("curing-notes.pdf" in name for name in names)


def test_document_lookup_keeps_its_retrieval_sentence():
    ask = "What are the delay damages under this contract?"
    answer = "The retrieved commentary indicates a cap is normally agreed."
    out = _pp(ask, answer)
    assert "commentary" in out.lower()


def _slab_messages() -> list[dict]:
    params = {
        "length_m": 15,
        "width_m": 8,
        "thickness_m": 0.2,
        "shape": "rectangular",
        "waste_factor": 0.05,
        "quantity": 1,
    }
    env = run_calculation("concrete_volume", dict(params))
    args = json.dumps({"calculation": "concrete_volume", "params": params})
    return [
        {"role": "user", "content": SLAB_ASK},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-slab",
                "type": "function",
                "function": {"name": "construction_calc", "arguments": args},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-slab",
            "name": "construction_calc",
            "content": json.dumps(env),
        },
    ]


def test_model_prose_is_not_a_calculator_source_title():
    row = _platform_calculator_source(MODEL_SOURCE)
    assert row is None or "geometry" not in (row.get("doc_name") or "")
    assert row is None or "allowance" not in (row.get("doc_name") or "").lower()


def _document_question(display: str) -> str:
    return f"What is the {display} stated in the project specification?"


def _explicit_calculation(display: str) -> str:
    return f"Calculate the {display}."


def _default_only_formulas():
    """Formulas a document question names, and that succeed with no operands."""
    from app.lib.construction_formulas import calculator_name_from_text

    rows = []
    for spec in all_specs():
        ask = _document_question(spec.display_name)
        if calculator_name_from_text(ask) != spec.name:
            continue
        env = run_calculation(spec.name, {})
        if not isinstance(env, dict) or env.get("status") != "success":
            continue
        rows.append((spec.name, spec.display_name))
    return rows


DEFAULT_ONLY = _default_only_formulas()


def test_default_only_document_questions_cover_the_registry():
    names = {name for name, _display in DEFAULT_ONLY}
    assert len(names) >= 4
    assert "formwork_striking_time" in names


@pytest.mark.parametrize(
    "name,display",
    DEFAULT_ONLY,
    ids=[name for name, _display in DEFAULT_ONLY],
)
def test_document_question_with_no_operands_does_not_select_a_calculator(name, display):
    """A specification question that supplies no figure is not a calculation."""
    from app.agents.runtime import (
        _forced_specific_tool,
        _message_is_formula_style_ask,
        _message_wants_named_calculator,
    )

    ask = _document_question(display)
    assert not any(ch.isdigit() for ch in ask)
    assert _message_is_formula_style_ask(ask) is False, ask
    assert _message_wants_named_calculator(ask) is False, ask
    assert _forced_specific_tool(
        [{"role": "user", "content": ask}], {"construction_calc"},
    ) is None


LIVE_FORMWORK_DOCUMENT = (
    "What is the formwork striking time for suspended slabs and the "
    "test cube frequency in the upload check note?"
)


def test_formwork_document_question_is_not_a_calculator():
    from app.agents.runtime import (
        _apply_rag_context,
        _forced_specific_tool,
        _message_is_formula_style_ask,
        _message_wants_named_calculator,
    )

    ask = LIVE_FORMWORK_DOCUMENT
    assert _message_is_formula_style_ask(ask) is False
    assert _message_wants_named_calculator(ask) is False
    assert _forced_specific_tool(
        [{"role": "user", "content": ask}], {"construction_calc"},
    ) is None
    msgs = [{"role": "user", "content": ask}]
    assert _apply_rag_context(msgs, {"content": "Strike after the cube test."}) is True
    assert "CALCULATION REQUEST" not in msgs[-1]["content"]


def test_formwork_document_answer_keeps_the_cited_document(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "upload-check-note.pdf"},
    )
    answer = (
        "The upload check note sets the striking time and the cube frequency. "
        "See upload-check-note.pdf."
    )
    out = _pp(
        LIVE_FORMWORK_DOCUMENT,
        answer,
        retrieval=(
            "[doc_id=d1 chunk=0 src=upload-check-note.pdf] "
            "Strike soffit formwork after the cube result."
        ),
    )
    assert "platform calculator" not in out.lower()
    assert "8.0" not in out
    assert "upload-check-note.pdf" in out
    sources = _build_sources_from_audit(
        {
            "chunks": [{
                "doc_id": "d1", "chunk_index": 0, "score": 0.91,
                "chunk_id": "c0", "layer": "own",
            }],
            "user_message_preview": LIVE_FORMWORK_DOCUMENT,
        },
        out,
    )
    names = [s.get("doc_name") or "" for s in sources]
    assert any("upload-check-note.pdf" in name for name in names)
    assert not any("platform calculator" in name for name in names)


def _result_figure(result: dict) -> float | None:
    for val in result.values():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        return float(val)
    return None


@pytest.mark.parametrize(
    "name,display",
    DEFAULT_ONLY,
    ids=[name for name, _display in DEFAULT_ONLY],
)
def test_explicit_calculation_may_run_on_defaults_and_labels_them(name, display):
    """An explicit calculation with no figures may use defaults, and says so."""
    from app.lib.construction_formulas import CALCULATORS
    from app.lib.formula_registry import get
    from app.lib.source_labels import _user_states_value, default_sentence

    from app.agents.runtime import (
        _forced_specific_tool,
        _message_is_formula_style_ask,
    )

    ask = _explicit_calculation(display)
    assert _message_is_formula_style_ask(ask) is True, ask
    assert _forced_specific_tool(
        [{"role": "user", "content": ask}], {"construction_calc"},
    ) == "construction_calc"
    env = run_calculation(name, {"text": ask})
    assert env.get("status") == "success", ask
    result = env["result"]
    figure = _result_figure(result)
    assert figure is not None, name
    signature = inspect.signature(CALCULATORS[name])
    spec = get(name)
    notes = result.get("notes") if isinstance(result.get("notes"), list) else []
    blob = " ".join(str(item) for item in notes)
    blob += " " + " ".join(str(val) for val in result.values() if not isinstance(val, list))
    messages = _calc_turn(name, {"text": ask}, env, ask)
    out = gate(f"The result is {figure}.\n", None, messages)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().lower().startswith("source:")]
    assert len(lines) == 1, out
    label = calculator_label(name, {})
    assert lines[0].startswith("Source: " + label)
    row = _platform_calculator_source(out)
    assert row is not None
    assert row["doc_name"] == label
    note_text = " ".join(str(item) for item in notes)
    if note_text:
        assert note_text not in (row["doc_name"] or "")
    labelled = False
    for key, param in signature.parameters.items():
        default = param.default
        if default is inspect.Parameter.empty or isinstance(default, bool):
            continue
        if isinstance(default, (int, float)) and float(default) == 0.0:
            continue
        if not isinstance(default, (int, float, str)):
            continue
        if not _user_states_value(blob, default):
            continue
        phrase = default_sentence(name, key, default)
        assert phrase in out, (name, phrase)
        labelled = True
    if labelled:
        assert "calculator's default" in out


def _calc_turn(name, params, env, ask):
    args = json.dumps({"calculation": name, "params": params})
    return [
        {"role": "user", "content": ask},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-default",
                "type": "function",
                "function": {"name": "construction_calc", "arguments": args},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-default",
            "name": "construction_calc",
            "content": json.dumps(env),
        },
    ]


def test_one_registry_source_line_and_defaults_are_not_user_inputs():
    answer = f"The concrete volume is 25.2 m3.\n{MODEL_SOURCE}\n"
    out = gate(answer, None, _slab_messages())
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().lower().startswith("source:")]
    assert len(lines) == 1
    label = calculator_label(
        "concrete_volume",
        {"length_m": 15, "width_m": 8, "thickness_m": 0.2},
    )
    assert lines[0] == "Source: " + label or lines[0].startswith("Source: " + label + " — ")
    assert label in lines[0]
    assert "shape" not in lines[0].lower()
    assert "rectangular" not in lines[0].lower()
    assert "waste" not in lines[0].lower()
    assert "geometry" not in lines[0].lower()
    assert "allowance" not in lines[0].lower()
    body = out.lower().replace(lines[0].lower(), "")
    assert "rectangular" in body
    assert "calculator's default" in body
    row = _platform_calculator_source(out)
    assert row is not None
    assert row["doc_name"] == label
