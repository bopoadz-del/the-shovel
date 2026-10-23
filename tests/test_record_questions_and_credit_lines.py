"""Record questions, calculator credit lines and the unverified-attribution note.

A calculator runs when the user asks for that calculation or supplies an
operand. A question about what a document or record says does neither, so
no calculator runs on its signature defaults and none is credited, whichever
way the record is pointed at: by a typed file name, as "my uploaded note",
or as "the memo I attached". An unnamed reference to the user's own upload
is resolved from the project's document list.

When a calculator is credited, the Source line is the registry label and
the user's inputs, nothing else; the defaults it rested on are stated,
currency included when the answer shows one, and prose denying defaults
does not stand beside them. A removal leaves no empty brackets or dangling
separators. A document the turn actually read is evidence, so citing it
never raises the unverified-attribution note; citing one it did not read
still does.

Every case is generated from the formula registry or from a class of
phrasing; no case names a project or a document.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

import pytest

from app.agents import runtime
from app.agents.citation_provenance import UNVERIFIED_NOTE, gate
from app.lib.construction_formulas import (
    CALCULATORS,
    calculator_name_from_text,
    describe_calculation_params,
    run_calculation,
)
from app.lib.formula_registry import all_specs
from app.lib.source_labels import (
    CURRENCY_TOKEN,
    calculator_label,
    guard_supplied_input_wording,
)

NOTE = UNVERIFIED_NOTE.strip()

# Ways a user points at a record without asking for a calculation. ``{q}``
# is what they ask about; ``{file}`` is a typed file name.
RECORD_FRAMES = (
    "What does my uploaded note say about the {q}?",
    "What does {file} say about the {q}?",
    "What does the memo I uploaded state about the {q}?",
    "What does our attached specification require for the {q}?",
)
FILE_NAMES = ("site_note_2026-03-14.txt", "level 2 memo v3.docx", "spec-07.md")


def _first_figure(result: dict) -> float | None:
    for key, val in result.items():
        if key in ("notes", "note") or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val:
            return float(val)
    return None


def _shown(value: float) -> str:
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _record_question(frame: str, display: str, index: int = 0) -> str:
    return frame.format(q=display.lower(), file=FILE_NAMES[index % len(FILE_NAMES)])


def _named_by_frames() -> list[tuple[str, str]]:
    """Formulas every record frame names (so a router could pick them)."""
    rows = []
    for spec in all_specs():
        display = spec.display_name or ""
        if not display:
            continue
        if all(
            calculator_name_from_text(_record_question(f, display, i)) == spec.name
            for i, f in enumerate(RECORD_FRAMES)
        ):
            rows.append((spec.name, display))
    return rows


NAMED = _named_by_frames()
DEFAULT_RUNS = [
    (name, display) for name, display in NAMED
    if run_calculation(name, {}).get("status") == "success"
]


def _scalar(param: inspect.Parameter) -> bool:
    if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
        return False
    ann = str(getattr(param.annotation, "__name__", "") or param.annotation).lower()
    if any(tok in ann for tok in ("str", "list", "dict", "bool", "tuple")):
        return False
    return not isinstance(param.default, (str, list, dict, tuple, bool))


def _supplied_runs() -> list[tuple[str, str, dict, dict]]:
    """``(name, display, inputs, envelope)``: each formula run on figures a
    user typed, one per required input, as its declared kind."""
    rows = []
    for spec in all_specs():
        sig = inspect.signature(spec.fn)
        required = [r for r in describe_calculation_params(spec.fn, name=spec.name)
                    if r.get("required")]
        if not required or not spec.display_name:
            continue
        if any(not _scalar(sig.parameters[r["name"]]) for r in required):
            continue
        inputs: dict = {}
        for index, row in enumerate(required):
            unit = str((spec.inputs or {}).get(row["name"]) or "").strip()
            inputs[row["name"]] = (0.05 + index * 0.01) if unit == "%" else 12.0 + index * 3
        env = run_calculation(spec.name, dict(inputs))
        if env.get("status") != "success" or _first_figure(env["result"]) is None:
            continue
        rows.append((spec.name, spec.display_name, inputs, env))
    return rows


SUPPLIED = _supplied_runs()


def _calc_turn(name: str, params: dict, env: dict, ask: str, text: str | None = None) -> list:
    args = {"calculation": name, "params": params}
    if text is not None:
        args["text"] = text
    return [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_rq", "function": {"name": "construction_calc",
                                          "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "call_rq", "name": "construction_calc",
         "content": json.dumps(env, default=str)},
    ]


def _source_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("Source:")]


# ── Item 1: a record question never runs or credits a defaulted calculator ──


def test_registry_cases_are_generated():
    assert len(NAMED) >= 10
    assert len(DEFAULT_RUNS) >= 3
    assert len(SUPPLIED) >= 20


@pytest.mark.parametrize("frame", RECORD_FRAMES)
@pytest.mark.parametrize("name,display", NAMED, ids=[n for n, _d in NAMED])
def test_record_question_is_not_routed_to_a_calculator(name, display, frame):
    from app.agents.runtime import (
        _forced_specific_tool,
        _message_is_formula_style_ask,
        _message_wants_named_calculator,
    )

    ask = _record_question(frame, display, RECORD_FRAMES.index(frame))
    assert _message_is_formula_style_ask(ask) is False, ask
    assert _message_wants_named_calculator(ask) is False, ask
    assert _forced_specific_tool(
        [{"role": "user", "content": ask}], {"construction_calc"},
    ) is None, ask


@pytest.mark.parametrize("frame", RECORD_FRAMES)
@pytest.mark.parametrize("name,display", NAMED, ids=[n for n, _d in NAMED])
def test_calculator_tool_does_not_run_a_record_question_on_defaults(name, display, frame):
    """The model may still call the calculator itself; the tool declines."""
    from app.agents.base.tools import handle_construction_calc
    from app.agents.core.tool_registry import ToolCall

    ask = _record_question(frame, display, RECORD_FRAMES.index(frame))
    call = ToolCall(agent=None, name="construction_calc",
                    args={"calculation": name}, user_message=ask)
    out = asyncio.run(handle_construction_calc(call))
    result = out["result"]
    assert result.get("status") == "not_run", (ask, result)
    assert out["ok"] is True
    assert not any(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in result.values())


@pytest.mark.parametrize("name,display", NAMED, ids=[n for n, _d in NAMED])
def test_calculator_tool_still_runs_an_explicit_ask_or_an_operand(name, display):
    from app.agents.base.tools import handle_construction_calc
    from app.agents.core.tool_registry import ToolCall

    explicit = ToolCall(agent=None, name="construction_calc", args={"calculation": name},
                        user_message=f"Calculate the {display.lower()}.")
    assert asyncio.run(handle_construction_calc(explicit))["result"].get("status") != "not_run"
    sig = inspect.signature(CALCULATORS[name])
    scalar = [k for k, p in sig.parameters.items() if _scalar(p)]
    if not scalar:
        return
    figure = 17.5 if sig.parameters[scalar[0]].default != 17.5 else 18.5
    operand = ToolCall(agent=None, name="construction_calc",
                       args={"calculation": name, "params": {scalar[0]: figure}},
                       user_message=_record_question(RECORD_FRAMES[0], display))
    assert asyncio.run(handle_construction_calc(operand))["result"].get("status") != "not_run"


@pytest.mark.parametrize("frame", RECORD_FRAMES)
@pytest.mark.parametrize("name,display", DEFAULT_RUNS, ids=[n for n, _d in DEFAULT_RUNS])
def test_defaulted_run_is_not_credited_to_a_record_question(name, display, frame):
    """A document answer that happens to quote a figure equal to a defaulted
    run's result gets no calculator credit and no default lines."""
    ask = _record_question(frame, display, RECORD_FRAMES.index(frame))
    env = run_calculation(name, {})
    figure = _first_figure(env["result"])
    assert figure is not None
    answer = f"The document gives {_shown(figure)} for this item."
    out = gate(answer, None, _calc_turn(name, {}, env, ask, text=ask))
    assert "platform calculator" not in out, out
    assert out == answer


def _resolution_docs(noun: str) -> tuple[list[dict], dict[str, str]]:
    """Three uploads: the one the ask is about, a sibling of the same kind,
    and an unrelated file. Names and bodies are built from the noun."""
    docs = [
        {"id": "d-other", "original_name": f"{noun}_procurement_log.txt",
         "file_path": "/data/a.txt", "doc_type": "document"},
        {"id": "d-target", "original_name": f"level2_{noun}.txt",
         "file_path": "/data/b.txt", "doc_type": "document"},
        {"id": "d-unrelated", "original_name": "invoice_register.xlsx",
         "file_path": "/data/c.xlsx", "doc_type": "document"},
    ]
    bodies = {
        "d-other": "Purchase order numbers and supplier lead times for rebar.",
        "d-target": "Striking of soffit formwork to suspended slabs: not before 10 days.",
        "d-unrelated": "Invoice 1 paid. Invoice 2 pending.",
    }
    return docs, bodies


def _wire_documents(monkeypatch, docs: list[dict], bodies: dict[str, str]) -> None:
    from app.core import projects as projects_mod

    monkeypatch.setattr(projects_mod, "list_document_names", lambda pid: list(docs))
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: list(docs))

    def fetch(project_id, doc_id, filename):
        for d in docs:
            if d["id"] == doc_id or (filename and d["original_name"] == filename):
                return ({"text": bodies[d["id"]], "truncated": False,
                         "source": "extracted"}, d, None)
        return None, None, "no document"

    monkeypatch.setattr(runtime, "_fetch_document_content", fetch)


def _agent():
    return runtime.Agent(name="project-assistant", description="", system_prompt="x",
                         allowed_blocks=["construction"])


OWN_UPLOAD_ASKS = (
    "What does my uploaded {noun} say about the formwork striking time for suspended slabs?",
    "What does the {noun} I uploaded say about striking formwork to suspended slabs?",
    "According to my attached {noun}, when can slab soffit formwork be struck?",
)


@pytest.mark.parametrize("noun", ["note", "memo", "report", "specification"])
@pytest.mark.parametrize("ask", OWN_UPLOAD_ASKS)
def test_unnamed_own_upload_is_read_from_the_project_documents(monkeypatch, noun, ask):
    docs, bodies = _resolution_docs(noun)
    _wire_documents(monkeypatch, docs, bodies)
    msgs = [{"role": "user", "content": ask.format(noun=noun)}]
    rec = asyncio.run(runtime._predispatch_file_tool(_agent(), msgs, "p1"))
    assert rec and rec["name"] == "fetch_document", rec
    assert rec["result"]["document_id"] == "d-target"
    assert bodies["d-target"] in msgs[-1]["content"]


def test_unnamed_upload_with_nothing_to_choose_by_reads_nothing(monkeypatch):
    docs, bodies = _resolution_docs("note")
    bodies = {k: "Unrelated text." for k in bodies}
    _wire_documents(monkeypatch, docs, bodies)
    msgs = [{"role": "user", "content": "What does my uploaded note say about the crane?"}]
    assert asyncio.run(runtime._predispatch_file_tool(_agent(), msgs, "p1")) is None
    assert len(msgs) == 1


# ── Item 2: the Source line is the registry label and the user's inputs ─────


NOTED = [row for row in SUPPLIED if row[3]["result"].get("notes")]


def test_noted_registry_cases_are_generated():
    assert len(NOTED) >= 3


@pytest.mark.parametrize("name,display,inputs,env", NOTED, ids=[r[0] for r in NOTED])
def test_source_line_is_the_label_and_inputs_only(name, display, inputs, env):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {_shown(figure)}."
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert _source_lines(out) == ["Source: " + calculator_label(name, inputs)], out
    for note in env["result"]["notes"]:
        assert all(str(note) not in line for line in _source_lines(out))


@pytest.mark.parametrize("name,display,inputs,env", NOTED, ids=[r[0] for r in NOTED])
def test_model_source_line_with_result_notes_is_reduced_to_the_label(name, display, inputs, env):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    label = "Source: " + calculator_label(name, inputs)
    notes = "; ".join(str(n) for n in env["result"]["notes"])
    answer = f"The {display.lower()} is {_shown(figure)}.\n{label} — {notes}"
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert _source_lines(out) == [label], out
    assert NOTE not in out


# ── Item 3: a removal leaves no empty brackets or dangling separators ───────


CITATIONS = ("FIDIC Sub-Clause 8.8", "Clause 14.3", "BS 8110", "ACI 318-19", "EN 1992-1")
# (template, expected) -- {c} is a citation nothing in the turn states.
DEBRIS_TEMPLATES = (
    ("Computed as 50,000 per day on the rate basis ({c}).",
     "Computed as 50,000 per day on the rate basis."),
    ("Delay damages 50,000 per day (rate × amount, {c} basis).",
     "Delay damages 50,000 per day (rate × amount)."),
    ("Delay damages 50,000 per day ({c}, rate × amount).",
     "Delay damages 50,000 per day (rate × amount)."),
    ("Delay damages 50,000 per day (per {c}).",
     "Delay damages 50,000 per day."),
    ("Delay damages 50,000 per day [{c}].",
     "Delay damages 50,000 per day."),
    ("Delay damages 50,000 per day, {c}.",
     "Delay damages 50,000 per day."),
)
_DEBRIS_RE = re.compile(
    r"\(\s*\)|\[\s*\]|\(\s*[,;:]|[,;:]\s*[)\]]|[,;]\s*[,;]|\s[,;:.)]|[,;]\s*$"
)


@pytest.mark.parametrize("citation", CITATIONS)
@pytest.mark.parametrize("template,expected", DEBRIS_TEMPLATES,
                         ids=[f"t{i}" for i in range(len(DEBRIS_TEMPLATES))])
def test_removed_citation_leaves_clean_wording(template, expected, citation):
    spec = next(iter(all_specs()))
    out = guard_supplied_input_wording(template.format(c=citation), "", "", spec.name, {}, {})
    assert citation not in out
    assert _DEBRIS_RE.search(out) is None, out
    assert out == expected


CONTRACT_ID = "AB-2001-101"
# (template, expected) -- an id the turn never read, cued inside brackets.
ID_TEMPLATES = (
    ("The rate is 0.1% (per {i}).", "The rate is 0.1%."),
    ("The rate is 0.1% (daily, per {i}).", "The rate is 0.1% (daily)."),
    ("The rate is 0.1% (see {i}; daily).", "The rate is 0.1% (daily)."),
    ("The rate is 0.1% [from {i}].", "The rate is 0.1%."),
)


@pytest.mark.parametrize("template,expected", ID_TEMPLATES,
                         ids=[f"t{i}" for i in range(len(ID_TEMPLATES))])
def test_removed_identifier_leaves_clean_wording(template, expected):
    out = gate(template.format(i=CONTRACT_ID), None,
               [{"role": "user", "content": "What is the daily rate?"}])
    body = out.replace(UNVERIFIED_NOTE, "")
    assert CONTRACT_ID not in body
    assert _DEBRIS_RE.search(body) is None, body
    assert body.strip() == expected
    assert NOTE in out


# ── Item 4: defaults are stated consistently ────────────────────────────────


def _currency_formulas() -> list[tuple[str, str, str, dict, dict]]:
    """``(name, display, code, inputs, envelope)`` for each supplied run of a
    formula whose signature defaults a currency code."""
    from app.lib.source_labels import _CURRENCY_CODES

    rows = []
    for spec in all_specs():
        sig = inspect.signature(spec.fn)
        codes = [p.default for p in sig.parameters.values()
                 if isinstance(p.default, str) and p.default.upper() in _CURRENCY_CODES]
        if not codes:
            continue
        required = [k for k, p in sig.parameters.items()
                    if not isinstance(p.default, str) and _scalar(p)]
        inputs = {k: (0.1 if (spec.inputs or {}).get(k) == "%" else 50_000_000.0)
                  for k in required}
        env = run_calculation(spec.name, dict(inputs))
        if env.get("status") != "success" or _first_figure(env["result"]) is None:
            continue
        rows.append((spec.name, spec.display_name, codes[0], inputs, env))
    return rows


CURRENCY = _currency_formulas()


def test_currency_default_cases_are_generated():
    assert CURRENCY


@pytest.mark.parametrize("name,display,code,inputs,env", CURRENCY, ids=[r[0] for r in CURRENCY])
def test_shown_currency_default_is_stated(name, display, code, inputs, env):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    assert re.search(CURRENCY_TOKEN, ask) is None
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {code} {_shown(figure)}."
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert f"The currency used, {code}, is the platform calculator's default" in out, out


@pytest.mark.parametrize("name,display,code,inputs,env", CURRENCY, ids=[r[0] for r in CURRENCY])
def test_unshown_currency_default_is_not_stated(name, display, code, inputs, env):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {_shown(figure)}."
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert code not in out


DENIALS = (
    "No other defaults were used.",
    "No defaults were applied.",
    "This was computed without any default values.",
    "The calculator did not use a default.",
    "Defaults were not needed here.",
)


def _defaulted_runs() -> list[tuple[str, str, dict, dict]]:
    """Supplied runs whose result still rests on a stated default."""
    from app.lib.source_labels import calculator_default_lines

    rows = []
    for name, display, inputs, env in SUPPLIED:
        if calculator_default_lines(name, env["result"], "", passed=inputs):
            rows.append((name, display, inputs, env))
    return rows


DEFAULTED = _defaulted_runs()


def test_defaulted_cases_are_generated():
    assert len(DEFAULTED) >= 3


@pytest.mark.parametrize("denial", DENIALS)
@pytest.mark.parametrize("name,display,inputs,env", DEFAULTED, ids=[r[0] for r in DEFAULTED])
def test_prose_denying_defaults_does_not_stand_beside_default_lines(name, display, inputs, env, denial):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {_shown(figure)}. {denial}"
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert "platform calculator's default" in out
    assert denial not in out, out


@pytest.mark.parametrize("name,display,inputs,env", DEFAULTED, ids=[r[0] for r in DEFAULTED])
def test_platform_bubble_values_are_not_user_inputs(name, display, inputs, env):
    """A pre-dispatch bubble repeats the calculator's own envelope, defaults
    included. Those values are the tool's, never figures the user stated."""
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {_shown(figure)}."
    msgs = _calc_turn(name, inputs, env, ask)
    msgs.append({"role": "user", "content": (
        f"{runtime._PREDISPATCH_PREFIX} construction_calc has ALREADY been run "
        f"from the operator facts in this turn. Authoritative draft:\n"
        f"{json.dumps(env, default=str)}\n")})
    out = gate(answer, None, msgs)
    assert "platform calculator's default" in out, out
    assert _source_lines(out) == ["Source: " + calculator_label(name, inputs)], out


# ── Item 5: a document the turn read never raises the note ──────────────────


DOC_BODY = "Soffit formwork to suspended slabs shall not be struck before 10 days."
OTHER_HIT = {"document_id": "d-hit", "filename": "general_conditions.pdf",
             "snippet": "Clause 1 definitions.", "score": 0.7, "origin": "own"}


def _search_turn() -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "s1", "function": {"name": "search_project_documents", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "s1", "name": "search_project_documents",
         "content": json.dumps({"name": "search_project_documents", "ok": True,
                                "result": {"results": [OTHER_HIT]}})},
    ]


def _read_driver(name: str) -> list[dict]:
    payload = {"name": "fetch_document", "ok": True,
               "result": {"document_id": "d-read", "filename": name, "content": DOC_BODY}}
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "f1", "function": {"name": "fetch_document", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "f1", "name": "fetch_document",
         "content": json.dumps(payload)},
    ]


def _read_legacy(name: str) -> list[dict]:
    inner = {"document_id": "d-read", "filename": name, "content": DOC_BODY}
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "f1", "function": {"name": "fetch_document", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "f1", "name": "fetch_document",
         "content": json.dumps(inner)},
    ]


def _read_replayed(name: str) -> list[dict]:
    inner = {"document_id": "d-read", "filename": name, "content": DOC_BODY}
    return [{"role": "user",
             "content": f"{runtime._TOOL_RESULT_PREFIX}fetch_document): {json.dumps(inner)}"}]


def _read_predispatched(name: str, monkeypatch) -> list[dict]:
    docs = [{"id": "d-read", "original_name": name, "file_path": "/data/x", "doc_type": "document"}]
    _wire_documents(monkeypatch, docs, {"d-read": DOC_BODY})
    msgs = [{"role": "user", "content": f"What does {name} say about striking slab formwork?"}]
    rec = asyncio.run(runtime._predispatch_file_tool(_agent(), msgs, "p1"))
    assert rec and rec["name"] == "fetch_document"
    return msgs[1:]


READ_SHAPES = ("driver", "legacy", "replayed", "predispatched")
READ_NAMES = ("site_note.txt", "slab memo rev2.docx", "notes/level-2 striking.md")


def _read_turn(shape: str, name: str, monkeypatch) -> list[dict]:
    ask = [{"role": "user", "content": "What does the note say about striking slab formwork?"}]
    if shape == "predispatched":
        return ask + _search_turn() + _read_predispatched(name, monkeypatch)
    reader = {"driver": _read_driver, "legacy": _read_legacy, "replayed": _read_replayed}[shape]
    return ask + _search_turn() + reader(name)


def _shown_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


@pytest.mark.parametrize("name", READ_NAMES)
@pytest.mark.parametrize("shape", READ_SHAPES)
def test_citing_the_read_document_keeps_the_line_and_no_note(monkeypatch, shape, name):
    msgs = _read_turn(shape, name, monkeypatch)
    for cite in (_shown_name(name), f"{_shown_name(name)} (uploaded note)"):
        answer = f"The note says formwork is not struck before 10 days.\n\nSource: {cite}"
        for kwargs in ({}, {"tool_passages": True}):
            out = gate(answer, None, msgs, **kwargs)
            assert NOTE not in out, (shape, cite, out)
            assert f"Source: {cite}" in out


@pytest.mark.parametrize("shape", READ_SHAPES)
def test_citing_a_document_nothing_read_still_raises_the_note(monkeypatch, shape):
    msgs = _read_turn(shape, READ_NAMES[0], monkeypatch)
    answer = "The note says formwork is not struck before 10 days.\n\nSource: unread_minutes.pdf"
    out = gate(answer, None, msgs, tool_passages=True)
    assert "unread_minutes.pdf" not in out.replace(UNVERIFIED_NOTE, "")
    assert NOTE in out


@pytest.mark.parametrize("name,display,inputs,env", SUPPLIED[:12], ids=[r[0] for r in SUPPLIED[:12]])
def test_model_wording_of_the_calculator_that_ran_is_not_an_unverified_source(name, display, inputs, env):
    ask = f"Calculate the {display.lower()} for " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in inputs.items()) + "."
    figure = _first_figure(env["result"])
    answer = f"The {display.lower()} is {_shown(figure)}.\nSource: {display} (platform calculator)"
    out = gate(answer, None, _calc_turn(name, inputs, env, ask))
    assert NOTE not in out, out
    assert _source_lines(out) == ["Source: " + calculator_label(name, inputs)], out


def test_every_registered_calculator_is_in_some_generated_case():
    covered = {n for n, _d in NAMED} | {r[0] for r in SUPPLIED}
    assert len(covered) >= len(CALCULATORS) // 2
