"""The exit check, second pass: what still reached a user.

Generated inputs, not known strings. Names come from the registries (tools,
blocks, the step tables) or are made up per seed, so a step added later is
covered without editing this file:

* a rewrite of machinery wording is grammatical: it never puts a
  determiner where the noun phrase already has one;
* a citation bracket left open closes on the name it gives, or goes whole;
* a tool, block, action or operation name never shows, in any written form
  (``snake_case``, ``block.operation``, a code chip, words ahead of "tool" or
  "action"); a sentence about the step goes, one about the work keeps its
  figures;
* a ``key: literal`` dump goes whatever markup wraps it, a fenced block of
  data goes, a code fence stays, a tick with no partner goes;
* the model's account of its own calls (arguments, the engine, what is
  implemented, which call failed) goes, and the platform says no sentence of
  that kind itself;
* site text using the same words (an engine, an action, a failed test, a
  payload) is left exactly as written.
"""
from __future__ import annotations

import importlib
import json
import random
import re

import pytest

try:
    answer_exit = importlib.import_module("app.agents.answer_exit")
except ImportError:  # pragma: no cover
    answer_exit = None

_SEEDS = range(20261009, 20261009 + 12)
_FACT = "The register lists 4 open risks."
_NOUNS = ("specification", "contract", "drainage", "tender", "survey", "method", "addendum", "schedule",
          "pile", "minutes", "programme", "drawing")


def _ax():
    if answer_exit is None:
        pytest.fail("no exit check")
    return answer_exit


@pytest.fixture(autouse=True)
def _no_document_store(monkeypatch):
    from app.core import projects

    monkeypatch.setattr(projects, "get_document", lambda _doc_id: None)


def _idempotent(out: str, **kw):
    assert _ax().check_text(out, **kw) == out, out


# ── 1. machinery wording rewrites are grammatical ───────────────────────────

_DETS = ("", "the ", "The ", "this ", "no ", "any ")
_MOD = ("", "specification ", "contract ", "drainage ")
_HEADS = ("excerpt", "excerpts", "chunk", "snippet")
_POSTS = (" retrieved", " provided", " supplied")
_GRAMMAR_CASES = [(d, m, h, p) for d in _DETS for m in _MOD for h in _HEADS for p in _POSTS]
_DETERMINER = r"(?:a|an|the|this|that|these|those|no|any)"


@pytest.mark.parametrize("det,mod,head,post", _GRAMMAR_CASES)
def test_machinery_rewrite_never_doubles_a_determiner(det, mod, head, post):
    ax = _ax()
    lead = "I checked: " if not det[:1].isupper() else ""
    text = f"{lead}{det}{mod}{head}{post} here does not restate the bond amount."
    out = ax.check_text(text)
    assert not re.search(rf"\b(?:{head}|{post.strip()})\b", out, re.IGNORECASE), out
    assert out.endswith(" here does not restate the bond amount."), out
    # one determiner per noun phrase: none after a determiner or a modifier
    assert not re.search(rf"\b{_DETERMINER}\s+(?:[a-z]+\s+)?{_DETERMINER}\s+(?:project|section|material)\b",
                         out, re.IGNORECASE), out
    if mod:
        plural = head.endswith("s")
        assert f"{det}{mod}{'sections' if plural else 'section'} here" in out, out
    _idempotent(out)


# ── 2. an open citation bracket closes or goes ──────────────────────────────

def _doc_name(rng: random.Random) -> str:
    sep = rng.choice(("_", " ", "-"))
    return sep.join(rng.sample(_NOUNS, rng.randint(2, 3))) + rng.choice((".pdf", ".pptx", ".docx"))


_LABELS = ("source: ", "Source: ", "sources: ", "Source - ", "ref: ", "cited: ")
_AFTER = ("\n\n_3 of 5 project documents indexed._", "", " The next point is the bond.")
_OPEN_CASES = [(s, lab, aft) for s in _SEEDS for lab in _LABELS for aft in _AFTER]


def _balanced(line: str) -> bool:
    depth = 0
    for ch in line:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


@pytest.mark.parametrize("seed,label,after", _OPEN_CASES)
def test_open_citation_bracket_closes_on_its_name(seed, label, after):
    ax = _ax()
    name = _doc_name(random.Random(seed))
    out = ax.check_text(f"The bond is 10% of the contract price (" + label + name + after)
    assert all(_balanced(ln) for ln in out.split("\n")), out
    assert f"(Source: {name})" in out, out
    if after.strip():
        assert after.strip() in out, out
    _idempotent(out)


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_open_citation_bracket_with_only_a_retrieval_key_goes_whole(seed):
    ax = _ax()
    rng = random.Random(seed)
    key = f"doc_id={rng.getrandbits(128):032x} chunk {rng.randint(0, 900)}"
    out = ax.check_text(f"The bond is 10% (source: {key}\n\nNext line.")
    assert "(" not in out and "doc_id" not in out and "chunk" not in out, out
    assert out.startswith("The bond is 10%") and out.endswith("Next line."), out


# ── 3. step names never show ────────────────────────────────────────────────

def _registry_steps():
    from app.agents.core import tool_registry
    from app.blocks import BLOCK_REGISTRY
    from app.core.cm_step_aliases import STEP_TO_TARGET

    tool_registry.load()
    named = {n for n, s in tool_registry._TOOLS.items() if not s.display_name}
    names = named | set(BLOCK_REGISTRY) | {a for _b, a in STEP_TO_TARGET.values() if a}
    return sorted(n for n in names if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", n)
                  and re.search(r"[a-z]{3}", n) and not _has_display(n))


def _has_display(name: str) -> bool:
    from app.lib.source_labels import formula_display_name, tool_display_name

    return bool(tool_display_name(name) or formula_display_name(name))


def _forms(name: str):
    words = name.replace("_", " ")
    return (
        (f"The {name} action returned errors.", False),
        (f"I routed it through {name} and {name}.status first.", False),
        (f"Checked `{name}` and `{words}` for this.", False),
        (f"The {words} tool ran.", False),
        (f"{name}.status reported nothing.", False),
        (f"The {name} step measured 45 m³ of concrete.", True),
    )


_STEP_CASES = [(n, i) for n in _registry_steps() for i in range(6)]


def _visible(name: str, out: str) -> bool:
    words = name.replace("_", " ")
    return name in out.lower() or words in out.lower()


@pytest.mark.parametrize("name,form", _STEP_CASES)
def test_step_name_never_shows(name, form):
    ax = _ax()
    sentence, carries_figure = _forms(name)[form]
    out = ax.check_text(f"{sentence} {_FACT}")
    assert not _visible(name, out) and "_" not in out and "`" not in out, out
    assert out.endswith(_FACT), out
    if carries_figure:
        assert "45 m³ of concrete" in out, out
    else:
        assert out == _FACT, out
    _idempotent(out)


def _made_up_ops(seed: int, n: int = 4) -> list[str]:
    rng = random.Random(seed)
    return [f"{rng.choice(_NOUNS)}_{rng.choice(('latest', 'history', 'sync', 'rebuild', 'summary'))}"
            for _ in range(n)]


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_operation_names_a_tool_listed_never_show(seed):
    """Operation names no registry holds: the turn learns them from the
    error the tool returned and from the call the model made."""
    ax = _ax()
    ops = sorted(set(_made_up_ops(seed)))
    block = f"{random.Random(seed).choice(_NOUNS)}_engine"
    turn = ax.Turn()
    ax.note_tool_result(block, {"status": "error", "error": f"Unknown operation: x. Use: {', '.join(ops)}"}, turn)
    ax.note_evidence(None, [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {
        "name": block, "arguments": json.dumps({"operation": ops[0]})}}]}], turn)
    words = ", ".join(f"`{o.replace('_', ' ')}`" for o in ops)
    for text in (f"Available operations: {', '.join(ops)}. {_FACT}",
                 f"The {block.replace('_', ' ')} block offers {words}. {_FACT}",
                 f"{block}.{ops[0]} returned nothing. {_FACT}"):
        out = ax.check_text(text, turn=turn)
        assert not any(_visible(o, out) for o in ops + [block]), (text, out)
        assert out == _FACT, (text, out)


# ── 4. dumps and stray ticks ────────────────────────────────────────────────

_LITERALS = ("false", "true", "null", "{}", "[]")
_WRAPS = ("{k}: {v}", "`{k}`: {v}", '"{k}": {v}', "**{k}**: {v}", "{k}={v}", "`{k}: {v}`")


def _dump_keys(seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f"{rng.choice(_NOUNS)}_{rng.choice(('flag', 'ok', 'status', 'enabled', 'state'))}" for _ in range(3)]


_DUMP_CASES = [(s, w, lit) for s in _SEEDS for w in range(len(_WRAPS)) for lit in _LITERALS]


@pytest.mark.parametrize("seed,wrap,literal", _DUMP_CASES)
def test_key_literal_dump_goes_in_any_markup(seed, wrap, literal):
    ax = _ax()
    key = _dump_keys(seed)[0]
    dump = _WRAPS[wrap].format(k=key, v=literal)
    for text in (f"Status:\n- {dump}\n- {_FACT}", f"{dump}. {_FACT}", f"{_FACT} Then {dump}."):
        out = ax.check_text(text)
        assert key not in out and key.replace("_", " ") not in out, (text, out)
        assert _FACT in out and "`" not in out, (text, out)


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_fenced_data_block_goes_and_code_fence_stays(seed):
    ax = _ax()
    keys = _dump_keys(seed)
    blob = json.dumps({k: v for k, v in zip(keys, (False, [], None))}, indent=2)
    yaml = "\n".join(f"{k}: {v}" for k, v in zip(keys, ("false", "true", "null")))
    for fence in (f"```json\n{blob}\n```", f"```\n{yaml}\n```", f"```yaml\n{yaml}\n```"):
        out = ax.check_text(f"{fence}\n{_FACT}")
        assert out == _FACT, out
    code = f"```python\nresult = compute({keys[0]}=10)\nprint(result)\n```"
    assert ax.check_text(code) == code


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_tick_with_no_partner_goes(seed):
    ax = _ax()
    rng = random.Random(seed)
    words = rng.sample(_NOUNS, 5)
    at = rng.randint(1, 4)
    sentence = " ".join(words[:at] + ["`"] + words[at:]) + "."
    out = ax.check_text(f"{sentence} {_FACT}")
    assert "`" not in out and all(w in out for w in words) and out.endswith(_FACT), out
    out = ax.check_text(f"The {words[0]} coefficients, `.")
    assert out == f"The {words[0]} coefficients.", out


# ── 5. the model's account of its own calls ─────────────────────────────────

_SUBJECTS = ("My first calls", "My earlier attempts", "My previous requests", "The first call")
_FAILS = ("failed", "returned errors", "errored")
_BECAUSE = ("because I passed the action inside the input JSON string",
            "because I sent the operation as a parameter", "so I retried with the action in params")
_MECH = (
    "The engine expects the action in params.",
    "The engine wants {op} in the args.",
    "({op}) isn't implemented in this engine version.",
    "That action is not implemented in this tool.",
    "One step of this request did not return a result.",
    "The tool call did not return a result.",
    "Unknown operation {op}; the call returned HTTP 400.",
    "I passed the wrong parameters to the call, so it failed.",
)
_MECH_CASES = ([(s, f, b) for s in _SUBJECTS for f in _FAILS for b in _BECAUSE]
               + [(m, "", "") for m in _MECH])


@pytest.mark.parametrize("subject,fail,because", _MECH_CASES)
def test_call_mechanics_never_reach_the_user(subject, fail, because):
    ax = _ax()
    op = random.Random(hash((subject, fail, because)) & 0xFFFF).choice(_NOUNS)
    sentence = f"{subject} {fail} {because}." if fail else subject.format(op=op)
    for text in (f"{sentence} {_FACT}", f"{_FACT}\n- {sentence}\n- The bond is 10%."):
        out = ax.check_text(text)
        assert _FACT in out and sentence not in out, (text, out)
        assert not re.search(r"\b(?:params|JSON|engine version|did not return)\b", out), out


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_platform_never_says_a_step_did_not_return(seed):
    """A copied tool error is replaced by an ask for the input it named, or
    by nothing; never by a sentence about a step."""
    ax = _ax()
    from app.lib import formula_registry

    rng = random.Random(seed)
    specs = [s for s in formula_registry.all_specs() if formula_registry.parameters(s)]
    spec = rng.choice(specs)
    param = rng.choice([p for p in formula_registry.parameters(spec) if "_" in p] or ["x_y_z"])
    tool = rng.choice(_registry_steps())
    turn = ax.Turn()
    errs = (f"Missing required parameter: {param}", "Traceback (most recent call last): boom at line 3",
            f"Error: {tool} could not open the cache")
    for err in errs:
        ax.note_tool_result(tool, {"status": "error", "error": err}, turn)
        out = ax.check_text(f"It said: {err}\n{_FACT}", turn=turn)
        assert "did not return" not in out and err not in out and tool not in out, out
        assert out.endswith(_FACT), out


# ── site text using the same words is untouched ─────────────────────────────

_SITE_TEXT = (
    "The excavator engine requires servicing every 250 hours.",
    "The cube test failed at 28 days; a retest was ordered.",
    "Corrective action was implemented on 3 June; the method statement was not implemented on level 2.",
    "The crane payload is 5 t at 30 m radius.",
    "We called the supplier twice and the delivery failed again.",
    "The status of works: substructure complete, frame 60%.",
    "Crane operations stop when wind exceeds 38 km/h.",
    "The safety compliance audit found 3 non-conformances.",
    "A risk register lists 12 risks; the top risk is working at height.",
    "Tool box talks are held daily at 07:00.",
    "The specification section 03 30 00 covers cast-in-place concrete.",
    "Use the IFC schema version IFC4 for the model exchange.",
    "```python\nvolume = length * width * depth\n```",
    "The block work is 200 mm thick; the engine room slab is 300 mm.",
)


@pytest.mark.parametrize("text", _SITE_TEXT)
def test_site_text_is_left_as_written(text):
    assert _ax().check_text(text) == text
