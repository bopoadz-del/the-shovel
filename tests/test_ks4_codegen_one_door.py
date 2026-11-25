"""KS-4: the CPM pass self-coding is told to call is the enveloped one.

Generated code runs in the sandbox, not as a file under ``app/``, so the
AST guard over product modules cannot see which pass it calls. The prompt
is what decides it. These tests read the prompt the code generator is
given, run the code it advertises through the real sandbox, and check that
every critical-path pass it reaches goes through ``run_calculation``.
"""

import inspect
import re

import pytest

import app.lib.construction_formulas as formulas
from app.core.sandbox import run_sandboxed
from app.lib import pm_computations
from app.prompts.codegen_system import build_codegen_prompt

_PM_SECTION = re.compile(
    r"PROJECT-MANAGEMENT LIBRARY.*?(?=\n\nModels in )", re.S
)
_ADVERTISED = re.compile(r"^- (\w+)\(", re.M)
_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _prompt() -> str:
    return build_codegen_prompt("critical path of the network", {})


def _advertised_functions() -> list:
    section = _PM_SECTION.search(_prompt())
    assert section, "the prompt has no project-management library section"
    return _ADVERTISED.findall(section.group(0))


def _top_level(params: str) -> list:
    """Split a parameter list at the commas outside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in params:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p.strip()]


@pytest.fixture
def envelope_calls(monkeypatch):
    """Record every calculation name that passes through the envelope."""
    seen = []
    real = formulas.run_calculation

    def spy(name, params=None):
        seen.append(name)
        return real(name, params)

    monkeypatch.setattr(formulas, "run_calculation", spy)
    return seen


def _series(*durations):
    lines = []
    prev = None
    for i, d in enumerate(durations):
        pid = f"T{i}"
        preds = (f"[Dependency(predecessor_id={prev!r})]" if prev else "[]")
        lines.append(f"    Activity(id={pid!r}, duration={d}, predecessors={preds}),")
        prev = pid
    return "\n".join(lines)


def _snippet(body: str, acts: str) -> str:
    return (
        "from app.lib.pm_computations import run_cpm, compress_schedule\n"
        "from app.schemas.cpm import CPMInput, Activity, Dependency\n"
        f"data = CPMInput(activities=[\n{acts}\n])\n"
        f"{body}\n"
    )


def test_every_advertised_function_exists_with_the_advertised_parameters():
    names = _advertised_functions()
    assert names, "no library function is advertised"
    section = _PM_SECTION.search(_prompt()).group(0)
    for name in names:
        fn = getattr(pm_computations, name, None)
        assert callable(fn), f"the prompt advertises {name}, which does not exist"
        line = re.search(rf"^- {name}\(([^)]*)\)", section, re.M).group(1)
        said = [p.split(":")[0].split("=")[0].strip() for p in _top_level(line)]
        params = inspect.signature(fn).parameters
        real = list(params)
        # the advertised parameters lead, in order; anything left unsaid
        # has a default, so the call as advertised is a valid call
        assert said == real[: len(said)], (name, said, real)
        unsaid = [p for p in real[len(said):]
                  if params[p].default is inspect.Parameter.empty]
        assert not unsaid, (name, unsaid)


def test_the_prompt_never_names_the_bare_pass():
    # compute_cpm is the pass the envelope wraps; naming it would send
    # generated code around the envelope.
    assert "compute_cpm" not in _prompt()
    assert "run_cpm" in _advertised_functions()


@pytest.mark.parametrize("durations, expected", [((3, 5, 2), 10), ((4, 1), 5)])
def test_advertised_pass_runs_inside_the_envelope_in_the_sandbox(
    envelope_calls, durations, expected
):
    out = run_sandboxed(_snippet("result = run_cpm(data).project_duration",
                                 _series(*durations)))
    assert out.success, out.error
    assert out.result == expected
    assert envelope_calls == ["cpm_network"]


def test_the_prompt_example_runs_in_the_sandbox_inside_the_envelope(envelope_calls):
    blocks = _CODE_BLOCK.findall(_prompt())
    assert blocks, "the prompt carries no example"
    out = run_sandboxed(blocks[0])
    assert out.success, out.error
    # A(3) -> B(5) -> C(2) in series, as the example states.
    assert out.result == 3 + 5 + 2
    assert envelope_calls == ["cpm_network"]


def test_the_bare_pass_does_not_touch_the_envelope(envelope_calls):
    # The reason the prompt must not name it: the same network through the
    # bare pass gives the same number with no envelope at all.
    code = _snippet("result = compute_cpm(data).project_duration", _series(3, 5, 2))
    code = code.replace("import run_cpm,", "import compute_cpm, run_cpm,")
    out = run_sandboxed(code)
    assert out.success, out.error
    assert out.result == 10
    assert envelope_calls == []


@pytest.mark.parametrize(
    "durations, cut, saved",
    [((3, 5, 2), {"T1": 2}, 2), ((4, 1), {"T0": 1, "T1": 1}, 2)],
)
def test_compress_schedule_returns_what_the_prompt_says(
    envelope_calls, durations, cut, saved
):
    section = _PM_SECTION.search(_prompt()).group(0)
    assert "tuple[CPMOutput, int]" in section
    out = run_sandboxed(_snippet(
        f"revised, days = compress_schedule(data, {cut!r})\n"
        "result = (revised.project_duration, days)",
        _series(*durations)))
    assert out.success, out.error
    assert out.result == (sum(durations) - saved, saved)
    # baseline and revised pass, both through the envelope
    assert envelope_calls == ["cpm_network", "cpm_network"]


def test_an_empty_network_is_not_sent_to_the_envelope(envelope_calls):
    out = run_sandboxed(
        "from app.lib.pm_computations import run_cpm\n"
        "from app.schemas.cpm import CPMInput\n"
        "result = run_cpm(CPMInput(activities=[])).project_duration\n")
    assert out.success, out.error
    assert out.result == 0
    assert envelope_calls == []
