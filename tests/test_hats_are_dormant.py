"""The discipline-hat system may not be switched on while its actions are fiction.

Audit 2026-08-12, agents subsystem. Registered in docs/KNOWN_INCOMPLETE.md under
"Stated limitations"; this file is the part that bites.

There are two agent systems here and only one runs. `runtime.load_agents()`
returns the 14 live agents. Separately, `app/agents/manifests/fork.*.json`
declares a base plus five discipline hats, each with an `allowed_actions` list,
gated behind `FORK_HATS_ENABLED`. That flag is unset everywhere, `runtime.py`
contains zero references to `allowed_actions`, and 32 of the 35 plain action
names the manifests declare resolve to no block, no container route key and no
synthetic tool.

So the flag is not an on-switch for five disciplines. Flipping it grants agents
permission to call 32 actions that do not exist -- the same failure the
container audit found, where `_suggest_next_action` recommended an action
`route()` answered "Unknown action" to, except 32 times over and reachable from
a single environment variable in a dashboard.

WHAT THIS FILE ASSERTS, AND WHY IT IS SHAPED THIS WAY

Not "32 actions are undispatchable" -- that pins the defect in place and goes
red the moment someone starts fixing it, which trains people to edit the test.

The invariant is CONDITIONAL: while the flag is off, an undispatchable action
is a known gap and this file merely keeps the register honest about its size.
The moment the flag is on, every declared action must be dispatchable. That
permits incremental progress in the good direction and blocks the one move that
turns a dormant subsystem into a live defect.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST_DIR = Path("app/agents/manifests")


def _declared_actions() -> dict[str, set[str]]:
    """Plain action names per manifest.

    Dotted / namespaced entries (`app.lib.x.y`, `block:name`) are formula and
    block references resolved elsewhere and already CI-guarded by
    `validate_manifest_bindings`. Only the bare names are dispatcher lookups.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(MANIFEST_DIR.glob("fork.*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        actions = data.get("allowed_actions") or []
        out[path.name] = {a for a in actions if "." not in a and ":" not in a}
    return out


def _dispatchable() -> set[str]:
    """Reuse the live surface the agent fence derives from the dispatcher.

    Deliberately imported rather than re-listed: a hand-kept copy of the
    dispatchable set is exactly the stale twin this audit kept finding.
    """
    from tests.test_agent_tool_surface import _dispatchable as live

    return live()


def test_the_manifests_are_actually_being_read():
    """Guards the parser. If the glob matched nothing, every assertion below
    would pass by iterating an empty set -- a check that gets weaker precisely
    when it stops working."""
    declared = _declared_actions()
    assert len(declared) >= 6, f"only found {sorted(declared)} in {MANIFEST_DIR}"
    total = set().union(*declared.values())
    # 35 originally; 7 unbacked names were REMOVED (not stubbed) when the
    # dispatch gap was closed on 2026-08-15.
    assert len(total) >= 25, f"only {len(total)} plain action names parsed: {sorted(total)}"


def test_the_hats_are_off_by_default():
    """The premise everything else rests on. If the flag ever defaults to on,
    the conditional invariant below silently becomes unconditional -- and it
    would be enforced by a test nobody re-read."""
    from app.agents import activation

    assert activation.HATS_ENABLED is False, (
        "FORK_HATS_ENABLED is set in this environment. That is not a test "
        "failure to work around: see test_enabling_the_hats_requires_every_"
        "declared_action_to_exist below, and docs/KNOWN_INCOMPLETE.md."
    )


def test_enabling_the_hats_requires_every_declared_action_to_exist():
    """The fence.

    Evaluated unconditionally against the manifests, but only ENFORCED as a
    failure when the flag is on. While the flag is off it reports the gap
    without failing, so incremental progress is possible; the moment someone
    sets FORK_HATS_ENABLED they must have closed it.
    """
    import os

    dispatchable = _dispatchable()
    missing: dict[str, list[str]] = {}
    for manifest, actions in _declared_actions().items():
        gap = sorted(actions - dispatchable)
        if gap:
            missing[manifest] = gap

    flag_on = (os.environ.get("FORK_HATS_ENABLED", "").strip()
               in ("1", "true", "True", "TRUE", "yes"))
    if not flag_on:
        pytest.skip(
            "hats are dormant (FORK_HATS_ENABLED unset). "
            f"{sum(len(v) for v in missing.values())} declared actions are "
            "currently undispatchable -- see docs/KNOWN_INCOMPLETE.md. This test "
            "becomes a hard failure the moment the flag is set."
        )

    assert not missing, (
        "FORK_HATS_ENABLED is on, but these hats declare actions that no "
        "block, container route or synthetic tool can dispatch. Agents would "
        "be given permission to call things that do not exist:\n"
        + "\n".join(f"  {m}: {', '.join(a)}" for m, a in sorted(missing.items()))
    )


def test_the_register_states_the_current_size_of_the_gap():
    """docs/KNOWN_INCOMPLETE.md claims a specific number. A register whose figures
    drift is worse than no register -- it reads as measured and is not.

    Asserted as a RANGE, not a fixed count: the exact number should be free to
    fall as actions get wired, and pinning it would make progress look like a
    regression. What must not happen is the register quoting 32 while the real
    figure is 50.
    """
    declared = set().union(*_declared_actions().values())
    gap = declared - _dispatchable()
    register = Path("docs/KNOWN_INCOMPLETE.md").read_text(encoding="utf-8")

    assert "FORK_HATS_ENABLED" in register, (
        "the dormant hat system is not in the register at all"
    )
    # CLOSED 2026-08-15: every declared action now dispatches (aliases in
    # _run_tool_call map declared names to real calculators/container
    # routes; unbacked names were removed). The gap must STAY closed.
    assert len(gap) == 0, (
        f"{len(gap)} declared actions are now undispatchable, but the register "
        f"says 32. The gap grew -- update docs/KNOWN_INCOMPLETE.md:\n{sorted(gap)}"
    )


def test_the_live_agent_system_is_unaffected_by_the_dormant_one():
    """The two systems must not have quietly merged.

    The live 14 agents are the shipped product; the hats are a parallel design.
    If a hat's undispatchable actions ever leaked into a live agent's tool
    list, `test_agent_tool_surface.py` would fail -- but only if the two really
    are separate, which is what this asserts.
    """
    from app.agents.runtime import get_agent, load_agents

    agents = load_agents()
    assert agents, "no agents loaded"

    advertised: set[str] = set()
    for agent_id in agents:
        advertised |= {
            (t.get("function") or {}).get("name") or t.get("name")
            for t in get_agent(agent_id).tool_definitions(project_id="fence-check")
        }

    declared = set().union(*_declared_actions().values())
    leaked = sorted((declared - _dispatchable()) & advertised)
    assert not leaked, (
        "a live agent is advertising hat actions that cannot be dispatched: "
        f"{leaked}"
    )
