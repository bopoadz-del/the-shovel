"""S-H1: one hydration pass says what it cost in model use.

Every pass logs one ``hydration_run`` line per recorded row (each project,
then ``scope=pass`` for the whole pass) with ``model_calls``,
``prompt_tokens`` and ``completion_tokens``, and the admin trigger returns the
run id and the same line. Accounting only: what the pass summarises and writes
back is unchanged. Every test builds its own conversations and its own fake
model; no stored corpus is read.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException

from tests.conftest import requires_construction_kit
from tests.test_hydration import _seed_conversation, isolated_data_dir  # noqa: F401

_DAY = "2026-05-26"
_GLOBAL_MARK = "global hydration rollup"


def _fake_chat_block(token_table, provider="deepseek"):
    """A ChatBlock stand-in whose provider usage depends on the prompt.

    ``token_table`` maps a substring of the prompt to (prompt, completion)
    token counts, so two scopes in one pass report different usage.
    """

    class _FakeChat:
        async def execute(self, data, params):
            prompt = data.get("text") or ""
            for marker, (p_tok, c_tok) in token_table.items():
                if marker in prompt:
                    return {
                        "text": "## summary\n- ok\n",
                        "provider": provider,
                        "tokens": {"prompt_tokens": p_tok, "completion_tokens": c_tok},
                    }
            return {"text": "## summary\n- ok\n", "provider": provider, "tokens": {}}

    return lambda: _FakeChat()


def _usage_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("hydration_run ")]


def _line(scope, project, run_id, calls, p_tok, c_tok):
    return (
        f"hydration_run scope={scope} project={project} run_id={run_id} "
        f"model_calls={calls} prompt_tokens={p_tok} completion_tokens={c_tok}"
    )


@pytest.mark.parametrize(
    "projects, global_tokens",
    [
        ({"proj-one": (1000, 200), "proj-two": (300, 50)}, (40, 7)),
        ({"site-a": (12, 3), "site-b": (7000, 900), "site-c": (55, 0)}, (210, 64)),
    ],
)
@pytest.mark.asyncio
async def test_each_project_reports_its_own_usage_and_the_pass_sums_them(
    isolated_data_dir, monkeypatch, caplog, projects, global_tokens,  # noqa: F811
):
    import app.blocks as _blocks
    from app.core import hydration_store
    from app.core.learning import hydration as hydration_module

    table = {f"project '{pid}'": toks for pid, toks in projects.items()}
    table[_GLOBAL_MARK] = global_tokens
    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")
    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _fake_chat_block(table))
    for i, pid in enumerate(projects):
        _seed_conversation(pid, f"{_DAY}T0{i + 1}:00:00Z", f"question {i} about {pid}?", "answer")

    caplog.set_level(logging.INFO, logger=hydration_module.logger.name)
    result = await hydration_module.run(target_date=_DAY)

    lines = _usage_lines(caplog)
    run_ids = {}
    for pid, (p_tok, c_tok) in projects.items():
        row = hydration_store.get_latest("project", pid)
        assert row["facts"]["usage"] == {
            "model_calls": 1, "prompt_tokens": p_tok, "completion_tokens": c_tok,
        }
        assert _line("project", pid, row["id"], 1, p_tok, c_tok) in lines
        run_ids[pid] = row["id"]

    total = {
        "model_calls": len(projects) + 1,
        "prompt_tokens": sum(t[0] for t in projects.values()) + global_tokens[0],
        "completion_tokens": sum(t[1] for t in projects.values()) + global_tokens[1],
    }
    glob = hydration_store.get_latest("global")
    assert result["usage"] == total
    assert glob["facts"]["usage"] == {
        "model_calls": 1, "prompt_tokens": global_tokens[0], "completion_tokens": global_tokens[1],
    }
    assert glob["facts"]["pass_usage"] == total
    assert result["run_id"] == glob["id"]
    assert result["project_run_ids"] == run_ids

    pass_line = _line(
        "pass", "all", glob["id"],
        total["model_calls"], total["prompt_tokens"], total["completion_tokens"],
    )
    assert pass_line in lines
    assert result["cost_line"] == pass_line


@pytest.mark.parametrize(
    "project_id, flag_value",
    [("proj-quiet", None), ("other-quiet", "false")],
)
@pytest.mark.asyncio
async def test_flag_off_pass_makes_no_model_call_and_says_so(
    isolated_data_dir, monkeypatch, caplog, project_id, flag_value,  # noqa: F811
):
    """Flag unset (the default) or false: the line still prints, zero calls."""
    import app.blocks as _blocks
    from app.core.learning import hydration as hydration_module

    if flag_value is None:
        monkeypatch.delenv("HYDRATION_LLM_SUMMARY", raising=False)
    else:
        monkeypatch.setenv("HYDRATION_LLM_SUMMARY", flag_value)

    class _Boom:
        def __init__(self):
            raise AssertionError("no model call when HYDRATION_LLM_SUMMARY is off")

    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _Boom)
    _seed_conversation(project_id, f"{_DAY}T08:00:00Z", "door schedule?", "see A-501")

    caplog.set_level(logging.INFO, logger=hydration_module.logger.name)
    result = await hydration_module.run(target_date=_DAY)

    zero = "model_calls=0 prompt_tokens=0 completion_tokens=0"
    assert result["usage"] == {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    assert result["cost_line"].endswith(zero)
    assert any(
        line.startswith(f"hydration_run scope=project project={project_id} ")
        and line.endswith(zero)
        for line in _usage_lines(caplog)
    )


@pytest.mark.parametrize(
    "provider, expected_calls",
    [("offline_template", 0), ("openrouter", 2)],
)
@pytest.mark.asyncio
async def test_only_a_provider_answer_counts_as_a_model_call(
    isolated_data_dir, monkeypatch, provider, expected_calls,  # noqa: F811
):
    """The offline template is not a model call; a provider answer is (one
    for the project, one for the global rollup)."""
    import app.blocks as _blocks
    from app.core.learning import hydration as hydration_module

    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")
    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _fake_chat_block({}, provider=provider))
    _seed_conversation("proj-off", f"{_DAY}T08:00:00Z", "cable tray length?", "unknown")

    result = await hydration_module.run(target_date=_DAY)
    assert result["usage"]["model_calls"] == expected_calls


def _route_setup(monkeypatch, tokens):
    import app.blocks as _blocks

    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")
    table = {f"project '{pid}'": toks for pid, toks in tokens.items()}
    table[_GLOBAL_MARK] = (20, 5)
    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _fake_chat_block(table))
    for i, pid in enumerate(tokens):
        _seed_conversation(pid, f"{_DAY}T09:0{i}:00Z", f"ask {i}?", "reply")


_ROUTE_TOKENS = {"proj-asked": (500, 90), "proj-other": (70, 11)}


@pytest.mark.parametrize(
    "auth",
    [{"role": "user", "user_id": "someone"}, {"user_id": "no-role"}],
)
@pytest.mark.asyncio
async def test_on_demand_route_refuses_a_non_admin(isolated_data_dir, monkeypatch, auth):  # noqa: F811
    """``POST /v1/hydration/run?project_id=`` without the admin role: 403, no row."""
    from app.core import hydration_store
    from app.routers.hydration import HydrationRunRequest, hydration_run

    _route_setup(monkeypatch, _ROUTE_TOKENS)
    with pytest.raises(HTTPException) as denied:
        await hydration_run(
            HydrationRunRequest(target_date=_DAY), project_id="proj-asked", auth=auth,
        )
    assert denied.value.status_code == 403
    assert hydration_store.get_latest("project", "proj-asked") is None
    assert hydration_store.get_latest("global") is None


@requires_construction_kit
@pytest.mark.parametrize("named, not_named", [("proj-asked", "proj-other"), ("proj-other", "proj-asked")])
@pytest.mark.asyncio
async def test_on_demand_route_runs_only_the_named_project_for_admin(
    isolated_data_dir, monkeypatch, named, not_named,  # noqa: F811
):
    """As an admin: only the named project runs, with its run id and the
    pass's cost line."""
    from app.core import hydration_store
    from app.routers.hydration import HydrationRunRequest, hydration_run

    _route_setup(monkeypatch, _ROUTE_TOKENS)
    envelope = await hydration_run(
        HydrationRunRequest(target_date=_DAY),
        project_id=named,
        auth={"role": "admin", "user_id": "ops"},
    )
    assert envelope["status"] == "success", envelope
    result = envelope["result"]
    p_tok, c_tok = _ROUTE_TOKENS[named]
    assert result["projects_processed"] == 1
    assert list(result["project_run_ids"]) == [named]
    assert result["usage"] == {
        "model_calls": 2, "prompt_tokens": p_tok + 20, "completion_tokens": c_tok + 5,
    }
    assert result["run_id"] == hydration_store.get_latest("global")["id"]
    assert result["cost_line"] == _line(
        "pass", "all", result["run_id"], 2, p_tok + 20, c_tok + 5,
    )
    assert hydration_store.get_latest("project", not_named) is None


# ── The pieces of the pass the usage accounting rides on ──────────────────


@pytest.mark.parametrize(
    "named, active",
    [(["proj-n1"], ["proj-n1", "proj-n2"]), (["site-y", "site-z"], ["site-x", "site-y", "site-z"])],
)
@pytest.mark.asyncio
async def test_named_project_ids_replace_auto_detection(
    isolated_data_dir, named, active,  # noqa: F811
):
    """``project_ids`` given: only those projects run, even when others were
    active in the window; each named project has its own usage row."""
    from app.core import hydration_store
    from app.core.learning import hydration as hydration_module

    for i, pid in enumerate(active):
        _seed_conversation(pid, f"{_DAY}T1{i}:00:00Z", f"ask {i}?", "reply")

    result = await hydration_module.run(target_date=_DAY, project_ids=named)

    assert sorted(result["project_run_ids"]) == sorted(named)
    assert result["projects_processed"] == len(named)
    for pid in active:
        row = hydration_store.get_latest("project", pid)
        assert (row is not None) == (pid in named), pid


@pytest.mark.parametrize("cap, conversations", [(3, 4), (5, 6)])
@pytest.mark.asyncio
async def test_project_messages_are_capped_before_summary(
    isolated_data_dir, monkeypatch, cap, conversations,  # noqa: F811
):
    """More messages than the per-project cap: the pass reads the cap, and
    the row says how many it saw."""
    from app.core import hydration_store
    from app.core.learning import hydration as hydration_module

    monkeypatch.setattr(hydration_module, "_MAX_MESSAGES_PER_PROJECT", cap)
    pid = f"proj-cap-{cap}"
    for i in range(conversations):
        _seed_conversation(pid, f"{_DAY}T0{i}:00:00Z", f"ask {i}?", "reply")

    result = await hydration_module.run(target_date=_DAY)

    assert 2 * conversations > cap
    assert hydration_store.get_latest("project", pid)["facts"]["messages_seen"] == cap
    assert result["projects_processed"] == 1


@pytest.mark.parametrize(
    "query_pid, body_pids, expected",
    [
        ("proj-q", None, ["proj-q"]),
        ("proj-q", ["proj-b"], ["proj-b", "proj-q"]),
    ],
)
@pytest.mark.asyncio
async def test_on_demand_route_dispatches_hydrate_with_the_named_projects(
    monkeypatch, query_pid, body_pids, expected,
):
    """The admin route hands ``learning_engine`` one ``hydrate`` op whose
    ``project_ids`` merge the query parameter into the body list, and returns
    what the block returned."""
    import app.blocks as _blocks
    from app.routers.hydration import HydrationRunRequest, hydration_run

    seen = []

    class _FakeEngine:
        @classmethod
        def shared_instance(cls):
            return cls()

        async def execute(self, data, params):
            seen.append(dict(data))
            return {"status": "success", "result": {"echo": list(data.get("project_ids") or [])}}

    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "learning_engine", _FakeEngine)
    envelope = await hydration_run(
        HydrationRunRequest(target_date=_DAY, project_ids=body_pids),
        project_id=query_pid,
        auth={"role": "admin", "user_id": "ops"},
    )
    assert seen == [{"operation": "hydrate", "target_date": _DAY, "project_ids": expected}]
    assert envelope == {"status": "success", "result": {"echo": expected}}


@pytest.mark.parametrize("project_id", ["proj-a", None])
@pytest.mark.asyncio
async def test_on_demand_route_says_503_when_learning_engine_is_absent(monkeypatch, project_id):
    """No ``learning_engine`` block loaded: the admin route answers 503, it
    does not crash on a missing class."""
    import app.blocks as _blocks
    from app.routers.hydration import HydrationRunRequest, hydration_run

    monkeypatch.delitem(_blocks.BLOCK_REGISTRY, "learning_engine", raising=False)
    with pytest.raises(HTTPException) as unavailable:
        await hydration_run(
            HydrationRunRequest(target_date=_DAY),
            project_id=project_id,
            auth={"role": "admin", "user_id": "ops"},
        )
    assert unavailable.value.status_code == 503
