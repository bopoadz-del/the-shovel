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


def _fake_chat_block(token_table):
    """A ChatBlock stand-in whose provider usage depends on the prompt.

    ``token_table`` maps a substring of the prompt to (prompt, completion)
    token counts, so two projects in one pass report different usage.
    """

    class _FakeChat:
        async def execute(self, data, params):
            prompt = data.get("text") or ""
            for marker, (p_tok, c_tok) in token_table.items():
                if marker in prompt:
                    return {
                        "text": "## summary\n- ok\n",
                        "provider": "deepseek",
                        "tokens": {"prompt_tokens": p_tok, "completion_tokens": c_tok},
                    }
            return {"text": "## summary\n- ok\n", "provider": "deepseek", "tokens": {}}

    return lambda: _FakeChat()


def _usage_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("hydration_run ")]


@pytest.mark.asyncio
async def test_two_projects_report_their_own_usage_and_the_pass_sums_them(
    isolated_data_dir, monkeypatch, caplog,  # noqa: F811
):
    import app.blocks as _blocks
    from app.core import hydration_store
    from app.core.learning import hydration as hydration_module

    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")
    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _fake_chat_block({
        "project 'proj-one'": (1000, 200),
        "project 'proj-two'": (300, 50),
        "global hydration rollup": (40, 7),
    }))
    _seed_conversation("proj-one", "2026-05-26T09:00:00Z", "rebar quantity level 2?", "120 t")
    _seed_conversation("proj-two", "2026-05-26T11:00:00Z", "slab pour date?", "Friday")

    caplog.set_level(logging.INFO, logger=hydration_module.logger.name)
    result = await hydration_module.run(target_date="2026-05-26")

    assert result["usage"] == {"model_calls": 3, "prompt_tokens": 1340, "completion_tokens": 257}
    one = hydration_store.get_latest("project", "proj-one")
    two = hydration_store.get_latest("project", "proj-two")
    glob = hydration_store.get_latest("global")
    assert one["facts"]["usage"] == {"model_calls": 1, "prompt_tokens": 1000, "completion_tokens": 200}
    assert two["facts"]["usage"] == {"model_calls": 1, "prompt_tokens": 300, "completion_tokens": 50}
    assert glob["facts"]["usage"] == {"model_calls": 1, "prompt_tokens": 40, "completion_tokens": 7}
    assert glob["facts"]["pass_usage"] == result["usage"]

    assert result["run_id"] == glob["id"]
    assert result["project_run_ids"] == {"proj-one": one["id"], "proj-two": two["id"]}

    lines = _usage_lines(caplog)
    assert (
        f"hydration_run scope=project project=proj-one run_id={one['id']} "
        "model_calls=1 prompt_tokens=1000 completion_tokens=200"
    ) in lines
    assert (
        f"hydration_run scope=project project=proj-two run_id={two['id']} "
        "model_calls=1 prompt_tokens=300 completion_tokens=50"
    ) in lines
    pass_line = (
        f"hydration_run scope=pass project=all run_id={glob['id']} "
        "model_calls=3 prompt_tokens=1340 completion_tokens=257"
    )
    assert pass_line in lines
    assert result["cost_line"] == pass_line


@pytest.mark.asyncio
async def test_default_pass_makes_no_model_call_and_says_so(
    isolated_data_dir, monkeypatch, caplog,  # noqa: F811
):
    """Flag off (the default): the line still prints, with zero calls."""
    import app.blocks as _blocks
    from app.core.learning import hydration as hydration_module

    monkeypatch.delenv("HYDRATION_LLM_SUMMARY", raising=False)

    class _Boom:
        def __init__(self):
            raise AssertionError("no model call when HYDRATION_LLM_SUMMARY is off")

    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _Boom)
    _seed_conversation("proj-quiet", "2026-05-26T08:00:00Z", "door schedule?", "see A-501")

    caplog.set_level(logging.INFO, logger=hydration_module.logger.name)
    result = await hydration_module.run(target_date="2026-05-26")

    assert result["usage"] == {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    assert result["cost_line"].endswith("model_calls=0 prompt_tokens=0 completion_tokens=0")
    assert any(
        line.startswith("hydration_run scope=project project=proj-quiet ")
        and line.endswith("model_calls=0 prompt_tokens=0 completion_tokens=0")
        for line in _usage_lines(caplog)
    )


@pytest.mark.asyncio
async def test_offline_template_is_not_counted_as_a_model_call(
    isolated_data_dir, monkeypatch,  # noqa: F811
):
    import app.blocks as _blocks
    from app.core.learning import hydration as hydration_module

    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")

    class _Offline:
        async def execute(self, data, params):
            return {"text": "offline", "provider": "offline_template"}

    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", lambda: _Offline())
    _seed_conversation("proj-off", "2026-05-26T08:00:00Z", "cable tray length?", "unknown")

    result = await hydration_module.run(target_date="2026-05-26")
    assert result["usage"]["model_calls"] == 0


def _route_setup(monkeypatch):
    import app.blocks as _blocks

    monkeypatch.setenv("HYDRATION_LLM_SUMMARY", "true")
    monkeypatch.setitem(_blocks.BLOCK_REGISTRY, "chat", _fake_chat_block({
        "project 'proj-asked'": (500, 90),
        "global hydration rollup": (20, 5),
    }))
    _seed_conversation("proj-asked", "2026-05-26T09:00:00Z", "pile count?", "48")
    _seed_conversation("proj-other", "2026-05-26T09:30:00Z", "crane hire?", "two weeks")


@pytest.mark.asyncio
async def test_on_demand_route_refuses_a_non_admin(isolated_data_dir, monkeypatch):  # noqa: F811
    """``POST /v1/hydration/run?project_id=`` as a plain user: 403, no row."""
    from app.core import hydration_store
    from app.routers.hydration import HydrationRunRequest, hydration_run

    _route_setup(monkeypatch)
    with pytest.raises(HTTPException) as denied:
        await hydration_run(
            HydrationRunRequest(target_date="2026-05-26"),
            project_id="proj-asked",
            auth={"role": "user", "user_id": "someone"},
        )
    assert denied.value.status_code == 403
    assert hydration_store.get_latest("project", "proj-asked") is None
    assert hydration_store.get_latest("global") is None


@requires_construction_kit
@pytest.mark.asyncio
async def test_on_demand_route_runs_one_project_for_admin(isolated_data_dir, monkeypatch):  # noqa: F811
    """The same call as an admin: the named project only, with its run id
    and the pass's cost line."""
    from app.core import hydration_store
    from app.routers.hydration import HydrationRunRequest, hydration_run

    _route_setup(monkeypatch)
    envelope = await hydration_run(
        HydrationRunRequest(target_date="2026-05-26"),
        project_id="proj-asked",
        auth={"role": "admin", "user_id": "ops"},
    )
    assert envelope["status"] == "success", envelope
    result = envelope["result"]
    assert result["projects_processed"] == 1
    assert list(result["project_run_ids"]) == ["proj-asked"]
    assert result["usage"] == {"model_calls": 2, "prompt_tokens": 520, "completion_tokens": 95}
    assert result["run_id"] == hydration_store.get_latest("global")["id"]
    assert result["cost_line"].startswith(
        f"hydration_run scope=pass project=all run_id={result['run_id']} "
    )
    # The project the admin did not name was not hydrated.
    assert hydration_store.get_latest("project", "proj-other") is None
