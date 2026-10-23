"""Live Phase 2: payment_certificate must not run with empty args.

Ask1 often MATCHES on labelled IPC numbers. Ask2 (measured works /
retention / MOS / contract sum) failed because the forced tool or
construction route emitted ``{}`` and never coerced the ask into the
action dict. Both asks must populate the invocation — not leave
``payment_certificate`` with an empty payload or the wrong route.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import Agent
from app.containers.construction import ConstructionContainer
from app.containers.construction.boq import _payment_figures_from_message
from tests.conftest import requires_construction_kit

# Ask1 — labelled progress / contract / retention (live MATCH shape).
ASK1 = (
    "Prepare an interim payment certificate for 42% progress on a "
    "SAR 25M contract with 10% retention."
)

# Ask2 — QS phrasing the live Phase 2 battery used. Synthetic numbers so
# the test never depends on a project corpus.
ASK2 = (
    "Issue the interim payment certificate. Measured works SAR 1,200,000, "
    "retention 5%, materials on site SAR 80,000, contract sum SAR 10,000,000."
)


def _agent():
    return Agent(
        name="project-assistant",
        description="",
        system_prompt="x",
        allowed_blocks=["construction"],
    )


def _empty_call(name: str, extra: dict | None = None):
    """The forced-tool / empty-payload shape the live fail used."""
    args = {} if extra is None else dict(extra)
    return {
        "id": "ipc-empty",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _merged(input_data, params) -> dict:
    out = {}
    if isinstance(params, dict):
        out.update(params)
    if isinstance(input_data, dict):
        out.update(input_data)
    return out


def _assert_ipc_fields_populated(merged: dict, *, ask: str) -> None:
    assert merged, f"payment_certificate invoked with empty args for: {ask}"
    assert merged != {}, f"payment_certificate invoked with {{}} for: {ask}"
    # The ask's figures must be on the invocation, not only inside a
    # leftover message the handler might never read.
    if ask is ASK1:
        assert float(merged.get("contract_value") or 0) == 25_000_000.0, merged
        assert float(merged.get("work_done_percent") or 0) == 42.0, merged
        assert float(merged.get("retention_percent") or 0) == 10.0, merged
    else:
        assert float(merged.get("contract_value") or 0) == 10_000_000.0, merged
        assert float(merged.get("retention_percent") or 0) == 5.0, merged
        measured = float(
            merged.get("measured_works")
            or merged.get("gross_valuation")
            or 0
        )
        assert measured >= 1_200_000.0, merged
        assert float(merged.get("materials_on_site") or 0) == 80_000.0, merged


@pytest.mark.parametrize("ask", [ASK1, ASK2])
def test_parser_extracts_synthetic_ipc_numbers(ask):
    fig = _payment_figures_from_message(ask)
    _assert_ipc_fields_populated(fig, ask=ask)


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("ask", [ASK1, ASK2])
@pytest.mark.parametrize(
    "tool_name,extra",
    [
        ("payment_certificate", None),
        ("construction", {"action": "payment_certificate"}),
        ("construction", {}),
        ("construction_calc", None),
        ("interim_certificate_generator", None),
    ],
)
async def test_empty_tool_payload_populates_payment_certificate_args(
    monkeypatch, ask, tool_name, extra,
):
    """Forced / empty payload → payment_certificate sees the ask's fields."""
    captured: list[tuple] = []
    real = ConstructionContainer.payment_certificate

    async def spy(self, input_data, params):
        captured.append((input_data, params))
        return await real(self, input_data, params)

    monkeypatch.setattr(ConstructionContainer, "payment_certificate", spy)

    result = await _agent()._run_tool_call(
        _empty_call(tool_name, extra),
        user_message=ask,
    )
    assert captured, (
        f"{tool_name} {extra!r} never invoked payment_certificate "
        f"(result={result!r})"
    )
    merged = _merged(*captured[0])
    _assert_ipc_fields_populated(merged, ask=ask)
    assert result.get("ok") is True, result
    inner = result.get("result") or {}
    # Envelope (execute) or direct container result.
    payload = inner.get("result") if isinstance(inner.get("result"), dict) else inner
    assert payload.get("status") == "success" or inner.get("status") == "success", result
