"""A valuation stated to date, with nothing said about earlier certificates, is an ask.

Live (chains class 4, scenario M33 in the PR body): an ask gave its valuation
as a running total ("valuation to date") and asked what to pay this month,
without saying what earlier certificates covered. The certificate took that
amount as nil, printed the whole running total less retention as this
month's payment, and only hedged in prose. Nothing was recorded as an
input the turn still needed.

The mechanism tested here:
- the reader reads a valuation labelled "to date" / "cumulative" and marks
  it as a running total;
- the certificate, given a running total and no amount certified before,
  returns the needs-input envelope that names that one input, instead of
  a figure;
- an amount certified before, once stated (a nil included), lets the
  certificate run; a valuation not labelled as a running total runs as before.

Every decision runs on two inputs, one each way. All figures are invented;
no project, document or scenario is named in product code.
"""
from __future__ import annotations

import asyncio

import pytest

from app.containers.construction import ConstructionContainer
from app.containers.construction.boq import _payment_figures_from_message


def _run(ask: str, params: dict | None = None) -> dict:
    container = ConstructionContainer()
    return asyncio.run(container.payment_certificate({"message": ask}, params or {}))


# --- the reader: a running-total label marks the valuation ------------------

@pytest.mark.parametrize("ask, gross", [
    ("Valuation to date is 3,450,000 on a 9 M contract, retention 5 %.", 3_450_000.0),
    ("cumulative value of work done 712,000, retention 10 %", 712_000.0),
    ("Gross valuation to date SAR 1,980,500. Retention 5%.", 1_980_500.0),
    ("work done to date 2.4 million, retention 10 %", 2_400_000.0),
])
def test_running_total_valuation_is_read_and_marked(ask, gross):
    fig = _payment_figures_from_message(ask)
    assert fig.get("gross_valuation") == gross
    assert fig.get("gross_to_date") is True


@pytest.mark.parametrize("ask", [
    # a valuation with no running-total word is this period's, as before
    "gross valuation 640,000, retention 5 %",
    # "to date" on a quantity, not on a valuation of work
    "Planned to date 1,200, done 1,050 of 3,000 m3",
    # "certified to date" names an amount certified, not a valuation
    "certified to date 3,100,000",
    # a percentage of work done to date is a rate, not a valuation
    "work done to date 40 %, contract 5,000,000",
])
def test_no_running_total_valuation_no_mark(ask):
    assert "gross_to_date" not in _payment_figures_from_message(ask)


# --- the certificate: a running total without earlier amount asks once -----

@pytest.mark.parametrize("ask", [
    "Valuation to date is 3,450,000 on a 9 M contract, retention 5 %. What do I pay this month?",
    "Cumulative valuation 1,275,000, retention 10 %. Net payment this period?",
])
def test_running_total_without_previous_amount_asks_for_it(ask):
    out = _run(ask)
    assert out.get("status") != "success", out
    assert out.get("needs_input") is True
    assert out.get("missing") == ["previous_certified"]
    assert "earlier" in out.get("question", "").lower()
    # no figure is offered as this period's payment
    assert "payment" not in out


@pytest.mark.parametrize("ask, previous, gross, ret_pct", [
    # the amount certified before is in the same ask
    ("Valuation to date is 3,450,000 on a 9 M contract, retention 5 %. "
     "Previously certified 2,100,000.", 2_100_000.0, 3_450_000.0, 5.0),
    # a stated nil is a statement: a first certificate runs
    ("Cumulative valuation 1,275,000, retention 10 %, previously certified 0.",
     0.0, 1_275_000.0, 10.0),
])
def test_running_total_with_previous_amount_stated_runs(ask, previous, gross, ret_pct):
    out = _run(ask)
    assert out.get("status") == "success", out
    assert out["valuation"]["gross_valuation"] == gross
    assert out["deductions"]["previous_payments"] == previous
    assert out["deductions"]["retention_percent"] == ret_pct


@pytest.mark.parametrize("params, runs", [
    # a caller that passes the earlier amount is a statement of it
    ({"previous_certified": 1_600_000}, True),
    # a zero a caller fills in on its own is not the operator's statement
    ({"previous_certified": 0}, False),
])
def test_caller_figure_counts_only_when_it_is_a_figure(params, runs):
    ask = "Valuation to date is 2,640,000 on a 7 M contract, retention 5 %."
    out = _run(ask, params)
    assert (out.get("status") == "success") is runs, out
    assert bool(out.get("needs_input")) is (not runs)


@pytest.mark.parametrize("ask, net", [
    # this-period gross, no running-total word: unchanged behaviour
    # 640,000 - 5 % (32,000) = 608,000
    ("gross valuation 640,000, retention 5 %, contract 4,000,000", 608_000.0),
    # 820,000 - 10 % (82,000) = 738,000
    ("work valued at 820,000, retention 10 %, contract 6,000,000", 738_000.0),
])
def test_period_valuation_without_running_total_runs_as_before(ask, net):
    out = _run(ask)
    assert out.get("status") == "success", out
    assert out["payment"]["net_due_this_period"] == net


def _real_container(monkeypatch) -> None:
    """The dispatcher looks the container up in the block registry, which
    only holds it under the construction profile; hand it the real one so
    both CI profiles run the same path."""
    container = ConstructionContainer()
    monkeypatch.setattr(
        "app.dependencies.get_block_instance",
        lambda name: container if name == "construction" else None,
    )


# --- the live tool path: the model's call, the ask recorded as turn state ----
# Live, the model chose payment_certificate itself and passed the gross it
# read; the ask's own words went along as the user message. This drives that
# entry (the runtime dispatcher every payment_certificate call goes through)
# and the turn record the answer and the route are read from.

@pytest.mark.parametrize("ask, model_args, asks", [
    # running total, nothing about earlier certificates: the turn asks
    ("Valuation to date is 3,450,000 on a 9 M contract, retention 5 %. "
     "What do I pay this month?",
     {"gross_valuation": 3_450_000, "contract_value": 9_000_000,
      "retention_percent": 5}, True),
    # the same call with the earlier amount in the ask: the turn computes
    # 3,450,000 - 5 % (172,500) - 2,100,000 = 1,177,500 under the calculator's
    # own rule (deductions on the stated gross, earlier amount subtracted)
    ("Valuation to date is 3,450,000 on a 9 M contract, retention 5 %, "
     "previously certified 2,100,000. What do I pay this month?",
     {"gross_valuation": 3_450_000, "contract_value": 9_000_000,
      "retention_percent": 5}, False),
])
async def test_model_call_records_the_missing_input_as_turn_state(
    monkeypatch, ask, model_args, asks,
):
    _real_container(monkeypatch)
    from app.agents.answer_exit import Turn, note_tool_result
    from app.agents.runtime import _dispatch_payment_certificate

    out = await _dispatch_payment_certificate(dict(model_args), user_message=ask)
    turn = Turn()
    note_tool_result("payment_certificate", out, turn)
    assert out["ok"] is (not asks), out
    assert turn.awaiting_inputs == (["previous_certified"] if asks else [])
    assert bool(turn.questions) is asks
    if not asks:
        assert out["result"]["payment"]["net_due_this_period"] == 1_177_500.0


async def test_reply_with_the_earlier_amount_lets_the_next_call_run(monkeypatch):
    _real_container(monkeypatch)
    # Turn 2 of the same thread: the reply is the bare figure, and the
    # model's call carries it under the name the ask recorded. The reply
    # has no running-total word of its own, and the figure is now stated.
    from app.agents.runtime import _dispatch_payment_certificate

    out = await _dispatch_payment_certificate(
        {"gross_valuation": 1_275_000, "retention_percent": 10,
         "previous_certified": 800_000},
        user_message="800,000.",
    )
    assert out["ok"] is True, out
    # 1,275,000 - 10 % (127,500) - 800,000 = 347,500
    assert out["result"]["payment"]["net_due_this_period"] == 347_500.0
