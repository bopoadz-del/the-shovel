"""A bare IPC request must reach the predefined payment_certificate path.

Live find: "Issue the interim payment certificate for this month on the PRJ2
Infrastructure Package 1 contract." routed with ``action=None,
reason=named_calculator`` and ran NO tool. Instead of the clean
missing-figures error the API returns, the turn fell through to RAG and
answered a payment question out of procurement-programme excerpts.

``_message_wants_named_calculator`` guards against stealing that deliverable,
but tested it with ``re.search(r"\\d", raw)`` -- ANY digit anywhere. A project
code (PRJ2), a package number (Package 1), a date or a revision is not a
figure, so the guard never fired and the calculator branch took the turn.

The same function must still keep the F5 behaviour its docstring describes:
an IPC request that DOES carry figures stays on ``construction_calc`` rather
than being rerouted to a container action with empty params.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _message_wants_named_calculator


# Incidental digits that are NOT figures. Each of these previously flipped the
# guard and cost the turn its deliverable.
@pytest.mark.parametrize(
    "message",
    [
        "Issue the interim payment certificate for this month on the "
        "PRJ2 Infrastructure Package 1 contract.",
        "issue an interim payment certificate from the contract",
        "Issue the interim payment certificate for IPC No. 7 dated 15 March 2025.",
        "Prepare the payment certificate per drawing INF-054-CPH-460 Rev 2.",
        "Generate the interim payment certificate for Package 1.",
    ],
)
def test_bare_ipc_request_is_not_stolen_by_the_calculator(message):
    assert _message_wants_named_calculator(message) is False, (
        "a bare IPC request was classified as a named calculator, which clears "
        "`action` and prevents the predefined payment_certificate dispatch"
    )


# Real figures -> the calculator path, exactly as the docstring's F5 case says.
@pytest.mark.parametrize(
    "message",
    [
        "interim payment certificate gross 750000 with 5 percent retention",
        "Issue an interim payment certificate, gross valuation SAR 10,000,000, "
        "retention 10%",
    ],
)
def test_ipc_request_carrying_figures_still_goes_to_the_calculator(message):
    assert _message_wants_named_calculator(message) is True, (
        "an IPC request with explicit figures must stay on construction_calc; "
        "rerouting it runs the container action with empty params (live F5)"
    )


def test_figure_test_uses_the_same_parser_the_action_uses():
    """The guard must agree with whatever ``payment_certificate`` can actually
    parse, or the two drift and the bug returns in a new shape."""
    from app.agents.runtime import _carries_ipc_figures
    from app.containers.construction.boq import _payment_figures_from_message

    for message in (
        "Issue the interim payment certificate for Package 1.",
        "gross valuation SAR 10,000,000, retention 10%",
        "issue an interim payment certificate from the contract",
    ):
        assert _carries_ipc_figures(message) is bool(
            _payment_figures_from_message(message)
        ), message
