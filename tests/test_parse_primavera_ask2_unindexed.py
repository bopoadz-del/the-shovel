"""Live tip 4ab55613: parse_primavera_schedule ask2 naming agent-d-valid-fixture.xer
must not short-circuit to the unindexed-reference refusal.

Exit evidence: tools=[], answer started with
"I could not confirm this reference in the indexed project sources for FIXTURE-d-..."
"""
from __future__ import annotations

from app.agents.runtime import (
    _message_wants_primavera_parse,
    _should_short_circuit_rag_miss,
)


ASK2 = (
    "parse_primavera_schedule: parse agent-d-valid-fixture.xer from this project's documents. "
    "Extract milestones MS-01/MS-02/MS-03 with dates."
)


def test_message_wants_primavera_parse_ask2():
    assert _message_wants_primavera_parse(ASK2)


def test_xer_named_parse_not_rag_miss_short_circuit():
    audit = {
        "identifier_miss": True,
        "threshold_fired": True,
        "extracted_identifiers": ["agent-d-valid-fixture.xer", "MS-01", "2026"],
    }
    assert _should_short_circuit_rag_miss(audit, None, ASK2) is False
