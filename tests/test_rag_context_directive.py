"""`_apply_rag_context` picks the directive that decides whether the model may answer.

Audit bar 3, 2026-08-12, agents subsystem. Replacing the whole body with
`return False` left all 150 agent tests green. One test file referenced the
function; none asserted which directive it chose.

That is the wrong thing to leave undetected, because the directive is not
formatting -- it is permission. The strict branch tells the model "if the
context does not contain the answer, say you don't have it", and the whole
reason the calculation branch exists is that the strict wording was observed
LIVE forbidding a correct answer:

    "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick and the
     number of 6m3 truck loads"
    -> "I cannot find that specific information in the provided reference
        context ... they do not contain a raft measuring 25 m x 18 m x 1.2 m"

25 x 18 x 1.2 = 540 m3, 540 / 6 = 90 loads. The model was not failing, it was
obeying. If the calculation branch silently stopped being selected, arithmetic
would start refusing itself again and every test in the repository would stay
green.

So the discriminating assertion throughout this file is not "some directive
was attached" but "the branch that fires does NOT carry the other branch's
wording". A regression that collapses three directives into one passes any
weaker check.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _apply_rag_context

CONTEXT = "REFERENCE CONTEXT\nFIDIC 20.1 requires notice within 28 days."

# The phrase that forbids answering from outside the context. Present in the
# strict branch by design, and fatal in the other two.
REFUSAL_WORDING = "using ONLY the reference context"

# A question carrying its own numbers and a compute verb. Must trip
# `_looks_like_self_contained_calculation`, which needs a calc verb plus at
# least `_MIN_DIMENSIONS_FOR_CALC` dimension tokens -- a bare "calculate the
# volume" would take the strict branch and prove nothing.
CALC_QUESTION = (
    "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick "
    "and the number of 6m3 truck loads"
)
GENERATE_QUESTION = "Generate a method statement for the raft pour"
LOOKUP_QUESTION = "What is the notice period under clause 20.1?"


def _turn(question: str) -> list[dict]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": question}]


def _applied_content(question: str) -> str:
    messages = _turn(question)
    applied = _apply_rag_context(messages, {"role": "system", "content": CONTEXT})
    assert applied is True, "_apply_rag_context reported it did nothing"
    return messages[-1]["content"]


# -- the branch selection, which is the whole point ------------------------

def test_a_self_contained_calculation_is_not_told_to_refuse(monkeypatch):
    """The live-observed bug. This branch existing is the fix; it firing is
    the behaviour."""
    monkeypatch.delenv("FORCE_CALC_ON_DIMENSIONS", raising=False)
    content = _applied_content(CALC_QUESTION)

    assert REFUSAL_WORDING not in content, (
        "a calculation whose inputs are all in the question was given the "
        "strict-grounding directive -- this is exactly the wording that made "
        "the platform refuse to multiply 25 x 18 x 1.2:\n" + content[-600:]
    )
    assert "CALCULATION REQUEST:" in content, content[-400:]
    assert "Do NOT refuse because the" in content, content[-400:]


def test_a_generate_request_may_use_domain_knowledge():
    """A method statement cannot be produced from context alone. The strict
    directive would make the model refuse or emit a stub."""
    content = _applied_content(GENERATE_QUESTION)

    assert REFUSAL_WORDING not in content, (
        "a generate request was told to answer only from context:\n" + content[-600:]
    )
    assert "REQUEST TO GENERATE" in content, content[-400:]
    # The protection that must survive in this branch: free to generate
    # standard content, never free to invent project facts.
    assert "Do NOT fabricate" in content, content[-400:]


def test_a_lookup_is_grounded_strictly():
    """The default, and the reason the other two branches have to be explicit
    exceptions rather than a loosened default."""
    content = _applied_content(LOOKUP_QUESTION)

    assert REFUSAL_WORDING in content, (
        "a factual lookup lost its strict-grounding directive, so the model is "
        "free to answer from its parametric prior:\n" + content[-600:]
    )
    assert "the context wins" in content, content[-400:]
    assert "QUESTION:" in content, content[-400:]


def test_the_three_directives_are_actually_different():
    """Guards the guard.

    Each test above passes if its branch fires. If a refactor collapsed the
    three into one string that happened to contain every asserted phrase, they
    would all still pass. Distinctness is the invariant they rely on.
    """
    calc = _applied_content(CALC_QUESTION)
    gen = _applied_content(GENERATE_QUESTION)
    lookup = _applied_content(LOOKUP_QUESTION)

    directives = {
        "calculation": calc[len(CONTEXT):],
        "generate": gen[len(CONTEXT):],
        "lookup": lookup[len(CONTEXT):],
    }
    for a, b in (("calculation", "generate"), ("calculation", "lookup"),
                 ("generate", "lookup")):
        assert directives[a] != directives[b], (
            f"the {a} and {b} directives are byte-identical -- the branch "
            "selection has collapsed and the tests above no longer "
            "distinguish anything"
        )


# -- placement, which is the reason the function exists at all -------------

def test_the_context_is_folded_into_the_user_turn_not_added_as_a_system_message():
    """The documented finding: with the context in a system block the model
    cites the doc-ids and still answers from its prior -- observed
    byte-identical answers whether the context was the correct KB note or
    unrelated templates. Folding it into the user turn is the fix, so a
    regression to a system message must fail here."""
    messages = _turn(LOOKUP_QUESTION)
    before = len(messages)
    _apply_rag_context(messages, {"role": "system", "content": CONTEXT})

    assert len(messages) == before, (
        "a message was inserted -- the context is being added as a separate "
        f"turn again rather than folded into the user turn: {messages}"
    )
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith(CONTEXT), (
        "the context is not at the head of the user turn"
    )
    assert LOOKUP_QUESTION in messages[-1]["content"], "the question was dropped"


def test_the_original_user_message_object_is_not_mutated_in_place():
    """`messages[-1] = {**messages[-1], ...}` replaces rather than mutates.

    Callers hold references to the conversation history; rewriting the stored
    turn would persist the injected context into the next turn's history and
    compound it every iteration.
    """
    original = {"role": "user", "content": LOOKUP_QUESTION}
    messages = [{"role": "system", "content": "sys"}, original]
    _apply_rag_context(messages, {"role": "system", "content": CONTEXT})

    assert original["content"] == LOOKUP_QUESTION, (
        f"the caller's own message object was rewritten: {original}"
    )


# -- the refusals ---------------------------------------------------------

@pytest.mark.parametrize("rag_sys_msg", [None, {}, {"content": ""}, {"content": None}])
def test_no_context_is_not_applied(rag_sys_msg):
    """Returning True with nothing attached would make the caller believe the
    turn was grounded when it was not."""
    messages = _turn(LOOKUP_QUESTION)
    assert _apply_rag_context(messages, rag_sys_msg) is False
    assert messages[-1]["content"] == LOOKUP_QUESTION, "content changed anyway"


def test_with_no_user_turn_to_attach_to_the_context_still_reaches_the_model():
    """The documented fallback. Unexpected shape, but dropping the retrieved
    context silently would produce an ungrounded answer with a sources panel
    -- worse than not retrieving at all."""
    messages = [{"role": "system", "content": "sys"},
                {"role": "assistant", "content": "prior answer"}]
    assert _apply_rag_context(messages, {"role": "system", "content": CONTEXT}) is True
    assert any(CONTEXT in (m.get("content") or "") for m in messages), (
        f"the context was reported as applied but is in no message: {messages}"
    )


def test_the_calculation_branch_can_be_switched_off_by_env():
    """`FORCE_CALC_ON_DIMENSIONS=0` is the documented kill-switch. If it
    stopped working there would be no way to revert the behaviour without a
    deploy."""
    import os
    from unittest import mock

    with mock.patch.dict(os.environ, {"FORCE_CALC_ON_DIMENSIONS": "0"}):
        content = _applied_content(CALC_QUESTION)
    assert "CALCULATION REQUEST:" not in content, (
        "the kill-switch did not disable the calculation branch"
    )
