"""Driver route catalogue: every workflow described by what it produces."""
from __future__ import annotations



def test_each_workflow_is_described_by_what_it_produces():
    """Lane 2 CORE: programme questions went to the WBS workflow. The catalogue
    now says what each workflow produces and that a figure a calculator
    computes goes to the calculator."""
    from app.agents.driver import routes

    d = routes.schema(routes.catalogue())["function"]["description"]
    assert "goes to that calculator, not to a workflow" in d
    for name in routes.catalogue():
        line = next(x for x in d.splitlines() if x.startswith(f"- {name}"))
        assert len(line) > len(f"- {name}") + 10, f"{name} has no description"
