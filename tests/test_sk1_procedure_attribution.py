"""
S-K1 -- a skill procedure's licence obligation is recorded where its lineage says so.

The design note's rule: a procedure whose method lineage is CC BY 4.0 or
Apache-2.0 carries an ``attribution`` (CC BY with the licence link; Apache with
the NOTICE retained); an MIT / common-practice procedure carries none. The rule
is read from each shipped procedure's own ``lineage`` text, on every hat that
carries procedures, so a newly imported hat is held to it without naming any
playbook here.
"""
from __future__ import annotations

import pytest

from app.agents import catalog
from app.agents.procedures import render_procedures

_ATTRIBUTION_LICENCES = ("CC BY 4.0", "Apache-2.0")


def _hats_with_procedures():
    return [h for h in catalog.list_hats() if h.playbook.procedures]


def test_two_hats_carry_procedures():
    assert len(_hats_with_procedures()) >= 2


@pytest.mark.parametrize("hat", _hats_with_procedures(), ids=lambda h: h.id)
def test_attribution_follows_lineage_licence(hat):
    rendered = render_procedures(hat)
    for proc in hat.playbook.procedures:
        named = [lic for lic in _ATTRIBUTION_LICENCES if lic in proc.lineage]
        if not named:
            assert proc.attribution is None, f"{hat.id}/{proc.id}: attribution without a licence that needs one"
            continue
        assert proc.attribution is not None, f"{hat.id}/{proc.id}: {named[0]} lineage without attribution"
        assert proc.attribution.licence == named[0]
        if named[0] == "CC BY 4.0":
            assert proc.attribution.link
        else:
            assert proc.attribution.notice
        assert f"Attribution: {proc.attribution.licence}" in rendered
