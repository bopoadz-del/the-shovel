"""#49 post-hoc: pin _safe_stream_margin's error fallback (survivor M5).

The post-hoc mutation sample on the landed #49 found the ``except -> return
128`` fallback unpinned: a mutant returning 0 or 64 on error survived. The
contract (docstring) is "on any doubt hold back MORE (128), never less", so a
smaller margin on error could leak a partial withheld party name. This pins it.
The dead acronym term (survivor M4) is deleted in the same change.
"""
from __future__ import annotations

import pytest

from app.agents.core.streaming import _safe_stream_margin


@pytest.mark.parametrize("excerpts", [
    "Engineer: Acme International Construction and Engineering Company Limited",
    "short excerpt with no party",
])
def test_margin_holds_back_more_on_error(monkeypatch, excerpts):
    def _boom(*_a, **_k):
        raise RuntimeError("party-name extraction failed")

    monkeypatch.setattr("app.core.party_names.extract_party_names", _boom)
    # On ANY error the margin must be the LARGER fallback (128), never the
    # 64 floor or 0 -- a partial name must never slip through a smaller hold.
    assert _safe_stream_margin(excerpts) == 128
