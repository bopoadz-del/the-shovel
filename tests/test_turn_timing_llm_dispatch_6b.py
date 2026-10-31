"""6b: llm_first_token is measured from the LLM-dispatch stamp, not from the
last completed stage.

THE DEFECT (live 1afd1b4, 2026-10-31): `llm_first_token` read ~0.0013 s median
on 31 warm asks while the real pre-first-token wait sat in `route` (~0.93 s).
`first_token()` computed `now - st["last"]`, and `st["last"]` is reset by every
`stage()` / `mark()` that runs — retrieval, pre-dispatch, hat — so a turn that
opened any stage between the model dispatch and its first token collapsed the
metric to the gap-to-that-stage, not the model-call wait.

THE FIX: `mark_llm_dispatch()` stamps the generation-turn start, and
`first_token()` measures from that stamp when present. These tests pin that the
figure tracks the real dispatch gap AND survives an intervening stage — the
mutant that reverts to `st["last"]` reddens the intervening-stage assertion.
"""
from __future__ import annotations

import time

import pytest

import app.core.turn_timing as tt


@pytest.mark.parametrize("gap", [0.03, 0.20])
def test_llm_first_token_tracks_the_dispatch_gap_through_an_intervening_stage(gap):
    # A turn: route runs (resets `last`), the model is dispatched, the real
    # wait elapses, THEN an intervening stage (retrieval) resets `last` to ~now
    # — exactly the live shape. The metric must still read the dispatch gap.
    tt.mark_arrival()
    with tt.stage("route"):
        pass
    tt.mark_llm_dispatch()
    time.sleep(gap)                      # the real model-call wait
    with tt.stage("retrieval"):          # resets st["last"] to ~now
        pass
    tt.first_token()

    lft = tt.snapshot()["llm_first_token"]
    # Tracks the real dispatch gap (two distinct gaps -> two distinct reads)...
    assert lft >= gap * 0.8, (lft, gap)
    assert lft <= gap + 0.5, (lft, gap)
    # ...and is NOT collapsed to the ~0 intervening-stage gap (the live bug /
    # the "measure from st['last']" mutant).
    assert lft >= 0.02, lft


@pytest.mark.parametrize("gap", [0.03, 0.20])
def test_without_a_dispatch_stamp_it_falls_back_to_last_stage(gap):
    # No mark_llm_dispatch(): the metric measures from the last stage, the
    # pre-fix behaviour, preserved as a fallback for any path that never stamps.
    tt.mark_arrival()
    with tt.stage("route"):
        pass
    time.sleep(gap)
    tt.first_token()
    lft = tt.snapshot()["llm_first_token"]
    assert lft >= gap * 0.8, (lft, gap)
    assert lft <= gap + 0.5, (lft, gap)
