"""6b latency lever (i): static-prefix-first ordering (ORDERS_31OCT_1525Z).

The driver system message leads with the STATIC block (intro, standing rules,
the hat catalogue, the chosen hat's guidance) -- identical across projects and
turns -- and ends with the per-project profile. Two different projects therefore
share that whole static lead as a common prefix, which is what the provider's
prefix cache keys on. A regression that moves the per-project profile back to
the front would shorten the shared prefix (it would diverge before the static
block), reddening these tests.
"""
from __future__ import annotations

import os

import pytest

# Dotted import so CI-M selects this for app.agents.driver.context::system_message.
import app.agents.driver.context as ctx

_HATS = sorted({h["discipline"] for h in ctx.hats()})
_HAT_WITH_GUIDANCE = next((h for h in _HATS if (ctx.hat_guidance(h) or "").strip()), None)


@pytest.mark.parametrize("proj_a, proj_b", [
    ("proj-aaaa", "proj-bbbb"),
    ("proj-cccc", "proj-dddd"),
])
def test_static_lead_is_the_shared_prefix_profile_is_last(proj_a, proj_b):
    a = ctx.system_message(proj_a, None, None)["content"]
    b = ctx.system_message(proj_b, None, None)["content"]
    common = os.path.commonprefix([a, b])
    # The static lead is shared verbatim between two different projects -- this
    # is the prefix the cache reuses. If the profile were first, the strings
    # would diverge before this block and the asserts would fail.
    assert "Standing rules" in common, common[-80:]
    assert "Hats you can wear" in common, common[-80:]
    # The per-project profile is past the shared lead (project-specific).
    assert proj_a in a and proj_a not in common


@pytest.mark.parametrize("proj", ["proj-eeee", "proj-ffff"])
def test_hat_guidance_still_precedes_the_profile(proj):
    if _HAT_WITH_GUIDANCE is None:
        pytest.skip("no hat declares system_prompt_guidance")
    content = ctx.system_message(proj, None, _HAT_WITH_GUIDANCE)["content"]
    guid_head = (ctx.hat_guidance(_HAT_WITH_GUIDANCE) or "").strip().split("\n")[0][:24]
    # guidance is present (kills the `if guidance:` drop_rule) and sits in the
    # static lead, before the per-project profile at the end.
    assert guid_head and guid_head in content
    assert content.index("Hats you can wear") < content.rindex(proj)
    assert content.index(guid_head) < content.rindex(proj)
