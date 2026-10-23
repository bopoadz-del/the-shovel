"""Live regression for the Engineer-identity answer on deepseek.

Defect (2026-09-14, deepseek-flash, build 2b48857): "Who is the Engineer?"
answered "The Engineer is 90 days of the effective date of a Letter of Award…"
4-5/5 — the appointment-timing clause in the identity slot instead of the firm
(EXAMPLAR / EXCON Example Limited). Fixed by rewording the inject hint to ENGINEER
IDENTITY (state the firm, not the date/period).

This test hits the LIVE deployed build, so it is SKIPPED unless FORK_API_KEY is
set (CI has no key). Run it after deploy:

    FORK_API_KEY=... python -m pytest tests/test_engineer_identity_live.py -q -s

It requires 5/5: the answer contains EXAMPLAR and never the appointment-timing
tokens ("Letter of Award", "90 days").
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

FORK_API_KEY = os.getenv("FORK_API_KEY")
BASE = os.getenv("FORK_BASE_URL", "https://theshovel.ai")

pytestmark = pytest.mark.skipif(
    not FORK_API_KEY, reason="FORK_API_KEY not set — live deploy test"
)


def _ask(message: str) -> str:
    body = json.dumps({"message": message, "project_id": "master_corpus"}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/stream", data=body,
        headers={"Authorization": f"Bearer {FORK_API_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    text = ""
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except Exception:
            continue
        if ev.get("type") == "token":
            text += ev.get("content", "")
    return " ".join(text.split())


def test_who_is_the_engineer_answers_jacobs_5x():
    """5/5 required: EXAMPLAR present, never the appointment-timing clause."""
    runs = []
    for _ in range(5):
        a = _ask("Who is the Engineer under this contract?")
        low = a.lower()
        ok = ("jacobs" in low or "ch2m" in low)
        junk = ("letter of award" in low or "90 days" in low)
        runs.append((ok and not junk, a[:120]))

    # `sum(1 for ok, _ in runs)` counts every tuple regardless of ok — it is
    # always len(runs), so the assert below could never fail. That vacuous form
    # shipped in #599 and reported green while the live answer was garbled
    # 10/10. Count only the passing runs.
    passes = sum(1 for ok, _ in runs if ok)
    detail = "\n".join(f"  {'PASS' if ok else 'FAIL'}: {snip}" for ok, snip in runs)
    assert passes == 5, f"Engineer identity {passes}/5 (need 5/5):\n{detail}"
