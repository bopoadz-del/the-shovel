"""A3 model test — scorer + offline stub (zero network).

The 10-row answer-style/model-quality table is exercised against a canned
stub provider so the harness can be proven with no model call, no key, no
spend. The verdict is COMPUTED from the scored passes and the five fixed
criteria — nothing is hardcoded from an expected table.

Run:  python -m tools.a3_model_test.a3_model_test
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Tuple

# --- fixed criteria (0625Z plan, advisor-fixed; never renegotiated) -----------
CRITERIA: Dict[str, Any] = {
    "flips_strong_min": 5,       # (a) >=5 of the 7 flip rows pass >=4/5 on arm B
    "flip_pass_of": 4,           #      >=4 of 5 per flip row
    "controls_pass_min": 15,     # (b) the three controls 15/15 on arm B
    "no_row_lower": True,        # (c) no row scores lower on B than on A
    "b_default_sampling": True,  #     arm B at the provider's default sampling
    "hard_stop": 100,            #     100-turn hard stop
}

FLIP_ROWS = ("A3", "A5", "A7", "B6", "B7", "B9", "B10")
CONTROL_ROWS = ("B1", "B2", "B5")
ROWS = FLIP_ROWS + CONTROL_ROWS
REPS = 5
ARMS = ("A", "B")


def pass_rule(row_id: str, answer: str) -> bool:
    """One answer's fixed pass rule (the tester's own criteria)."""
    t = answer.lower()
    if row_id == "A3":
        if "88" not in t or "3,013,000" not in t:
            return False
        return not ("72" in t and "88" in t and ("partial" in t or "no choice" in t))
    if row_id == "A5":
        return "40" in t and "50" in t and "in-situ" in t and "precast" in t
    if row_id == "A7":
        return "75" in t and "src" in t and "25" in t
    if row_id == "B6":
        return "st-100-2000103" in t and "aci 318-14" in t and "vol 2" in t and "specification" in t
    if row_id == "B7":
        if "aci 318-19" in t:
            return False
        return "aci 318-14" in t and "25" in t and "40" in t
    if row_id == "B9":
        if re.search(r"documents differ.*7\.5.*9", t):
            return False
        return "12.5" in t and "7.5" in t and "load-bearing" in t
    if row_id == "B10":
        return "500" in t and "420" in t and "bs 4449" in t and "differ" in t
    if row_id == "B1":
        return "type 2" in t and "high yield" in t and "420" in t
    if row_id == "B2":
        return "1.0" in t
    if row_id == "B5":
        return ("9,136,000" in t and "4,568" in t) or ("350,000" in t and "100" in t)
    raise ValueError(f"unknown row {row_id}")


# --- offline stub provider: canned answers, no network, no key -----------------

_PASS = {
    "A3": "The total is 88 No, and the contract sum is SAR 3,013,000.",
    "A5": "In-situ is SRC 40 and precast is SRC 50.",
    "A7": "Blinding is 75 mm thick, grade SRC 25.",
    "B6": "Drawing ST-100-2000103, ACI 318-14, Vol 2 Specification.",
    "B7": "ACI 318-14 governs. Cover: slabs/walls 25 mm or bar dia; beams/columns 40 mm.",
    "B9": "Solid blocks: 12.5 N/mm2 load-bearing, 7.5 non-load-bearing.",
    "B10": "The drawing states 500 N/mm2 (BS 4449); the method statement states 420. The documents differ.",
    "B1": "Type 2 high yield deformed bars, grade 420 N/mm2.",
    "B2": "The dimension is 1.0 m.",
    "B5": "Lump sum SAR 9,136,000 on about 4,568 m2.",
}
_FAIL = "No figure was found."

#: passes per row each arm's stub returns (data, not expected verdicts).
_STUB_PASSES: Dict[str, Dict[str, int]] = {
    "A": {"A3": 2, "A5": 2, "A7": 3, "B6": 2, "B7": 2, "B9": 3, "B10": 1,
          "B1": 3, "B2": 4, "B5": 3},
    "B": {"A3": 5, "A5": 4, "A7": 4, "B6": 5, "B7": 5, "B9": 5, "B10": 4,
          "B1": 5, "B2": 5, "B5": 5},
}


def stub_answer(arm: str, row_id: str, rep: int) -> str:
    """The stub's canned answer for one rep (offline, deterministic)."""
    passes = _STUB_PASSES[arm][row_id]
    return _PASS[row_id] if rep < passes else _FAIL


def run_table(answers: Dict[str, Dict[str, List[str]]]) -> Tuple[Dict[str, Dict[str, int]], bool, str]:
    """Score every cell and compute the verdict from CRITERIA as data."""
    passes: Dict[str, Dict[str, int]] = {"A": {}, "B": {}}
    for arm in ARMS:
        for row in ROWS:
            passes[arm][row] = sum(1 for a in answers[arm][row] if pass_rule(row, a))
    b = passes["B"]
    a = passes["A"]
    strong = sum(1 for r in FLIP_ROWS if b[r] >= CRITERIA["flip_pass_of"])
    if strong < CRITERIA["flips_strong_min"]:
        return passes, False, f"only {strong}/{len(FLIP_ROWS)} flips >= {CRITERIA['flip_pass_of']}/5 on B"
    controls = sum(b[r] for r in CONTROL_ROWS)
    if controls < CRITERIA["controls_pass_min"]:
        return passes, False, f"controls {controls}/{CRITERIA['controls_pass_min']} on B"
    for r in ROWS:
        if b[r] < a[r]:
            return passes, False, f"{r} lower on B ({b[r]}) than A ({a[r]})"
    return passes, True, "GAIN"


def print_table(passes: Dict[str, Dict[str, int]], verdict: bool, why: str) -> None:
    print(f"{'row':<6}{'kind':<8}{'A passes':<10}{'B passes':<10}")
    for row in ROWS:
        kind = "flip" if row in FLIP_ROWS else "control"
        print(f"{row:<6}{kind:<8}{passes['A'][row]:<10}{passes['B'][row]:<10}")
    print(f"verdict: {'GAIN' if verdict else 'NO GAIN'} ({why})")


def main() -> int:
    answers = {
        arm: {row: [stub_answer(arm, row, rep) for rep in range(REPS)] for row in ROWS}
        for arm in ARMS
    }
    passes, verdict, why = run_table(answers)
    print_table(passes, verdict, why)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
