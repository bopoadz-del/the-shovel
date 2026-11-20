"""Formula Layer 1, batch 15 (exam 5): the five procedure-rule calculators
that joined the registry on 19 Nov (U-1, #470) -- document number,
critical-rules gate, NCR next status, design status check, design review
timeline check.

These are rule checks with text / status inputs and boolean / string
outputs, not physical formulas, so there is no U (unit) case: the U kind
converts lengths, areas and stresses only, and none of these inputs carries
a physical unit.  The three other kinds stand:

  T  typical -- a normal input and the rule's expected verdict, worked by
     hand from the rule text;
  M  one required input missing -- the calculator must ASK for it by the
     input's declared label and must NOT return the verdict;
  B  boundary -- a value on the rule's limit, a value past it, or an input
     the rule cannot read.

Every expected value is derived by hand from the rule function and written
in the comment beside its parametrize row; none was copied from the
product's output.

  * document number     the register templates: RFI/IR/JR/PR/WP pad to 4,
                        NCR/PDN take an optional year then pad to 3, and the
                        rest (VO/RFM/DD, plus any unknown type) pad to 3;
                        an unknown type falls back to ``TYPE-0000`` with a
                        4-digit pad.
  * critical-rules gate the single rule: the word APPROVED (or approval)
                        in text that also carries a design-context keyword
                        (design / drawing / package / review / consultant /
                        architect / engineer / document) is one violation.
  * NCR next status     the fixed workflow RAISED -> ACKNOWLEDGED ->
                        DISPOSITION_PROPOSED -> DISPOSITION_REVIEWED ->
                        APPROVED -> IMPLEMENTING -> VERIFYING -> CLOSED;
                        CLOSED has no next step.
  * design status       valid = {FOR_COMMENT, ACCEPTANCE, BUY_OFF,
                        PENDING_DEBRIEF, SUPERSEDED}; forbidden = {APPROVED,
                        APPROVAL, SIGN_OFF}; anything else is invalid.
  * review timeline     period = workshop - distribution in calendar days;
                        under 7 days fails the minimum, 7 to 14 inclusive is
                        compliant, over 14 is compliant with a "consider
                        issuing closer" note, and an unparseable date is an
                        error.

All inputs are invented; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", env
    return env["result"]


def _asks(name: str, params: dict, missing: str, withheld: tuple[str, ...]) -> dict:
    """The call must ask for ``missing`` (by its declared label) and withhold
    every key in ``withheld`` -- it must not return the verdict it could not
    have formed without the input."""
    env = run_calculation(name, params)
    assert env.get("status") in ("needs_input", "error"), env
    text = " ".join(str(env.get(k) or "") for k in ("error", "question", "message", "note"))
    text += " " + " ".join(str(m) for m in (env.get("missing") or []))
    assert missing.lower() in text.lower(), (missing, env)
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


def _refuses(name: str, params: dict, withheld: tuple[str, ...]) -> dict:
    """A refusal: not success, no verdict in the withheld keys, no exception."""
    env = run_calculation(name, params)
    assert env.get("status") != "success", env
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


# ---------------------------------------------------------------- G1 document number
G1 = "generate_doc_number"
_G1_OUT = ("document_number", "doc_type")


@pytest.mark.parametrize("doc_type, sequence, year, number", [
    ("RFI", 42, None, "RFI-0042"),        # RFI pads to 4, no year
    ("NCR", 1, 2024, "NCR-2024-001"),     # NCR takes the year, pads to 3
    ("VO", 15, None, "VO-015"),           # VO pads to 3, no year
    ("DD", 23, None, "DD-023"),           # DD pads to 3
    ("WP", 7, None, "WP-0007"),           # WP pads to 4
])
def test_g1_t(doc_type, sequence, year, number):
    params = {"doc_type": doc_type, "sequence": sequence}
    if year is not None:
        params["year"] = year
    assert _ok(G1, params)["document_number"] == number


@pytest.mark.parametrize("missing", ["document type", "sequence number"])
def test_g1_m(missing):
    if missing == "document type":
        params = {"sequence": 42}
    else:
        params = {"doc_type": "RFI"}
    _asks(G1, params, missing, _G1_OUT)


@pytest.mark.parametrize("sequence, number", [
    (1, "RFI-0001"),      # the declared floor (1) still pads to 4
    (99999, "RFI-99999"),  # the declared ceiling (99,999)
])
def test_g1_b_sequence_bounds(sequence, number):
    assert _ok(G1, {"doc_type": "RFI", "sequence": sequence})["document_number"] == number


@pytest.mark.parametrize("sequence", [0, -1, 100000])
def test_g1_b_past_declared_range_refused(sequence):
    _refuses(G1, {"doc_type": "RFI", "sequence": sequence}, _G1_OUT)


# ---------------------------------------------------------------- G2 critical-rules gate
G2 = "enforce_critical_rules"
_G2_OUT = ("compliant", "violation_count", "violations")


@pytest.mark.parametrize("body, count", [
    ("The architect approved the drawing package.", 1),  # approved + design keyword
    ("The engineer approved the design.", 1),            # approved + engineer keyword
    ("Blockwork to grid C is complete.", 0),             # no forbidden word
])
def test_g2_t(body, count):
    r = _ok(G2, {"document_text": body})
    assert r["violation_count"] == count
    assert r["compliant"] is (count == 0)


@pytest.mark.parametrize("body", [
    "The client approved the budget.",   # approved, but no design-context keyword
    "Approval of the invoice is complete.",  # approval alone, no design context
])
def test_g2_b_no_design_context(body):
    r = _ok(G2, {"document_text": body})
    assert r["violation_count"] == 0
    assert r["compliant"] is True


def test_g2_m():
    _asks(G2, {}, "document text", _G2_OUT)


# ---------------------------------------------------------------- G3 NCR next status
G3 = "next_ncr_status"
_G3_OUT = ("next_status",)


@pytest.mark.parametrize("current, following", [
    ("RAISED", "ACKNOWLEDGED"),            # first step of the workflow
    ("DISPOSITION_REVIEWED", "APPROVED"),  # middle step
    ("IMPLEMENTING", "VERIFYING"),         # near the end
])
def test_g3_t(current, following):
    r = _ok(G3, {"ncr_status": current})
    assert r["next_status"] == following
    assert r["workflow_complete"] is False


def test_g3_b_end_of_sequence():
    # CLOSED is the terminal step: there is no next status and the workflow
    # is flagged complete.
    r = _ok(G3, {"ncr_status": "CLOSED"})
    assert r["next_status"] is None
    assert r["workflow_complete"] is True


@pytest.mark.parametrize("status", ["BOGUS", "CL0SED"])
def test_g3_b_unknown_status_refused(status):
    _refuses(G3, {"ncr_status": status}, _G3_OUT)


def test_g3_m():
    _asks(G3, {}, "current NCR status", _G3_OUT)


# ---------------------------------------------------------------- G4 design status check
G4 = "validate_design_status"
_G4_OUT = ("valid",)


@pytest.mark.parametrize("given", [
    "FOR_COMMENT",     # a valid status, exactly as declared
    "pending debrief",  # a valid status written lowercase with a space
    "BUY_OFF",         # another valid status
])
def test_g4_t(given):
    assert _ok(G4, {"design_status": given})["valid"] is True


@pytest.mark.parametrize("given, forbidden", [
    ("APPROVED", True),   # forbidden: APPROVED on a design document
    ("sign-off", True),   # forbidden: sign-off normalises to SIGN_OFF
    ("BOGUS", False),     # neither valid nor forbidden: just invalid
])
def test_g4_b_invalid_or_forbidden(given, forbidden):
    r = _ok(G4, {"design_status": given})
    assert r["valid"] is False
    if forbidden:
        assert "forbidden" in r["message"].lower()
    else:
        assert "not a valid" in r["message"].lower()


def test_g4_m():
    _asks(G4, {}, "design review status", _G4_OUT)


# ---------------------------------------------------------------- G5 design review timeline
G5 = "check_review_timeline"
_G5_OUT = ("compliant", "distribution_period_days")


@pytest.mark.parametrize("dist, workshop, days", [
    ("2026-01-01", "2026-01-10", 9),    # 9 days: inside the 7..14 window
    ("2026-01-01", "2026-01-08", 7),    # 7 days: exactly the minimum
    ("2026-01-01", "2026-01-15", 14),   # 14 days: exactly the maximum
])
def test_g5_t_compliant(dist, workshop, days):
    r = _ok(G5, {"distribution_date": dist, "workshop_date": workshop})
    assert r["distribution_period_days"] == days
    assert r["compliant"] is True


@pytest.mark.parametrize("dist, workshop, days", [
    ("2026-01-01", "2026-01-07", 6),     # 6 days: under the 7-day minimum
    ("2026-01-01", "2025-12-25", -7),    # workshop before distribution
])
def test_g5_b_under_minimum(dist, workshop, days):
    r = _ok(G5, {"distribution_date": dist, "workshop_date": workshop})
    assert r["distribution_period_days"] == days
    assert r["compliant"] is False


@pytest.mark.parametrize("workshop, days", [
    ("2026-01-20", 19),   # 19 days: still compliant, the rule names its 14-day max
    ("2026-01-22", 21),   # 21 days: same, farther past the maximum
])
def test_g5_b_over_maximum_notes_not_fails(workshop, days):
    r = _ok(G5, {"distribution_date": "2026-01-01", "workshop_date": workshop})
    assert r["distribution_period_days"] == days
    assert r["compliant"] is True
    assert "maximum" in r["message"].lower()


def test_g5_b_unparseable_date_refused():
    _refuses(G5, {"distribution_date": "1st April", "workshop_date": "2026-01-10"}, _G5_OUT)


@pytest.mark.parametrize("missing", ["distribution date", "workshop date"])
def test_g5_m(missing):
    if missing == "distribution date":
        params = {"workshop_date": "2026-01-10"}
    else:
        params = {"distribution_date": "2026-01-01"}
    _asks(G5, params, missing, _G5_OUT)
