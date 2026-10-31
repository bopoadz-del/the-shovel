"""A calculator's status must reflect whether it actually calculated.

Found by the 2026-08-15 all-76 calculator sweep. ``score_risk`` rejected
out-of-range inputs and returned {"error": "Probability and impact must each
be 1-5"} -- and ``run_calculation`` wrapped that in::

    {"status": "success", "result": {"error": "Probability and impact ..."}}

An error envelope inside a green one. Callers branch on ``status``, so the
agent tool loop read a refused calculation as a working answer.

Mechanism: ``run_calculation`` treated "did not raise" as "succeeded". But not
every calculator raises to signal failure -- several in
app/core/construction_knowledge.py validate inputs and RETURN an error dict.
The fix lives in run_calculation so all 76 are covered, rather than in the one
calculator the sweep happened to catch.

Fenced both ways: a real calculation must still report success, and a numeric
field named "error" (a tolerance or deviation, not a failure) must not be
mistaken for one.
"""
from __future__ import annotations

import app.lib.construction_formulas as cf


def test_rejected_inputs_do_not_report_success():
    """The regression: RED before the fix, which returned status success."""
    out = cf.run_calculation("score_risk", {"probability": 10, "impact": 10})
    assert out["status"] == "error", (
        "a calculator that refused its inputs must not report success; got "
        f"{out}"
    )
    assert "1 to 5" in out.get("error", ""), out


def test_a_real_calculation_still_reports_success():
    """The other direction, so the fix cannot degrade into refusing everything."""
    out = cf.run_calculation("score_risk", {"probability": 3, "impact": 4})
    assert out["status"] == "success", out
    assert out["result"]["score"] == 12
    assert out["result"]["band"] == "RED"


def test_no_successful_result_smuggles_a_string_error():
    """Sweep every calculator: status success must never carry an error string.

    This is the property the one-off assertion above generalises to. It runs
    over the live registry, so a calculator added later with the same
    return-an-error-dict habit is caught without editing this test.
    """
    offenders = []
    for name in cf.available_calculations():
        fn = cf.CALCULATORS.get(name)
        if fn is None:
            continue
        # Call with no params: calculators either run on their own defaults or
        # report a parameter error. Either is fine -- what must never happen is
        # status success carrying a string "error".
        out = cf.run_calculation(name, {})
        if out.get("status") == "success":
            result = out.get("result") or {}
            if isinstance(result, dict) and isinstance(result.get("error"), str):
                offenders.append((name, result["error"][:60]))
    assert not offenders, (
        "these reported success while returning an error message: " + str(offenders)
    )


def test_numeric_error_field_is_not_treated_as_a_failure():
    """A tolerance named "error" is data, not a failure signal."""
    assert cf._result_is_failure({"error": "bad input"}) is True
    assert cf._result_is_failure({"error": 0.02}) is False
    assert cf._result_is_failure({"value": 5}) is False
