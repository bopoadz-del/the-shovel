"""Checker and runner for exam C1 (conflict between documents).

The exam reads an answer in contract form:

    {"text": str, "conflicts": [conflict, ...]}

    conflict = {
        "values": [{"value": <as written>, "document": <doc name>}, ...],
        "governs": <doc name> | None,          # C3: only from a precedence
        "basis_document": <doc name> | None,   #     statement in the evidence
    }

A case where documents disagree passes only when exactly one conflict is
reported, its (document, value) pairs are exactly the expected ones, every
value is said in the text (no single silent pick), and the winner is the
expected one -- ``None`` where no precedence statement covers the pair. A case
where they agree, only look different, or answer different questions passes
only when no conflict is reported.

The checker compares strings as written (whitespace collapsed, case folded).
It has no unit or date logic of its own, so it cannot share a normalisation
mistake with the product.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).with_name("cases.json")


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def _norm(s: Any) -> str:
    return " ".join(str(s or "").split()).casefold()


def _pairs(values: Any) -> Counter:
    out: Counter = Counter()
    for v in values or []:
        if isinstance(v, dict):
            out[(_norm(v.get("document")), _norm(v.get("value")))] += 1
    return out


def check(case: dict[str, Any], output: dict[str, Any]) -> tuple[bool, list[str]]:
    """``(passed, reasons)`` for one case. Reasons name what failed."""
    exp = case["expected"]
    conflicts = [c for c in (output.get("conflicts") or []) if isinstance(c, dict)]
    text = _norm(output.get("text"))
    reasons: list[str] = []
    if not exp["conflict"]:
        if conflicts:
            reasons.append(f"false conflict reported ({len(conflicts)})")
        return not reasons, reasons
    if not conflicts:
        return False, ["no conflict reported"]
    if len(conflicts) != 1:
        reasons.append(f"expected one conflict, got {len(conflicts)}")
    want = _pairs(exp["values"])
    match = next((c for c in conflicts if _pairs(c.get("values")) == want), None)
    if match is None:
        reasons.append("no conflict carries exactly the expected values and sources")
    for v in {_norm(v["value"]) for v in exp["values"]}:
        if v not in text:
            reasons.append(f"value not said in the text: {v!r}")
    if match is not None:
        got = match.get("governs")
        if _norm(got) != _norm(exp.get("governs")):
            reasons.append(f"governs {got!r}, expected {exp.get('governs')!r}")
        if exp.get("governs") and _norm(match.get("basis_document")) != _norm(exp.get("basis_document")):
            reasons.append("precedence basis not the document that states it")
    return not reasons, reasons


def good_output(case: dict[str, Any]) -> dict[str, Any]:
    """The output a correct product would give, built from the expectation
    alone (used to prove the checker passes what it should)."""
    exp = case["expected"]
    if not exp["conflict"]:
        return {"text": case["model_answer"], "conflicts": []}
    said = "; ".join(f"{v['value']} ({v['document']})" for v in exp["values"])
    return {
        "text": f"The documents differ: {said}.",
        "conflicts": [{
            "values": [dict(v) for v in exp["values"]],
            "governs": exp.get("governs"),
            "basis_document": exp.get("basis_document"),
        }],
    }


def score(cases: list[dict[str, Any]], produce: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        try:
            out = produce(case)
            ok, why = check(case, out)
        except Exception as exc:  # noqa: BLE001 -- a crash is a failed case, not a crashed exam
            ok, why = False, [f"crashed: {type(exc).__name__}: {exc}"]
        results.append({"id": case["id"], "kind": case["kind"], "passed": ok, "reasons": why})
    by_kind: dict[str, list[int]] = {}
    for r in results:
        k = by_kind.setdefault(r["kind"], [0, 0])
        k[0] += int(r["passed"])
        k[1] += 1
    return {"passed": sum(r["passed"] for r in results), "total": len(results),
            "by_kind": by_kind, "results": results}


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def product_output(case: dict[str, Any]) -> dict[str, Any]:
    """The product's answer contract on the case: the model's answer (which
    silently picks one value) and the evidence rows. The question is passed
    when the contract accepts it."""
    from app.agents import answer_contract

    kwargs: dict[str, Any] = {"sources": case["rows"]}
    if "question" in inspect.signature(answer_contract.apply).parameters:
        kwargs["question"] = case["question"]
    res = answer_contract.apply(case["model_answer"], **kwargs)
    conflicts = [_plain(c) for c in (getattr(res, "conflicts", None) or [])]
    return {"text": res.text, "conflicts": conflicts}


def main() -> int:
    report = score(load_cases(), product_output)
    print(f"C1 conflict exam: {report['passed']}/{report['total']}")
    for kind, (p, t) in sorted(report["by_kind"].items()):
        print(f"  {kind}: {p}/{t}")
    for r in report["results"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']}: {'; '.join(r['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
