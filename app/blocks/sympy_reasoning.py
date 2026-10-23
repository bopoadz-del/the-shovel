"""SymPy Reasoning Block — symbolic variance analysis + construction recommendations.

This block lives up to its name: it builds real symbolic expressions for
each metric using sympy.symbols, lets sympy simplify them once at module
import, then substitutes the per-item numerics in to produce both:
- the closed-form formula string (so the caller can see the math)
- the evaluated numeric value

A previous version of this block wrapped every float in ``sp.Float()`` then
immediately unwrapped via ``float()`` with no symbolic manipulation in
between, which made the "symbolic" naming a lie. The current version emits
formulas alongside numbers so a user can see exactly what was computed.
"""

from typing import Any, Dict, List, Optional, Tuple
import ast
import re
from app.core.universal_base import UniversalBlock

# Symbolic expressions are built ON FIRST USE, then cached for the process.
#
# They used to be built at import time, which pulled sympy into every boot
# (~1.0s) even though app/blocks/__init__.py imports this module eagerly and
# most requests never evaluate a formula. See
# tests/test_boot_does_not_import_heavy_libraries.py for why boot weight
# mattered: it cost the service its Render deploy.
#
# The simplified strings are NOT hardcoded here on purpose — sympy renders
# variance_pct as "100*actual/avg - 100", not the naive form, and the caller is
# shown the real expression so the computation stays auditable.
_SYMBOLIC: Optional[Dict[str, Any]] = None


def _symbolic() -> Dict[str, Any]:
    """Build (once) and return the symbolic forms this block reasons with.

    Keys: ``available`` (bool), ``exprs`` (tag -> sympy expression, empty when
    sympy is missing), and ``strings`` (tag -> human-readable formula).
    """
    global _SYMBOLIC
    if _SYMBOLIC is not None:
        return _SYMBOLIC
    try:
        import sympy as _sp
    except ImportError:  # pragma: no cover — sympy is in requirements.txt
        _SYMBOLIC = {
            "available": False,
            "exprs": {},
            "strings": {
                "variance_pct": "(actual - avg) / avg * 100",
                "z_score": "(actual - avg) / std_dev",
                "cost_impact": "(actual - avg) * quantity",
            },
        }
        return _SYMBOLIC

    actual, avg, std_dev, quantity = _sp.symbols(
        "actual avg std_dev quantity", real=True
    )
    exprs = {
        # variance_pct = (actual - avg) / avg * 100   — simplified canonical form
        "variance_pct": _sp.simplify((actual - avg) / avg * 100),
        # z_score      = (actual - avg) / std_dev
        "z_score": _sp.simplify((actual - avg) / std_dev),
        # cost_impact  = (actual - avg) * quantity
        "cost_impact": _sp.simplify((actual - avg) * quantity),
    }
    _SYMBOLIC = {
        "available": True,
        "exprs": exprs,
        "strings": {tag: str(expr) for tag, expr in exprs.items()},
        # Callers substitute by plain name; _eval_symbolic maps those to these
        # objects. It must be THESE instances: they carry real=True, and
        # sympy treats Symbol("actual") and Symbol("actual", real=True) as
        # different symbols, so a bare string key would silently fail to
        # substitute and leave the expression unevaluated.
        "symbols": {
            "actual": actual,
            "avg": avg,
            "std_dev": std_dev,
            "quantity": quantity,
        },
    }
    return _SYMBOLIC


def _eval_symbolic(formula_name: str, subs: Dict) -> float:
    """Substitute numerics for a named formula and return a float.

    The formula is identified by a string tag — `"variance_pct"`, `"z_score"`,
    or `"cost_impact"` — rather than a sympy expression object, so the
    no-sympy fallback never references any symbol that only exists inside the
    sympy-only branch.

    `subs` may be keyed by sympy Symbols (when sympy is available) OR by the
    plain string names ``"actual"``, ``"avg"``, ``"std_dev"``, ``"quantity"``.
    The pure-Python fallback uses the latter.
    """
    symbolic = _symbolic()
    if symbolic["available"]:
        expr = symbolic["exprs"].get(formula_name)
        if expr is None:
            raise RuntimeError(f"Unknown formula: {formula_name}")
        # Accept string keys as documented above by mapping them to the real
        # (real=True) Symbol instances; pass through anything already a Symbol.
        names = symbolic["symbols"]
        resolved = {
            (names.get(key, key) if isinstance(key, str) else key): value
            for key, value in subs.items()
        }
        return float(expr.subs(resolved))

    # Pure-Python fallback. Read values from the subs dict by string key so we
    # don't depend on the sympy Symbol objects existing.
    def _lookup(key_str, default):
        # Accept either string keys or sympy-symbol-style keys with that name.
        for k, val in subs.items():
            if getattr(k, "name", None) == key_str or k == key_str:
                return val
        return default

    a = _lookup("actual", 0)
    v = _lookup("avg", 0)
    s = _lookup("std_dev", 0)
    q = _lookup("quantity", 1)

    if formula_name == "variance_pct":
        return (a - v) / v * 100 if v else 0.0
    if formula_name == "z_score":
        return (a - v) / s if s else 0.0
    if formula_name == "cost_impact":
        return (a - v) * q
    raise RuntimeError(f"Unknown formula: {formula_name}")


class SymPyReasoningBlock(UniversalBlock):
    name = "sympy_reasoning"
    version = "1.1.0"
    description = (
        "Symbolic variance analysis: builds real sympy expressions for "
        "variance %, z-score, and cost impact; emits both the formula "
        "strings and the evaluated values."
    )
    layer = 3
    tags = ["domain", "construction", "reasoning", "math", "symbolic", "sympy"]
    requires = []
    allow_empty_input = True

    default_config = {
        "variance_threshold_pct": 10.0,
        "confidence_floor": 0.7,
    }

    ui_schema = {
        "input": {
            "type": "json",
            "placeholder": '{"boq_data": [...], "drawing_data": {}, "spec_data": {}, "historical_benchmarks": {}}',
            "multiline": True,
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "recommendations", "type": "list", "label": "Recommendations"},
                {"name": "variances", "type": "list", "label": "Variances"},
                {"name": "cost_impacts", "type": "list", "label": "Cost Impacts"},
                {"name": "formulas", "type": "json", "label": "Formulas"},
            ],
        },
        "quick_actions": [
            {"icon": "📊", "label": "Analyze BOQ", "prompt": "Analyze BOQ vs historical benchmarks"},
            {"icon": "⚠️", "label": "Find Variances", "prompt": "Find cost variances above 10%"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        expr = data.get("expression") or data.get("expr") or params.get("expression") or params.get("expr")
        if expr:
            return self._eval_free_expression(str(expr))

        boq_data = data.get("boq_data", [])
        drawing_data = data.get("drawing_data", {})
        spec_data = data.get("spec_data", {})
        historical_benchmarks = data.get("historical_benchmarks", {})

        if not _symbolic()["available"]:
            return {"status": "error", "error": "sympy not installed. Run: pip install sympy"}

        threshold = float(
            params.get("variance_threshold_pct",
                        self.config.get("variance_threshold_pct", 10.0))
        )

        # Two variance sources, both stay in the same `variances` list so
        # downstream consumers don't need to branch:
        #   1. BOQ rate vs historical_benchmarks (the original design — needs
        #      a populated benchmarks dict to fire).
        #   2. BOQ quantity vs drawing_data quantity (the construction-domain
        #      headline use case; what the heavy-reasoning prompt expects).
        variances = self._compute_variances(boq_data, historical_benchmarks, threshold)
        variances.extend(self._compute_qty_variances(boq_data, drawing_data, threshold))
        cost_impacts = self._compute_cost_impacts(variances)
        recommendations = self._generate_recommendations(variances, spec_data, drawing_data)

        return {
            "status": "success",
            "recommendations": recommendations,
            "variances": variances,
            "cost_impacts": cost_impacts,
            "items_analyzed": len(boq_data),
            "high_variance_count": sum(1 for v in variances if v.get("severity") == "high"),
            # Expose the closed-form formulas the block evaluated. This makes
            # the math auditable — anyone reading the response can verify the
            # block isn't doing something different from what its name suggests.
            "formulas": dict(_symbolic()["strings"]),
        }

    def _eval_free_expression(self, expr: str) -> Dict:
        """Evaluate a user-supplied arithmetic expression (18.4 - 16) * 2850.

        Only digits and + - * / ** () . are allowed. This is the leftover-hat
        L7 path: heavy-reasoning asked sympy for 18.4−16 then ×2850 and got
        empty variance metadata because boq_data was missing.
        """
        cleaned = expr.replace(",", "").replace("×", "*").replace("−", "-").strip()
        if not cleaned or not re.fullmatch(r"[0-9+\-*/().eE\s]+", cleaned):
            return {
                "status": "error",
                "error": (
                    "expression must be arithmetic digits and + - * / ** () only"
                ),
                "expression": expr,
            }
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as e:
            return {"status": "error", "error": f"invalid expression: {e}", "expression": expr}
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.FloorDiv,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return {
                    "status": "error",
                    "error": f"disallowed syntax: {type(node).__name__}",
                    "expression": expr,
                }
        try:
            value = float(self._eval_ast_number(tree))
        except Exception as e:
            return {"status": "error", "error": str(e), "expression": expr}
        # Binary floats like (18.4-16)*2850 land on 6839.999999999996;
        # round through a short significant-figure form so the tool
        # result is the 6840 the user asked for.
        value = float(format(value, ".12g"))
        return {
            "status": "success",
            "expression": cleaned,
            "value": value,
            "variances": [],
            "cost_impacts": [],
            "formulas": {"expression": cleaned},
        }

    @staticmethod
    def _eval_ast_number(node: ast.AST) -> float:
        """Walk a vetted AST; no eval/exec — security_scan forbids both."""
        if isinstance(node, ast.Expression):
            return SymPyReasoningBlock._eval_ast_number(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric constant")
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            val = SymPyReasoningBlock._eval_ast_number(node.operand)
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("disallowed unary op")
        if isinstance(node, ast.BinOp):
            left = SymPyReasoningBlock._eval_ast_number(node.left)
            right = SymPyReasoningBlock._eval_ast_number(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12 or abs(left) > 1e6:
                    raise ValueError("exponent out of range")
                return left ** right
            raise ValueError("disallowed binary op")
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    def _compute_variances(
        self, boq_data: List, benchmarks: Dict, threshold: float
    ) -> List[Dict]:
        variances = []
        for item in boq_data:
            key = (
                item.get("item_key")
                or item.get("description", "").lower().replace(" ", "_")
            )
            # An item with `unit_cost: 0` was previously silently falling
            # through to `rate` — explicit None checks now disambiguate.
            actual_raw = item.get("unit_cost")
            if actual_raw is None:
                actual_raw = item.get("rate", 0)
            actual = float(actual_raw or 0)
            benchmark = benchmarks.get(key, {})
            avg_cost = float(benchmark.get("avg_cost", 0))
            std_dev = float(benchmark.get("std_dev", 0))

            if avg_cost <= 0:
                continue

            subs = {"actual": actual, "avg": avg_cost, "std_dev": std_dev}
            variance_pct = _eval_symbolic("variance_pct", subs)
            z_score = _eval_symbolic("z_score", subs) if std_dev > 0 else 0.0

            if abs(variance_pct) > threshold * 2:
                severity = "high"
            elif abs(variance_pct) > threshold:
                severity = "medium"
            else:
                severity = "low"

            variances.append(
                {
                    "item_key": key,
                    "description": item.get("description", key),
                    "actual_cost": actual,
                    "benchmark_avg": avg_cost,
                    "variance_pct": round(variance_pct, 2),
                    "z_score": round(z_score, 3),
                    "severity": severity,
                    "quantity": item.get("quantity", 1),
                    "unit": item.get("unit", ""),
                }
            )

        return sorted(variances, key=lambda x: abs(x["variance_pct"]), reverse=True)

    def _compute_qty_variances(
        self, boq_data: List, drawing_data: Dict, threshold: float
    ) -> List[Dict]:
        """Quantity variance: each BOQ row matched against a drawing_data entry.

        Match key precedence: ``item_key`` on the row -> ``description``
        lowered+snake-cased -> ``item``. Drawing values can be either a
        bare scalar (``"concrete": 1380``) or a dict
        (``"concrete": {"qty": 1380}``). Variance % is computed off the
        BOQ value so the sign matches "drawing minus BOQ" intuition.
        """
        out: List[Dict] = []
        for item in boq_data:
            key = (
                item.get("item_key")
                or (item.get("description", "") or "").lower().replace(" ", "_")
                or item.get("item")
            )
            if not key:
                continue
            dval = drawing_data.get(key)
            if dval is None:
                # also try the original ``item`` / ``description`` verbatim
                dval = drawing_data.get(item.get("item")) or drawing_data.get(item.get("description"))
            if dval is None:
                continue
            drawing_qty = float(dval.get("qty", dval.get("quantity", 0))) if isinstance(dval, dict) else float(dval)
            boq_qty = float(item.get("quantity") or 0)
            if boq_qty <= 0:
                continue
            qty_diff = drawing_qty - boq_qty
            variance_pct = (qty_diff / boq_qty) * 100.0
            rate = float(item.get("rate") or item.get("unit_cost") or 0)
            cost_impact = qty_diff * rate

            if abs(variance_pct) > threshold * 2:
                severity = "high"
            elif abs(variance_pct) > threshold:
                severity = "medium"
            else:
                severity = "low"

            out.append({
                "item_key": key,
                "description": item.get("description", key),
                "source": "boq_vs_drawing",
                "boq_quantity": boq_qty,
                "drawing_quantity": drawing_qty,
                "qty_diff": qty_diff,
                "variance_pct": round(variance_pct, 2),
                "rate": rate,
                "cost_impact": round(cost_impact, 2),
                "severity": severity,
                "quantity": boq_qty,
                "unit": item.get("unit", ""),
            })
        return out

    def _compute_cost_impacts(self, variances: List[Dict]) -> List[Dict]:
        impacts = []
        for v in variances:
            subs = {
                "actual": float(v.get("actual_cost", 0)),
                "avg": float(v.get("benchmark_avg", 0)),
                "quantity": float(v.get("quantity", 1)),
            }
            impact = _eval_symbolic("cost_impact", subs)
            if abs(impact) < 0.01:
                continue
            impacts.append(
                {
                    "item_key": v["item_key"],
                    "cost_impact_usd": round(impact, 2),
                    "direction": "over" if impact > 0 else "under",
                    "severity": v["severity"],
                }
            )
        return sorted(impacts, key=lambda x: abs(x["cost_impact_usd"]), reverse=True)

    def _generate_recommendations(
        self, variances: List[Dict], spec_data: Dict, drawing_data: Dict
    ) -> List[Dict]:
        templates = {
            "high_over": {
                "text": "URGENT: {item} is {pct:.1f}% over benchmark. Review supplier pricing and re-tender.",
                "severity": "critical",
                "action_items": [
                    "Re-tender to 3 suppliers",
                    "Check market rates",
                    "Negotiate volume discount",
                ],
            },
            "high_under": {
                "text": "ALERT: {item} is {pct:.1f}% under benchmark. Verify scope and quality compliance.",
                "severity": "warning",
                "action_items": [
                    "Verify specification compliance",
                    "Check material grade",
                    "Audit scope inclusion",
                ],
            },
            "medium_over": {
                "text": "REVIEW: {item} is {pct:.1f}% above benchmark. Monitor and negotiate.",
                "severity": "medium",
                "action_items": ["Request quote breakdown", "Compare with market index"],
            },
            "medium_under": {
                "text": "NOTE: {item} is {pct:.1f}% below benchmark. Confirm quality.",
                "severity": "low",
                "action_items": [
                    "Confirm material specification",
                    "Check labor inclusion",
                ],
            },
        }

        recs = []
        for v in variances:
            if v["severity"] == "low":
                continue
            direction = "over" if v["variance_pct"] > 0 else "under"
            tpl = templates.get(f"{v['severity']}_{direction}", templates["medium_over"])
            recs.append(
                {
                    "item_key": v["item_key"],
                    "recommendation": tpl["text"].format(
                        item=v["description"], pct=abs(v["variance_pct"])
                    ),
                    "severity": tpl["severity"],
                    "action_items": tpl["action_items"],
                    "variance_pct": v["variance_pct"],
                }
            )
        return recs
