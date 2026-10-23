"""Construction Knowledge Base loader and evaluator.

A utility module (not a UniversalBlock) that exposes the construction
knowledge base shipped at ``app/knowledge/construction_kb.json``.

Two entry kinds:

* ``formula`` entries carry a sympy-parseable ``expression``; call
  :func:`evaluate` with the formula's variables to get a numeric result
  tagged with provenance and credibility.
* ``workflow`` entries carry a list of guarded ``transitions`` between
  named ``states``; call :func:`validate_transition` to check whether
  a proposed transition is allowed given the current ``context``.

Workflow guards are author-controlled JSON strings but are still
SECURITY-SENSITIVE: they are evaluated with :func:`_safe_guard_eval`,
an AST allowlist walker that refuses anything beyond attribute /
subscript access on ``context``, comparisons, boolean ops, unary ops,
and literal constants. ``ast.Call``, ``ast.Lambda``, imports, names
other than ``context``, and every other node type are rejected.

The KB JSON is loaded once and cached by mtime in the same style as
``app/core/usage_tracker._load_pricing``: the operator can edit the
file without a restart and the next call picks up the change.
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

# sympy is imported lazily in evaluate() — it costs ~1.0s at BOOT and only
# `formula` entries need it, so a service that never evaluates a formula should
# not pay for it. Same pattern as app/blocks/sympy_reasoning.py:22.


_KB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "construction_kb.json"
)
_KB_OVERRIDE_ENV = "CONSTRUCTION_KB_FILE"

_LOCK = threading.RLock()
_KB_CACHE: Optional[Dict[str, Any]] = None
_KB_MTIME: float = 0.0


def _kb_path() -> str:
    return os.getenv(_KB_OVERRIDE_ENV) or _KB_PATH


def _load_kb() -> Dict[str, Any]:
    """Reload the KB JSON when its mtime changes.

    Returns the parsed top-level dict (``schema_version``, ``kb_version``,
    ``entries``). On error returns an empty-but-structured dict so callers
    never crash.
    """
    global _KB_CACHE, _KB_MTIME
    path = _kb_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {"schema_version": "0", "kb_version": "missing", "entries": []}
    with _LOCK:
        if _KB_CACHE is not None and mtime == _KB_MTIME:
            return _KB_CACHE
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "entries" not in data:
                data = {"schema_version": "0", "kb_version": "invalid", "entries": []}
            _KB_CACHE = data
            _KB_MTIME = mtime
        except (OSError, ValueError):
            _KB_CACHE = _KB_CACHE or {
                "schema_version": "0",
                "kb_version": "error",
                "entries": [],
            }
        return _KB_CACHE


def load_knowledge(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return entries; filter by domain (in applicability.applies_to) if given."""
    entries = list(_load_kb().get("entries", []))
    if domain is None:
        return entries
    return [
        e
        for e in entries
        if domain in (e.get("applicability", {}).get("applies_to") or [])
    ]


def get_rule(rule_id: str) -> Optional[Dict[str, Any]]:
    """Return entry by id, or None."""
    for entry in _load_kb().get("entries", []):
        if entry.get("id") == rule_id:
            return entry
    return None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def search_knowledge(
    query: str, top_k: int = 5, domain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Free-text retrieval over the KB: rank entries by token overlap between
    the query and each entry's id + title + statement, return the top-K.

    Lightweight + dependency-free (no vector index needed) so the construction
    blocks can map a natural-language question to the relevant rule(s). Empty /
    no-match query returns []. ``domain`` restricts to one applies_to namespace.
    """
    qt = _tokens(query)
    if not qt:
        return []
    scored: List[tuple] = []
    for e in load_knowledge(domain):
        hay = (
            _tokens((e.get("id") or "").replace(".", " "))
            | _tokens(e.get("title"))
            | _tokens(e.get("statement"))
        )
        score = len(qt & hay)
        if score:
            # secondary key: id-token hits weigh a touch more (title relevance)
            id_hits = len(qt & _tokens((e.get("id") or "").replace(".", " ")))
            scored.append((score + 0.5 * id_hits, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


def _build_warnings(entry: Dict[str, Any]) -> List[str]:
    """Standard warning list applied to every evaluator response.

    Entries at credibility tier 3 or below (site-experience priors or
    unverified) surface a "verify against your project spec or applicable
    standards" reminder. Entries flagged as region- or project-specific
    in the applicability block surface the same reminder so a prior
    sourced from one project is never silently applied to another.
    """
    warnings: List[str] = []
    tier = entry.get("credibility_tier")
    applic = entry.get("applicability", {}) or {}
    region = applic.get("region_specific")
    project = applic.get("project_specific")
    if isinstance(tier, int) and tier <= 3:
        warnings.append(
            f"credibility tier {tier}; verify against your project spec or applicable standards"
        )
    if region:
        warnings.append(
            f"region_specific={region}; verify against your project spec or applicable standards"
        )
    if project:
        warnings.append(
            f"project_specific={project}; verify against your project spec or applicable standards"
        )
    return warnings


def evaluate(rule_id: str, **values: Any) -> Dict[str, Any]:
    """Evaluate a formula entry.

    For ``type=formula``: parse the sympy expression, substitute the
    supplied variable values, and return a numeric result tagged with the
    threshold unit, provenance, credibility tier, and warnings.

    For ``type=workflow``: refuses with ``ValueError`` — callers must use
    :func:`validate_transition` instead.
    """
    entry = get_rule(rule_id)
    if entry is None:
        raise KeyError(f"unknown rule_id: {rule_id}")
    kind = entry.get("type")
    if kind == "workflow":
        raise ValueError("use validate_transition for workflows")
    if kind != "formula":
        raise ValueError(f"evaluate() only supports type=formula (got {kind!r})")

    import sympy

    expr_str = entry.get("expression")
    if not expr_str:
        raise ValueError(f"entry {rule_id!r} has no expression")
    # evaluate=False keeps sympy from collapsing the parsed tree before
    # substitution; we run .evalf() once at the end for the final float.
    expr = sympy.sympify(expr_str, evaluate=False)
    subs = {sympy.Symbol(name): value for name, value in values.items()}
    result = expr.subs(subs).evalf()
    try:
        result_f = float(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"formula did not reduce to a scalar — missing variables? got {result!r}"
        ) from exc

    return {
        "rule_id": rule_id,
        "result": result_f,
        "unit": (entry.get("thresholds") or {}).get("unit"),
        "provenance": entry.get("provenance", {}),
        "credibility_tier": entry.get("credibility_tier"),
        "warnings": _build_warnings(entry),
    }


# ---------------------------------------------------------------------------
# Safe guard evaluator
# ---------------------------------------------------------------------------

# Only these comparison and boolean operator nodes are allowed. Anything
# else (e.g. arithmetic ``ast.BinOp``) is rejected as out-of-scope for
# transition guards.
_ALLOWED_CMP_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)
_ALLOWED_UNARY_OPS = (ast.Not,)


class GuardEvalError(ValueError):
    """Raised when a guard cannot be parsed or contains a forbidden node."""


def _safe_guard_eval(guard_str: str, context: Dict[str, Any]) -> bool:
    """Evaluate a transition guard expression against ``context``.

    Guards are author-controlled JSON strings, but are still treated as
    untrusted input. We parse the expression with :mod:`ast` and walk the
    tree with a strict allowlist. ``ast.Call``, ``ast.Lambda``, imports,
    attribute access on anything other than ``context``, and any node
    type not explicitly listed below all raise :class:`GuardEvalError`
    BEFORE any value is produced.

    Allowed:
      * literals (``ast.Constant``) — including ``True``/``False``/``None``
      * the bare name ``context``
      * ``context.attr`` (resolved as ``context.get("attr")`` so missing
        keys become ``None`` instead of ``AttributeError``)
      * ``context["attr"]`` (resolved as ``context.get("attr")``)
      * comparisons (Eq, NotEq, Lt, LtE, Gt, GtE, Is, IsNot, In, NotIn)
      * boolean ops (And, Or, Not)
    """
    if not isinstance(guard_str, str) or not guard_str.strip():
        raise GuardEvalError("empty guard")
    try:
        tree = ast.parse(guard_str, mode="eval")
    except SyntaxError as exc:
        raise GuardEvalError(f"guard parse failed: {exc}") from exc

    def _walk(node: ast.AST) -> Any:
        # ast.Expression wrapper from mode="eval".
        if isinstance(node, ast.Expression):
            return _walk(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id != "context":
                raise GuardEvalError(
                    f"name {node.id!r} not allowed in guard"
                )
            return context

        if isinstance(node, ast.Attribute):
            target = _walk(node.value)
            if target is not context:
                raise GuardEvalError(
                    "attribute access only allowed on `context`"
                )
            return context.get(node.attr)

        if isinstance(node, ast.Subscript):
            target = _walk(node.value)
            if target is not context:
                raise GuardEvalError(
                    "subscript access only allowed on `context`"
                )
            key = _walk(node.slice)
            if isinstance(context, dict):
                return context.get(key)
            return None

        if isinstance(node, ast.Compare):
            left = _walk(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if not isinstance(op, _ALLOWED_CMP_OPS):
                    raise GuardEvalError(
                        f"comparison op {type(op).__name__} not allowed"
                    )
                right = _walk(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Is):
                    ok = left is right
                elif isinstance(op, ast.IsNot):
                    ok = left is not right
                elif isinstance(op, ast.In):
                    ok = left in right
                elif isinstance(op, ast.NotIn):
                    ok = left not in right
                else:  # pragma: no cover — guarded above
                    raise GuardEvalError(
                        f"comparison op {type(op).__name__} not allowed"
                    )
                if not ok:
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _ALLOWED_BOOL_OPS):
                raise GuardEvalError(
                    f"boolean op {type(node.op).__name__} not allowed"
                )
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = _walk(v)
                    if not result:
                        return result
                return result
            # Or
            result = False
            for v in node.values:
                result = _walk(v)
                if result:
                    return result
            return result

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise GuardEvalError(
                    f"unary op {type(node.op).__name__} not allowed"
                )
            return not _walk(node.operand)

        raise GuardEvalError(f"node {type(node).__name__} not allowed in guard")

    return bool(_walk(tree))


def validate_transition(
    rule_id: str,
    state: str,
    event: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a proposed state transition for a workflow entry.

    ``event`` must include ``to`` (the target state). For each transition
    in the entry matching ``from=state`` and ``to=event["to"]``, the
    guard is evaluated with :func:`_safe_guard_eval`. The first guard
    that passes makes the transition allowed.

    ``missing_documents`` and ``missing_approvals`` are present-but-best
    -effort: they list the entry's declared ``required_documents`` and
    ``approval_roles`` minus whatever the caller flagged as supplied via
    ``context["supplied_documents"]`` / ``context["supplied_approvals"]``.
    The presence of the keys is the contract; richer document inference
    is intentionally deferred.
    """
    entry = get_rule(rule_id)
    if entry is None:
        raise KeyError(f"unknown rule_id: {rule_id}")
    if entry.get("type") != "workflow":
        raise ValueError("validate_transition only supports type=workflow")

    target = event.get("to") if isinstance(event, dict) else None
    if not target:
        raise ValueError("event must include a 'to' field")

    allowed = False
    guard_used: Optional[str] = None
    last_guard_error: Optional[str] = None
    for tr in entry.get("transitions", []):
        if tr.get("from") != state or tr.get("to") != target:
            continue
        guard_str = tr.get("guard", "True")
        guard_used = guard_str
        try:
            if _safe_guard_eval(guard_str, context):
                allowed = True
                break
        except GuardEvalError as exc:
            last_guard_error = str(exc)
            continue

    supplied_docs = set(context.get("supplied_documents") or [])
    supplied_approvals = set(context.get("supplied_approvals") or [])
    missing_documents = [
        d for d in (entry.get("required_documents") or []) if d not in supplied_docs
    ]
    missing_approvals = [
        r for r in (entry.get("approval_roles") or []) if r not in supplied_approvals
    ]

    warnings = _build_warnings(entry)
    if last_guard_error and not allowed:
        warnings.append(f"guard parse failed: {last_guard_error}")

    return {
        "rule_id": rule_id,
        "allowed": allowed,
        "guard": guard_used,
        "missing_documents": missing_documents,
        "missing_approvals": missing_approvals,
        "provenance": entry.get("provenance", {}),
        "credibility_tier": entry.get("credibility_tier"),
        "warnings": warnings,
    }
