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
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

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
    # The entry's declared variables are symbols, whatever their name: without
    # ``locals`` sympify reads a name sympy reserves (S, Q, E, I, N, O) as its
    # own object, so the stated value is never substituted.
    declared = {name: sympy.Symbol(name) for name in (entry.get("variables") or {})}
    expr = sympy.sympify(expr_str, locals=declared, evaluate=False)
    subs = {sympy.Symbol(name): value for name, value in values.items()}
    result = expr.subs(subs).evalf()
    try:
        result_f = float(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"formula did not reduce to a scalar — missing variables? got {result!r}"
        ) from exc

    out = {
        "rule_id": rule_id,
        "result": result_f,
        "unit": (entry.get("thresholds") or {}).get("unit"),
        "provenance": entry.get("provenance", {}),
        "credibility_tier": entry.get("credibility_tier"),
        "warnings": _build_warnings(entry),
    }
    from_course = course_inputs(entry, values)
    if from_course:
        out["course_inputs"] = from_course
    # A count rounded UP (machines, layers, floors) shows the figure it was
    # rounded from, so a reader sees 2.03 -> 3 and not only 3. Read from the
    # expression's own structure: one ceiling() node, its argument evaluated.
    ceilings = [n for n in sympy.preorder_traversal(expr) if isinstance(n, sympy.ceiling)]
    if len(ceilings) == 1:
        out["unrounded"] = float(ceilings[0].args[0].subs(subs).evalf())
    return out


def lookup_table(rule_id: str, **match: Any) -> Dict[str, Any]:
    """Return the rows of a ``type=table`` entry whose cells equal every given value.

    The columns come from the entry's own ``table.columns``; a filter on a
    column the table does not declare is refused, not silently matched to
    nothing. A cell held for the owner's reading is stored as null and named
    in ``table.held``, which is returned with the rows so an answer can say
    the cell is not read rather than present a gap as a value.
    """
    entry = get_rule(rule_id)
    if entry is None:
        raise KeyError(f"unknown rule_id: {rule_id}")
    kind = entry.get("type")
    if kind != "table":
        raise ValueError(f"lookup_table() only supports type=table (got {kind!r})")
    table = entry.get("table") or {}
    columns = table.get("columns") or []
    names = {c.get("name") for c in columns}
    unknown = sorted(set(match) - names)
    if unknown:
        raise ValueError(f"table {rule_id!r} has no column(s) {unknown}")
    rows = [
        row
        for row in table.get("rows") or []
        if all(row.get(name) == value for name, value in match.items())
    ]
    return {
        "rule_id": rule_id,
        "columns": columns,
        "rows": rows,
        "held": table.get("held") or [],
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


# ---------------------------------------------------------------------------
# A formula entry the ask names AND supplies, chosen without a model.
# Same contract as construction_formulas.formula_completed_by_user_text for
# the registry: the ask is a calculation request, every content word of the
# entry's title is in it, and every declared variable binds to exactly one
# figure. A tie, a missing or an ambiguous input is no selection (None): the
# turn is never guessed.
# ---------------------------------------------------------------------------
_FIGURE_RE = re.compile(
    r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(%|[^\W\d_][\w³²]*(?:\s+[^\W\d_]+)?)?"
)
_POWER_SUFFIX_RE = re.compile(r"^([^\W\d_]+)([23])$")


def _unit_registry():
    import pint
    global _UREG
    try:
        return _UREG
    except NameError:
        _UREG = pint.UnitRegistry()
        return _UREG


def _parse_unit(text: str):
    """pint units for ``text`` or None. ``m3`` reads as ``m**3``."""
    raw = (text or "").strip()
    if not raw:
        return None
    m = _POWER_SUFFIX_RE.match(raw)
    if m:
        raw = f"{m.group(1)}**{m.group(2)}"
    try:
        return _unit_registry().parse_units(raw)
    except Exception:  # noqa: BLE001 -- not a unit: the figure has none
        _LOG.debug("not a unit: %r", raw, exc_info=True)
        return None


def _figures(text: str) -> List[Dict[str, Any]]:
    """Every figure in ``text`` with its span and the unit written after it.

    A two-word unit (``cubic meter``) is tried before its first word, so
    the longest reading wins; a word that is not a unit leaves the figure
    without one (it can then bind only by name).
    """
    out: List[Dict[str, Any]] = []
    for m in _FIGURE_RE.finditer(text or ""):
        value = float(m.group(1).replace(",", ""))
        unit = None
        tail = m.group(2) or ""
        words = tail.split()
        for n in range(len(words), 0, -1):
            unit = _parse_unit(" ".join(words[:n]))
            if unit is not None:
                break
        out.append({"value": value, "unit": unit, "start": m.start(1)})
    return out


def _bind_by_name(text: str, name: str, figures: List[Dict[str, Any]]):
    """The figure written as ``name = n`` / ``name: n`` / ``name of n``."""
    flags = 0 if len(name) < 3 else re.IGNORECASE
    pat = re.compile(rf"(?<![\w.]){re.escape(name)}\s*(?:=|:|\bof\b|\bis\b)\s*", flags)
    hits = []
    for m in pat.finditer(text):
        hits += [f for f in figures if f["start"] == m.end()]
    return hits[0] if len(hits) == 1 else None


def bind_entry_variables(entry: Dict[str, Any], text: str) -> Optional[Dict[str, float]]:
    """Every declared variable of ``entry`` bound from ``text``, or None.

    By name first. A variable left over binds by its declared unit only
    when it is the one left-over variable of that unit and exactly one
    unused figure carries that unit. No unit conversion: a figure in a
    different unit is not this variable's figure. A variable with no
    figure of its unit takes the value its course declares, if one
    applies (:func:`_course_value`) and every figure in ``text`` is
    already bound to another variable.
    """
    variables = entry.get("variables") or {}
    if not isinstance(variables, dict) or not variables:
        return None
    figures = _figures(text)
    bound: Dict[str, float] = {}
    used: set = set()
    for name in variables:
        fig = _bind_by_name(text, name, figures)
        if fig is not None and id(fig) not in used:
            bound[name] = fig["value"]
            used.add(id(fig))
    left = [n for n in variables if n not in bound]
    from_course: Dict[str, float] = {}
    declared = {n: _parse_unit(str((variables[n] or {}).get("unit") or "")) for n in left}
    for name in left:
        unit = declared[name]
        if unit is None or sum(1 for n in left if declared[n] == unit) != 1:
            return None
        cands = [f for f in figures if id(f) not in used and f["unit"] == unit]
        if not cands:
            course = _course_value(variables[name], text)
            if course is None:
                return None
            from_course[name] = course
            continue
        if len(cands) != 1:
            return None
        bound[name] = cands[0]["value"]
        used.add(id(cands[0]))
    # The course fills an input only when every typed figure is already an
    # input: a figure left over may be the user's own value for it, written
    # in a form not read here, and a typed value wins over the course's.
    if from_course and len(used) != len(figures):
        return None
    bound.update(from_course)
    return bound


def _course_value(spec: Any, text: str) -> Optional[float]:
    """The value the course declares for an input the ask gives no figure for.

    ``course_values`` lists the course's values for the input; one with
    ``when`` words applies when every word is in the ask (the layer, the
    case it is stated for), one without applies always. Exactly one value
    must apply, or the input stays unbound. A typed figure never reaches
    here: the course fills only what the user left out.
    """
    from app.lib.construction_formulas import _words_all_present
    declared = (spec or {}).get("course_values") if isinstance(spec, dict) else None
    if not isinstance(declared, list):
        return None
    applies = [
        c for c in declared
        if isinstance(c, dict) and isinstance(c.get("value"), (int, float))
        and _words_all_present(text, list(c.get("when") or []))
    ]
    return float(applies[0]["value"]) if len(applies) == 1 else None


def course_inputs(entry: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    """Inputs of ``entry`` whose value is one the course declares, named
    with the course reference, so an answer says where the figure came
    from."""
    ref = (entry.get("provenance") or {}).get("course_reference")
    out: Dict[str, Any] = {}
    for name, spec in (entry.get("variables") or {}).items():
        declared = spec.get("course_values") if isinstance(spec, dict) else None
        if name not in values or not isinstance(declared, list):
            continue
        for c in declared:
            if isinstance(c, dict) and c.get("value") == values[name]:
                out[name] = {"value": values[name], "when": c.get("when") or [], "source": ref}
                break
    return out


# A formula title names its result first, then how it is worked: "X from
# A and B". A bracketed aside, anywhere, is another name or the inputs:
# "X (A vs B)", "P (also Q) X". The words left before "from" are
# the title's head.
_TITLE_ASIDE_RE = re.compile(r"\s*\([^)]*\)")
_TITLE_QUALIFIER_RE = re.compile(r"\s+from\s+", re.IGNORECASE)
# A counting head word names no thing of its own: "number of X" is asked
# as "how many X". The phrase the ask uses instead of the head word.
_COUNT_HEAD_ASKED_AS = {"number": "how many", "amount": "how much"}


def _title_head_words(title: str) -> List[str]:
    from app.lib.construction_formulas import _display_content_words
    head = _TITLE_QUALIFIER_RE.split(_TITLE_ASIDE_RE.sub("", title or ""), maxsplit=1)[0]
    return _display_content_words(head)


def _head_named(raw: str, head: List[str]) -> bool:
    """Every head word is in the ask; a leading counting word may be
    asked by its question phrase instead."""
    from app.lib.construction_formulas import _words_all_present
    words = list(head)
    if words and words[0] in _COUNT_HEAD_ASKED_AS:
        asked = _COUNT_HEAD_ASKED_AS[words[0]]
        if re.search(rf"\b{re.escape(asked)}\b", raw.lower()):
            words = words[1:]
    return bool(words) and _words_all_present(raw, words)


def formula_completed_by_user_text(text: str) -> Optional[tuple]:
    """``(rule_id, values)`` for the one formula entry the ask names and
    supplies, or None.

    The ask names an entry when it carries the head of the entry's title
    (its words before "from", bracketed asides left out). Among named entries whose inputs
    all bind, the one with the most title words in the ask wins, then the
    one with fewest title words missing; any tie is no selection.
    """
    raw = text or ""
    from app.lib.construction_formulas import (
        _display_content_words, _words_all_present, explicit_calculation_request,
    )
    if not raw.strip() or not explicit_calculation_request(raw):
        return None
    candidates = []
    for entry in load_knowledge():
        if entry.get("type") != "formula":
            continue
        title = str(entry.get("title") or "")
        if not _head_named(raw, _title_head_words(title)):
            continue
        values = bind_entry_variables(entry, raw)
        if values is None:
            continue
        words = _display_content_words(title)
        present = sum(1 for w in words if _words_all_present(raw, [w]))
        candidates.append((entry["id"], values, (present, present - len(words))))
    if not candidates:
        return None
    best = max(c[2] for c in candidates)
    tied = [c for c in candidates if c[2] == best]
    if len(tied) != 1:
        return None
    return tied[0][0], tied[0][1]
