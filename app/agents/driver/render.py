"""What the model and the user read of a calculation: prose, never raw results.

One renderer for every registry formula. It reads the formula's declared
display name, its inputs' display labels and units, and its outputs' units,
so a formula added tomorrow renders the same way with no new code. Parameter
and formula identifiers (``contract_amount``, ``delay_damages_daily``) never
appear in its output.

The exit check uses the same declarations the other way round: an answer
that still writes an identifier the registry knows is rewritten to the name a
user reads (``readable``) before it leaves.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.lib.source_labels import CALCULATOR_SUFFIX, input_phrase

_LOG = logging.getLogger(__name__)
_MAX_LIST = 8
_NOT_OUTPUTS = frozenset({"note", "notes", "standard", "defaults_used", "needs_input", "not_computed"})
#: The keys a result repeats its primary figure under, beside primary_result.
_PRIMARY_MIRROR = frozenset({"primary_result", "value", "unit", "label"})


def _label_without_unit(key: str, unit: str) -> str:
    """A key in words, without trailing words that only repeat its unit
    ("bank_volume_m3", "m3" -> "bank volume")."""
    words = input_phrase(key, "").strip().split()
    flat = re.sub(r"[^a-z0-9]", "", str(unit or "").lower())
    while len(words) > 1 and flat and flat.endswith(words[-1].lower()):
        words.pop()
    # A currency amount's key ends in its currency code ("price_sar"): that
    # word repeats the unit, whatever the code.
    tokens = {t for t in re.split(r"[^a-z0-9]+", str(unit or "").lower()) if t}
    if len(words) > 1 and (flat.startswith("currency") and re.fullmatch(r"[a-z]{3}", words[-1].lower())
                           or words[-1].lower() in tokens):
        words.pop()
        while len(words) > 1 and flat.endswith(words[-1].lower()):
            words.pop()
    # "selling price per m3" less its unit leaves a dangling "per".
    if len(words) > 1 and words[-1].lower() == "per" and len(words) < len(input_phrase(key, "").split()):
        words.pop()
    return " ".join(words)


def headline(envelope: Dict[str, Any]) -> str:
    """The one sentence a calculation's answer leads with: its primary figure,
    labelled, with its unit. Empty when the result names no primary figure."""
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    spec0 = _spec(str(envelope.get("calculation") or ""))
    main = getattr(spec0, "main_output", "") if spec0 else ""
    primary = result.get("primary_result")
    if main and isinstance(result.get(main), (int, float)) and not isinstance(result.get(main), bool):
        # The formula declares the output its answer leads with.
        unit = currency_unit(main, dict(spec0.outputs).get(main, ""), result.get("currency"))
        primary = {"value": result[main], "unit": unit, "label": _label_without_unit(main, unit)}
    if not isinstance(primary, dict) or primary.get("value") in (None, ""):
        # No primary figure named: the formula's first declared output the
        # result carries as a number.
        from app.lib import formula_registry

        spec = _spec(str(envelope.get("calculation") or ""))
        declared = dict(spec.outputs) if spec else {}
        inputs = set(formula_registry.parameters(spec)) if spec else set()
        figures = [k for k in declared if isinstance(result.get(k), (int, float))
                   and not isinstance(result.get(k), bool)]
        # A declared output that is also an input is the input echoed back,
        # not what the formula found.
        key = next((k for k in figures if k not in inputs), figures[0] if figures else None)
        if key is None:
            return ""
        unit = currency_unit(key, declared[key], result.get("currency"))
        primary = {"value": result[key], "unit": unit, "label": _label_without_unit(key, unit)}
    label = str(primary.get("label") or "").strip() or "result"
    unit = str(primary.get("unit") or "").strip()
    # A unit is a compact token (m3, degC, N/mm2); a declared "unit" that is a
    # description ("same unit as specified") is not printed.
    unit = "" if unit in ("-", "") or re.search(r"\s", unit) else " " + unit
    value = primary["value"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        shown = f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) >= 1000 else f"{value:g}"
    else:
        shown = str(value)
    return f"Headline: {label[:1].upper() + label[1:]} is {shown}{unit}."


def _spec(calculation: str):
    from app.lib import formula_registry
    return formula_registry.get(calculation or "")


def _value(value: Any) -> Optional[str]:
    if isinstance(value, (dict, list, tuple, set)):
        return None
    if value is None or value == "":
        return None
    return input_phrase("", value).strip()


def _shown_unit(unit: str) -> str:
    """A declared unit as it is printed beside a figure. A unit written as
    words that is not a unit ("same unit as specified") is a note about the
    figure, not its unit, and is not printed."""
    unit = str(unit or "").strip()
    if not unit or " " not in unit:
        return unit
    try:
        from app.agents.base.formulas.construction_formulas_planning import parse_unit

        return unit if parse_unit(unit) is not None else ""
    except Exception:  # noqa: BLE001 -- rendering never fails on a unit
        _LOG.warning("unit parser failed on declared unit %r; not printed", unit, exc_info=True)
        return ""


def _phrases(values: Dict[str, Any], labels: Dict[str, str], units: Dict[str, str]) -> List[str]:
    out = []
    for key, value in values.items():
        if _value(value) is None:
            continue
        unit = _shown_unit(units.get(key, ""))
        phrase = input_phrase(key, value, unit)
        label = labels.get(key)
        if label:  # the declared label replaces the words built from the key
            head = input_phrase(key, "", unit).strip()
            phrase = label + phrase[len(head):] if phrase.startswith(head) else f"{label} {phrase}"
        out.append(phrase)
    return out


def render_calculation(envelope: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None) -> str:
    """A calculator envelope (``run_calculation``'s result) as prose."""
    from app.lib import formula_registry

    calc = str(envelope.get("calculation") or "")
    spec = _spec(calc)
    name = spec.display_name if spec and spec.display_name else "Calculation"
    head = f"{name} — {CALCULATOR_SUFFIX}"
    question = envelope.get("question")
    if envelope.get("status") != "success":
        if question:
            return f"{head}: needs input. Ask the user exactly this: {question}"
        return f"{head}: did not run. Ask the user for the inputs it needs, by their labels."
    params = formula_registry.parameters(spec) if spec else {}
    labels = {k: p.label for k, p in params.items() if p.label}
    in_units = {k: p.unit for k, p in params.items()}
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    used = {k: v for k, v in (inputs or {}).items() if k in params}
    lines = [head + "."]
    lead = headline(envelope)
    if lead:
        lines.append(lead)
    if used:
        lines.append("Inputs: " + "; ".join(_phrases(used, labels, in_units)) + ".")
    lines += result_lines(envelope, inputs)
    for key in ("standard", "note"):
        if isinstance(result.get(key), str) and result[key].strip():
            lines.append(result[key].strip())
    notes = result.get("notes")
    if isinstance(notes, list):
        lines += [str(n) for n in notes[:_MAX_LIST] if str(n).strip()]
    return "\n".join(lines)


def _outputs(envelope: Dict[str, Any], inputs: Optional[Dict[str, Any]]):
    from app.lib import formula_registry

    spec = _spec(str(envelope.get("calculation") or ""))
    params = formula_registry.parameters(spec) if spec else {}
    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    used = {k for k in (inputs or {}) if k in params}
    outputs = {k: v for k, v in result.items() if k not in used and k not in _NOT_OUTPUTS}
    if isinstance(result.get("primary_result"), dict):
        # value / unit / label beside primary_result repeat it; they are not
        # results of their own.
        outputs = {k: v for k, v in outputs.items() if k not in _PRIMARY_MIRROR}
    labels = {k: p.label for k, p in params.items() if p.label}
    units = {k: currency_unit(k, u, result.get("currency")) for k, u in (dict(spec.outputs) if spec else {}).items()}
    return outputs, labels, units


def currency_unit(key: str, unit: str, currency: Any = None) -> str:
    """A declared money unit in the currency the output's own key names:
    "currency/m3" on a key like selling_sar_m3 reads "SAR/m3". The code comes from
    the platform's currency table, never a fixed string. A key that names no
    currency takes the result's own ``currency`` output when that is a known
    code; otherwise it keeps "currency"."""
    text = str(unit or "")
    if not text.lower().startswith("currency"):
        return text
    from app.lib.construction_formulas import _CURRENCY_CODES

    codes = {c.lower(): c for c in _CURRENCY_CODES}
    named = [codes[part] for part in re.split(r"[^a-z]+", str(key).lower()) if part in codes]
    if len(set(named)) == 1:
        return named[0] + text[len("currency"):]
    given = codes.get(str(currency or "").strip().lower())
    if not named and given:
        return given + text[len("currency"):]
    return text


def _items(value: Any) -> List[str]:
    """A list output's rows, each as written (rows of plain values only)."""
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value[:_MAX_LIST]
            if not isinstance(v, (dict, list, tuple, set)) and str(v).strip()]


def _title(key: str, labels: Dict[str, str]) -> str:
    return labels.get(key) or input_phrase(key, "").strip().capitalize() or str(key)


def _mapping_rows(value: Any) -> List[str]:
    """A mapping output as labelled rows, one per entry with a plain value."""
    if not isinstance(value, dict):
        return []
    rows = []
    for key, item in list(value.items())[:_MAX_LIST]:
        shown = _value(item)
        if shown is not None:
            rows.append(f"- {_title(str(key), {})}: {shown}")
    return rows + _more(len(value))


def _record_table(value: Any) -> List[str]:
    """A list of records as one table: the same columns, in the same order,
    on every row (a field a record lacks is left blank)."""
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, dict) for v in value):
        return []
    columns: List[str] = []
    for record in value[:_MAX_LIST]:
        for key, item in record.items():
            if key not in columns and _value(item) is not None:
                columns.append(key)
    if not columns:
        return []
    out = ["| " + " | ".join(_title(c, {}) for c in columns) + " |",
           "|" + "|".join(" --- " for _ in columns) + "|"]
    for record in value[:_MAX_LIST]:
        out.append("| " + " | ".join(_value(record.get(c)) or "" for c in columns) + " |")
    return out + _more(len(value))


def _more(count: int) -> List[str]:
    """What a capped list left out, said rather than dropped."""
    return [f"- and {count - _MAX_LIST} more"] if count > _MAX_LIST else []


def _defaults_line(envelope: Dict[str, Any]) -> Optional[str]:
    from app.lib import formula_registry

    result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    rows = result.get("defaults_used") or envelope.get("defaults_used") or []
    if isinstance(rows, dict):
        rows = [{"parameter": k, "value": v} for k, v in rows.items()]
    spec = _spec(str(envelope.get("calculation") or ""))
    declared = formula_registry.parameters(spec) if spec else {}
    parts = []
    for row in rows if isinstance(rows, list) else []:
        # Only an input the formula declares to its users: a switch or an
        # internal option the formula keeps to itself is not "assumed".
        if not isinstance(row, dict) or str(row.get("parameter")) not in declared:
            continue
        if isinstance(row.get("value"), bool):
            continue
        shown = _value(row.get("value"))
        if shown is not None:
            label = row.get("label") or input_phrase(str(row.get("parameter") or ""), "").strip()
            parts.append(f"{label} {shown}")
    return ("Assumed, not given: " + "; ".join(parts) + ".") if parts else None


def result_lines(envelope: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None) -> List[str]:
    """A successful calculation's result: its figures; then each list output
    as rows, each mapping output as labelled rows, each list of records as a
    table; then any optional input the run took from a default."""
    outputs, labels, units = _outputs(envelope, inputs)
    lines = []
    shown = _phrases(outputs, labels, units)
    if shown:
        lines.append("Result: " + "; ".join(shown) + ".")
    for key, value in outputs.items():
        rows = _items(value) and ([f"- {row}" for row in _items(value)] + _more(len(value)))
        rows = rows or _mapping_rows(value) or _record_table(value)
        if rows:
            lines.append(f"{_title(key, labels)}:")
            lines += rows
    defaults = _defaults_line(envelope)
    if defaults:
        lines.append(defaults)
    return lines


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")


def _numbers(text: str) -> List[float]:
    out = []
    for raw in _NUMBER.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def carried(envelope: Dict[str, Any], inputs: Optional[Dict[str, Any]], answer: str) -> bool:
    """Does the answer carry this calculation's result: every figure it
    produced (to the rounding a writer would use) and every row it listed?"""
    outputs, _labels, _units = _outputs(envelope, inputs)
    written = _numbers(answer)
    text = (answer or "").lower()
    for value in outputs.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not any(abs(w - value) <= max(abs(value) * 0.005, 0.005) for w in written):
                return False
        elif any(row.lower() not in text for row in _items(value)):
            return False
        elif not _figures_written(value, written, text):
            return False
    return True


def record_lists(result: Any) -> List[list]:
    """Every list of records (rows of the same kind, each a mapping) directly
    inside a tool's result."""
    if not isinstance(result, dict):
        return []
    return [v for v in result.values() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]


def _row_figures(record: Dict[str, Any]) -> List[float]:
    return [float(v) for v in record.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 0.005, 0.005)


def rows_presented(rows: list, text: str) -> bool:
    """The answer shows this list: a figure of one of its rows is written."""
    written = _numbers(text)
    return any(any(_near(w, f) for w in written for f in _row_figures(r) if f) for r in rows[:_MAX_LIST])


def rows_carried(rows: list, text: str) -> bool:
    """Each shown row is a row of the answer: one line carries at least two of
    its figures (or its one figure). A sideways table puts one field of many
    rows on a line, and bullets put one figure on a line -- neither passes."""
    lines = [_numbers(line) for line in (text or "").splitlines()]
    for record in rows[:_MAX_LIST]:
        figures = [f for f in _row_figures(record)]
        need = min(2, len(figures))
        if need == 0:
            continue
        if not any(_matched(figures, line) >= need for line in lines):
            return False
    return True


def _matched(figures: List[float], written: List[float]) -> int:
    """How many of ``figures`` a line writes, each written number counted for
    one figure only (a month whose billing and cumulative are equal is not
    two figures on a line that writes the number once)."""
    left = list(written)
    count = 0
    for f in figures:
        for i, w in enumerate(left):
            if _near(w, f):
                del left[i]
                count += 1
                break
    return count


def record_table(rows: list) -> List[str]:
    """``rows`` as one table, one row per record (the renderer's own form)."""
    return _record_table(rows)


def _figures_written(value: Any, written: List[float], text: str) -> bool:
    """The rows a mapping or a list of records renders (the same capped rows
    the renderer shows) are all in the answer: every figure and name, and for
    a mapping every row's label as well -- a table with the right total and
    invented row names does not carry the result (LANE2)."""
    if isinstance(value, dict):
        rows = [(key, item) for key, item in list(value.items())[:_MAX_LIST] if _value(item) is not None]
        for key, _item in rows:
            if not _label_written(str(key), text):
                return False
        records = [dict(rows)]
    elif isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
        records = list(value[:_MAX_LIST])
    else:
        records = []
    for record in records:
        for item in record.values():
            if isinstance(item, bool) or isinstance(item, (dict, list, tuple, set)):
                continue
            if isinstance(item, (int, float)):
                if not any(abs(w - item) <= max(abs(item) * 0.005, 0.005) for w in written):
                    return False
            elif isinstance(item, str) and item.strip() and item.strip().lower() not in text:
                return False
    return True


def _label_written(key: str, text: str) -> bool:
    """A mapping row's label is in the answer: as the renderer titles it, or
    as the key itself written in words."""
    forms = {_title(key, {}).lower(), key.lower(), key.replace("_", " ").lower()}
    return any(form and form in text for form in forms)


def _identifier_map() -> Dict[str, str]:
    """Every identifier-shaped registry key -> the words a user reads.
    Identifier-shaped = contains an underscore: plain words a key happens to
    share ("span", "load") are never touched."""
    from app.agents.core import tool_registry
    from app.lib import formula_registry

    out: Dict[str, str] = {}
    for spec in formula_registry.all_specs():
        if "_" in spec.name:
            out[spec.name] = spec.display_name or spec.name.replace("_", " ")
        for key, p in formula_registry.parameters(spec).items():
            if "_" in key and key not in out:
                out[key] = p.label or input_phrase(key, "", p.unit).strip() or key.replace("_", " ")
    tool_registry.load()
    for name, t in tool_registry._TOOLS.items():
        if "_" in name:
            out[name] = t.display_name or name.replace("_", " ")
    return out


_MAP: Dict[int, tuple] = {}


def _compiled():
    from app.lib import formula_registry

    key = len(formula_registry.all_specs())
    hit = _MAP.get(key)
    if hit is None:
        mapping = _identifier_map()
        names = sorted(mapping, key=len, reverse=True)
        rx = re.compile(r"(?<![A-Za-z0-9_])`?(" + "|".join(map(re.escape, names)) + r")`?(?![A-Za-z0-9_])") if names else None
        hit = (rx, mapping)
        _MAP.clear()
        _MAP[key] = hit
    return hit


def readable(text: str) -> tuple:
    """``text`` with every registry identifier replaced by its display name or
    label, and how many were replaced."""
    rx, mapping = _compiled()
    if not text or rx is None:
        return text, 0
    count = 0

    def sub(m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return mapping[m.group(1)]
    return rx.sub(sub, text), count
