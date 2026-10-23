"""What a user reads when an answer credits a calculation or a tool.

Every formula and tool declares a ``display_name`` where it is registered;
this module is the one place that turns a credit into words. Internal names
(``construction_calc``, ``delay_damages_daily``) are identifiers for code and
logs and never reach an answer. Input names are read as words, and an input
whose declared unit the name already spells ("length_m", unit "m") is shown
as "length 12 m".

The same reading applies to a finished answer. A registered id, or a
``key=value`` assignment of one, is shown as its display name. On a turn
whose inputs the user already supplied, a currency, unit, clause or
comparison is kept only when the request, the retrieved excerpts or the
calculator result actually state it.
"""
from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_LOG = logging.getLogger(__name__)

#: Shown after a formula's display name, so the reader knows it was computed.
CALCULATOR_SUFFIX = "platform calculator"

#: Units that carry no reading of their own after a number.
_UNITLESS = frozenset({"", "-", "currency", "ratio", "count", "no", "nr", "fraction"})


def _norm(unit: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", (unit or "").lower())


#: An exponent written as a digit after a unit symbol ("m2", "kg/m3", "mm4");
#: not the digit of a chemical formula ("CO2", "kgCO2e").
_UNIT_EXPONENT_RE = re.compile(r"(?<=[A-Za-z])(?<![Cc][Oo])([234])(?![0-9A-Za-z])")
_SUPERSCRIPT = {"2": "²", "3": "³", "4": "⁴"}


def unit_words(unit: str) -> str:
    """A declared unit as a reader writes it after a figure: "currency/t" is
    "per t", "m2" is "m²", "kN/m3" is "kN/m³"."""
    raw = (unit or "").strip()
    if raw.lower().startswith("currency/"):
        raw = "per " + raw.split("/", 1)[1]
    return _UNIT_EXPONENT_RE.sub(lambda m: _SUPERSCRIPT[m.group(1)], raw)


def _unit_value(value: Any) -> Optional[str]:
    """A string input that is a unit ("ft2", "kN/m"), as a reader writes it."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z°µμ][\w°µμ/.^²³⁴ -]{0,15}", value.strip()):
        return None
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    return unit_words(value) if parse_unit(value) is not None else None


def _is_fraction(unit: str) -> bool:
    return (unit or "").strip().lower() == "fraction"


def readable_value(value: Any) -> str:
    """A code-style value as words: "simply_supported" is "simply supported"."""
    if isinstance(value, str):
        return re.sub(r"[_\s]+", " ", value).strip()
    return _fmt_value(value)


def formula_display_name(calculation: str) -> Optional[str]:
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    return spec.display_name if spec and spec.display_name else None


def tool_display_name(tool: str) -> Optional[str]:
    from app.agents.core import tool_registry

    spec = tool_registry.get(tool or "")
    return spec.display_name if spec and spec.display_name else None


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def parameter_words(key: str, unit: str = "") -> str:
    """``length_m`` with unit "m" -> "length", ``bearing_kn_m2`` with unit
    "kN/m2" -> "bearing": the name without the unit its last words spell."""
    words = [w for w in str(key).split("_") if w]
    norm = _norm(unit)
    percent = norm == "%" or _is_fraction(unit)
    if len(words) > 1 and percent and words[-1].lower() in ("pct", "percent"):
        return " ".join(words[:-1])
    for start in range(len(words) - 1, 0, -1):
        if norm and _norm("".join(words[start:])) == norm:
            return " ".join(words[:start])
    return " ".join(words)


def input_phrase(key: str, value: Any, unit: str = "") -> str:
    """``contract_amount, 50000000`` -> "contract amount 50,000,000";
    ``length_m, 12, "m"`` -> "length 12 m"; ``rate_percent, 0.1, "%"`` -> "rate 0.1%";
    ``waste_pct, 0.05, "fraction"`` -> "waste 5%"."""
    words = parameter_words(key, unit).split()
    norm = _norm(unit)
    shown = _unit_value(value) or _fmt_value(value)
    if norm == "%":
        shown += "%"
    elif _is_fraction(unit) and isinstance(value, (int, float)) and not isinstance(value, bool):
        shown = f"{_fmt_value(_rounded(value * 100))}%"
    elif norm not in _UNITLESS and not isinstance(value, (bool, str)):
        shown += f" {unit_words(unit)}"
    return f"{' '.join(words)} {shown}".strip()


def parameter_label(calculation: str, key: str) -> str:
    """The words a user reads for one input: its declared label, else its name as words."""
    from app.lib import formula_registry

    from app.agents.base.formulas.construction_formulas_planning import CONVERSION_CONTEXT

    spec = formula_registry.get(calculation or "")
    declared = spec.params.get(key) if spec else None
    if declared is not None and declared.label:
        return declared.label
    if declared is None and key in CONVERSION_CONTEXT:
        return CONVERSION_CONTEXT[key]
    return parameter_words(key, declared.unit if declared is not None else "")


def _is_fraction_percent(row: Dict[str, Any]) -> bool:
    hi = row.get("hi")
    unit = row.get("unit") or ""
    return _is_fraction(unit) or (_norm(unit) == "%" and hi is not None and hi <= 1)


def _shown_quantity(value: Any, row: Dict[str, Any]) -> str:
    unit = row.get("unit") or ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_shown_quantity(v, row) for v in value)
    if _is_fraction_percent(row) and isinstance(value, (int, float)) and not isinstance(value, bool):
        # A fraction input's value is a fraction by the time it is checked
        # (150% arrives as 1.5) and reads as a percentage; a figure above 100
        # can only be a percentage the caller wrote unscaled (150).
        return f"{_fmt_value(value * 100 if abs(value) <= 100 else value)}%"
    if isinstance(value, str):
        return f"\u201c{value}\u201d"
    shown = _fmt_value(value)
    if _norm(unit) == "%":
        return shown + "%"
    if _norm(unit) in _UNITLESS:
        return shown
    return f"{shown} {unit_words(unit)}"


# Unitless kinds of input a range still names, so "0 to 1" reads as a ratio.
_NAMED_KINDS = frozenset({"ratio", "count", "fraction"})


def _accepted_range(row: Dict[str, Any]) -> str:
    shown = f"{_shown_quantity(row['lo'], row)} to {_shown_quantity(row['hi'], row)}"
    kind = _norm(row.get("unit") or "")
    if kind in _NAMED_KINDS and not _is_fraction_percent(row):
        shown += f", as a {kind}"
    return shown


def input_question(calculation: str, rejected: List[Dict[str, Any]],
                   missing: Iterable[str] = ()) -> str:
    """Ask the user again for inputs the calculator would not run, by their labels."""
    display = formula_display_name(calculation) or "calculation"
    sentences: List[str] = []
    labels: List[str] = []
    for row in rejected:
        label = parameter_label(calculation, row["parameter"])
        value = row.get("value")
        reason = row.get("reason")
        if reason == "needs_conversion_input":
            needs = [parameter_label(calculation, key) for key in row.get("needs") or ()]
            sentences.append(f"To use {value} as the {label}, I need the {_joined(needs)}.")
            labels.extend(n for n in needs if n not in labels)
            continue
        labels.append(label)
        if reason == "wrong_unit":
            expected = row.get("expected_dimension")
            wanted = unit_words(row.get("unit") or "")
            sentences.append(
                f"The {label} is {_with_article(_dimension_words(expected))} in {wanted}, and {value} is "
                f"{_with_article(_dimension_words(row.get('given_dimension')))}, so it does not convert."
                if expected else f"The {label} needs a figure in {wanted}, not {value}.")
        elif reason == "unknown_unit":
            sentences.append(f"I do not recognise the unit in {value}; the {label} is in "
                             f"{unit_words(row.get('unit') or '')}.")
        elif reason == "out_of_range":
            verb = "are" if row.get("kind") == "series" else "is"
            sentences.append(f"The {label} given ({_shown_quantity(value, row)}) {verb} outside what the "
                             f"{display} calculator accepts ({_accepted_range(row)}).")
            scale = row.get("scale")
            if scale:
                exponent = scale["exponent"]
                step = (f"multiply by 1e{exponent}" if exponent > 0 else f"divide by 1e{-exponent}")
                sentences.append(f"If that figure is in {scale['unit']}, it is "
                                 f"{_shown_quantity(_rounded(scale['value']), row)} ({step}).")
        elif reason == "not_whole":
            sentences.append(f"The {label} needs a whole number, not {_fmt_value(value)}.")
        elif reason == "empty":
            sentences.append(f"The {label} needs at least one number.")
        elif reason == "not_yes_no":
            sentences.append(f"The {label} needs a yes or a no, not \u201c{value}\u201d.")
        else:
            sentences.append(f"The {label} needs a number, not {_shown_quantity(value, row)}.")
    needed = [label for label in (parameter_label(calculation, key) for key in missing)
              if label not in labels]
    if needed:
        also = " also" if sentences else ""
        sentences.append(f"The {display} calculator{also} needs the {_joined(needed)}.")
    asked = labels + needed
    if len(asked) == 1:
        sentences.append(f"What {asked[0]} should I use?")
    elif asked:
        sentences.append(f"What values should I use for the {_joined(asked)}?")
    return " ".join(sentences)


def _rounded(number: float) -> float:
    return float(f"{number:.6g}")


def _dimension_words(dimension: Any) -> str:
    """"currency/mass" is "price per mass"; "force/length" is "force per length"."""
    parts = [("price" if part == "currency" else part) for part in str(dimension or "").split("/")]
    return " per ".join(p for p in parts if p) or "quantity"


def _with_article(words: str) -> str:
    return ("an " if words[:1].lower() in "aeiou" else "a ") + words


def _joined(words: List[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def calculator_label(calculation: str, inputs: Optional[Dict[str, Any]] = None) -> str:
    """"Delay damages per day -- platform calculator (rate 0.1%, contract amount 50,000,000)"."""
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    name = spec.display_name if spec and spec.display_name else "Calculation"
    units = spec.inputs if spec else {}
    shown = ", ".join(_credit_phrase(calculation, k, v, units.get(k, "")) for k, v in (inputs or {}).items()
                      if v not in (None, ""))
    return f"{name} — {CALCULATOR_SUFFIX}" + (f" ({shown})" if shown else "")


def _credit_phrase(calculation: str, key: str, value: Any, unit: str) -> str:
    """One input in a credit line: its declared label (else its name as
    words) and its value in its unit -- "core temperature 68 °C", never the
    key's own words when a label is declared."""
    value_part = input_phrase("", value, unit).strip()
    return f"{parameter_label(calculation, key)} {value_part}".strip()


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return False


def _user_states_value(user: str, value: Any) -> bool:
    """True when the user's own words already carry this input's value."""
    text = user or ""
    if isinstance(value, str):
        token = value.strip()
        if len(token) < 2:
            return False
        return re.search(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        ) is not None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    target = float(value)
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            found = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if abs(found - target) <= max(1e-9, abs(target) * 1e-9):
            return True
        window = text[max(0, match.start() - 1):match.end() + 1]
        if 0 < abs(target) < 1 and "%" in window and abs(found - target * 100.0) <= 1e-6:
            return True
    return False


def stated_calculator_inputs(
    calculation: str,
    inputs: Optional[Dict[str, Any]],
    user_text: str = "",
) -> Dict[str, Any]:
    """Inputs the user stated.

    A value equal to the formula's signature default, which the user did
    not write, is the calculator's own default and is not an input.
    """
    raw = dict(inputs or {})
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    if spec is None:
        return raw
    try:
        signature = inspect.signature(spec.fn)
    except (TypeError, ValueError):
        return raw
    stated: Dict[str, Any] = {}
    for key, val in raw.items():
        param = signature.parameters.get(key)
        if (
            param is not None
            and param.default is not inspect.Parameter.empty
            and _same_value(val, param.default)
            and not _user_states_value(user_text, val)
        ):
            continue
        stated[key] = val
    return stated


def _result_mentions(result: Any, value: Any) -> bool:
    """True when this result's figures or notes already show ``value``."""
    parts: List[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            parts.append(str(obj))

    walk(result)
    return _user_states_value("\n".join(parts), value)


def _call_supplied_an_operand(
    signature: inspect.Signature,
    stated_keys: Iterable[str],
    given: Dict[str, Any],
) -> bool:
    """True when the call carried a figure the user, not the signature, chose."""
    if stated_keys:
        return True
    for key, val in given.items():
        param = signature.parameters.get(key)
        if param is None:
            continue
        if param.default is inspect.Parameter.empty or not _same_value(val, param.default):
            return True
    return False


def calculator_default_lines(
    calculation: str,
    result: Any,
    user_text: str = "",
    stated_keys: Optional[Iterable[str]] = None,
    passed: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """One sentence per signature default this result used and the user did not state.

    A zero default is the absence of a figure. A currency the formula
    inserts only because the parameter was left blank is not restated.
    When the call carried no figure of the user's, a default the result
    shows under any field is still that default.
    """
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    if spec is None or not isinstance(result, dict):
        return []
    try:
        signature = inspect.signature(spec.fn)
    except (TypeError, ValueError):
        _LOG.debug("signature unreadable for %s", calculation, exc_info=True)
        return []
    known = set(stated_keys or ())
    given = dict(passed or {})
    supplied = _call_supplied_an_operand(signature, known, given)
    lines: List[str] = []
    for key, param in signature.parameters.items():
        if param.default is inspect.Parameter.empty or key in known:
            continue
        default = param.default
        if isinstance(default, bool):
            continue
        echoed = result.get(key) if key in result else given.get(key)
        shown = echoed is not None and _same_value(echoed, default)
        if not shown:
            if supplied or not _result_mentions(result, default):
                continue
            echoed = default
        if _user_states_value(user_text, default):
            continue
        if isinstance(default, (int, float)):
            if float(default) == 0.0:
                continue
        elif isinstance(default, str):
            token = default.strip()
            if not token or token.upper() in _CURRENCY_CODES:
                continue
        else:
            continue
        lines.append(default_sentence(calculation, key, default if isinstance(default, str) else echoed))
    return lines


def default_sentence(calculation: str, key: str, value: Any) -> str:
    """One sentence naming a default the calculator used, by the input's label
    and a value as a reader writes it: "The shape used, rectangular, is ..."."""
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    declared = formula_registry.parameters(spec).get(key) if spec else None
    row = {"unit": declared.unit if declared else (spec.inputs or {}).get(key, "") if spec else "",
           "hi": declared.hi if declared else None}
    shown = readable_value(value) if isinstance(value, str) else _shown_quantity(value, row)
    return (f"The {parameter_label(calculation, key)} used, {shown}, is the platform "
            "calculator's default, not a figure from the project documents.")


def calculator_currency_defaults(
    calculation: str,
    result: Any,
    user_text: str = "",
    passed: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """``(code, default line)`` for each currency the formula filled in itself.

    The parameter was left blank and the user's words name no currency, so
    the code is the signature default. ``calculator_default_lines`` does
    not restate it, because an answer that shows no currency rests on no
    currency; the caller states it when the answer does show it.
    """
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    if spec is None:
        return []
    try:
        signature = inspect.signature(spec.fn)
    except (TypeError, ValueError):
        _LOG.debug("signature unreadable for %s", calculation, exc_info=True)
        return []
    given = dict(passed or {})
    found: List[Tuple[str, str]] = []
    for key, param in signature.parameters.items():
        default = param.default
        if not isinstance(default, str):
            continue
        code = default.strip().upper()
        if code not in _CURRENCY_CODES:
            continue
        chosen = given.get(key)
        if chosen not in (None, "") and not _same_value(chosen, default):
            continue
        if any(_user_states_value(user_text, c) for c in _CURRENCY_CODES):
            continue
        found.append((code, default_sentence(calculation, key, code)))
    return found


def calculator_parameter_words(
    calculation: str,
    stated: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, bool]]:
    """``(parameter, words, defaulted)`` for each input the formula takes.

    ``words`` is the parameter read as a user would write it ("contract
    amount" for ``contract_amount``); ``defaulted`` is True when the user
    did not state it.
    """
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    if spec is None:
        return []
    try:
        signature = inspect.signature(spec.fn)
    except (TypeError, ValueError):
        _LOG.debug("signature unreadable for %s", calculation, exc_info=True)
        return []
    known = set((stated or {}).keys())
    units = spec.inputs or {}
    out: List[Tuple[str, str, bool]] = []
    for key, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        words = parameter_words(key, units.get(key, "")).lower()
        out.append((key, words, key not in known))
    return out


#: Stands where a removal took text out, until the line is tidied.
_REMOVED_MARK = "\x00"
#: Brackets around nothing but separators and space.
_EMPTY_BRACKETS_RE = re.compile(r"[ \t]*(?:\([\s,;:/–—-]*\)|\[[\s,;:/–—-]*\])")
#: A word that only makes sense attached to what follows it.
_HEAD_PREPOSITION = (
    r"(?:on|under|using|via|by|per|in|of|from|to|at|see|with|and|or)"
)


def tidy_removal_debris(line: str, mark: str = "\x00") -> str:
    """The line with the debris a removal left behind cleaned up.

    ``mark`` sits where text was removed. A list element the removal emptied,
    or cut down to a lone word, goes with its separator ("rate, Sub-Clause
    8.8 basis" -> "rate"); a preposition left pointing at nothing goes; then
    brackets left empty, separators left doubled or dangling inside a
    bracket or before a sentence end.
    """
    if not line:
        return line
    m = re.escape(mark)
    element_end = r"(?=[ \t]*(?:[),;\]]|$))"
    # Element reduced to the removed span, a preposition, and/or one word.
    line = re.sub(
        rf"[ \t]*[,;][ \t]*(?:{_HEAD_PREPOSITION}[ \t]+)?{m}(?:[ \t]*{m})*"
        rf"(?:[ \t]+[A-Za-z][\w-]*)?{element_end}",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        rf"(?<=[(\[])[ \t]*(?:{_HEAD_PREPOSITION}[ \t]+)?{m}(?:[ \t]*{m})*"
        rf"(?:[ \t]+[A-Za-z][\w-]*)?[ \t]*(?:[,;][ \t]*|(?=[)\]]))",
        "",
        line,
        flags=re.IGNORECASE,
    )
    # A preposition whose object was the removed span.
    line = re.sub(
        rf"[ \t]+{_HEAD_PREPOSITION}[ \t]*{m}(?=[ \t]*(?:[),;:.?!\]]|$))",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = line.replace(mark, "")
    line = _EMPTY_BRACKETS_RE.sub("", line)
    line = re.sub(r"([(\[])[ \t]*[,;:][ \t]*", r"\1", line)
    line = re.sub(r"[ \t]*[,;:][ \t]*([)\]])", r"\1", line)
    line = re.sub(r"([,;])(?:[ \t]*[,;])+", r"\1", line)
    line = re.sub(r"[ \t]*[,;:][ \t]*(?=[.?!](?:\s|$))", "", line)
    line = re.sub(r"[ \t]*[,;][ \t]*$", "", line)
    line = re.sub(r"[ \t]{2,}", " ", line)
    line = re.sub(r"[ \t]+([.,;:!?)\]])", r"\1", line)
    line = re.sub(r"([(\[])[ \t]+", r"\1", line)
    return line.rstrip()


def tool_label(tool: str, inputs: Optional[Dict[str, Any]] = None) -> str:
    """"Payment certificate (gross valuation 2,400,000, retention 10%)"."""
    name = tool_display_name(tool) or "Platform tool"
    shown = ", ".join(input_phrase(k, v) for k, v in (inputs or {}).items()
                      if v not in (None, "") and not isinstance(v, (dict, list)))
    return name + (f" ({shown})" if shown else "")


#: Currency tokens the cost gate already treats as money. One list, so a
#: supplied-input answer and that gate agree on what a currency is.
CURRENCY_TOKEN = r"(?:SAR|SR|USD|US\$|AED|EUR|GBP|QAR|KWD|OMR|BHD|﷼|\$|€|£)"

_CURRENCY_CODES = frozenset({
    "SAR", "SR", "USD", "AED", "EUR", "GBP", "QAR", "KWD", "OMR", "BHD",
})
_CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "﷼", "US$"})

# A demonstrative comparison ("above that limit") whose noun was never
# introduced. The noun is the kind of limit a sentence can point at.
_COMPARISON_RE = re.compile(
    r"\b(?:below|above|under|over|within|outside|beneath|beyond|exceed(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:that|this|those|these)\s+([a-z][a-z-]{2,})\b",
    re.IGNORECASE,
)
_COMPARISON_NOUNS = frozenset({
    "band", "range", "limit", "threshold", "allowance", "cap", "ceiling",
    "floor", "minimum", "maximum", "bracket", "tolerance", "margin",
    "interval", "window", "criterion", "criteria",
})
_CLAUSE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9/+.&-]{1,40}\s+){0,4}"
    r"(?:[Ss]ub-)?[Cc]lause\s+\d+(?:\.\d+)*"
)
_CODE_RE = re.compile(
    r"\b([A-Z]{2,}(?:\s*[/\-]\s*[A-Z0-9]+)*)\s+(\d+(?:\.\d+)*(?:[-/]\d+)?)"
)
_CODE_STOP = frozenset({
    "THE", "AND", "FOR", "PER", "NOT", "BUT", "ALL", "ANY", "ARE", "WAS",
    "HAS", "HAD", "YOU", "OUR", "DAY", "NET", "MAX", "MIN", "NO", "OK",
    "ID", "ITS", "ONE", "TWO",
})
_ASSIGN_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)=([^\s),]+)"
)
_UNIT_AFTER_RE = re.compile(
    r"(?<=\d)\s+([A-Za-z][A-Za-z0-9²³/.+-]{0,16})\b"
)
_BASIS_TAIL_RE = re.compile(
    r"\s+(?:on|under|using|via)\s+the\s+basis\b",
    re.IGNORECASE,
)
_DANGLING_PREP_RE = re.compile(
    r"\s+(?:on|under|using|via|by)\s+(?=[.?!]|$)",
    re.IGNORECASE,
)

_display_by_id: Optional[Dict[str, str]] = None
_param_units: Optional[Dict[str, str]] = None
_known_units: Optional[set] = None
_id_pattern: Optional[re.Pattern[str]] = None


def _registry_maps() -> Tuple[Dict[str, str], Dict[str, str], set]:
    """Display names, snake_case parameter units, and concrete unit tokens.

    Built once from the formula and tool registries. A token is rewritten
    only when it is one of those ids, so a file name or an engineering
    symbol that is not registered is left as written.
    """
    global _display_by_id, _param_units, _known_units, _id_pattern
    if _display_by_id is not None and _param_units is not None and _known_units is not None:
        return _display_by_id, _param_units, _known_units
    from app.agents.core.tool_registry import OWNERS, get, tools_of
    from app.lib import formula_registry

    displays: Dict[str, str] = {}
    params: Dict[str, str] = {}
    units: set = set()
    for spec in formula_registry.all_specs():
        if spec.name and spec.display_name and "_" in spec.name:
            displays[spec.name.lower()] = spec.display_name
        unitless = {_norm(item) for item in _UNITLESS}
        declared = list((spec.inputs or {}).items()) + list((spec.outputs or {}).items())
        for key, unit in declared:
            if "_" in key:
                held = params.get(key.lower())
                if held is None or (_norm(held) in unitless and _norm(unit or "") not in unitless):
                    params[key.lower()] = unit or ""
            norm = _norm(unit)
            raw = (unit or "").strip()
            if raw and norm and norm not in {_norm(item) for item in _UNITLESS}:
                units.add(raw.lower())
    for owner in OWNERS:
        for name in tools_of(owner):
            spec = get(name)
            if spec is None or not spec.display_name or "_" not in spec.name:
                continue
            displays[spec.name.lower()] = spec.display_name
            _collect_schema_params(spec.schema, params)
    _display_by_id = displays
    _param_units = params
    _known_units = units
    if displays:
        alt = "|".join(re.escape(name) for name in sorted(displays, key=len, reverse=True))
        _id_pattern = re.compile(
            rf"(?<![A-Za-z0-9_./])({alt})(?![A-Za-z0-9_])(?!\.[A-Za-z0-9])",
            re.IGNORECASE,
        )
    else:
        _id_pattern = None
    return displays, params, units


def _collect_schema_params(schema: Any, params: Dict[str, str]) -> None:
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if isinstance(props, dict):
        for key, row in props.items():
            if "_" not in str(key):
                continue
            unit = ""
            if isinstance(row, dict):
                unit = str(row.get("unit") or "")
            params.setdefault(str(key).lower(), unit)
    for value in schema.values():
        if isinstance(value, dict):
            _collect_schema_params(value, params)


def _coerce_assignment_value(raw: str) -> Any:
    cleaned = (raw or "").rstrip(".,;:")
    if re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", cleaned):
        number = cleaned.replace(",", "")
        if "." in number:
            value = float(number)
            return int(value) if value.is_integer() else value
        return int(number)
    return cleaned


def plain_registry_text(text: str) -> str:
    """Show registered ids as display names. Leave every other token alone.

    A formula or tool id, including one wrapped as code or written
    ``slot=<id>``, becomes that registration's display name. A snake_case
    parameter written ``key=value`` is read as words, the same way a credit
    reads an input. A file name (the id followed by an extension) and an
    assignment whose key and value are not registered are unchanged.
    """
    if not text or ("_" not in text and "=" not in text):
        return text
    displays, params, _units = _registry_maps()
    if not displays and not params:
        return text

    def replace_assignment(match: re.Match[str]) -> str:
        key, raw = match.group(1), match.group(2)
        low_key = key.lower()
        low_val = raw.rstrip(".,;:").lower()
        if low_val in displays:
            return displays[low_val]
        if low_key in displays and low_key not in params:
            value = _coerce_assignment_value(raw)
            shown = _fmt_value(value)
            return f"{displays[low_key]} {shown}".strip()
        if low_key in params:
            return input_phrase(low_key, _coerce_assignment_value(raw), params[low_key])
        return match.group(0)

    text = _ASSIGN_RE.sub(replace_assignment, text)

    def replace_code_span(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        display = displays.get(inner.lower())
        if display is None:
            return match.group(0)
        return display

    text = re.sub(r"`([^`\n]+)`", replace_code_span, text)
    pattern = _id_pattern
    if pattern is None:
        return text

    def replace_id(match: re.Match[str]) -> str:
        return displays.get(match.group(1).lower(), match.group(0))

    return pattern.sub(replace_id, text)


#: Public name for the pure display-name transform (ORDERS_0150Z PR-A0): show
#: registered formula/tool ids as their display names, leave everything else
#: alone. Pure, idempotent (display names contain spaces so they never re-match
#: an id token), no I/O beyond the cached registry maps, no state.
apply_display_names = plain_registry_text

#: A trailing run of identifier characters at the very end of a chunk — it may
#: be the first half of a registry id whose second half is in the next chunk.
_ID_TAIL_RE = re.compile(r"[A-Za-z0-9_.]+$")


class DisplayNameStreamer:
    """Apply :func:`apply_display_names` across STREAMED segments without
    splitting a registry id that straddles a segment boundary.

    The transform is pure; streaming needs a small per-stream tail buffer
    because an id can arrive in two deltas (``"delay_"`` then
    ``"damages_daily"``). :meth:`feed` converts everything up to a trailing
    run of identifier characters and holds that run back; the next
    :meth:`feed` re-joins it; :meth:`flush` converts whatever remains at
    end-of-stream. One instance per stream — the only state is the local tail
    buffer, so the module stays stateless and the transform stays pure.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, segment: str) -> str:
        s = self._buf + (segment or "")
        m = _ID_TAIL_RE.search(s)
        if m is not None and m.end() == len(s):
            # Ends mid-identifier-run: convert the safe prefix, hold the run.
            self._buf = s[m.start():]
            return apply_display_names(s[: m.start()])
        self._buf = ""
        return apply_display_names(s)

    def flush(self) -> str:
        out = apply_display_names(self._buf)
        self._buf = ""
        return out


def _result_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found: List[str] = []
        for item in value.values():
            found.extend(_result_strings(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_result_strings(item))
        return found
    return []


def _unsupplied_currency_defaults(calculation: str, supplied: Dict[str, Any]) -> set:
    """Currency codes a formula stamps when the user did not give one.

    The code is a signature default, not a unit the calculation measured
    and not a currency the user typed. It does not authorise writing that
    code next to the user's figures.
    """
    from app.lib import formula_registry

    spec = formula_registry.get(calculation or "")
    if spec is None:
        return set()
    try:
        signature = inspect.signature(spec.fn)
    except (TypeError, ValueError):
        return set()
    bound = supplied or {}
    found = set()
    for key, param in signature.parameters.items():
        if key in bound and bound[key] not in (None, ""):
            continue
        default = param.default
        if not isinstance(default, str):
            continue
        code = default.strip().upper()
        if code in _CURRENCY_CODES:
            found.add(code)
    return found


def _supplied_tokens(supplied: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for value in (supplied or {}).values():
        if isinstance(value, bool) or value in (None, ""):
            continue
        if isinstance(value, str) and len(value.strip()) >= 2:
            tokens.append(value.strip().casefold())
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            rendered = f"{value:g}"
            if isinstance(value, int) and abs(value) >= 10:
                tokens.append(f"{value:,}")
            if len(rendered) >= 2:
                tokens.append(rendered.casefold())
    return tokens


def _citation_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    spans.extend((match.start(), match.end(), match.group(0)) for match in _CLAUSE_RE.finditer(text))
    for match in _CODE_RE.finditer(text):
        head = match.group(1).split()[0].upper()
        if head in _CURRENCY_CODES or head in _CODE_STOP or head in _CURRENCY_SYMBOLS:
            continue
        spans.append((match.start(), match.end(), match.group(0)))
    spans.sort(key=lambda item: item[0])
    return spans


def _citation_stated(span: str, user: str, retrieval: str, result: Any,
                     supplied: Dict[str, Any]) -> bool:
    """True when this citation is in the request, the excerpts, or a result
    string that also carries a value the user supplied.

    A fixed label the formula returns for every call does not name the
    user's inputs, so it is not a source for the basis of this result.
    """
    needle = re.sub(r"\s+", " ", span).strip().casefold()
    if len(needle) < 4:
        return True
    blob = re.sub(r"\s+", " ", f"{user or ''}\n{retrieval or ''}").casefold()
    if needle in blob:
        return True
    tokens = _supplied_tokens(supplied)
    if not tokens:
        return False
    for value in _result_strings(result):
        folded = re.sub(r"\s+", " ", value).casefold()
        if needle in folded and any(token in folded for token in tokens):
            return True
    return False


def _known_unit_in(text: str, known: Iterable[str]) -> set:
    found = set()
    for unit in known:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(unit)}(?![A-Za-z0-9])", text or "", re.IGNORECASE):
            found.add(unit.lower())
    return found


def _allowed_currencies(user: str, retrieval: str, supplied: Dict[str, Any],
                        result: Any, calculation: str) -> set:
    allowed = set()
    blob = f"{user or ''}\n{retrieval or ''}"
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", blob, re.IGNORECASE):
            allowed.add(code)
    for symbol in _CURRENCY_SYMBOLS:
        if symbol in blob:
            allowed.add(symbol)
    for value in (supplied or {}).values():
        if isinstance(value, str) and value.strip().upper() in _CURRENCY_CODES:
            allowed.add(value.strip().upper())
    skip = _unsupplied_currency_defaults(calculation, supplied)
    for value in _result_strings(result):
        for code in _CURRENCY_CODES:
            if code in skip:
                continue
            if re.search(rf"\b{code}\b", value):
                allowed.add(code)
        for symbol in _CURRENCY_SYMBOLS:
            if symbol in value:
                allowed.add(symbol)
    return allowed


def _allowed_units(user: str, retrieval: str, calculation: str, result: Any,
                   known: set) -> set:
    from app.lib import formula_registry

    allowed = _known_unit_in(f"{user or ''}\n{retrieval or ''}", known)
    spec = formula_registry.get(calculation or "")
    if spec is not None:
        for unit in list((spec.inputs or {}).values()) + list((spec.outputs or {}).values()):
            raw = (unit or "").strip().lower()
            if raw in known:
                allowed.add(raw)
    for value in _result_strings(result):
        allowed.update(_known_unit_in(value, known))
    return allowed


def _currency_allowed(token: str, allowed: set) -> bool:
    if token in allowed or token.upper() in allowed:
        return True
    return False


def _strip_ungrounded_qualifiers(text: str, allowed_ccy: set, allowed_units: set,
                                 known_units: set) -> str:
    def drop_currency(match: re.Match[str]) -> str:
        if _currency_allowed(match.group(1), allowed_ccy):
            return match.group(0)
        return ""

    text = re.sub(
        rf"(?<![A-Za-z0-9])({CURRENCY_TOKEN})\s*(?=\d)",
        drop_currency,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(?<=\d)\s*({CURRENCY_TOKEN})(?![A-Za-z0-9])",
        drop_currency,
        text,
        flags=re.IGNORECASE,
    )

    def drop_unit(match: re.Match[str]) -> str:
        token = match.group(1).lower()
        if token not in known_units:
            return match.group(0)
        if token in allowed_units:
            return match.group(0)
        return ""

    text = _UNIT_AFTER_RE.sub(drop_unit, text)
    return text


def _strip_ungrounded_citations(text: str, user: str, retrieval: str, result: Any,
                                supplied: Dict[str, Any]) -> str:
    kept: List[str] = []
    for line in text.split("\n"):
        spans = _citation_spans(line)
        removed = False
        for start, end, span in reversed(spans):
            if _citation_stated(span, user, retrieval, result, supplied):
                continue
            line = line[:start] + _REMOVED_MARK + line[end:]
            removed = True
        if not removed:
            kept.append(line)
            continue
        line = tidy_removal_debris(line, _REMOVED_MARK).strip()
        line = _BASIS_TAIL_RE.sub("", line)
        line = _DANGLING_PREP_RE.sub("", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if line and re.search(r"\d", line):
            kept.append(line)
    return "\n".join(kept)


def _sentence_is_dangling(sentence: str, user: str, retrieval: str, earlier: str) -> bool:
    if _COMPARISON_RE.search(sentence) is None:
        return False
    nouns = [
        match.group(1).lower()
        for match in _DEMONSTRATIVE_RE.finditer(sentence)
        if match.group(1).lower() in _COMPARISON_NOUNS
    ]
    if not nouns:
        return False
    without = _DEMONSTRATIVE_RE.sub(" ", sentence)
    blob = f"{user or ''}\n{retrieval or ''}\n{earlier}\n{without}"
    for noun in nouns:
        if re.search(rf"\b{re.escape(noun)}\b", blob, re.IGNORECASE) is None:
            return True
    return False


def _drop_dangling_comparisons(text: str, user: str, retrieval: str) -> str:
    earlier = ""
    kept_lines: List[str] = []
    for line in text.split("\n"):
        parts = re.split(r"((?<=[.!?])\s+)", line)
        rebuilt: List[str] = []
        for part in parts:
            if re.fullmatch(r"\s+", part or ""):
                rebuilt.append(part)
                continue
            if _sentence_is_dangling(part, user, retrieval, earlier):
                if rebuilt and re.fullmatch(r"\s+", rebuilt[-1] or ""):
                    rebuilt.pop()
                continue
            earlier += " " + part
            rebuilt.append(part)
        kept_lines.append("".join(rebuilt).strip())
    return "\n".join(line for line in kept_lines if line.strip())


def _tidy_wording(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guard_supplied_input_wording(
    text: str,
    user: str,
    retrieval: str,
    calculation: str,
    supplied: Optional[Dict[str, Any]] = None,
    result: Any = None,
) -> str:
    """Drop currency, unit, clause and comparison text this turn does not state.

    A qualifier attached to a figure stays when the user's words, the
    retrieved excerpts or the calculator result state it. A currency the
    formula inserts only because the user left that parameter blank does
    not count as the calculator stating it. A clause or standard counts
    when the excerpts state it, or when a result string states it and that
    string also carries a value the user supplied. A comparison to "that
    <limit>" stays when the limit was named earlier in the turn.
    """
    if not text:
        return text
    bound = dict(supplied or {})
    _displays, _params, known = _registry_maps()
    text = _strip_ungrounded_citations(text, user, retrieval, result, bound)
    allowed_ccy = _allowed_currencies(user, retrieval, bound, result, calculation)
    text = _strip_ungrounded_qualifiers(
        text,
        allowed_ccy,
        _allowed_units(user, retrieval, calculation, result, known),
        known,
    )
    for code, line in calculator_currency_defaults(calculation, result, user, bound):
        if not _currency_allowed(code, allowed_ccy):
            # The figures no longer show this currency, so nothing rests on it.
            text = "\n".join(ln for ln in text.split("\n") if ln.strip() != line)
    text = _drop_dangling_comparisons(text, user, retrieval)
    return _tidy_wording(text)
