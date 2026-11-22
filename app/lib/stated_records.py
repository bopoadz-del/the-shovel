"""Weighted records an ask states in its own words.

An ask can state several candidates, each with figures under criterion
labels, plus the weight of each criterion ("<name> <figure> <criterion>
<figure>, <name> ..., weights <criterion> <w> / <criterion> <w>"). This
module reads that shape and binds it to the one registered formula whose
signature takes a list of records and a ``weights`` mapping, so the
platform can run the scoring instead of the model writing it in prose.

Every rule reads the ask's own words: a criterion is a label the weights
name, a record is a clause that opens on a name and carries figures, and a
figure with no label of its own belongs to the one weighted criterion no
record labels. Anything in doubt binds nothing.
"""
from __future__ import annotations

import inspect
import logging
import re
import typing
from typing import Any

_LOG = logging.getLogger(__name__)

_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+"

# Words that scale a written amount. A lone lower-case "m" is a metre, not
# a million, so it is not here.
_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "M": 1e6, "mn": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9,
}
_SCALE_RE = r"(?:k|K|M|mn|million|thousand|bn|billion)\b"

_WEIGHT_WORD_RE = re.compile(r"\bweight(?:s|ings?)?\b\s*(?:of|are|is|:|=)?\s*", re.IGNORECASE)
# One weight: a label of letters and a figure, either order, optional %.
_WEIGHT_ITEM_RE = re.compile(
    rf"^\s*(?:and\s+)?(?:(?P<label>[A-Za-z][A-Za-z ]*?)\s*[:=]?\s*(?P<num>{_NUM})\s*%?"
    rf"|(?P<num2>{_NUM})\s*%?\s*(?:for\s+)?(?P<label2>[A-Za-z][A-Za-z ]*?))\s*$",
    re.IGNORECASE,
)
_WEIGHT_SEP_RE = re.compile(r"\s*(?:/|;|,\s|\band\b)\s*", re.IGNORECASE)

# A stated scoring rule for a criterion ("price score = lowest/price*100").
_RULE_RE = re.compile(
    r"\b(?P<crit>[A-Za-z]+)\s+score\s*(?:=|is|:)\s*(?P<rule>[^,;?\n]+?)(?=[,;?\n]|\.\s|\.$|$)",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z]+")


def _words(label: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD_RE.findall(label.lower())]


def _key(label: str) -> str:
    return "_".join(_words(label))


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def stated_weights(text: str) -> dict[str, float] | None:
    """``{criterion: weight}`` from the clause that follows a weight word.

    The clause runs to the sentence end or the next comma that opens a new
    phrase; every item in it must be a label with a figure. Fewer than two
    criteria, or two items under one label, is no reading.
    """
    raw = text or ""
    for match in _WEIGHT_WORD_RE.finditer(raw):
        tail = raw[match.end():]
        stop = re.search(r"\.(?!\d)|\?|\n|,\s+(?=[a-z]+\s+score\b)", tail, re.IGNORECASE)
        clause = tail[: stop.start()] if stop else tail
        items = [p for p in _WEIGHT_SEP_RE.split(clause) if p.strip()]
        weights: dict[str, float] = {}
        for item in items:
            m = _WEIGHT_ITEM_RE.match(item)
            if m is None:
                weights = {}
                break
            label = m.group("label") or m.group("label2") or ""
            key = _key(label)
            if not key or key in weights:
                weights = {}
                break
            weights[key] = _number(m.group("num") or m.group("num2"))
        if len(weights) >= 2:
            return weights
    return None


def _criterion_for(label: str, criteria: list[str]) -> str | None:
    """The one criterion whose words the label's words match.

    A label with no words matches nothing: a criterion key always has at
    least one word.
    """
    words = set(_words(label))
    hits = [c for c in criteria if set(c.split("_")) == words]
    return hits[0] if len(hits) == 1 else None


_FIGURE_RE = re.compile(
    rf"(?P<label>[A-Za-z][A-Za-z ]*?)?\s*(?P<num>{_NUM})\s*(?P<scale>{_SCALE_RE})?(?P<pct>\s*%)?",
)


def _parse_record(segment: str, criteria: list[str]) -> tuple[str, dict[str, float], list[float]] | None:
    """``(name, labelled figures, unlabelled figures)`` for one clause.

    The clause opens on its name (one token that is not a criterion and not
    a figure) and the rest is figures, each either under a criterion label
    or bare. Any other word makes it no record.
    """
    seg = re.sub(r"^\s*(?:and\s+)?", "", segment).strip()
    head = re.match(r"(?P<name>[A-Z][A-Za-z0-9]*|\d+[A-Za-z]+)\s+(?P<rest>.+)$", seg)
    if head is None:
        return None
    name = head.group("name")
    if _criterion_for(name, criteria):
        return None
    rest = head.group("rest")
    labelled: dict[str, float] = {}
    bare: list[float] = []
    pos = 0
    for m in _FIGURE_RE.finditer(rest):
        if rest[pos:m.start()].strip():
            return None
        pos = m.end()
        value = _number(m.group("num"))
        scale = m.group("scale")
        if scale:
            value *= _SCALE.get(scale, _SCALE.get(scale.lower(), 1.0))
        label = (m.group("label") or "").strip()
        if label:
            crit = _criterion_for(label, criteria)
            if crit is None or crit in labelled:
                return None
            labelled[crit] = value
        else:
            bare.append(value)
    if rest[pos:].strip() or not (labelled or bare):
        return None
    return name, labelled, bare


def _candidate_segments(text: str) -> list[list[str]]:
    """Comma/semicolon runs, one per sentence, after the sentence's colon."""
    runs: list[list[str]] = []
    for sentence in re.split(r"\.(?!\d)\s*|\?\s*|\n+", text or ""):
        body = sentence.rsplit(":", 1)[-1]
        parts = [p for p in re.split(r";\s*|,\s+", body) if p.strip()]
        if len(parts) >= 2:
            runs.append(parts)
    return runs


def stated_records(text: str, criteria: list[str]) -> list[dict[str, Any]] | None:
    """Records the ask states, every one carrying every criterion.

    A run of at least two clauses that each read as a record. A bare figure
    goes to the single criterion that no record labels; with no such single
    criterion, or a record with more than one bare figure, nothing binds.
    """
    for parts in _candidate_segments(text):
        parsed = [_parse_record(p, criteria) for p in parts]
        if any(p is None for p in parsed) or len(parsed) < 2:
            continue
        labelled_anywhere = {c for _n, lab, _b in parsed for c in lab}
        free = [c for c in criteria if c not in labelled_anywhere]
        records: list[dict[str, Any]] = []
        for name, labelled, bare in parsed:
            row: dict[str, Any] = {"name": name, **labelled}
            if bare:
                if len(bare) != 1 or len(free) != 1:
                    records = []
                    break
                row[free[0]] = bare[0]
            if any(c not in row for c in criteria):
                records = []
                break
            records.append(row)
        names = [r["name"] for r in records]
        if len(records) >= 2 and len(set(names)) == len(names):
            return records
    return None


def _rule_disagrees(text: str, criteria: list[str]) -> bool:
    """True when the ask states a score rule other than lowest / amount.

    The scorer this binds to scores an amount as lowest / amount x 100. An
    ask that writes any other rule for a criterion is not that scorer's ask.
    """
    for m in _RULE_RE.finditer(text or ""):
        crit = _criterion_for(m.group("crit"), criteria)
        if crit is None:
            continue
        rule = re.sub(r"\s+", "", m.group("rule").lower())
        own = re.escape(crit.replace("_", ""))
        if not re.fullmatch(rf"lowest(?:{own})?/{own}(?:[x*×]100)?", rule):
            return True
    return False


def _takes_records_and_weights(fn: Any) -> str | None:
    """The records parameter of a formula that takes records and weights."""
    try:
        params = inspect.signature(fn).parameters
        hints = typing.get_type_hints(fn)
    except (TypeError, ValueError, NameError) as exc:
        _LOG.info("records-and-weights check skipped %r: signature unreadable (%s)", fn, exc)
        return None
    if "weights" not in params:
        return None
    records = [name for name in params if name != "weights" and _is_list_of_dicts(hints.get(name))]
    return records[0] if len(records) == 1 else None


def _is_list_of_dicts(hint: Any) -> bool:
    if typing.get_origin(hint) is not list:
        return False
    args = typing.get_args(hint)
    return bool(args) and (args[0] is dict or typing.get_origin(args[0]) is dict)


def records_completed_by_user_text(text: str) -> tuple[str, dict[str, Any]] | None:
    """``(formula name, inputs)`` when the ask states weighted records.

    The one registered formula that takes a list of records and a weights
    mapping, with both read from the ask. Several such formulas is no
    selection.
    """
    raw = text or ""
    weights = stated_weights(raw)
    if not weights:
        return None
    criteria = list(weights)
    if _rule_disagrees(raw, criteria):
        return None
    records = stated_records(raw, criteria)
    if not records:
        return None
    from app.lib.formula_registry import all_specs

    takers = [
        (spec.name, param) for spec in all_specs()
        for param in [_takes_records_and_weights(spec.fn)] if param
    ]
    if len(takers) != 1:
        return None
    name, param = takers[0]
    return name, {param: records, "weights": weights}
