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

The bound inputs must also be inputs the scorer can take as written: the
weights are fractions or percentages, an amount is a positive sum, every
other criterion is a 0-100 score where higher is better, and no rule the
ask states for a criterion differs from the scorer's own. A criterion's
direction is known only from the scorer's own criteria or from the ask's
words; a score of unknown direction binds nothing.
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

# Words that call a figure a score. The scorer reads every criterion that is
# not an amount as a 0-100 score, so a figure binds there only when the ask
# itself calls it one.
_SCORE_WORD = r"(?:scores?|scored|points?|marks?|ratings?)"
_SCORE_SUFFIX_RE = re.compile(rf"(?<=[A-Za-z])\s+{_SCORE_WORD}$", re.IGNORECASE)
_SCORE_MIN, _SCORE_MAX = 0.0, 100.0
# Any word that says a figure was scored, for the amount-rule guard.
_SCORED_RE = rf"(?:{_SCORE_WORD}|scoring)"
# A sentence that says which way a criterion is better: "higher is better",
# "lower scores are better", "the highest rating wins".
_BETTER_RE = re.compile(
    r"\b(?P<dir>higher|highest|lower|lowest)\b(?:\s+[A-Za-z]+){0,3}?\s+"
    r"(?:(?:is|are)\s+)?(?:better|best|preferred|wins?)\b",
    re.IGNORECASE,
)
# Weights the scorer can apply: fractions of one or percentages of 100,
# with the tolerances the scorer itself reads percentages with.
_WEIGHT_TOTALS = ((1.0, 0.005), (100.0, 0.5))


def _words(label: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD_RE.findall(label.lower())]


def _key(label: str) -> str:
    return "_".join(_words(label))


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _weight_clause_end(tail: str) -> int:
    """Where the weight clause that opens ``tail`` ends."""
    stop = re.search(r"\.(?!\d)|\?|\n|,\s+(?=[a-z]+\s+score\b)", tail, re.IGNORECASE)
    return stop.start() if stop else len(tail)


def stated_weights(text: str) -> dict[str, float] | None:
    """``{criterion: weight}`` from the clause that follows a weight word.

    The clause runs to the sentence end or the next comma that opens a new
    phrase; every item in it must be a label with a figure. Fewer than two
    criteria, or two items under one label, is no reading.
    """
    raw = text or ""
    for match in _WEIGHT_WORD_RE.finditer(raw):
        tail = raw[match.end():]
        clause = tail[: _weight_clause_end(tail)]
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
        label = _SCORE_SUFFIX_RE.sub("", (m.group("label") or "").strip())
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


def _crit_re(crit: str) -> str:
    """The criterion's words as written, each optionally plural."""
    return r"\s+".join(rf"{re.escape(w)}s?" for w in crit.split("_"))


def _agrees(rule: str, crit: str) -> bool:
    rule = re.sub(r"\s+", "", rule.lower())
    own = re.escape(crit.replace("_", ""))
    return re.fullmatch(rf"lowest(?:{own})?/{own}(?:[x*×]100)?", rule) is not None


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"\.(?!\d)\s*|\?\s*|\n+", text or "") if s.strip()]


def _rule_disagrees(text: str, criteria: list[str], amounts: set[str]) -> bool:
    """True when the ask states a score rule other than the scorer's.

    The scorer this binds to scores an amount as lowest / amount x 100. An
    ask that writes any other rule for a criterion is not that scorer's ask,
    in either form: "<criterion> score = <rule>", or a sentence that says
    how an amount criterion is scored ("<criterion> is scored ...",
    "scoring of <criterion> ...") without writing the scorer's rule.
    """
    for m in _RULE_RE.finditer(text or ""):
        crit = _criterion_for(m.group("crit"), criteria)
        if crit is not None and not _agrees(m.group("rule"), crit):
            return True
    for crit in amounts:
        c = _crit_re(crit)
        said = re.compile(
            rf"\b{c}\b[^0-9,;]*?\b{_SCORED_RE}\b|\b{_SCORED_RE}\s+(?:of|for|on)\s+(?:the\s+)?{c}\b",
            re.IGNORECASE,
        )
        for sentence in _sentences(text):
            if not said.search(sentence):
                continue
            stated = [
                r for r in _RULE_RE.finditer(sentence)
                if _criterion_for(r.group("crit"), criteria) == crit
            ]
            if not stated or not all(_agrees(r.group("rule"), crit) for r in stated):
                return True
    return False


def _called_a_score(text: str, crit: str) -> bool:
    """True when the ask calls the criterion's figures scores.

    "<criterion> score(s) / scored / points / marks / rating", optionally
    with "is"/"are" between, or "scores for / of / on <criterion>".
    """
    c = _crit_re(crit)
    return re.search(
        rf"\b{c}\s+(?:(?:is|are)\s+)?{_SCORE_WORD}\b|\b{_SCORE_WORD}\s+(?:for|of|on)\s+(?:the\s+)?{c}\b",
        text or "",
        re.IGNORECASE,
    ) is not None


def _directions(text: str, crit: str, criteria: list[str]) -> set[str]:
    """Which way the ask says the criterion is better: "higher", "lower".

    A sentence with a better-word applies to the criteria it names, or,
    when it names none, to every score ("higher scores are better").
    """
    found: set[str] = set()
    for sentence in _sentences(text):
        named = {
            c for c in criteria
            if re.search(rf"\b{_crit_re(c)}\b", sentence, re.IGNORECASE)
        }
        if named and crit not in named:
            continue
        if not named and not re.search(rf"\b{_SCORE_WORD}\b", sentence, re.IGNORECASE):
            continue
        for m in _BETTER_RE.finditer(sentence):
            found.add("higher" if m.group("dir").lower().startswith("high") else "lower")
    return found


def _criterion_named(crit: str, text: str) -> bool:
    return re.search(rf"\b{_crit_re(crit)}\b", text, re.IGNORECASE) is not None


# The one scale the scorer reads: "out of 100", "0 to 100", "0-100".
_SCORER_SCALE_RE = re.compile(
    r"\bout\s+of\s+100\b|\b(?:from\s+)?0\s*(?:to|-|–)\s*100\b",
    re.IGNORECASE,
)

# The words a sentence about weighted records may use and still be read in
# full: the scorer's own reading of a score (a 0-100 score, higher better),
# which way is better, and the question of who wins. Any other word in a
# sentence that is not the records, the weights or the scorer's own rule
# says something this reader does not read, so the ask binds nothing.
_READ_WORDS = frozenset("""
    a an the is are be all each every both for of on and in as with its their it they
    score scored scoring point mark marked rating rated run range
    higher highest lower lowest better best preferred prefer win winner
    who which what should we rank ranking ranked recommend award overall weighted total
    use using apply evaluate
    tender bid offer bidder supplier candidate option contractor
""".split())
# Candidate counts, read only in the heading before the records ("Two offers:").
_COUNT_WORDS = frozenset(
    "two three four five six seven eight nine ten eleven twelve".split()
)
_TOKEN_RE = re.compile(rf"[A-Za-z]+(?:'s)?|{_NUM}")
# A figure that carries its own unit: a percentage or a scaled amount.
_UNIT_FIGURE_RE = re.compile(rf"(?:{_NUM})\s*(?:%|{_SCALE_RE})", re.IGNORECASE)


def _strip_read_parts(sentence: str, criteria: list[str]) -> tuple[str, str]:
    """``(heading, rest)``: the sentence without its records, weights and rule.

    The scorer's own rule ("<criterion> score = lowest/<criterion>*100") and
    the weight clause are taken out; a run of records takes the sentence
    after its colon, and the words before that colon are the heading.
    """
    for m in reversed(list(_RULE_RE.finditer(sentence))):
        crit = _criterion_for(m.group("crit"), criteria)
        if crit is not None and _agrees(m.group("rule"), crit):
            sentence = sentence[: m.start()] + sentence[m.end():]
    weight = _WEIGHT_WORD_RE.search(sentence)
    if weight is not None:
        tail = sentence[weight.end():]
        sentence = sentence[: weight.start()] + " " + tail[_weight_clause_end(tail):]
    head, colon, tail = sentence.rpartition(":")
    if colon and stated_records(tail, criteria):
        return head, ""
    if stated_records(sentence, criteria):
        return "", ""
    return "", sentence


def _read_in_full(tokens: list[str], criteria: list[str], extra: frozenset[str] = frozenset()) -> bool:
    """True when every token is a read word; a figure is never one."""
    crit_words = {w for c in criteria for w in c.split("_")}
    for token in tokens:
        words = _words(token)
        if not words:
            return False
        if any(w not in _READ_WORDS and w not in crit_words and w not in extra for w in words):
            return False
    return True


def _further_instruction(sentence: str, criteria: list[str]) -> bool:
    """True for a sentence that asks a further step with figures of its own.

    It names no criterion and no score, and every figure in it carries its
    own unit (a percentage, a scaled amount), so none of them can be read as
    a scale or a count of the weighted criteria ("pay 20 % on the winner").
    """
    if any(_criterion_named(c, sentence) for c in criteria):
        return False
    if re.search(rf"\b{_SCORED_RE}\b", sentence, re.IGNORECASE):
        return False
    figures = re.findall(_NUM, sentence)
    return bool(figures) and len(_UNIT_FIGURE_RE.findall(sentence)) == len(figures)


def _every_sentence_read(text: str, criteria: list[str]) -> bool:
    """True when every sentence of the ask is read in full.

    A sentence outside the records, the weights and the scorer's own rule
    may only restate the scorer's reading (a score out of 100, higher is
    better) and ask who wins, or ask a further step with figures of its own.
    A sentence that says anything else - a scale, a maximum, what the
    figures count, where they come from - may change what they mean, and
    this reader does not read it, so nothing binds.
    """
    for sentence in _sentences(text):
        heading, rest = _strip_read_parts(sentence, criteria)
        if not _read_in_full(_TOKEN_RE.findall(heading), criteria, _COUNT_WORDS):
            return False
        rest = _SCORER_SCALE_RE.sub(" ", rest)
        if not _read_in_full(_TOKEN_RE.findall(rest), criteria) and not _further_instruction(rest, criteria):
            return False
    return True


def _scorer_criteria() -> frozenset[str]:
    """The scorer's own criteria: 0-100 scores where higher is better."""
    from app.agents.hats.procurement.formulas.construction_knowledge import TENDER_DEFAULT_WEIGHTS

    return frozenset(TENDER_DEFAULT_WEIGHTS)


def _weights_apply(weights: dict[str, float]) -> bool:
    """True when the weights are fractions of one or percentages of 100."""
    total = sum(weights.values())
    return any(abs(total - want) <= tol for want, tol in _WEIGHT_TOTALS)


def _figures_fit(text: str, records: list[dict[str, Any]], criteria: list[str], amounts: set[str]) -> bool:
    """True when every figure is one the scorer can take as written.

    An amount is a positive sum (the scorer divides by it). Any other
    criterion is a 0-100 score the scorer ranks higher-first, so it binds
    only when that is its direction: one of the scorer's own criteria, or a
    criterion the ask calls a score and says is better higher. A score
    word alone does not say the direction (a lower score can be the better
    one), and a figure the ask never calls a score may be days, weeks or a
    dose. The ask saying lower is better, or both ways, binds nothing.
    """
    own = _scorer_criteria()
    for crit in criteria:
        values = [r[crit] for r in records]
        if crit in amounts:
            if any(v <= 0 for v in values):
                return False
            continue
        directions = _directions(text, crit, criteria)
        if "lower" in directions:
            return False
        if crit not in own and not (_called_a_score(text, crit) and directions == {"higher"}):
            return False
        if any(not (_SCORE_MIN <= v <= _SCORE_MAX) for v in values):
            return False
    return True


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
    if not _weights_apply(weights):
        return None
    criteria = list(weights)
    from app.agents.hats.procurement.formulas.construction_knowledge import _names_an_amount

    # The scorer's own test of which criteria are amounts; one copy.
    amounts = {c for c in criteria if _names_an_amount(c)}
    if _rule_disagrees(raw, criteria, amounts):
        return None
    records = stated_records(raw, criteria)
    if not records or not _figures_fit(raw, records, criteria, amounts):
        return None
    if not _every_sentence_read(raw, criteria):
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
