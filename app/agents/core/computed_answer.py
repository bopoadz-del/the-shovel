"""The answer a computed tool's result states, in the language of the question.

When the model's synthesis is missing, a tool declared ``kind="computed"``
answers with its own statements (``tool_registry.Statement``). This module
turns those statements -- and a computed call that FAILED or found NO
matching rows -- into plain text:

* ONE ANSWER PER TURN (``answer_turn``): the calls that returned a result
  lead, in call order; identical results (same tool, same stated lines) are
  one result; a call that returned no figure is shown only when no result of
  the same tool answered the same part of the question (``covers``), and
  then after every result, in one plain line;
* a part that did not return (one bill of several) is NAMED, never skipped,
  so a partial result is never shown as complete -- after the parts that did;
* a turn where no call returned a result says what was asked and why there is
  no figure, instead of the generic "unable to generate" sentence;
* every label and every connecting word comes from ``LABELS``, keyed by the
  language of the user's question. Figures, units, sources, row labels and the
  words the user asked about are data and are shown as they are.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Labels per language. A language is added by adding one entry; nothing
#: below writes a word of its own.
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "heading": "Computed result",
        "count": "Number of rows",
        "sum": "{what} summed over the rows",
        "holds": " whose description holds {terms}",
        "within": " in the section {terms}",
        "source": "source",
        "rows": "Rows",
        "working": "Working",
        "no_rows": "no matching rows",
        "not_returned": "did not return",
        "asked": "Asked",
        "no_figure": "the tool returned no figure",
        "failed": "the tool could not complete the request",
        "several": "{n} computed results in this turn, all shown below.",
        "sep": ", ",
        "list_sep": "; ",
        "quantity": "quantity",
        "amount": "amount",
        "tool_error": "the tool reported an error",
        "code": "code",
        "unanswered": "Not answered in this turn: {parts}.",
    },
    "ar": {
        "heading": "نتيجة محسوبة",
        "count": "عدد البنود",
        "sum": "مجموع {what} للبنود",
        "holds": " التي يتضمن وصفها {terms}",
        "within": " في القسم {terms}",
        "source": "المصدر",
        "rows": "البنود",
        "working": "طريقة الحساب",
        "no_rows": "لا توجد بنود مطابقة",
        "not_returned": "لم تُرجع نتيجة",
        "asked": "المطلوب",
        "no_figure": "لم تُرجع الأداة أي رقم",
        "failed": "تعذّر على الأداة إتمام الطلب",
        "several": "{n} نتائج محسوبة في هذا الدور، وجميعها معروضة أدناه.",
        "sep": "، ",
        "list_sep": "؛ ",
        "quantity": "الكمية",
        "amount": "المبلغ",
        "tool_error": "أبلغت الأداة عن خطأ",
        "code": "الرمز",
        "unanswered": "لم تتم الإجابة عنه في هذا الدور: {parts}.",
    },
}

#: Known reason codes, worded per language. A tool's error is looked up
#: ONLY here (never in LABELS, so an error that happens to equal a label key
#: is not shown as that label). A code is added by adding it to every
#: language below.
REASONS: Dict[str, Dict[str, str]] = {
    "en": {
        "no_derived_bill_sheet": "no bill sheet has been derived for this project",
        "derived_sheet_has_no_readable_bytes": "the bill sheet file could not be read",
        "derived_sheet_is_not_a_readable_workbook": "the bill sheet is not a readable workbook",
    },
    "ar": {
        "no_derived_bill_sheet": "لم يُستخرج جدول كميات لهذا المشروع",
        "derived_sheet_has_no_readable_bytes": "تعذّرت قراءة ملف جدول الكميات",
        "derived_sheet_is_not_a_readable_workbook": "ملف جدول الكميات ليس مصنفًا مقروءًا",
    },
}

_LOG = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"

#: The letters of each script a LABELS language is written in.
_SCRIPT_LETTERS: Dict[str, "re.Pattern[str]"] = {
    "ar": re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"),
    "en": re.compile(r"[A-Za-z]"),
}


def has_words(text: str) -> bool:
    """True when ``text`` carries a letter or digit in any script; markdown
    alone (rules, bullets, emphasis marks, blank lines) says nothing."""
    return bool(re.sub(r"[\W_]+", "", text or ""))


def question_language(question: str) -> str:
    """The LABELS language whose script holds most letters of ``question``."""
    counts = {lang: len(rx.findall(question or "")) for lang, rx in _SCRIPT_LETTERS.items()}
    best = max(counts, key=lambda lang: counts[lang])
    return best if counts[best] else DEFAULT_LANGUAGE


def labels_for(language: str) -> Dict[str, str]:
    return LABELS.get(language) or LABELS[DEFAULT_LANGUAGE]


def _figure_text(value: float) -> str:
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _word(lab: Dict[str, str], key: str) -> str:
    """A data value shown through LABELS when it is a known code, else as is."""
    return lab.get(key, key)


#: What must never reach a user from the echoed call arguments (a tool's
#: error text is never shown at all, see ``reason_text``):
#: a stack trace and everything after it, a file path, a SQL table/relation
#: name, an internal id (uuid, long hex, a word joined to a hex key by an
#: underscore). A document's own codes (drawing S-101, lot L-01) are data.
_TRACE_RE = re.compile(r"(?:Traceback \(most recent call last\)|\bFile \"[^\"]*\", line \d+).*", re.S)
_PATH_RE = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\|(?<![\w.])/)(?:[\w.\-]+[\\/])+[\w.\-]*")
_TABLE_RE = re.compile(
    r"\b(?:relation|table|column|schema|index|constraint)\s+"
    r"(?:[\"'`][^\"'`\s]+[\"'`]|[a-z]\w*[_.][\w.]+)"
    r"|\b(?:from|into|update|join)\s+[\"'`]?[a-z_]\w*\.[\w.]+[\"'`]?",
    re.I)
#: A dotted module or exception path (``psycopg.errors.UndefinedTable``).
_DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}:?")
_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b"
    r"|\b[a-z]+_(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{6,}\b",
    re.I)


def scrub(text: str) -> str:
    """``text`` without a stack trace, file path, table name or internal id.
    Applied to the echoed call arguments only."""
    out = _TRACE_RE.sub("", str(text or ""))
    for rx in (_PATH_RE, _TABLE_RE, _DOTTED_RE, _ID_RE):
        out = rx.sub("", out)
    out = re.sub(r"[{}\[\]]", "", out)
    out = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"\(\s*\)", "", out))
    return re.sub(r"\s{2,}", " ", out).strip(" ,;:-\n")


#: The longest reason code ever shown; a longer string is not a code.
MAX_CODE_CHARS = 64


def reason_text(error: str, lab: Dict[str, str], language: str,
                declared: Sequence[str] = ()) -> str:
    """A tool's error as the user reads it, in the question's language.

    No free text a tool wrote reaches an answer, in any language. A reason
    code in ``REASONS`` reads as its wording there. A code the tool's
    registry entry DECLARES (``declared``) and ``REASONS`` does not word is
    shown as a code after the plain sentence that the tool reported an
    error. Anything else -- free text, a table, column or path, an
    undeclared token -- is the plain sentence alone."""
    raw = str(error or "").strip()
    if not raw:
        return ""
    worded = (REASONS.get(language) or REASONS[DEFAULT_LANGUAGE]).get(raw)
    if worded:
        return worded
    if len(raw) <= MAX_CODE_CHARS and raw in tuple(declared or ()):
        return f"{lab['tool_error']} ({lab['code']}: {raw})"
    return lab["tool_error"]


def _what(st: Any, lab: Dict[str, str]) -> str:
    basis = getattr(st, "basis", "")
    if basis not in ("count", "sum"):
        what = str(st.what or "")
        return what[:1].upper() + what[1:]
    head = lab["count"] if basis == "count" else lab["sum"].format(what=_word(lab, st.what))
    terms = tuple(getattr(st, "terms", ()) or ())
    within = tuple(getattr(st, "within", ()) or ())
    if terms:
        head += lab["holds"].format(terms=lab["sep"].join(terms))
    if within:
        head += lab["within"].format(terms=lab["sep"].join(within))
    return head[:1].upper() + head[1:]


def _statement_lines(st: Any, lab: Dict[str, str], language: str = DEFAULT_LANGUAGE,
                     declared: Sequence[str] = ()) -> List[str]:
    if st.figure is None:
        reason = (reason_text(str(getattr(st, "failure", "") or ""), lab, language, declared)
                  or lab["failed"])
        line = f"- {_what(st, lab)}: {lab['not_returned']} ({reason})"
    elif st.count == 0:
        line = f"- {_what(st, lab)}: **0** ({lab['no_rows']})"
    else:
        figure = str(st.count) if st.count is not None else _figure_text(st.figure)
        figure = f"{figure} {st.unit}".strip()
        line = f"- {_what(st, lab)}: **{figure}**"
    if st.source:
        line += f" ({lab['source']}: {st.source})"
    lines = [line]
    if st.rows:
        lines.append(f"  {lab['rows']}: " + lab["list_sep"].join(st.rows))
    if getattr(st, "working", ""):
        lines.append(f"  {lab['working']}: {st.working}")
    return lines


def _typed_by_user(value: Any, question: str) -> bool:
    """True when ``value`` occurs in the user's question (case-folded)."""
    forms = {str(value)}
    if isinstance(value, (int, float)):
        forms.add(_figure_text(float(value)))
    folded = (question or "").casefold()
    return any(f.strip() and f.strip().casefold() in folded for f in forms)


def _asked_line(asked: Dict[str, Any], lab: Dict[str, str], question: str = "") -> str:
    # Only what the USER typed is echoed: an argument the model sent is shown
    # only when its value occurs in the question, so a path, a table, an id
    # or a name the model put in a call never reaches the answer. The scrub
    # stays as a second layer. No value left -> no line.
    values = [scrub(str(v)) for v in (asked or {}).values()
              if isinstance(v, (str, int, float)) and not isinstance(v, bool)
              and _typed_by_user(v, question)]
    values = [v for v in values if v]
    return f"{lab['asked']}: " + lab["sep"].join(values) if values else ""


def render_call(heading: str, statements: Sequence[Any], *, asked: Dict[str, Any],
                error: str, language: str, declared: Sequence[str] = (),
                question: str = "") -> str:
    """One computed call as text. With no statement the call still answers:
    what was asked, and why there is no figure (a known or declared reason
    code, else the plain sentence that the tool reported an error).
    ``declared``: the reason codes the tool's registry entry declares.
    ``question``: the user's question; only argument values it holds are
    echoed on the asked line."""
    lab = labels_for(language)
    lines: List[str] = [f"{heading}:"]
    lines.extend(_body_lines(statements, lab, language, declared))
    if not statements:
        asked_line = _asked_line(asked, lab, question)
        if asked_line:
            lines.append(f"- {asked_line}")
        reason = reason_text(error, lab, language, declared) if error else lab["no_figure"]
        lines.append(f"- {lab['not_returned']}: {reason}")
    return "\n".join(lines)


def _body_lines(statements: Sequence[Any], lab: Dict[str, str], language: str,
                declared: Sequence[str]) -> List[str]:
    """The stated lines of one call: the parts that returned a figure first,
    then the parts that did not (each still named), each group in its order."""
    ordered = ([st for st in statements if st.figure is not None]
               + [st for st in statements if st.figure is None])
    lines: List[str] = []
    for st in ordered:
        lines.extend(_statement_lines(st, lab, language, declared))
    return lines


def heading_for(display_name: str, language: str) -> str:
    """The tool's own name where the question is in the default language;
    otherwise the language's label for a computed result."""
    if language == DEFAULT_LANGUAGE and (display_name or "").strip():
        return display_name.strip()
    return labels_for(language)["heading"]


def join_calls(rendered: Iterable[str], language: str) -> str:
    """Every computed call of the turn; with more than one, a first line says
    how many there are, so none reads as the whole answer by omission."""
    parts = [r for r in rendered if r]
    if len(parts) > 1:
        parts.insert(0, labels_for(language)["several"].format(n=len(parts)))
    return "\n\n".join(parts)


_TOKEN_RE = re.compile(r"\w+(?:[.,]\w+)*")


def selection(asked: Any) -> frozenset:
    """The part of the question a call asks about: the case-folded words and
    numbers of every argument VALUE (keys are the model's names, not the
    user's words), nested lists and objects included, order ignored."""
    if isinstance(asked, dict):
        return frozenset().union(*(selection(v) for v in asked.values()))
    if isinstance(asked, (list, tuple, set, frozenset)):
        return frozenset().union(*(selection(v) for v in asked))
    if asked is None or isinstance(asked, bool):
        return frozenset()
    return frozenset(_TOKEN_RE.findall(str(asked).casefold()))


@dataclass(frozen=True)
class Call:
    """One computed call of the turn, as the answer reads it."""
    tool: str
    heading: str
    statements: Tuple[Any, ...] = ()
    asked: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    declared: Tuple[str, ...] = ()

    @property
    def answered(self) -> bool:
        """True when the call returned at least one figure."""
        return any(st.figure is not None for st in self.statements)

    @property
    def stated(self) -> bool:
        """True when the call says something: a statement or its failure."""
        return bool(self.statements or self.error)


def covers(result: Call, other: Call) -> bool:
    """True when ``result`` answered the part of the question ``other`` asked:
    the same tool, and every word of ``other``'s selection is in ``result``'s
    (the same selection, or the same one asked again with more detail)."""
    return result.tool == other.tool and selection(other.asked) <= selection(result.asked)


def unanswered(calls: Sequence[Call]) -> List[Call]:
    """The calls with no figure whose part no result of the turn answered,
    one per (tool, selection), in call order."""
    results = [c for c in calls if c.answered]
    out: List[Call] = []
    for c in calls:
        if c.answered or any(covers(r, c) for r in results):
            continue
        if any(o.tool == c.tool and selection(o.asked) == selection(c.asked) for o in out):
            continue
        out.append(c)
    return out


def answer_turn(calls: Sequence[Call], language: str, question: str = "") -> str:
    """The one answer of a turn from its computed calls.

    Results lead, in call order; a result identical to one already shown
    (same tool, same stated lines) is not repeated. When one tool shows more
    than one different result, each says what was asked. A call with no
    figure is shown only when no result answered its part (``covers``):
    after the results in one plain line, or -- when no call returned a
    result -- as the answer, saying what was asked and why there is none."""
    lab = labels_for(language)
    shown: List[Tuple[Call, List[str]]] = []
    for c in calls:
        if not c.answered:
            continue
        body = _body_lines(c.statements, lab, language, c.declared)
        if any(o.tool == c.tool and ob == body for o, ob in shown):
            continue
        shown.append((c, body))
    missing = unanswered(calls)
    if not shown:
        return "\n\n".join(render_call(c.heading, c.statements, asked=c.asked, error=c.error,
                                        language=language, declared=c.declared,
                                        question=question) for c in missing)
    per_tool: Dict[str, int] = {}
    for c, _ in shown:
        per_tool[c.tool] = per_tool.get(c.tool, 0) + 1
    blocks: List[str] = []
    for c, body in shown:
        head = [f"{c.heading}:"]
        asked_line = _asked_line(c.asked, lab, question) if per_tool[c.tool] > 1 else ""
        if asked_line:
            head.append(f"- {asked_line}")
        blocks.append("\n".join(head + body))
    if len(blocks) > 1:
        blocks.insert(0, lab["several"].format(n=len(blocks)))
    if missing:
        parts = []
        for c in missing:
            asked_line = _asked_line(c.asked, lab, question)
            reason = reason_text(c.error, lab, language, c.declared) if c.error else lab["no_figure"]
            what = f"{c.heading} ({asked_line})" if asked_line else c.heading
            parts.append(f"{what}, {lab['not_returned']}: {reason}")
        blocks.append(lab["unanswered"].format(parts=lab["list_sep"].join(parts)))
    return "\n\n".join(blocks)


def tool_call_args(messages: Sequence[Dict[str, Any]], call_id: str) -> Dict[str, Any]:
    """The arguments the model sent with tool call ``call_id``, or {}."""
    import json

    if not call_id:
        return {}
    for m in messages or []:
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            if not isinstance(tc, dict) or tc.get("id") != call_id:
                continue
            raw = (tc.get("function") or {}).get("arguments")
            if isinstance(raw, dict):
                return raw
            try:
                parsed = json.loads(raw or "{}")
            except (TypeError, ValueError):
                _LOG.info("tool call %s: arguments are not JSON", call_id, exc_info=True)
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def failure_text(payload: Optional[Dict[str, Any]]) -> str:
    """The plain reason a failed result gives: its ``error`` or ``reason``."""
    for key in ("error", "reason"):
        val = (payload or {}).get(key)
        if isinstance(val, str) and val.strip():
            # The code or text as given; reason_text decides what is shown.
            return re.sub(r"[{}\[\]]", "", val).strip()[:2000]
    return ""


__all__: Tuple[str, ...] = (
    "LABELS", "REASONS", "has_words", "question_language", "render_call", "heading_for", "join_calls",
    "Call", "selection", "covers", "unanswered", "answer_turn",
    "tool_call_args", "failure_text", "scrub", "reason_text", "MAX_CODE_CHARS",
)
