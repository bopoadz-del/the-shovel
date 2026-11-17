"""The answer a computed tool's result states, in the language of the question.

When the model's synthesis is missing, a tool declared ``kind="computed"``
answers with its own statements (``tool_registry.Statement``). This module
turns those statements -- and a computed call that FAILED or found NO
matching rows -- into plain text:

* every computed call of the turn is shown, in call order (none is dropped);
* a part that did not return (one bill of several) is NAMED, never skipped,
  so a partial result is never shown as complete;
* a call that failed, or stated nothing, says what was asked and why there is
  no figure, instead of the generic "unable to generate" sentence;
* every label and every connecting word comes from ``LABELS``, keyed by the
  language of the user's question. Figures, units, sources, row labels and the
  words the user asked about are data and are shown as they are.
"""
from __future__ import annotations

import logging
import re
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
        "no_derived_bill_sheet": "no bill sheet has been derived for this project",
        "derived_sheet_has_no_readable_bytes": "the bill sheet file could not be read",
        "derived_sheet_is_not_a_readable_workbook": "the bill sheet is not a readable workbook",
        "tool_error": "the tool reported an error",
        "code": "code",
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
        "no_derived_bill_sheet": "لم يُستخرج جدول كميات لهذا المشروع",
        "derived_sheet_has_no_readable_bytes": "تعذّرت قراءة ملف جدول الكميات",
        "derived_sheet_is_not_a_readable_workbook": "ملف جدول الكميات ليس مصنفًا مقروءًا",
        "tool_error": "أبلغت الأداة عن خطأ",
        "code": "الرمز",
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


#: What must never reach a user from a tool's error or its echoed arguments:
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
_CODE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def scrub(text: str) -> str:
    """``text`` without a stack trace, file path, table name or internal id."""
    out = _TRACE_RE.sub("", str(text or ""))
    for rx in (_PATH_RE, _TABLE_RE, _DOTTED_RE, _ID_RE):
        out = rx.sub("", out)
    out = re.sub(r"[{}\[\]]", "", out)
    out = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"\(\s*\)", "", out))
    return re.sub(r"\s{2,}", " ", out).strip(" ,;:-\n")


def reason_text(error: str, lab: Dict[str, str], language: str) -> str:
    """A tool's error as the user reads it, in the question's language.

    A reason code LABELS knows is translated. Another code is kept, after a
    plain sentence in the question's language that the tool reported an
    error. Free text is shown scrubbed where the question is in the language
    the tools write in; in another language it is never pasted raw: the
    plain sentence stands in its place."""
    raw = str(error or "").strip()
    if not raw:
        return ""
    if raw in lab:
        return lab[raw]
    if _CODE_RE.fullmatch(raw):
        return f"{lab['tool_error']} ({lab['code']}: {raw})"
    if language == DEFAULT_LANGUAGE:
        return scrub(raw)[:180] or lab["tool_error"]
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


def _statement_lines(st: Any, lab: Dict[str, str], language: str = DEFAULT_LANGUAGE) -> List[str]:
    if st.figure is None:
        reason = reason_text(str(getattr(st, "failure", "") or ""), lab, language) or lab["failed"]
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


def _asked_line(asked: Dict[str, Any], lab: Dict[str, str]) -> str:
    # Echoed arguments are scrubbed like an error: an id, path or table name
    # the model passed is never shown back.
    values = [scrub(str(v)) for v in (asked or {}).values()
              if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
    values = [v for v in values if v]
    return f"{lab['asked']}: " + lab["sep"].join(values) if values else ""


def render_call(heading: str, statements: Sequence[Any], *, asked: Dict[str, Any],
                error: str, language: str) -> str:
    """One computed call as text. With no statement the call still answers:
    what was asked, and the tool's error (or that it returned no figure)."""
    lab = labels_for(language)
    lines: List[str] = [f"{heading}:"]
    for st in statements:
        lines.extend(_statement_lines(st, lab, language))
    if not statements:
        asked_line = _asked_line(asked, lab)
        if asked_line:
            lines.append(f"- {asked_line}")
        reason = reason_text(error, lab, language) if error else lab["no_figure"]
        lines.append(f"- {lab['not_returned']}: {reason}")
    return "\n".join(lines)


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
            # The code or text as given; reason_text words it and scrubs it.
            return re.sub(r"[{}\[\]]", "", val).strip()[:2000]
    return ""


__all__: Tuple[str, ...] = (
    "LABELS", "has_words", "question_language", "render_call", "heading_for", "join_calls",
    "tool_call_args", "failure_text", "scrub", "reason_text",
)
