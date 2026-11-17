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
    },
    "ar": {
        "heading": "نتيجة محسوبة",
        "count": "عدد البنود",
        "sum": "مجموع {what} للبنود",
        "holds": " التي يتضمن وصفها {terms}",
        "within": " في القسم {terms}",
        "source": "المصدر",
        "rows": "البنود",
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
    },
}

DEFAULT_LANGUAGE = "en"

#: The letters of each script a LABELS language is written in.
_SCRIPT_LETTERS: Dict[str, "re.Pattern[str]"] = {
    "ar": re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"),
    "en": re.compile(r"[A-Za-z]"),
}


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


def _statement_lines(st: Any, lab: Dict[str, str]) -> List[str]:
    if st.figure is None:
        reason = _word(lab, str(getattr(st, "failure", "") or "")) or lab["failed"]
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
    return lines


def _asked_line(asked: Dict[str, Any], lab: Dict[str, str]) -> str:
    values = [str(v).strip() for v in (asked or {}).values()
              if isinstance(v, (str, int, float)) and str(v).strip()]
    return f"{lab['asked']}: " + lab["sep"].join(values) if values else ""


def render_call(heading: str, statements: Sequence[Any], *, asked: Dict[str, Any],
                error: str, language: str) -> str:
    """One computed call as text. With no statement the call still answers:
    what was asked, and the tool's error (or that it returned no figure)."""
    lab = labels_for(language)
    lines: List[str] = [f"{heading}:"]
    for st in statements:
        lines.extend(_statement_lines(st, lab))
    if not statements:
        asked_line = _asked_line(asked, lab)
        if asked_line:
            lines.append(f"- {asked_line}")
        reason = _word(lab, error) if error else lab["no_figure"]
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
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def failure_text(payload: Optional[Dict[str, Any]]) -> str:
    """The plain reason a failed result gives: its ``error`` or ``reason``."""
    for key in ("error", "reason"):
        val = (payload or {}).get(key)
        if isinstance(val, str) and val.strip():
            return re.sub(r"[{}\[\]]", "", val).strip()[:180]
    return ""


__all__: Tuple[str, ...] = (
    "LABELS", "question_language", "render_call", "heading_for", "join_calls",
    "tool_call_args", "failure_text",
)
