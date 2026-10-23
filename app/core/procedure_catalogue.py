"""Procedure catalogue: generic procedure KINDS, live document codes at run time.

The shipped catalogue (``app/data/procedures/procedures_db.json``) describes
procedure kinds -- change management, design review, inspection request, HSE
audit ... -- keyed by a neutral kind id, with generic rules, roles and
workflows and the generic title phrases (``match_phrases``) a document of
that kind carries in its name. It ships no organisation's document codes or
document titles.

An organisation's own procedure documents live in its corpus, named the way
it names them ("XYZ-101_Change Control.pdf"). Which document implements a
kind, and what code it carries, is resolved here at RUN TIME from the
document names of the active project -- never written into the product:

* ``kind_for_name(name)``      -> the kind a document name describes, or None
* ``document_code(name)``      -> the leading controlled-document code, or None
* ``resolve_procedure(kind, names)`` -> the ``procedure_id`` to report: the
  live document's own code when one is found, else the kind id
* ``expand_codes(message, names)``   -> a message where a bare code the user
  typed is followed by its kind's label, so keyword routing sees the kind
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "data" / "procedures" / "procedures_db.json"

#: A controlled company document code at the START of a file name:
#: letters, a separator, three or four digits, an optional letter suffix,
#: then a separator before the title ("XYZ-101_Change Control.pdf",
#: "ab_240 Design Review.docx"). A contract id like "AB-2001-101" is not one:
#: its digits are followed by another hyphenated group, not by a title; a
#: year ("FIDIC 2017 ...") is not a document number either.
_LEADING_CODE_RE = re.compile(
    r"^\s*([A-Za-z]{2,6}[-_ ](?!(?:19|20)\d\d(?!\d))\d{3,4}[A-Za-z]?)(?=[-_ .]+[A-Za-z(]|\.[A-Za-z0-9]+$|$)"
)
#: The same code shape anywhere in a user's message.
_CODE_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,6}-\d{3,4}[A-Za-z]?)(?![A-Za-z0-9-])")


@lru_cache(maxsize=1)
def _catalogue() -> Dict[str, Any]:
    try:
        return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("procedure catalogue unreadable at %s", CATALOGUE_PATH, exc_info=True)
        return {}


def procedures() -> Dict[str, Dict[str, Any]]:
    """``{kind: record}`` from the shipped catalogue."""
    return dict(_catalogue().get("procedures") or {})


def get_kind(kind: str) -> Optional[Dict[str, Any]]:
    return procedures().get(kind or "")


def label(kind: str) -> str:
    """Human label for a kind ("change management procedure"); the id if unknown."""
    rec = get_kind(kind) or {}
    return str(rec.get("label") or (kind or "").replace("_", " "))


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9&]+", " ", (text or "").lower()).strip() + " "


@lru_cache(maxsize=256)
def _phrase_re(phrase: str) -> "re.Pattern[str]":
    """Phrase words in order, each tolerating a plural ``s`` ("design reviews")."""
    words = _norm(phrase).split()
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) + "s?" for w in words) + r"(?![a-z0-9])")


def kind_for_name(name: str) -> Optional[str]:
    """The procedure kind a document name describes, by its title phrases.

    The longest matching phrase wins, so "Design Directive" is not read as
    the broader "design review" family and "HSE Audit" beats "audit".
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", (name or "").strip())
    code = document_code(stem)
    title = _norm(stem[len(code):] if code else stem)
    best: Optional[str] = None
    best_len = 0
    for kind, rec in procedures().items():
        for phrase in rec.get("match_phrases") or []:
            p = _norm(phrase).strip()
            if p and len(p) > best_len and _phrase_re(p).search(title):
                best, best_len = kind, len(p)
    return best


def document_code(name: str) -> Optional[str]:
    """Leading controlled-document code of a name, upper-cased with ``-``."""
    m = _LEADING_CODE_RE.match(name or "")
    if not m:
        return None
    return re.sub(r"[-_ ]", "-", m.group(1), count=1).upper()


def resolve_procedure(kind: str, document_names: Optional[Iterable[str]] = None) -> Dict[str, Optional[str]]:
    """``procedure_id`` for ``kind``: a live document's code, else the kind id.

    Only a document that both describes this kind and carries a leading
    code lends its code; the first such name (in the order given) wins.
    """
    for name in document_names or ():
        if kind_for_name(name) == kind:
            code = document_code(name)
            if code:
                return {"procedure_id": code, "procedure_document": name, "procedure_kind": kind}
    return {"procedure_id": kind, "procedure_document": None, "procedure_kind": kind}


def kind_for_code(code: str, document_names: Optional[Iterable[str]] = None) -> Optional[str]:
    """The kind of the live document whose leading code is ``code``."""
    want = (code or "").upper().replace("_", "-").replace(" ", "-")
    for name in document_names or ():
        if document_code(name) == want:
            kind = kind_for_name(name)
            if kind:
                return kind
    return None


def has_code(message: str) -> bool:
    """True when the message contains a controlled-document-code shape."""
    return bool(_CODE_IN_TEXT_RE.search(message or ""))


def expand_codes(message: str, document_names: Optional[Iterable[str]] = None) -> str:
    """Append each typed code's kind label so keyword routing sees the kind."""
    names = list(document_names or ())
    if not message or not names:
        return message
    extra: List[str] = []
    for code in _CODE_IN_TEXT_RE.findall(message):
        kind = kind_for_code(code, names)
        if kind:
            extra.append(label(kind))
    return f"{message} ({'; '.join(dict.fromkeys(extra))})" if extra else message


def live_document_names(project_id: Optional[str]) -> List[str]:
    """Names of the project's documents, read at run time ([] on any failure)."""
    if not project_id:
        return []
    try:
        from app.core.projects import list_documents

        return [d.get("original_name") or "" for d in list_documents(project_id)]
    except Exception:  # noqa: BLE001 -- resolution is advisory; the kind id still answers
        logger.warning("procedure resolution: could not list documents for %s", project_id, exc_info=True)
        return []
