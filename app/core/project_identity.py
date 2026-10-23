"""A project's identity, and the one rule that keeps it out of every output.

Order 2210Z (LANE4 name flag) and 0735Z item 3a: a project's real name, its
short forms and initials, the codes in its record and in its own document
file names, and those file names themselves never leave the platform in an
answer, a citation, a Sources line, an upload or listing response. A project
document is shown by its neutral display title -- its type and a short id
("Specification abc123"); library documents keep their real titles.

Nothing here is a list of words. What is "generic" -- a word or code the
platform itself uses, so never a project's identity -- comes from the
platform's own data: the formula, hat and unit registries, the currency
table, and the library corpus (the general-knowledge projects' passages).

The set is derived from the project's record and documents when they change
(``refresh``) and stored with the project (``projects.identity``). A turn only
reads it (``for_project``), off the event loop. If it cannot be read, the
redactor fails closed: every file-name-shaped word and every code-shaped word
the platform does not use is replaced.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

_LOG = logging.getLogger(__name__)

#: A code: 2-8 capitals and digits with at least two letters ("HTD", "AB12").
_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(?=(?:[0-9]*[A-Z]){2})[A-Z][A-Z0-9]{1,7}(?![A-Za-z0-9])")
#: A file name as written in text: a word with an extension ("x_BOQ_rev3.xlsx").
_FILE_RE = re.compile(r"(?<![\w/.-])[\w][\w .()&+,'-]*?\.(?:[A-Za-z][A-Za-z0-9]{1,4})(?![\w])")
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")
_SEP_RE = re.compile(r"[-_.\s]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_MIN_LEN = 3


def neutral_id(project_id: str) -> str:
    """What a project identity member becomes: ``PRJ-<first six of the id>``."""
    short = re.sub(r"[^0-9A-Za-z]", "", project_id or "")[:6]
    return f"PRJ-{short}" if short else "PRJ"


def display_title(doc_id: str, doc_type: str = "") -> str:
    """A project document as the user reads it: its type and a short id."""
    kind = re.sub(r"[_\-]+", " ", (doc_type or "").strip()).strip().capitalize() or "Document"
    short = re.sub(r"[^0-9A-Za-z]", "", doc_id or "")[:6]
    return f"{kind} {short}".strip() if short else kind


# -- what the platform itself uses (data, never a list) ---------------------

_REGISTRY_WORDS: set[str] | None = None
_LIBRARY_SEEN: dict[str, bool] = {}


def _registry_words() -> set[str]:
    """Every word and code the platform's registries write, lower-cased:
    formula names, labels, descriptions and units; hat texts; currencies."""
    global _REGISTRY_WORDS
    if _REGISTRY_WORDS is None:
        texts: list[str] = []
        try:
            from app.lib import formula_registry

            for spec in formula_registry.all_specs():
                texts += [spec.name, spec.description, spec.display_name,
                          *map(str, spec.outputs.values()), *map(str, spec.inputs.values())]
                for key, p in formula_registry.parameters(spec).items():
                    texts += [key, p.label or "", p.unit or ""]
        except Exception:  # noqa: BLE001 -- a missing source narrows the vocabulary
            _LOG.warning("project_identity: formula registry unavailable", exc_info=True)
        try:
            from app.agents import catalog

            texts += [f"{h.name} {h.description} {h.system_prompt_guidance or ''}" for h in catalog.list_hats()]
        except Exception:  # noqa: BLE001
            _LOG.warning("project_identity: hat catalogue unavailable", exc_info=True)
        words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", " ".join(texts))}
        try:
            from app.lib.construction_formulas import _CURRENCY_CODES

            words |= {c.lower() for c in _CURRENCY_CODES}
        except Exception:  # noqa: BLE001
            _LOG.warning("project_identity: currency table unavailable", exc_info=True)
        _REGISTRY_WORDS = words
    return _REGISTRY_WORDS


def _in_library(word: str) -> bool:
    """True when ``word`` is a whole word of the library corpus. Cached."""
    key = word.lower()
    if key in _LIBRARY_SEEN:
        return _LIBRARY_SEEN[key]
    found = False
    try:
        from sqlalchemy import select

        from app.core import projects
        from app.core.models import RagChunk
        from app.core.rag.retriever import _general_knowledge_project_ids

        ids = [i for i in _general_knowledge_project_ids() if i]
        if ids:
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            with projects.SessionLocal() as session:
                rows = session.execute(
                    select(RagChunk.text).where(RagChunk.project_id.in_(ids),
                                                RagChunk.text.ilike(f"%{word}%")).limit(20)
                ).scalars().all()
            found = any(pattern.search(t or "") for t in rows)
    except Exception:  # noqa: BLE001 -- unknown is "not generic": the word stays an identity member
        _LOG.warning("project_identity: library lookup failed for %r", word, exc_info=True)
    _LIBRARY_SEEN[key] = found
    return found


def is_generic(word: str) -> bool:
    """A word or code the platform itself uses -- never a project's identity."""
    return word.lower() in _registry_words() or _in_library(word)


# -- deriving the set (on change, never per turn) ---------------------------

def _distinctive(value: str) -> bool:
    """True when ``value`` carries a capitalised word or a code the platform
    does not use -- a proper name, not a description ("FIDIC Red Book" is
    generic; an unknown estate name is not)."""
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", value):
        if (token[0].isupper() or _CODE_RE.fullmatch(token)) and not is_generic(token):
            return True
    return False


def _short_form(value: str) -> str:
    """``value`` without its trailing words the platform uses ("Harbour Tower
    Development Phase 2" -> the part before the generic tail)."""
    words = value.split()
    while words and (is_generic(re.sub(r"[^A-Za-z0-9]", "", words[-1]) or "x") or not re.search(r"[A-Za-z]", words[-1])):
        words.pop()
    return " ".join(words).strip(" -,")


def _initials(value: str) -> str:
    """The initials of the capitalised words of a multi-word name, three or
    more letters ("Harbour Tower Development" -> "HTD"). Lower-case function
    words ("of", "the") are not capitalised, so no word list is needed."""
    caps = [w for w in _WORD_RE.findall(value) if w[0].isupper()]
    letters = "".join(w[0] for w in caps)
    return letters.upper() if len(caps) >= 2 and len(letters) >= _MIN_LEN else ""


def _document_codes(names: Iterable[str]) -> set[str]:
    """The code leading the project's own file names ("ABC-ST-0042.pdf" ->
    "ABC"), when the platform does not use it."""
    leads = set()
    for name in names:
        stem = _EXT_RE.sub("", (name or "").replace(chr(92), "/").split("/")[-1])
        lead = _SEP_RE.split(stem)[0] if stem else ""
        if _CODE_RE.fullmatch(lead or "") and not is_generic(lead):
            leads.add(lead)
    return leads


def derive(project: dict[str, Any], document_names: Iterable[str] = ()) -> list[str]:
    """The identity members the project's record and documents give."""
    names = list(document_names)
    values = [str(project.get(k) or "").strip() for k in ("name", "client", "location")]
    profile = project.get("profile") if isinstance(project.get("profile"), dict) else {}
    values += [str(v).strip() for v in (profile or {}).values() if isinstance(v, (str, int)) and str(v).strip()]
    members: set[str] = set()
    for value in values:
        if len(value) < _MIN_LEN or not _distinctive(value):
            continue
        members.add(value)
        short = _short_form(value)
        if short and short != value and len(short) >= _MIN_LEN and _distinctive(short):
            members.add(short)
        for form in {value, short}:
            initials = _initials(form) if form else ""
            if initials and not is_generic(initials):
                members.add(initials)
        members.update(c for c in _CODE_RE.findall(value) if not is_generic(c))
    members |= _document_codes(names)
    return sorted(m for m in members if len(m) >= 2)


# -- the redactor -------------------------------------------------------------

class Redactor:
    """One project's rule: identity members -> the neutral id; its document
    file names -> their display titles. Whole words only."""

    def __init__(self, project_id: str, members: Iterable[str] = (),
                 documents: Iterable[dict[str, Any]] = (), *, closed: bool = False) -> None:
        self.project_id = project_id
        self.closed = closed
        self.replacement = neutral_id(project_id)
        self.titles: dict[str, str] = {}
        for doc in documents:
            doc_id = str(doc.get("id") or doc.get("document_id") or "").strip()
            name = str(doc.get("original_name") or doc.get("filename") or "").strip()
            if not doc_id or not name:
                continue
            shown = display_title(doc_id, str(doc.get("doc_type") or ""))
            base = name.replace(chr(92), "/").split("/")[-1].strip()
            stem = _EXT_RE.sub("", base)
            for variant in {name, base, stem, _SEP_RE.sub(" ", stem).strip()}:
                if len(variant) >= 4:
                    self.titles[variant] = shown
        names = sorted(self.titles, key=len, reverse=True)
        self._title_re = (re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, names)) + r")(?![\w])")
                          if names else None)
        self.members = sorted({m for m in members if m and len(m) >= 2}, key=len, reverse=True)
        parts = []
        for member in self.members:
            body = re.escape(member).replace(r"\ ", r"[\s_-]+")
            parts.append(body if _CODE_RE.fullmatch(member) else f"(?i:{body})")
        self._member_re = (re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])")
                           if parts else None)
        #: The longest text one replacement can span: a stream holds back
        #: this many characters so a name split across chunks is still caught.
        self.span = max([len(n) for n in names] + [len(m) for m in self.members] + [64])

    def text(self, value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        out = value
        if self._title_re is not None:
            out = self._title_re.sub(lambda m: self.titles[m.group(0)], out)
        if self._member_re is not None:
            out = self._member_re.sub(self.replacement, out)
        if self.closed:
            out = _FILE_RE.sub("Document", out)
            out = _CODE_RE.sub(lambda m: m.group(0) if is_generic(m.group(0)) else self.replacement, out)
        return out

    def deep(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.deep(v) for v in value]
        if isinstance(value, dict):
            return {k: self.deep(v) for k, v in value.items()}
        return value


# -- stored set: written on change, read per turn ---------------------------

def refresh(project_id: str) -> list[str]:
    """Derive the set from the project's record and documents and store it.
    Called when the record, profile or documents change -- never per turn."""
    from app.core import projects
    from app.core.models import Project

    project = projects.get_project(project_id) or {}
    if not project:
        return []
    docs = projects.list_document_names(project_id) or []
    derived = derive(project, [str(d.get("original_name") or "") for d in docs])
    with projects.SessionLocal() as session:
        row = session.get(Project, project_id)
        if row is not None:
            stored = _parse(getattr(row, "identity", None))
            added = [str(m) for m in stored.get("added") or [] if str(m).strip()]
            value = json.dumps({"derived": derived, "added": added})
            if getattr(row, "identity", None) != value:
                row.identity = value
                session.commit()
    return derived


def _parse(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        _LOG.warning("project_identity: stored identity set is not JSON; treated as empty")
        value = {}
    return value if isinstance(value, dict) else {}


def for_project(project_id: str | None) -> Redactor | None:
    """The project's redactor from its stored set: one read, no derivation.
    None when no project is in scope. If the set cannot be read, a closed
    redactor -- never one that lets the identity through."""
    if not project_id:
        return None
    try:
        from sqlalchemy import select

        from app.core import projects
        from app.core.models import Project

        with projects.SessionLocal() as session:
            raw = session.execute(select(Project.identity).where(Project.id == project_id)).scalar_one_or_none()
        stored = _parse(raw)
        if not stored or stored.get("stale") or "derived" not in stored:
            # Changed since it was last derived (or never derived): derive
            # once now and store it; later turns only read.
            refresh(project_id)
            with projects.SessionLocal() as session:
                raw = session.execute(select(Project.identity).where(Project.id == project_id)).scalar_one_or_none()
            stored = _parse(raw)
        docs = projects.list_document_names(project_id) or []
        members = [str(m) for m in (stored.get("derived") or []) + (stored.get("added") or [])]
        return Redactor(project_id, members, docs)
    except Exception:  # noqa: BLE001 -- unreadable set: fail closed, never open
        _LOG.error("project_identity: identity set unreadable for %s; redacting closed", project_id, exc_info=True)
        return Redactor(project_id, closed=True)


_CURRENT: contextvars.ContextVar[Redactor | None] = contextvars.ContextVar("project_identity", default=None)


def set_current(redactor: Redactor | None) -> contextvars.Token:
    return _CURRENT.set(redactor)


def reset(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def current() -> Redactor | None:
    return _CURRENT.get()
