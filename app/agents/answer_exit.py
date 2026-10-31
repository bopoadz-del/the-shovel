"""The exit check: nothing internal leaves in an answer.

Every answer a user reads, streamed, returned or reopened from history,
passes through :func:`check_text` last, and its Sources rows through
:func:`check_sources`. The rules are structural; none names a project, a
document or a question.

* A citation is resolved to the document's name and page, or removed. A
  citation is any span carrying a retrieval key (``doc_id``, ``chunk``,
  ``src=``, ``score=``), bracketed or not, closed or not, and any
  ``Source:`` mention that carries one. The turn's evidence (the retrieval
  markers the model was given, the passages tools returned, the Sources rows)
  says which document and page a chunk number or document id stands for.
  A citation nothing resolves is taken out.
* Chunk numbers and document ids never appear. A Sources row shows the page
  (``p. N``) or nothing; a row with no document name is not shown.
* A registered formula, tool or parameter id reads as its display name
  (``source_labels.plain_registry_text``). A ``key=value`` whose key is not
  registered reads as words; one whose value is an empty or literal dump
  (``{}``, ``[]``, ``true``, ``null``) goes. Any other code identifier
  (``snake_case``, ``dotted.name``) reads as words.
* A tool's error text copied into the answer is replaced by one plain
  sentence naming the step, when that text carries an internal token. An
  ``error:`` label or an exception name is replaced the same way.
* Refusal wording that names the platform's machinery (retrieved excerpts,
  provided chunks, injected context, tool calls, supplied parameters) reads
  in the user's terms.

Never raises: a check that can break an answer is a check that gets
switched off.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)

#: Keys the retrieval marker (``app.core.rag.inject``) writes on every
#: excerpt. ``tests/test_answer_exit_registry.py`` checks this against the
#: marker the injector actually builds, so a new key cannot slip past.
MARKER_KEYS: tuple[str, ...] = ("doc_id", "chunk", "score", "class", "layer", "page", "rev", "src")

#: Stands where text was removed until the line is tidied.
_MARK = "\x00"


# ── The turn's evidence ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Place:
    """One place in one document the turn read."""

    name: str
    doc_id: str = ""
    chunk: int | None = None
    page: int | None = None


@dataclass
class Turn:
    """What this turn read and which tools failed, for the exit check."""

    places: list[Place] = field(default_factory=list)
    #: (step label, error text) for every tool that returned an error.
    tool_errors: list[tuple[str, str]] = field(default_factory=list)
    #: Tool, block, action and operation names this turn's calls used.
    step_names: set[str] = field(default_factory=set)
    #: A tool's question for inputs it needs and did not get, in the user's words.
    questions: list[str] = field(default_factory=list)
    #: The question asked in place of the error text that came with it.
    error_questions: dict[str, str] = field(default_factory=dict)


_TURN: contextvars.ContextVar[Turn | None] = contextvars.ContextVar("answer_exit_turn", default=None)


def open_turn() -> Turn:
    """Start a turn's record. Tasks and threads started afterwards share it."""
    turn = Turn()
    _TURN.set(turn)
    return turn


def current_turn() -> Turn | None:
    return _TURN.get()


def reset_turn(token: contextvars.Token) -> None:
    _TURN.reset(token)


def begin_turn() -> tuple[Turn, contextvars.Token]:
    """``open_turn`` for a caller that restores the previous record afterwards."""
    turn = Turn()
    return turn, _TURN.set(turn)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def _page(value: Any) -> int | None:
    page = _as_int(value)
    return page if page is not None and page > 0 else None


def _chunk(value: Any) -> int | None:
    return _as_int(value)


def _step_label(tool: str, payload: Any) -> str:
    """The step's display name, or '' (an internal name is never a label)."""
    action = payload.get("action") if isinstance(payload, dict) else None
    from app.lib.source_labels import formula_display_name, tool_display_name

    for name in (action, tool):
        if not isinstance(name, str) or not name.strip():
            continue
        shown = tool_display_name(name) or formula_display_name(name)
        if shown:
            return shown
    return ""


#: Argument keys whose value names a step (an action, an operation, a block).
_NAME_KEYS = ("action", "operation", "op", "tool", "block", "block_name", "mode", "calculation")
_NAME_VALUE_RE = re.compile(r"[a-z][a-z0-9]*(?:[_.][a-z0-9]+)*")


def _call_names(tool: str, args: Any) -> set[str]:
    names = {tool.strip().lower()} if isinstance(tool, str) and tool.strip() else set()
    if isinstance(args, dict):
        for key in _NAME_KEYS:
            value = args.get(key)
            if isinstance(value, str) and _NAME_VALUE_RE.fullmatch(value.strip().lower()):
                names.add(value.strip().lower())
    return names


def _error_step_names(err: str) -> set[str]:
    """Code names in a tool's error text that are not parameters: the
    operations and actions it lists."""
    from app.lib.source_labels import _registry_maps

    _displays, params, _units = _registry_maps()
    return {t.lower() for t in re.findall(rf"(?<![\w.]){_IDENT}(?![\w])", err or "")
            if t.lower() not in params and _is_word_ident(t.lower(), dotted=True)}


def _error_texts(payload: Any) -> list[str]:
    """Error strings an error envelope carries, at any depth."""
    found: list[str] = []
    if isinstance(payload, dict):
        failed = payload.get("status") in ("error", "needs_input") or payload.get("ok") is False
        err = payload.get("error")
        if isinstance(err, str) and err.strip() and (failed or "status" not in payload):
            found.append(err.strip())
        elif isinstance(err, dict):
            found.extend(_error_texts({"status": "error", **err}))
        for key in ("result", "data"):
            if isinstance(payload.get(key), dict):
                found.extend(_error_texts(payload[key]))
    return found


def _input_questions(payload: Any) -> list[tuple[str, str]]:
    """(question, the error text beside it) for every envelope that asks for
    inputs it needs, at any depth."""
    found: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        question = payload.get("question")
        if (isinstance(question, str) and question.strip()
                and (payload.get("missing") or payload.get("needs_input"))):
            err = payload.get("error")
            found.append((question.strip(), err.strip() if isinstance(err, str) else ""))
        for key in ("result", "data"):
            found.extend(_input_questions(payload.get(key)))
    return found


def note_tool_result(tool: str, payload: Any, turn: Turn | None = None) -> None:
    """Record a tool's error text so a copy of it in the answer is recognised,
    and its question for missing inputs."""
    turn = turn or _TURN.get()
    if turn is None:
        return
    try:
        for question, err in _input_questions(payload):
            if question not in turn.questions:
                turn.questions.append(question)
            if err and err != question:
                turn.error_questions.setdefault(err, question)
        turn.step_names.update(_call_names(tool, payload))
        for err in _error_texts(payload):
            turn.step_names.update(_error_step_names(err))
            label = _step_label(tool, payload)
            if (label, err) not in turn.tool_errors:
                turn.tool_errors.append((label, err))
    except Exception:  # noqa: BLE001 -- bookkeeping must never break a tool
        _LOG.warning("answer_exit: could not record a tool error", exc_info=True)


def note_evidence(
    rag_sys_msg: dict[str, Any] | None,
    messages: Iterable[dict[str, Any]] | None,
    turn: Turn | None = None,
) -> None:
    """Record the places this turn read (retrieval markers, tool passages) and
    the tools that failed (tool messages)."""
    turn = turn or _TURN.get()
    if turn is None:
        return
    try:
        from app.agents import citation_provenance as cp

        msgs = list(messages or [])
        evidence = cp.build_evidence(rag_sys_msg, msgs, tool_passages=True)
        for rec in evidence.records:
            if rec.kind != "retrieval" or not (rec.source_name or rec.doc_id):
                continue
            name = rec.source_name if rec.source_name != rec.doc_id else ""
            place = Place(name=_display_name(name), doc_id=str(rec.doc_id or ""),
                          chunk=rec.chunk_index, page=_page(rec.page))
            if place not in turn.places:
                turn.places.append(place)
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                for call in m.get("tool_calls") or []:
                    fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                    raw = fn.get("arguments")
                    try:
                        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                    except (TypeError, ValueError):
                        args = {}
                    turn.step_names.update(_call_names(str(fn.get("name") or ""), args))
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            try:
                payload = json.loads(m.get("content") or "")
            except (TypeError, ValueError):
                continue
            note_tool_result(str(m.get("name") or ""), payload, turn)
    except Exception:  # noqa: BLE001 -- bookkeeping must never break an answer
        _LOG.warning("answer_exit: could not record the turn's evidence", exc_info=True)


def places_from_sources(rows: Iterable[dict[str, Any]] | None) -> list[Place]:
    places: list[Place] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = _display_name(str(row.get("doc_name") or ""))
        doc_id = str(row.get("doc_id") or "")
        if not (name or doc_id):
            continue
        places.append(Place(name=name, doc_id=doc_id, chunk=_chunk(row.get("chunk_index")),
                            page=_page(row.get("page"))))
    return places


# ── Resolving a citation ─────────────────────────────────────────────────────

_HEX_ID = r"[0-9a-f]{6,}(?:-[0-9a-f]{2,}){0,4}"
_UUID_RE = re.compile(r"(?<![\w-])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![\w-])",
                      re.IGNORECASE)
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{2,5}$")
#: A hex token at least this long is an id (stored ids carry 32).
_ID_LENGTH = 24


def _display_name(name: str) -> str:
    """A document name as a user reads it: the base name, no path."""
    name = (name or "").strip().strip("\"'“”‘’`")
    if not name:
        return ""
    parts = [p for p in name.replace("\\", "/").split("/") if p.strip()]
    return (parts[-1] if parts else name).strip()


def _name_key(name: str) -> str:
    low = _display_name(name).lower()
    low = re.sub(r"^(?:…|\.\.\.)\s*", "", low)
    low = re.sub(r"\s*(?:…|\.\.\.)$", "", low)
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        low = low.replace(dash, "-")
    low = _EXT_RE.sub("", low)
    return re.sub(r"[\s_]+", " ", low).strip(" .,;:-")


def _names_match(a: str, b: str) -> bool:
    ka, kb = _name_key(a), _name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    shorter, longer = sorted((ka, kb), key=len)
    return len(shorter) >= 5 and shorter in longer


def _looks_like_id(token: str) -> bool:
    token = (token or "").strip()
    if _UUID_RE.fullmatch(token):
        return True
    return bool(re.fullmatch(_HEX_ID, token, re.IGNORECASE)) and bool(re.search(r"\d", token))


def _document_name(doc_id: str) -> str:
    """The stored name of a document id, or ''."""
    if not doc_id:
        return ""
    try:
        from app.core import projects

        doc = projects.get_document(doc_id) or {}
    except Exception:  # noqa: BLE001 -- a lookup failure only means "unknown"
        _LOG.debug("answer_exit: document lookup failed for %s", doc_id, exc_info=True)
        return ""
    return _display_name(str(doc.get("original_name") or ""))


class CitationIndex:
    """Which document and page a cited chunk number or document id stands for."""

    def __init__(self, places: Iterable[Place] = (), *, lookup: bool = True) -> None:
        self.places = [p for p in places if p.name or p.doc_id]
        self._lookup = lookup
        self._looked_up: dict[str, str] = {}

    def _by_doc(self, doc: str) -> list[Place]:
        doc = doc.lower()
        out = []
        for p in self.places:
            pid = p.doc_id.lower()
            if pid and (pid == doc or (len(doc) >= 6 and pid.startswith(doc))):
                out.append(p)
        return out

    def _name_for_doc(self, doc: str, places: list[Place]) -> str:
        names = [p.name for p in places if p.name and not _looks_like_id(p.name)]
        if names:
            return max(names, key=len)
        if not self._lookup:
            return ""
        if doc not in self._looked_up:
            self._looked_up[doc] = _document_name(doc)
        return self._looked_up[doc]

    def known_names(self) -> list[str]:
        return [p.name for p in self.places if p.name]

    def is_document_id(self, token: str) -> bool:
        return bool(token) and bool(self._by_doc(token))

    def resolve(
        self,
        *,
        doc: str = "",
        name: str = "",
        chunks: Iterable[int] = (),
        pages: Iterable[int] = (),
    ) -> tuple[str, list[int]] | None:
        """``(document name, pages)`` for a citation, or None when nothing
        backs it. ``pages`` is empty when the document has no page there."""
        chunks = [c for c in chunks if c is not None]
        stated = sorted({p for p in (_page(x) for x in pages) if p})
        if doc and name and not self._by_doc(doc) and not self._name_for_doc(doc, []):
            doc = ""
        cands = self._by_doc(doc) if doc else list(self.places)
        if name:
            named = [p for p in cands if p.name and _names_match(p.name, name)]
            cands = named if (named or not doc) else cands
        matched = [p for p in cands if p.chunk in chunks] if chunks else []
        if chunks and not matched and not doc and not name:
            return None
        pool = matched or cands
        if doc:
            shown = self._name_for_doc(doc, pool)
        else:
            names = {p.name for p in pool if p.name and not _looks_like_id(p.name)}
            if name and not names:
                shown = _display_name(name)
            elif not names:
                return None
            else:
                keys = {_name_key(n) for n in names}
                if len(keys) > 1 and not name:
                    return None
                shown = max(names, key=len)
        if not shown or _looks_like_id(shown):
            return None
        found = sorted({p.page for p in matched if p.page})
        if not found and not chunks and (doc or name):
            # The document is cited without a chunk: its page is known only
            # when the turn read it at one page.
            pages_read = {p.page for p in pool if p.page}
            found = sorted(pages_read) if len(pages_read) == 1 else []
        return shown, (stated or found)


def render_place(name: str, pages: list[int]) -> str:
    if not pages:
        return name
    if len(pages) == 1:
        return f"{name}, p. {pages[0]}"
    return f"{name}, pp. {', '.join(str(p) for p in pages)}"


# ── Citation grammar ─────────────────────────────────────────────────────────

_NUMS = r"\d+(?:\s*(?:,|and|&|[-–—]|to)\s*\d+)*"
_DOC_KEY = r"\bdoc(?:ument)?[ _-]?id\b"
_DOC_TOKEN_RE = re.compile(rf"{_DOC_KEY}\s*[=:#]?\s*(?P<doc>{_HEX_ID})", re.IGNORECASE)
_CHUNK_TOKEN_RE = re.compile(rf"\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*(?P<nums>{_NUMS})",
                             re.IGNORECASE)
_PAGE_TOKEN_RE = re.compile(rf"(?:\bpages?\b|\bpp?\.)[ \t]*[=:#]?[ \t]*(?P<nums>{_NUMS})", re.IGNORECASE)
_SRC_TOKEN_RE = re.compile(r"\bsrc\s*=\s*(?P<src>.+?)(?=\s+[a-z_]+\s*=|\s*[\]】)]|$)", re.IGNORECASE)
_TELEMETRY_KEYS = tuple(k for k in MARKER_KEYS if k not in ("doc_id", "chunk", "page", "src"))
#: Marker telemetry: a key written ``key=value``, or a score written with a
#: space. "Class A" or "Rev 3" inside a document name is not telemetry.
_TELEMETRY_RE = re.compile(
    rf"\b(?:{'|'.join(_TELEMETRY_KEYS)}|cosine|rank|relevance)\b[ \t]*[=:][ \t]*"
    r"(?:-?\d+(?:\.\d+)?|[A-Za-z_][\w-]*)"
    r"|\b(?:score|cosine|relevance)\b[ \t]+-?\d+(?:\.\d+)?|\bSUPERSEDED\b",
    re.IGNORECASE,
)
_SOURCE_LABEL_RE = re.compile(r"^\s*\**\s*(?:sources?|cited|citations?|ref(?:erence)?s?)\s*\**\s*[:=-]\s*",
                              re.IGNORECASE)
_SOURCE_LABEL_AT_RE = re.compile(_SOURCE_LABEL_RE.pattern.lstrip("^"), re.IGNORECASE)
_CUE_RE = re.compile(r"^(?:see|per|from|in|at|cf\.?|ref\.?|refer\s+to|according\s+to|as\s+stated\s+in)\b\s*",
                     re.IGNORECASE)
#: A retrieval key strong enough to make a span a citation.
_CITATION_KEY_RE = re.compile(
    rf"{_DOC_KEY}\s*[=:#]?\s*[0-9a-f]{{4}}|\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*\d"
    r"|\bsrc\s*=|\bscore\s*[=:]\s*-?\d",
    re.IGNORECASE,
)


def _numbers(blob: str) -> list[int]:
    out: list[int] = []
    for m in re.finditer(r"(\d+)\s*(?:[-–—]|to)\s*(\d+)|(\d+)", blob or ""):
        if m.group(3):
            out.append(int(m.group(3)))
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        if 0 <= hi - lo <= 50:
            out.extend(range(lo, hi + 1))
        else:
            out.extend((lo, hi))
    return out


@dataclass
class _Cite:
    doc: str = ""
    name: str = ""
    chunks: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    labelled: bool = False

    @property
    def machine(self) -> bool:
        return bool(self.doc or self.chunks)


def _parse_cite(inner: str, index: CitationIndex, *, labelled: bool = False) -> _Cite:
    """Pull doc id, chunk numbers, pages and a document name out of one citation.
    ``labelled``: the span was introduced as a source, so its remaining text
    is the document's name."""
    cite = _Cite(labelled=labelled)
    rest = inner or ""
    label = _SOURCE_LABEL_RE.match(rest)
    if label:
        cite.labelled = True
        rest = rest[label.end():]
    m = _SRC_TOKEN_RE.search(rest)
    if m:
        cite.name = m.group("src").strip().rstrip(".,;")
        rest = rest[:m.start()] + " " + rest[m.end():]
    for m in _DOC_TOKEN_RE.finditer(rest):
        cite.doc = cite.doc or m.group("doc")
    rest = _DOC_TOKEN_RE.sub(" ", rest)
    for m in _CHUNK_TOKEN_RE.finditer(rest):
        cite.chunks.extend(_numbers(m.group("nums")))
    rest = _CHUNK_TOKEN_RE.sub(" ", rest)
    for m in _PAGE_TOKEN_RE.finditer(rest):
        cite.pages.extend(_numbers(m.group("nums")))
    rest = _PAGE_TOKEN_RE.sub(" ", rest)
    rest = _TELEMETRY_RE.sub(" ", rest)
    rest = re.sub(r"\s+", " ", rest).strip(" ,;:.-–—|\"'“”‘’`*()[]【】")
    rest = _CUE_RE.sub("", rest).strip(" ,;:.-–—")
    if not cite.name and rest:
        if _looks_like_id(rest):
            cite.doc = cite.doc or rest
        elif (cite.labelled or _EXT_RE.search(rest)
              or any(_names_match(n, rest) for n in index.known_names())):
            cite.name = rest
    return cite


def _resolve(cite: _Cite, index: CitationIndex) -> str | None:
    hit = index.resolve(doc=cite.doc, name=cite.name, chunks=cite.chunks, pages=cite.pages)
    return render_place(*hit) if hit else None


_OPEN_TO_CLOSE = {"[": "]", "(": ")", "【": "】"}
_BRACKET_OPEN_RE = re.compile(r"[\[(【]")
#: The run of retrieval tokens an unclosed bracket carries.
_UNCLOSED_RUN_RE = re.compile(
    rf"(?:[ \t,;:|]*(?:{_DOC_KEY}\s*[=:#]?\s*{_HEX_ID}"
    rf"|\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*{_NUMS}"
    rf"|(?:\bpages?\b|\bpp?\.)[ \t]*[=:#]?[ \t]*{_NUMS}"
    r"|\bsrc\s*=\s*[^\n\]】)]+?(?=[ \t]+[a-z_]+\s*=|[.;]?\s*$)"
    rf"|{_TELEMETRY_RE.pattern}|(?:sources?)\s*:))+",
    re.IGNORECASE,
)


def _rewrite_brackets(text: str, index: CitationIndex) -> str:
    """Bracketed citations, closed or not, become ``(Name, p. N)`` or go."""
    out: list[str] = []
    pos = 0
    while True:
        m = _BRACKET_OPEN_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        start = m.start()
        opener = m.group(0)
        closer = _OPEN_TO_CLOSE[opener]
        line_end = text.find("\n", start)
        line_end = len(text) if line_end == -1 else line_end
        close = _matching_close(text, start, line_end, opener, closer)
        if close is not None:
            inner = text[start + 1:close]
            end = close + 1
        elif not text[start + 1:line_end].replace(_MARK, "").strip(" \t.,;:"):
            # An opener with nothing after it is what a removal left.
            out.append(text[pos:start] + _MARK)
            pos = line_end
            continue
        elif _SOURCE_LABEL_AT_RE.match(text, start + 1, line_end):
            # An unclosed "(Source: name" runs to the end of its sentence:
            # it is closed on the name it gives, or removed whole.
            cut = _NEXT_SENTENCE_RE.search(text, start + 1, line_end)
            end = cut.start("gap") if cut else line_end
            inner = text[start + 1:end]
            out.append(text[pos:start])
            out.append(_close_cite(inner, index))
            pos = end
            continue
        else:
            run = _UNCLOSED_RUN_RE.match(text, start + 1)
            inner = run.group(0) if run else ""
            end = run.end() if run else start + 1
        if not _CITATION_KEY_RE.search(inner) or len(inner) > 600:
            out.append(text[pos:start + 1])
            pos = start + 1
            continue
        out.append(text[pos:start])
        cite = _parse_cite(inner, index)
        place = _resolve(cite, index)
        if place:
            lead = "Source: " if cite.labelled else ""
            out.append(f"({lead}{place})")
        else:
            out.append(_MARK)
        pos = end
    return "".join(out)


_FILE_EXTS = r"pdf|docx?|xlsx?|xlsm|csv|txt|md|pptx?|dwg|dxf|ifc|xer|xml|json|zip|rar|png|jpe?g"
#: Where an open citation's name ends: a sentence end, or a file name's
#: extension ahead of a new sentence.
_NEXT_SENTENCE_RE = re.compile(rf"(?:(?<=[.;!?])|\.(?:{_FILE_EXTS})\b)(?P<gap>[ \t]+)(?=[A-Z])")


def _close_cite(inner: str, index: CitationIndex) -> str:
    stop = re.search(r"[ \t.,;:]*$", inner)
    body = inner[:stop.start()] if stop else inner
    tail = "." if "." in inner[len(body):] else ""
    cite = _parse_cite(body, index, labelled=True)
    if not (cite.machine or cite.name):
        return _MARK
    place = _resolve(cite, index)
    if place:
        return f"(Source: {place}){tail}"
    if cite.machine:
        return _MARK
    return f"(Source: {cite.name}){tail}"


def _matching_close(text: str, start: int, limit: int, opener: str, closer: str) -> int | None:
    depth = 0
    for i in range(start, limit):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return None


_SOURCE_LINE_RE = re.compile(
    r"(?P<lead>(?:^|(?<=\n))[ \t]*(?:[-*•][ \t]+)?\**[ \t]*|(?<=[\s(]))"
    r"(?P<label>Sources?)[ \t]*\**[ \t]*:[ \t]*\**[ \t]*"
    r"(?P<body>[^\n]*?)(?P<tail>\**[ \t]*(?:\.(?=\s|$)|$|(?=\n)))",
    re.IGNORECASE,
)


def _rewrite_source_mentions(text: str, index: CitationIndex) -> str:
    """``Source: <name>, chunk 0`` style mentions carry a retrieval key; each
    part becomes ``Name, p. N`` or goes. A mention with no key is left alone
    (a calculator credit is one)."""

    def repl(m: re.Match[str]) -> str:
        body = m.group("body")
        if not _CITATION_KEY_RE.search(body or ""):
            return m.group(0)
        kept: list[str] = []
        for part in re.split(r"\s*;\s*", body):
            if not part.strip():
                continue
            cite = _parse_cite(part, index, labelled=True)
            place = _resolve(cite, index) if (cite.machine or cite.name) else None
            if place and place not in kept:
                kept.append(place)
        if not kept:
            return m.group("lead") + _MARK
        return f"{m.group('lead')}{m.group('label')}: {'; '.join(kept)}{m.group('tail')}"

    return _SOURCE_LINE_RE.sub(repl, text)


_CHUNK_TAIL = rf"[ \t]*,?[ \t]*\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*(?P<nums>{_NUMS})"
_FILE_CHUNK_RE = re.compile(
    rf"(?P<name>(?<![\w./-])[\w][\w.&()'+-]{{0,160}}?\.[A-Za-z0-9]{{2,5}}){_CHUNK_TAIL}",
    re.IGNORECASE,
)
_INLINE_DOC_RE = re.compile(
    rf"(?:\b(?:see|per|from|in)\s+)?{_DOC_KEY}\s*[=:#]?\s*(?P<doc>{_HEX_ID})"
    rf"(?:[ \t]*[,;]?[ \t]*\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*(?P<nums>{_NUMS}))?",
    re.IGNORECASE,
)
_INLINE_CHUNK_RE = re.compile(
    rf"(?P<cue>\b(?:see|per|from|in|at|of)\s+)?\bchunks?\b(?:[ \t]*index)?[ \t]*[=:#]?[ \t]*#?[ \t]*"
    rf"(?P<nums>{_NUMS})",
    re.IGNORECASE,
)


def _rewrite_inline(text: str, index: CitationIndex) -> str:
    """Retrieval tokens loose in prose: a named file's chunk, a document id,
    a bare chunk number, marker telemetry, a raw id."""

    def file_chunk(m: re.Match[str]) -> str:
        name = m.group("name").strip()
        hit = index.resolve(name=name, chunks=_numbers(m.group("nums")))
        return render_place(*hit) if hit else _display_name(name)

    for known in sorted(set(index.known_names()), key=len, reverse=True):
        stem = _EXT_RE.sub("", known)
        if len(stem) < 5:
            continue
        pattern = rf"(?P<name>{re.escape(stem)}(?:\.[A-Za-z0-9]{{2,5}})?){_CHUNK_TAIL}"
        text = re.sub(pattern, file_chunk, text, flags=re.IGNORECASE)
    text = _FILE_CHUNK_RE.sub(file_chunk, text)

    def doc_ref(m: re.Match[str]) -> str:
        hit = index.resolve(doc=m.group("doc"), chunks=_numbers(m.group("nums") or ""))
        return render_place(*hit) if hit else _MARK

    text = _INLINE_DOC_RE.sub(doc_ref, text)

    def chunk_ref(m: re.Match[str]) -> str:
        hit = index.resolve(chunks=_numbers(m.group("nums")))
        if not hit:
            return _MARK
        return (m.group("cue") or "") + render_place(*hit)

    text = _INLINE_CHUNK_RE.sub(chunk_ref, text)
    text = re.sub(
        rf"\b(?:{'|'.join(_TELEMETRY_KEYS)}|src)\s*=\s*[^\s,;)\]】]+", _MARK, text, flags=re.IGNORECASE,
    )

    def raw_id(m: re.Match[str]) -> str:
        hit = index.resolve(doc=m.group(0))
        return hit[0] if hit else _MARK

    text = _UUID_RE.sub(raw_id, text)

    def known_id(m: re.Match[str]) -> str:
        token = m.group(0)
        if not (_looks_like_id(token) and re.search(r"[a-f]", token, re.IGNORECASE)):
            return token
        # A short hex token may be a code or a colour; one of id length is an
        # id whether or not this turn read it.
        if not (index.is_document_id(token) or len(token.replace("-", "")) >= _ID_LENGTH):
            return token
        hit = index.resolve(doc=token)
        return hit[0] if hit else _MARK

    return re.sub(rf"(?<![\w./-]){_HEX_ID}(?![\w/-])(?!\.\w)", known_id, text, flags=re.IGNORECASE)


# ── Identifiers and dumps ────────────────────────────────────────────────────

#: A code name: lowercase segments joined by ``_`` or ``.``. It is internal
#: only when it has an underscore (dotted alone is also a web domain) and one
#: segment is a word of three letters (``f_ck`` is an engineering symbol).
_IDENT = r"[a-z][a-z0-9]*(?:[_.][a-z0-9]+)+"
_LITERAL = (r"\{\s*\}|\[\s*\]|\{[^{}\n]{0,240}\}|\[[^\[\]\n]{0,240}\]|"
            r"true\b|false\b|null\b|none\b|nan\b|undefined\b")
#: Markup a model wraps a key in: code ticks, quotes, bold.
_KEY_WRAP = r"[`\"'*]{0,2}"
_DUMP_RE = re.compile(
    rf"(?<![\w\\./@-]){_KEY_WRAP}(?P<key>{_IDENT}|[a-z]{{3,}}){_KEY_WRAP}"
    rf"(?:[ \t]*(?P<sep>=|:)[ \t]*{_KEY_WRAP}|[ \t]+)(?P<val>{_LITERAL})[`*]{{0,2}},?",
    re.IGNORECASE,
)
#: ``code_key: word`` -- a payload field and a word value (a figure after a
#: code key reads as words instead).
_FIELD_RE = re.compile(
    rf"(?<![\w\\./@-]){_KEY_WRAP}(?P<key>{_IDENT}){_KEY_WRAP}[ \t]*:[ \t]*{_KEY_WRAP}"
    r"(?P<val>[A-Za-z][\w/&+-]*)(?!\w|\.\w)[`\"'*]{0,2},?",
)
_ASSIGN_RE = re.compile(rf"(?<![\w\\./@-])(?P<key>{_IDENT})=(?P<val>[^\s,;)\]]+)")
_IDENT_RE = re.compile(rf"(?<![\w\\./@#-])(?P<id>{_IDENT})(?![\w/@-])(?!\.[A-Za-z0-9])")
_PROTECTED_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+\.\w+|```.*?```"
    rf"|(?<![\w-])[\w][\w.&()'+-]{{0,160}}?\.(?:{_FILE_EXTS})\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_word_ident(token: str, *, dotted: bool = False) -> bool:
    if "_" not in token and not dotted:
        return False
    return any(len(seg) >= 3 and seg.isalpha() for seg in re.split(r"[_.]", token))


def _words(token: str) -> str:
    return re.sub(r"[_.]+", " ", token).strip()


def _protect(text: str, index: CitationIndex) -> tuple[str, list[str]]:
    """Hide URLs, file names and code fences behind placeholders."""
    held: list[str] = []

    def hold(m: re.Match[str]) -> str:
        held.append(m.group(0))
        return f"\x01{len(held) - 1}\x02"

    text = _PROTECTED_RE.sub(hold, text)
    for name in sorted({n for n in index.known_names() if "_" in n}, key=len, reverse=True):
        stem = _EXT_RE.sub("", name)
        if len(stem) >= 5:
            text = re.sub(re.escape(stem), lambda m: hold(m), text)
    return text, held


def _restore(text: str, held: list[str]) -> str:
    return re.sub(r"\x01(\d+)\x02", lambda m: held[int(m.group(1))], text)


def _drop_dumps(text: str) -> str:
    def dump(m: re.Match[str]) -> str:
        key = m.group("key")
        plain_key = "_" not in key and "." not in key
        if plain_key and m.group("sep") != "=" and not re.fullmatch(r"\{\s*\}|\[\s*\]", m.group("val")):
            return m.group(0)
        if plain_key and m.group("val").lower() in ("none", "nan", "undefined"):
            return m.group(0)
        if not plain_key and not _is_word_ident(key, dotted=True):
            return m.group(0)
        val = m.group("val")
        if val.startswith(("{", "[")) and len(val) > 2 and not re.search(r"[\"':]|^\[\s*\d", val):
            return m.group(0)
        return _DUMP

    from app.lib.source_labels import _registry_maps

    displays, params, _units = _registry_maps()

    def field(m: re.Match[str]) -> str:
        # A payload field shown with its value: a registered name reads as
        # its label elsewhere; any other code key is the platform's own.
        key = m.group("key").lower()
        if key in displays or key in params or not _is_word_ident(key, dotted=True):
            return m.group(0)
        return _DUMP

    return _FIELD_RE.sub(field, _DUMP_RE.sub(dump, text))


#: Stands where a tool, block, action or operation name was until the
#: sentence around it is settled (:func:`_settle_steps`).
_STEP = "\x03"
#: Stands where a ``key=literal`` dump was: a sentence that held one says
#: nothing to a reader unless it also carries a figure.
_DUMP = "\x04"
_STEP_NOUN = (r"(?:tools?|blocks?|actions?|operations?|ops|functions?|modules?|engines?|endpoints?"
              r"|handlers?|routines?|methods?|calls?|steps?)")
_step_cache: dict[Any, Any] = {}


def _step_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    from app.agents.core import tool_registry
    from app.blocks import BLOCK_REGISTRY

    tool_registry.load()
    return tool_registry._TOOLS, BLOCK_REGISTRY


def _static_step_names() -> frozenset[str]:
    """Every name the platform runs a step by: agent tools, blocks, and the
    actions the step tables route to. Read from the registries, so a tool or
    block added later is covered."""
    try:
        tools, blocks = _step_sources()
    except Exception:  # noqa: BLE001 -- a missing registry only narrows the set
        _LOG.debug("answer_exit: step registries unavailable", exc_info=True)
        tools, blocks = {}, {}
    key = ("static", len(tools), len(blocks))
    cached = _step_cache.get("static")
    if cached and cached[0] == key:
        return cached[1]
    names: set[str] = set(tools) | set(blocks)
    try:
        from app.core.action_router import ACTION_HINTS
        from app.core.cm_step_aliases import ACTION_ALIASES, STEP_TO_TARGET

        names.update(ACTION_HINTS)
        names.update(STEP_TO_TARGET)
        names.update(n for target in STEP_TO_TARGET.values() for n in target if n)
        names.update(ACTION_ALIASES)
        names.update(ACTION_ALIASES.values())
    except Exception:  # noqa: BLE001
        _LOG.debug("answer_exit: step tables unavailable", exc_info=True)
    found = frozenset(str(n).strip().lower() for n in names if str(n).strip())
    _step_cache["static"] = (key, found)
    return found


def _step_words_re(names: frozenset[str]) -> re.Pattern[str] | None:
    """A multi-word step name written as words, or a list of them, ahead of
    a step noun: "bim extractor block", "audit and register actions"."""
    multi = tuple(sorted({n for n in names if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", n)
                          and _is_word_ident(n)}, key=len, reverse=True))
    if not multi:
        return None
    cached = _step_cache.get("words")
    if cached and cached[0] == multi:
        return cached[1]
    one = "|".join(r"[ _-]+".join(map(re.escape, n.split("_"))) for n in multi)
    pattern = re.compile(
        rf"(?<![\w-])(?:{one})(?:[ \t]*(?:,[ \t]*(?:and[ \t]+|or[ \t]+)?|and[ \t]+|or[ \t]+|&[ \t]*)"
        rf"(?:{one}))*(?=[ \t]+{_STEP_NOUN}\b)",
        re.IGNORECASE,
    )
    _step_cache["words"] = (multi, pattern)
    return pattern


def _is_step_name(token: str, steps: frozenset[str]) -> bool:
    low = token.lower()
    if low in steps:
        return True
    if "." in low:
        # block.operation, module.attribute
        return low.split(".", 1)[0] in steps or "_" in low
    return False


_CHIP_RE = re.compile(r"`([^`\n]+)`")
_EMPTY_CHIP_RE = re.compile(rf"(?<!`)`[ \t{_MARK}]*`(?!`)")
_PLAIN_WORDS_RE = re.compile(r"[A-Za-z][A-Za-z0-9 -]*[A-Za-z0-9]")


def _unwrap_chips(text: str, steps: frozenset[str], called: frozenset[str]) -> str:
    """A code chip around a code name or plain words loses its ticks; one
    that names a step becomes a step mark. Ticks with no partner go."""

    from app.lib.source_labels import _registry_maps

    displays, params, _units = _registry_maps()

    def chip(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        bare = inner.rstrip("()").strip()
        if re.fullmatch(_IDENT, bare):
            low = bare.lower()
            # A code name shown as code is the platform's own name for a
            # step or a field, unless it is a registered name with words.
            if low in displays or low in params or not _is_word_ident(low, dotted=True):
                return bare
            return _STEP
        if _PLAIN_WORDS_RE.fullmatch(bare):
            key = re.sub(r"[ -]+", "_", bare.lower())
            if ("_" in key and key in steps) or key in called:
                return _STEP
            return bare
        return m.group(0)

    lines = []
    for line in text.split("\n"):
        # A span with nothing left in it shows the user two ticks.
        line = _EMPTY_CHIP_RE.sub(_MARK, line)
        line = _CHIP_RE.sub(chip, line)
        if "`" in line:
            parts = re.split(r"(`[^`\n]+`)", line)
            line = "".join(p if p.startswith("`") and p.endswith("`") and len(p) > 1 else
                           p.replace("`", _MARK) for p in parts)
        lines.append(line)
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```[\w+-]*[ \t]*\n?(?P<body>.*?)```", re.DOTALL)
_DATA_LINE_RE = re.compile(
    rf"[ \t]*(?:[-*][ \t]+)?{_KEY_WRAP}[A-Za-z_][\w.-]*{_KEY_WRAP}[ \t]*[:=][ \t]*\S.*"
    r"|[ \t]*[\[\]{}(),]+[ \t]*"
    rf"|[ \t]*(?:[-*][ \t]+)?{_IDENT}(?:[ \t]*,[ \t]*{_IDENT})*[ \t]*,?[ \t]*",
)
_CODE_KEY_RE = re.compile(rf"[\"']|(?<![\w.]){_IDENT}(?![\w])|[:=][ \t]*{_KEY_WRAP}(?:{_LITERAL})")


def _drop_dump_fences(text: str) -> str:
    """A fenced block that is only data (JSON, ``key: value`` lines with a
    code key or a literal value, a list of code names) is a dump and goes.
    A fence with code in it is kept as written."""

    def fence(m: re.Match[str]) -> str:
        body = m.group("body").strip()
        if not body:
            return _MARK
        try:
            if isinstance(json.loads(body), (dict, list)):
                return _MARK
        except ValueError:
            pass
        lines = [ln for ln in body.split("\n") if ln.strip()]
        if all(_DATA_LINE_RE.fullmatch(ln) for ln in lines) and any(_CODE_KEY_RE.search(ln) for ln in lines):
            # Lines that carry figures are a report: they are read as text
            # (each line checked on its own), not thrown away whole.
            return body if any(re.search(r"(?<![\w.])\d", ln) for ln in lines) else _MARK
        return m.group(0)

    return _FENCE_RE.sub(fence, text) if "```" in text else text


def _plain_identifiers(text: str, steps: frozenset[str] = frozenset(),
                       called: frozenset[str] = frozenset()) -> str:
    from app.lib.source_labels import _registry_maps, parameter_words, plain_registry_text

    text = plain_registry_text(_unwrap_chips(_drop_dumps(text), steps, called))
    _displays, params, _units = _registry_maps()

    def assign(m: re.Match[str]) -> str:
        key = m.group("key")
        if not _is_word_ident(key):
            return m.group(0)
        return f"{_words(key)} {m.group('val')}"

    text = _ASSIGN_RE.sub(assign, text)

    def ident(m: re.Match[str]) -> str:
        token = m.group("id")
        if _is_step_name(token, steps):
            return _STEP
        if not _is_word_ident(token):
            return token
        if token in params:
            return parameter_words(token, params[token]) or _words(token)
        return _words(token)

    text = _IDENT_RE.sub(ident, text)
    words = _step_words_re(steps)
    return words.sub(_STEP, text) if words else text


# ── Raw tool errors ──────────────────────────────────────────────────────────

_INTERNAL_TOKEN_RE = re.compile(
    rf"(?<![\w./-])(?:{_IDENT})(?![\w/-])|[{{}}\[\]]|\w=\S|\b\w+(?:Error|Exception)\b|[\\/][\w.-]+[\\/]",
)
_STRUCTURAL_ERROR_RE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]+)?\**[ \t]*(?:error|exception|traceback)\b[ \t]*\**[ \t]*[:=](?![ \t]*\**[ \t]*[-+±]?\d)"
    r"|\bTraceback \(most recent call last\)"
    r"|\b[A-Z][A-Za-z]*(?:Error|Exception)\b[ \t]*[:(]"
    r"|[\"']?status[\"']?[ \t]*[:=][ \t]*[\"']?error\b",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+(?=\S)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower().rstrip(".")


def _error_sentence(label: str, err: str) -> str:
    from app.lib.source_labels import _registry_maps, parameter_words

    _displays, params, _units = _registry_maps()
    needs = []
    for token in re.findall(rf"(?<![\w.]){_IDENT}(?![\w])", err or "", re.IGNORECASE):
        low = token.lower()
        if low in params:
            words = parameter_words(low, params[low])
            if words and words not in needs:
                needs.append(words)
    if not needs:
        # That a step failed is the platform's business; the answer goes on
        # without the copied error.
        return ""
    wanted = needs[0] if len(needs) == 1 else f"{', '.join(needs[:-1])} and {needs[-1]}"
    return f"{label} needs the {wanted}." if label else f"I need the {wanted} to work this out."


def _error_forms(err: str) -> list[str]:
    """The error as the tool wrote it and as an answer copies it after
    registry ids were already read out as display names."""
    from app.lib.source_labels import plain_registry_text

    forms = [err]
    try:
        plain = plain_registry_text(err)
    except Exception:  # noqa: BLE001 -- the raw form still matches
        _LOG.debug("answer_exit: plain form of a tool error unavailable", exc_info=True)
        plain = err
    if plain.strip() and plain != err:
        forms.append(plain)
    return forms


def _replace_tool_errors(text: str, turn: Turn | None) -> str:
    errors = [(label, err, form) for label, err in (turn.tool_errors if turn else [])
              if _INTERNAL_TOKEN_RE.search(err) for form in _error_forms(err)]
    lines = []
    said: set[str] = set()
    for line in text.split("\n"):
        parts = _SENTENCE_SPLIT_RE.split(line)
        kept: list[str] = []
        for part in parts:
            body = _norm(part)
            plain = None
            for label, err, form in errors:
                norm_err = _norm(form)
                pieces = [_norm(p) for p in _SENTENCE_SPLIT_RE.split(form) if len(_norm(p)) >= 16]
                if len(body) >= 16 and (body in norm_err or any(p in body for p in pieces)):
                    plain = turn.error_questions.get(err) or _error_sentence(label, err)
                    whole = re.search(r"\s+".join(map(re.escape, form.split())), part)
                    if whole and part[whole.end():].strip(" .") and plain not in said:
                        # The error ran into the next sentence: only its own
                        # words are replaced.
                        said.add(plain)
                        part = f"{part[:whole.start()]}{plain or _MARK} {part[whole.end():].lstrip(' .')}".strip()
                        plain = None
                    break
            if plain is None and _STRUCTURAL_ERROR_RE.search(part):
                plain = _error_sentence("", "")
            if plain is None:
                kept.append(part)
                continue
            if plain not in said:
                said.add(plain)
                lead = re.match(r"[ \t]*(?:[-*•][ \t]+)?", part)
                kept.append((lead.group(0) if lead else "") + (plain or _MARK))
        lines.append(" ".join(kept) if kept else "")
    return "\n".join(lines)


# ── Machinery wording ────────────────────────────────────────────────────────

_MACHINE_ADJ = r"(?:retrieved|provided|supplied|injected|returned|rag|search|retrieval|indexed)"
_MACHINE_PHRASE_RE = re.compile(
    r"(?P<det>\b(?:the|a|an|any|no|this|that|these|those|each|every|some|all(?:\s+of)?(?:\s+the)?)\s+)?"
    rf"(?P<adjs>(?:(?:{_MACHINE_ADJ}|available|relevant|above|given|attached|top)\s+){{0,3}})"
    r"(?P<mid>(?:(?:document|project|corpus|context|knowledge[- ]base)\s+){0,2})"
    r"(?P<noun>excerpts?|chunks?|snippets?|passages?|context|search\s+results?|retrieval\s+results?)\b"
    rf"(?P<post>\s+{_MACHINE_ADJ}\b(?!\s+(?:to|by|in)\b))?",
    re.IGNORECASE,
)
_STRONG_NOUNS = ("excerpt", "chunk", "snippet")
_TOOL_PHRASE_RE = re.compile(
    r"\b(?:tool|function)\s+(?P<what>calls?|results?|outputs?|responses?)\b", re.IGNORECASE,
)
_PARAM_PHRASE_RE = re.compile(
    rf"\b(?P<adj>{_MACHINE_ADJ}|passed|required|missing|given)\s+parameters?\b"
    r"|\bparameters?\s+(?P<post>supplied|provided|passed|given)\b",
    re.IGNORECASE,
)


def _cap_like(original: str, new: str) -> str:
    return new[:1].upper() + new[1:] if original[:1].isupper() else new


#: Words that cannot modify a noun: what precedes a noun phrase rather than
#: sitting inside one. English grammar, not a vocabulary of the corpus.
_CLOSED_CLASS = frozenset((
    "a an the this that these those any no all some each every either neither both such what which whose "
    "in on at of from for to by with within into onto about as than via per across among between through "
    "under over below above after before during without upon against toward towards beyond "
    "and or but nor so yet if because while although though since unless whether "
    "is are was were be been being am has have had do does did not can could may might must shall should "
    "will would it its they their them there here i we you he she my our your his her me us "
    "see find found show shows showed state states stated say says said contain contains mention mentions "
    "give gives gave use used using read include includes cite cites quote quotes"
).split())


def _modified(before: str) -> bool:
    """The noun that follows ``before`` has a modifier: the last word is a
    noun or adjective, not a determiner, preposition, verb or pronoun."""
    m = re.search(r"([A-Za-z][A-Za-z'-]*)[ \t]+$", before)
    if not m:
        return False
    word = m.group(1).lower()
    return word not in _CLOSED_CLASS and not re.fullmatch(_MACHINE_ADJ, word) and not word.endswith("ly")


def _machine_wording(text: str) -> str:
    def phrase(m: re.Match[str]) -> str:
        noun = m.group("noun").lower()
        adjs = (m.group("adjs") or "").lower()
        machine = bool(re.search(rf"\b{_MACHINE_ADJ}\b", adjs) or m.group("post"))
        det = (m.group("det") or "").strip().lower()
        if noun.startswith(("search", "retrieval")):
            machine = True
        if (m.group("mid") or "").strip() and noun.startswith(_STRONG_NOUNS):
            machine = True
        # "the excerpts" is the context's own label for what it holds;
        # "concrete chunks" or "the chunks of slab" is about the works.
        strong = (noun.startswith(_STRONG_NOUNS) and bool(det)
                  and not re.match(r"\s+of\b", m.string[m.end():]))
        if not (strong or machine):
            return m.group(0)
        plural = noun.endswith("s") or noun.startswith(("search", "retrieval"))
        if not det and _modified(m.string[:m.start()]):
            # "the specification excerpt": the noun phrase began earlier,
            # with its own determiner; only the head noun is replaced.
            new = "material" if noun == "context" else ("sections" if plural else "section")
        elif noun == "context":
            new = f"{det if det in ('no', 'any', 'all the') else 'the'} project material"
        elif plural:
            keep = det if det in ("no", "any", "some", "all", "all the", "all of the") else "the"
            new = f"{keep} project document sections"
        elif not det:
            new = "project document section"
        else:
            keep = {"an": "a", "these": "the", "those": "the"}.get(det, det)
            new = f"{keep} project document section"
        return _cap_like(m.group(0), new)

    text = _MACHINE_PHRASE_RE.sub(phrase, text)
    return _PARAM_PHRASE_RE.sub(
        lambda m: _cap_like(m.group(0), (f"{m.group('adj')} inputs" if m.group("adj") else
                                         f"inputs {m.group('post')}")),
        text,
    )


# ── Step names and tool-call mechanics ───────────────────────────────────────

#: How the platform's calls worked, said to the user: the arguments, the
#: engine's expectations, what is implemented. Never part of an answer.
_MECHANICS_RE = re.compile(
    r"\bJSON[ \t]+(?:strings?|objects?|bod(?:y|ies)|arguments?|inputs?|keys?|fields?)\b|\binput[ \t]+JSON\b"
    r"|\bJSON\b[^.!?\n]{0,60}\b(?:params|parameters|arguments)\b"
    r"|\b(?:params|kwargs|args)\b"
    r"|\bengine(?:'s)?[ \t]+version\b"
    r"|\bimplemented[ \t]+(?:in|by|on)[ \t]+(?:this|the)[ \t]+(?:engine|version|tool|block|platform|api|build)\b"
    r"|\bunimplemented\b"
    r"|\btraceback\b|\bstack[ \t]+trace\b|\bHTTP[ \t]+\d{3}\b|\bstatus[ \t]+code\b"
    r"|\bunknown[ \t]+(?:operation|action|tool|parameter|argument|function|block)s?\b"
    r"|\bpre-?dispatch(?:ed|es|ing)?\b"
    r"|\b(?:steps?|requests?|calls?|tools?|blocks?)\b[^.!?\n]{0,40}\bdid(?:n't|[ \t]+not)[ \t]+return\b",
    re.IGNORECASE,
)
#: The model telling the user about its own calls.
_PROCESS_RE = re.compile(
    r"\b(?:I|we)(?:'ve|[ \t]+(?:had|have|then|first|initially|also|just|originally))?[ \t]+"
    r"(?:called|invoked|passed|retried|re-?ran|re-?tried|sent|queried|routed|dispatched|"
    r"tried[ \t]+(?:calling|invoking|again|to[ \t]+(?:call|invoke|run)))\b"
    r"|\bmy[ \t]+(?:(?:first|second|third|earlier|previous|initial|last|original)[ \t]+)?"
    r"(?:calls?|attempts?|requests?|quer(?:y|ies)|tries)\b"
    r"|\b(?:calls?|attempts?|requests?)[ \t]+(?:both[ \t]+|all[ \t]+)?(?:failed|errored|returned[ \t]+(?:an[ \t]+)?errors?)\b",
    re.IGNORECASE,
)
#: What the model's calls are made of; process narration names one.
_CALL_NOUN_RE = re.compile(
    r"\b(?:calls?|tools?|functions?|actions?|operations?|parameters?|arguments?|inputs?|requests?|"
    r"quer(?:y|ies)|engine|blocks?|steps?|JSON|schema|searche?s?|retriev\w*)\b",
    re.IGNORECASE,
)
_FAILURE_RE = re.compile(
    r"\b(?:fail(?:ed|s|ing|ure)?|errors?|errored|rejected|wrong|incorrect(?:ly)?|"
    r"did(?:n't|[ \t]+not)[ \t]+work|retry|again|instead|crashed|timed[ \t]+out|"
    r"(?:returned|reported|gave|produced)[ \t]+nothing|no[ \t]+results?|empty[ \t]+results?)\b",
    re.IGNORECASE,
)
_JOINED_STEPS_RE = re.compile(
    rf"{_STEP}(?:[ \t]*(?:,[ \t]*(?:and[ \t]+|or[ \t]+)?|and[ \t]+|or[ \t]+|&[ \t]*)[ \t]*{_STEP})+",
    re.IGNORECASE,
)
_LINE_LEAD_RE = re.compile(r"[ \t]*(?:(?:[-*•+]|\d+[.)])[ \t]+)?(?:\*\*)?")
#: The model announcing that it is about to repeat a step.
_RETRY_RE = re.compile(
    r"^[ \t]*(?:(?:[-*•+]|\d+[.)])[ \t]+)?(?:\*\*)?"
    r"(?:re-?trying|re-?running|re-?attempting|(?:trying|attempting)[ \t]+again)\b"
    r"|^[ \t]*(?:(?:[-*•+]|\d+[.)])[ \t]+)?(?:\*\*)?[A-Za-z]+ing\b[^.!?\n]*\b(?:once[ \t]+more|one[ \t]+more[ \t]+time)\b"
    r"|\b(?:I|we|let[ \t]+me)(?:'ll|[ \t]+will|'m[ \t]+going[ \t]+to|[ \t]+am[ \t]+going[ \t]+to)?[ \t]+(?:now[ \t]+)?"
    r"(?:re-?try|re-?run|try[ \t]+again)\b",
    re.IGNORECASE,
)
_ID_NOUN = r"[ _-]?(?:ids?|identifiers?|uuids?)\b"
#: An identifier only the platform has: no user knows or needs one.
_PLATFORM_ID_RE = re.compile(
    rf"\b(?:formula|tool|block|operation|action|chunk|registry|calculator|function|conversation|session|request)"
    rf"{_ID_NOUN}",
    re.IGNORECASE,
)
#: An identifier the platform keeps for a project or a document; asking the
#: user for one asks them for the platform's own key.
_KEPT_ID_RE = re.compile(rf"\b(?:project|doc(?:ument)?|file|record|user){_ID_NOUN}", re.IGNORECASE)
_ASK_RE = re.compile(
    r"\?|\b(?:please|provide|share|send|give|tell|confirm|specify|supply|let[ \t]+me[ \t]+know)\b",
    re.IGNORECASE,
)
_STEPS = (rf"(?:\b(?:the|a|an|this|that|these|those|its|their|our|my)[ \t]+)?{_STEP}+"
          rf"(?:[ \t]*\([ \t]*\))?(?:[ \t]+{_STEP_NOUN}\b)?")
#: A step named in a phrase the sentence can lose ("using the X,", "(via X)").
_STEP_ADJUNCT_RE = re.compile(
    rf"(?P<lc>[ \t]*,)?[ \t]*\b(?:via|using|through|with|by|from|per|in|on|according[ \t]+to|based[ \t]+on)"
    rf"[ \t]+{_STEPS}(?:(?P<rc>[ \t]*,)|(?=[ \t]*(?:[.;:!?)]|$)|[ \t]+(?:and|but|so|then)\b))",
    re.IGNORECASE,
)
_STEP_PAREN_RE = re.compile(rf"[ \t]*\([ \t]*(?:(?:via|using|from|by|with)[ \t]+)?{_STEPS}[ \t]*\)", re.IGNORECASE)
#: What a step did, said as what was found: the verb read with "I".
_STEP_VERBS = {
    "returned": "found", "returns": "found", "reported": "found", "reports": "found", "gave": "found",
    "gives": "found", "showed": "found", "shows": "found", "produced": "found", "produces": "found",
    "yielded": "found", "yields": "found", "listed": "found", "lists": "found", "found": "found",
    "finds": "found", "extracted": "extracted", "extracts": "extracted", "identified": "identified",
    "identifies": "identified", "detected": "detected", "detects": "detected", "counted": "counted",
    "counts": "counted", "computed": "computed", "computes": "computed", "calculated": "calculated",
    "calculates": "calculated", "measured": "measured", "measures": "measured", "recorded": "recorded",
    "records": "recorded", "read": "read", "reads": "read", "located": "located", "locates": "located",
    "matched": "matched", "matches": "matched", "priced": "priced", "prices": "priced",
}
#: A step as the subject of what it did, with an object after the verb.
_STEP_SUBJECT_RE = re.compile(
    rf"^(?P<lead>{_LINE_LEAD_RE.pattern}){_STEPS}[ \t]+(?:(?:has|have|had)[ \t]+)?(?:also[ \t]+)?"
    rf"(?P<verb>{'|'.join(sorted(_STEP_VERBS, key=len, reverse=True))})\b"
    r"(?=[ \t]+(?!(?:on|in|at|for|with|by|to|from|that|nothing|it|them|and|or|but|nor|so|then)\b)[\w$€£(])",
    re.IGNORECASE,
)


def _mechanics(sentence: str) -> bool:
    if _MECHANICS_RE.search(sentence) or _RETRY_RE.search(sentence) or _PLATFORM_ID_RE.search(sentence):
        return True
    if _KEPT_ID_RE.search(sentence) and _ASK_RE.search(sentence):
        return True
    if _DUMP in sentence and not re.search(r"\d", sentence.replace(_DUMP, "")):
        return True
    if _PROCESS_RE.search(sentence) and _FAILURE_RE.search(sentence) and _CALL_NOUN_RE.search(sentence):
        return True
    # A sentence that names a step and says how it ran is about the platform.
    return _STEP in sentence and bool(_FAILURE_RE.search(sentence) or _PROCESS_RE.search(sentence))


def _without_step(sentence: str) -> str:
    """``sentence`` with the step it names taken out, or '' when the step
    cannot leave without the sentence: a step is never named in its place.

    The phrase that names it goes when the sentence stands without it
    ("Using X, the total is 5" -> "The total is 5"); a step that did the
    finding is said as what was found ("X found 42 items" -> "I found 42
    items"). A sentence whose subject or object is the step goes whole.
    """
    lead = _LINE_LEAD_RE.match(sentence)
    head = lead.group(0) if lead else ""

    def adjunct(m: re.Match[str]) -> str:
        return " " if (m.group("lc") and m.group("rc")) else ""

    out = _STEP_PAREN_RE.sub("", sentence)
    out = _STEP_ADJUNCT_RE.sub(adjunct, out)
    out = _STEP_SUBJECT_RE.sub(lambda m: f"{m.group('lead')}I {_STEP_VERBS[m.group('verb').lower()]}", out)
    if _STEP in out:
        return ""
    body = out[len(head):].lstrip(" \t,;:") if out.startswith(head) else out.lstrip(" \t,;:")
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+([,.;:!?])", r"\1", body)
    if len(re.findall(r"[A-Za-z]{2,}", body)) < 2:
        return ""
    return head + body[:1].upper() + body[1:]


def _settle_steps(text: str) -> str:
    """Sentences about the platform's calls go. A step name leaves a
    sentence about the work the way :func:`_without_step` takes it out,
    and the sentence goes when it cannot."""
    if (_STEP not in text and _DUMP not in text and not (_MECHANICS_RE.search(text) or _PROCESS_RE.search(text)
            or _PLATFORM_ID_RE.search(text) or _KEPT_ID_RE.search(text)
            or any(_RETRY_RE.search(p) for line in text.split("\n") for p in _SENTENCE_SPLIT_RE.split(line)))):
        return text
    text = _JOINED_STEPS_RE.sub(_STEP * 2, text)
    out: list[str] = []
    for line in text.split("\n"):
        parts = _SENTENCE_SPLIT_RE.split(line)
        kept: list[str] = []
        for part in parts:
            if _mechanics(part):
                continue
            if _STEP in part:
                part = _without_step(part)
                if not part:
                    continue
            kept.append(part)
        if kept == parts:
            out.append(line)
            continue
        if not kept:
            continue
        if kept[0] is not parts[0]:
            lead = _LINE_LEAD_RE.match(parts[0])
            kept[0] = (lead.group(0) if lead else "") + kept[0][len(_LINE_LEAD_RE.match(kept[0]).group(0)):]
        out.append(" ".join(kept))
    return "\n".join(out).replace(_DUMP, _MARK)


# ── One Source line per credit ───────────────────────────────────────────────

_WHOLE_SOURCE_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]+)?\**[ \t]*Sources?[ \t]*\**[ \t]*:[ \t]*\**[ \t]*(?P<body>.*?)[ \t]*\**[ \t]*\.?[ \t]*$",
    re.IGNORECASE,
)
_LABEL_JOINS_ONLY_RE = re.compile(r"(?:[\s.*`\"'→›»>|·;,/=-]|\bvia\b|\band\b)*", re.IGNORECASE)


def _dedupe_source_lines(text: str) -> str:
    """A Source line that repeats another, or that only names platform
    labels a calculator credit in the same answer already gives (``Platform
    calculator → Unit conversion`` beside ``Unit conversion — platform
    calculator (…)``), is removed."""
    from app.lib.source_labels import CALCULATOR_SUFFIX

    lines = text.split("\n")
    found = [(i, m.group("body")) for i, ln in enumerate(lines) if (m := _WHOLE_SOURCE_LINE_RE.match(ln))]
    if len(found) < 2:
        return text
    suffix = CALCULATOR_SUFFIX.lower()
    credit_mark = f" — {suffix}"
    credited = {body.lower().split(credit_mark)[0].strip() for _i, body in found if credit_mark in body.lower()}
    names = sorted(credited | {suffix}, key=len, reverse=True)
    seen: set[str] = set()
    drop: set[int] = set()
    for i, body in found:
        key = re.sub(r"\s+", " ", body.lower()).strip(" .")
        if key in seen:
            drop.add(i)
            continue
        seen.add(key)
        if not credited or credit_mark in body.lower():
            continue
        rest = body.lower()
        for name in names:
            rest = rest.replace(name, " ")
        if rest != body.lower() and _LABEL_JOINS_ONLY_RE.fullmatch(rest):
            drop.add(i)
    if not drop:
        return text
    return "\n".join(ln for i, ln in enumerate(lines) if i not in drop)


# ── The check ────────────────────────────────────────────────────────────────

_JOINED_MARKS_RE = re.compile(rf"{_MARK}(?:[ \t]*(?:[,;&]|\band\b|\bor\b)?[ \t]*{_MARK})+", re.IGNORECASE)
#: A sentence that opens with removed text and runs on in lower case: its
#: subject was the internal text, so what is left says nothing to a reader.
_HEADLESS_SENTENCE_RE = re.compile(
    rf"(?:(?<=^)|(?<=\n)|(?<=[.!?:][ \t]))[ \t]*{_MARK}[ \t]*(?:[,;:]?[ \t]*[a-z][^.!?\n]*)?[.!?]?[ \t]*(?=\S|$)",
)


# A source or page marker whose reference is not a page (-1, 0, none, n/a)
# points at nothing a reader can open: it goes, silently -- it names no source.
_NO_PAGE_MARKER_RE = re.compile(
    r"[ \t]*[(\[]\s*(?:source|src|page|pg|p)\.?\s*[:#=]?\s*"
    r"(?:-\d+|0+|none|null|nil|n/?a|unknown|undefined)\s*[)\]]",
    re.IGNORECASE,
)


def _tidy(text: str) -> str:
    if _MARK not in text:
        return text
    from app.agents.citation_provenance import SENTINEL
    from app.agents.citation_provenance import _tidy as tidy

    text = _JOINED_MARKS_RE.sub(_MARK, text)
    text = _HEADLESS_SENTENCE_RE.sub(_MARK, text)
    return tidy(text.replace(_MARK, SENTINEL))


def build_index(
    turn: Turn | None = None,
    sources: Iterable[dict[str, Any]] | None = None,
    *,
    lookup: bool = True,
) -> CitationIndex:
    places = list(turn.places) if turn else []
    places.extend(places_from_sources(sources))
    return CitationIndex(places, lookup=lookup)


def check_text(
    text: str,
    *,
    turn: Turn | None = None,
    sources: Iterable[dict[str, Any]] | None = None,
    index: CitationIndex | None = None,
) -> str:
    """The answer as it may leave: citations resolved to name and page or
    removed, no chunk numbers, ids, dumps, raw tool errors or machinery words."""
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        turn = turn if turn is not None else _TURN.get()
        index = index or build_index(turn, sources)
        trailing = text[len(text.rstrip()):]
        out = _NO_PAGE_MARKER_RE.sub("", text)
        out = _rewrite_source_mentions(out, index)
        out = _rewrite_brackets(out, index)
        out = _rewrite_inline(out, index)
        out = _replace_tool_errors(out, turn)
        out = _drop_dump_fences(out)
        out, held = _protect(out, index)
        called = frozenset(turn.step_names) if turn else frozenset()
        out = _plain_identifiers(out, _static_step_names() | called, called)
        out = _TOOL_PHRASE_RE.sub(_STEP, out)
        out = _settle_steps(out)
        out = _machine_wording(out)
        out = _restore(out, held)
        out = _tidy(out)
        out = _dedupe_source_lines(out)
        if out == text:
            return text
        out = re.sub(r"[ \t]+\n", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out + trailing if out else out
    except Exception:  # noqa: BLE001 -- the exit must never break an answer
        _LOG.exception("answer_exit: check failed; answer passed through")
        return text


_SENTENCE_END_RE = re.compile(r"[.!?…](?:[\"'”’)\]*_`]+)?(?=[ \t]|$)")
_ABBREV_END_RE = re.compile(
    r"(?:^|[\s(])(?:e\.g|i\.e|vs|cf|approx|nos?|cl|fig|incl|mr|ms|dr)\.$",
    re.IGNORECASE,
)
_LEAD_IN_RE = re.compile(r"[ \t]*(?:#{1,6}[ \t].*|\*\*[^*\n]+\*\*:?|[^\n]*:[ \t]*(?:\*\*)?)[ \t]*")


def end_on_a_sentence(text: str) -> str:
    """``text`` cut back to its last whole sentence, for an answer the model
    stopped writing part-way (a length limit, a dropped stream). Only the
    last line was being written: it keeps what ends a sentence, or goes
    whole (a half list item or table row is not a sentence either). A
    heading or lead-in left with nothing under it goes too."""
    if not isinstance(text, str) or not text.strip():
        return text
    body = text.rstrip()
    if body.count("```") % 2:
        body = body[:body.rfind("```")].rstrip()
    lines = body.split("\n")
    last = lines.pop()
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(last)
            if not _ABBREV_END_RE.search(last[:m.start() + 1])]
    if last.rstrip().endswith("|") and last.lstrip().startswith("|"):
        lines.append(last)
    elif ends:
        lines.append(last[:ends[-1]].rstrip())
    while lines and (not lines[-1].strip() or (len(lines) > 1 and _LEAD_IN_RE.fullmatch(lines[-1]))):
        lines.pop()
    return "\n".join(lines).rstrip()


def remove_dumps(text: str) -> str:
    """``text`` without ``key=literal`` dumps (``delta_ok=false``,
    ``summary {}``). For a step that rewrites ``key=value`` as words before
    the exit: a dump read as words is still a dump."""
    if not isinstance(text, str) or not text:
        return text
    try:
        held_text, held = _protect(text, CitationIndex(lookup=False))
        out = _drop_dumps(held_text).replace(_DUMP, _MARK)
        if out == held_text:
            return text
        return _tidy(_restore(out, held))
    except Exception:  # noqa: BLE001 -- the exit must never break an answer
        _LOG.exception("answer_exit: dump removal failed; text passed through")
        return text


def check_text_or_fallback(text: str, **kwargs: Any) -> str:
    """``check_text`` for a whole answer: one that was nothing but internals,
    or the platform's empty-turn reply, becomes the turn's own question for
    the inputs a tool still needs when there is one; otherwise the empty-turn
    reply, never an empty bubble (the client would show the raw streamed
    tokens in its place)."""
    from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK

    checked = check_text(text, **kwargs)
    if isinstance(text, str) and text.strip() and (
            not (checked or "").strip() or checked.strip() == _EMPTY_RESPONSE_FALLBACK):
        turn = kwargs.get("turn") or _TURN.get()
        asked = check_text("\n".join(turn.questions), **kwargs) if turn and turn.questions else ""
        return asked.strip() or _EMPTY_RESPONSE_FALLBACK
    return checked


def check_sources(
    rows: list[dict[str, Any]] | None,
    *,
    turn: Turn | None = None,
    index: CitationIndex | None = None,
) -> list[dict[str, Any]] | None:
    """Sources rows as they may leave: a document name, and the page or nothing."""
    if not isinstance(rows, list):
        return rows
    try:
        turn = turn if turn is not None else _TURN.get()
        index = index or build_index(turn, rows)
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            fixed = dict(row)
            name = _display_name(str(row.get("doc_name") or ""))
            doc_id = str(row.get("doc_id") or "")
            if not name or _looks_like_id(name):
                hit = index.resolve(doc=doc_id or name) if (doc_id or name) else None
                name = hit[0] if hit else ""
            if not name:
                _LOG.info("answer_exit: Sources row with no document name dropped (doc %s)", doc_id)
                continue
            fixed["doc_name"] = name
            page = _page(row.get("page"))
            section = str(row.get("page_or_section") or "")
            if page:
                fixed["page_or_section"] = f"p. {page}"
            elif _CITATION_KEY_RE.search(section) or re.search(r"\bchunk\b|#", section, re.IGNORECASE):
                fixed["page_or_section"] = ""
            elif re.fullmatch(r"\s*\d+\s*", section):
                fixed["page_or_section"] = f"p. {int(section)}"
            else:
                fixed["page_or_section"] = check_text(section, turn=turn, index=index) if section else section
            for key in ("layer_label", "source_class_label"):
                if isinstance(fixed.get(key), str):
                    fixed[key] = check_text(fixed[key], turn=turn, index=index)
            out.append(fixed)
        return out
    except Exception:  # noqa: BLE001 -- the exit must never break an answer
        _LOG.exception("answer_exit: Sources check failed; rows passed through")
        return rows


def check_result(result: Any, turn: Turn | None = None) -> Any:
    """A whole-turn result (``answer`` and ``sources``) as it may leave."""
    if not isinstance(result, dict):
        return result
    fixed = dict(result)
    rows = fixed.get("sources")
    index = build_index(turn, rows if isinstance(rows, list) else None)
    if isinstance(fixed.get("answer"), str):
        fixed["answer"] = check_text_or_fallback(fixed["answer"], turn=turn, index=index)
    if isinstance(rows, list):
        fixed["sources"] = check_sources(rows, turn=turn, index=index)
    return fixed


def check_end_event(event: dict[str, Any], streamed: str = "", turn: Turn | None = None) -> dict[str, Any]:
    """The terminal stream event as it may leave. ``streamed`` is the text the
    tokens carried; the client shows ``content`` in its place, so an event
    without content gets the checked streamed text."""
    if not isinstance(event, dict) or event.get("type") != "end":
        return event
    turn = turn if turn is not None else _TURN.get()
    rows = event.get("sources")
    index = build_index(turn, rows if isinstance(rows, list) else None)
    fixed = dict(event)
    content = event.get("content")
    if not isinstance(content, str) or not content.strip():
        content = streamed
    if isinstance(content, str) and content.strip():
        checked = check_text_or_fallback(content, turn=turn, index=index)
        if checked != content or "content" in event or streamed:
            fixed["content"] = checked
    if isinstance(rows, list):
        fixed["sources"] = check_sources(rows, turn=turn, index=index)
    return fixed
