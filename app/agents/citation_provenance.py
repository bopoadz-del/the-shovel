"""Citation provenance — an attribution the turn cannot back is not shipped.

THE INCIDENT (gate battery ``13b2bf7``, 2026-08-31, question F2). The WBS
answer closed with::

    BOQ context: Bill 03 - Example Works (AB-2001-101)

on a project whose contract is AB-2002-202. Three literal checks, not inference:

* ``generate_wbs`` (``app/containers/construction/schedule.py:2235``)
  documents itself "Deterministic template-based: no LLM"; its body contains
  no ``boq``/``bill``/``quantit``/``contract`` token at all. It has no BOQ
  input to read.
* The string ``BOQ context`` does not exist anywhere in this repository.
  Nothing composed that line; the model wrote it.
* The contract id it named belongs to a different contract year than the
  project under discussion.

So the sentence was not a mis-scoped retrieval that a fence could re-scope.
It was prose shaped like a citation, with no retrieval or tool run behind it.

THE RULE (owner's ruling, 2026-09-01): *the model never writes a Source
line.* Citations are rendered from EVIDENCE OBJECTS — retrieval records and
tool-run records with their actual inputs. Any Source / BOQ / contract
attribution in model prose that matches no evidence record is stripped and
the answer flagged "unverified attribution removed".

This module is the sibling of the cost-grounding gate in ``runtime.py``.
That one grounds the FIGURES; this one grounds the claim about WHERE the
figures came from. Same contract: flag-controlled, never raises, and it
edits the attribution only — an answer's content is never rewritten by it.

SCOPE, stated so the edges are known rather than discovered:

* Attributions are policed; content claims are not. "Clause 8.8.1 says X" is
  the numeric/­retrieval guards' business. "Source: AB-2001-101" is this one's.
* When the turn read the corpus at all, only the attribution surface is
  policed (Source lines, ``BOQ context:`` fragments, parenthesised ids), so a
  working answer that discusses a contract in prose is never mangled.
* When the turn read NO corpus — the F2 shape — every attribution in the
  answer is unbacked by construction, and ids the user did not themselves
  name are stripped wherever they appear.
* ``source_class`` is carried on every record and enforced here, and the
  retrieval marker now emits ``class=`` (the owner's numbered item 2), so
  the classes are live rather than a forward fence. ``project_corpus`` and
  ``master_corpus`` may back an identifier attribution; ``knowledge_base``
  and ``template`` may not. A reference note or a blank form is a real
  document and may still be NAMED as a source — what it may not do is lend
  its identifiers to a claim about this project's contract, which is the
  template-quoted-as-the-contract shape.
* A figure a calculator actually returned is credited to that calculation
  and the tool's own notes (the inputs it was given, and the note lines it
  emitted). A retrieved chunk that did not produce the figure — a ``Source:``
  line or an inline "chunk N" — is not that credit. A chunk that supplied a
  governing input may still be named for that input. An answer with no
  calculator result is left untouched.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_LOG = logging.getLogger(__name__)

FLAG_ENV = "CITATION_PROVENANCE_GATE"

UNVERIFIED_NOTE = (
    "\n\n_Unverified attribution removed: this answer named a source that "
    "nothing in this turn actually read. The answer's content is unchanged; "
    "only the unbacked attribution was taken out._"
)

# PREFIX-YEAR-SEQ, identical in shape to
# ``app.core.rag.retriever._CONTRACT_DOC_ID_RE`` so an id this module strips
# is exactly an id that module would have scoped on. Not a \b pattern:
# underscore-glued filenames ("AB-2002-202_Vol 1.pdf") must still match.
_CONTRACT_ID_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-\d{4}-\d+)(?![A-Za-z0-9])")

# The F2 form, and the general "BOQ context: ..." attribution it belongs to.
# Leading separator is swallowed so removing the fragment does not leave a
# dangling pipe or bullet behind.
_BOQ_ATTRIB_RE = re.compile(
    r"(?:BOQ|Bill\s+of\s+Quantities)\s+context\s*:[^|\n]*",
    re.IGNORECASE,
)

# A markdown table row. Inside one, every pipe is structure: a stripped cell
# must be left EMPTY, never closed up, or the row loses a column and the table
# stops rendering. Found by the table-row fence test, which the first cut of
# this module failed.
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")

# A "Source:" / "Sources:" attribution line. Only whole lines — an inline
# "source: ..." mid-sentence is prose, and prose is out of scope.
_SOURCE_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]+)?\**[ \t]*Sources?[ \t]*:?\**[ \t]*:?[ \t]*"
    r"\**[ \t]*(?P<body>.+?)[ \t]*\**[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# An id in running prose is only an attribution when the grammar makes it one.
# Blanket id-stripping would eat standards references -- "ISO-9001-2015"
# matches the contract-id shape exactly -- so the cue is required.
_ATTRIB_CUE_ID_RE = re.compile(
    r"\b(?P<cue>per|from|source|sources|ref\.?|refer\s+to|see|under|cited\s+in|"
    r"according\s+to|as\s+stated\s+in|as\s+set\s+out\s+in|taken\s+from)"
    r"[ \t]+(?P<id>[A-Za-z]{2,}-\d{4}-\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# A parenthesised id directly after a name is an attribution:
# "Bill 03 - Example Works (AB-2001-101)".
_PAREN_ID_RE = re.compile(
    r"[ \t]*\((?:(?:from|per|source|ref\.?|see)[ \t]+)?"
    r"([A-Za-z]{2,}-\d{4}-\d+)\)",
    re.IGNORECASE,
)

# Per-chunk retrieval marker emitted by
# ``app.core.rag.inject.format_chunks_as_system_message``. ``class=`` does not
# exist yet (numbered item 2); it is read here so that when it lands, this
# gate enforces it without another change.
_MARKER_RE = re.compile(
    r"\[doc_id=(?P<doc>[^\s\]]+)\s+chunk=(?P<chunk>\d+)"
    r"(?P<attrs>[^\]]*)\]",
)
_MARKER_SRC_RE = re.compile(r"\bsrc=([^\]]+?)(?=\s+\w+=|$)")
_MARKER_CLASS_RE = re.compile(r"\bclass=([A-Za-z_]+)")
_MARKER_PAGE_RE = re.compile(r"\bpage=(\d+)")
_MARKER_LAYER_RE = re.compile(r"\blayer=([A-Za-z_]+)")

# Chunks that actually carry priced-bill material. A "BOQ context" claim is
# backed only by one of these, never by a chunk that merely says "bill".
_BOQ_SEMANTIC_RE = re.compile(
    r"\b(?:bill\s+of\s+quantit|\bboq\b|priced\s+bill|rate\s+schedule|"
    r"schedule\s+of\s+rates|unit\s+rate)\b",
    re.IGNORECASE,
)

# Tools PROVEN not to retrieve from the project corpus. Each was verified by
# reading its body on 13b2bf7 -- zero corpus/retriev/search_project/rag_/chunk
# references in any of them:
#   generate_wbs            schedule.py:2235   (template scheduler)
#   construction_calc       __init__.py:2164   (deterministic formula registry)
#   resource_histogram      schedule.py:724
#   commissioning_checklist schedule.py:652
#   cash_flow_forecast      boq.py:1709        (reads a BOQ only from its OWN
#                                               payload -- see boq_backed())
#   wir_form                documents.py:2112
# The default for an unlisted tool is "assume it read the corpus", so a new
# tool can never have a legitimate attribution stripped by omission. Adding a
# tool here is a claim about its body and must be verified the same way.
NON_CORPUS_TOOLS: frozenset[str] = frozenset({
    "generate_wbs",
    "construction_calc",
    "resource_histogram",
    "commissioning_checklist",
    "cash_flow_forecast",
    "wir_form",
})

# Source classes whose material may be cited as the project's own words.
#
# ``master_corpus`` is here and ``knowledge_base``/``template`` are not, and
# the asymmetry is deliberate. The master corpus is the DISCLOSED empty/thin
# fallback (STEP 0b): when it is in play it is the only corpus there is, the
# runtime already labels the answer as a fallback, and refusing its
# identifiers would strip the attribution off every answer on that path. A
# curated reference note or a blank form is the opposite case -- the project
# corpus is right there, and lending a standard form's identifiers to a claim
# about this contract is exactly the template-quoted-as-the-contract defect.
CITABLE_CLASSES: frozenset[str] = frozenset(
    {"project_corpus", "master_corpus", "user"}
)


@dataclass
class EvidenceRecord:
    """One thing that actually happened this turn.

    ``kind`` is "retrieval" (a chunk came back), "tool_run" (a tool executed,
    with the inputs it was actually given) or "user" (the operator's own
    words -- an id the user named is theirs, not a fabrication).
    """

    kind: str
    text: str = ""
    tool: str | None = None
    inputs: str = ""
    source_name: str = ""
    source_class: str = "project_corpus"
    chunk_index: int | None = None
    doc_id: str | None = None
    page: int | None = None
    layer: str | None = None

    @property
    def reads_corpus(self) -> bool:
        if self.kind == "retrieval":
            return bool(self.text.strip())
        if self.kind == "tool_run":
            return (self.tool or "") not in NON_CORPUS_TOOLS
        return False

    @property
    def citable(self) -> bool:
        """May an identifier in this record back an attribution?

        A corpus-reading tool run can: ``search_project_documents`` returns
        real document ids. A template scheduler cannot -- that is the whole
        of F2. The operator's own words always can: an id the user typed is
        theirs, not an invention.
        """
        if self.kind == "user":
            return True
        if self.kind == "tool_run":
            return self.reads_corpus
        return self.source_class in CITABLE_CLASSES

    def identifiers(self) -> set[str]:
        blob = " ".join((self.text, self.inputs, self.source_name))
        return {m.group(1).lower() for m in _CONTRACT_ID_RE.finditer(blob)}

    def carries_boq(self) -> bool:
        """True when this record actually holds priced-bill material.

        Retrieval: the chunk is BOQ-semantic. Tool run: the tool was HANDED a
        non-empty ``boq`` payload -- ``cash_flow_forecast`` legitimately works
        that way, and its citation is honest. ``generate_wbs`` has no such
        field at all, which is why F2's line had nothing behind it.
        """
        if self.kind == "retrieval":
            return bool(_BOQ_SEMANTIC_RE.search(self.text))
        if self.kind == "tool_run":
            try:
                args = json.loads(self.inputs) if self.inputs else {}
            except (ValueError, TypeError):
                return bool(_BOQ_SEMANTIC_RE.search(self.inputs))
            if isinstance(args, dict):
                for key in ("boq", "bill_of_quantities", "priced_boq", "rates"):
                    if args.get(key):
                        return True
            return False
        return False


@dataclass
class Evidence:
    """Every record for one turn, and the questions the gate asks of them."""

    records: list[EvidenceRecord] = field(default_factory=list)

    def any_corpus_read(self) -> bool:
        return any(r.reads_corpus for r in self.records)

    def citable_ids(self) -> set[str]:
        ids: set[str] = set()
        for r in self.records:
            if r.citable:
                ids |= r.identifiers()
        return ids

    def user_ids(self) -> set[str]:
        ids: set[str] = set()
        for r in self.records:
            if r.kind == "user":
                ids |= r.identifiers()
        return ids

    def boq_backed(self) -> bool:
        return any(r.carries_boq() for r in self.records)

    def source_names(self, citable_only: bool = False) -> set[str]:
        """Filenames the turn actually read.

        ``citable_only`` narrows to records whose class may back a claim
        about this project. It matters because a filename can CONTAIN a
        contract id: ``AB-2002-202 Contract Template Vol 4.pdf`` is a
        template, and letting its name back a bare ``Source: AB-2002-202``
        is the template-as-contract defect arriving by the back door -- the id
        would be rescued by the
        very document that must not lend it.
        """
        return {
            r.source_name.lower()
            for r in self.records
            if r.source_name and (r.citable or not citable_only)
        }

    def tool_names(self) -> set[str]:
        return {r.tool for r in self.records if r.kind == "tool_run" and r.tool}


def _enabled() -> bool:
    return os.getenv(FLAG_ENV, "1") not in ("0", "false", "False", "")


def _retrieval_records(rag_sys_msg: dict[str, Any] | None) -> list[EvidenceRecord]:
    """One record per retrieved chunk.

    Per-chunk, never one flat blob: the cost gate learned on 2026-07-14 that a
    single block mixing a rate table with a drawing dimension-table reads as
    rate-semantic as a whole and grounds numbers it should not. The same
    applies to a BOQ claim -- a bundle containing one bill chunk must not make
    every other chunk in it look like a bill.
    """
    content = ((rag_sys_msg or {}).get("content") or "") if rag_sys_msg else ""
    if not content.strip():
        return []
    records: list[EvidenceRecord] = []
    matches = list(_MARKER_RE.finditer(content))
    if not matches:
        # Retrieval happened but not in marker form (a non-agent path passing
        # a flat context). One record, corpus-reading, whole blob.
        return [EvidenceRecord(kind="retrieval", text=content)]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        attrs = m.group("attrs") or ""
        src_m = _MARKER_SRC_RE.search(attrs)
        cls_m = _MARKER_CLASS_RE.search(attrs)
        page_m = _MARKER_PAGE_RE.search(attrs)
        layer_m = _MARKER_LAYER_RE.search(attrs)
        records.append(EvidenceRecord(
            kind="retrieval",
            text=content[start:end].strip(),
            source_name=(src_m.group(1).strip() if src_m else m.group("doc")),
            source_class=(cls_m.group(1).lower() if cls_m else "project_corpus"),
            chunk_index=int(m.group("chunk")),
            doc_id=m.group("doc"),
            page=(int(page_m.group(1)) if page_m else None),
            layer=(layer_m.group(1).lower() if layer_m else None),
        ))
    return records


_TOOL_NAME_RE = re.compile(r"\s*(?:the\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def _platform_bubble(content: str) -> tuple[str, str] | None:
    """``(tool, body)`` for a user-role bubble the platform wrote, else None.

    The platform replays tool output and pre-dispatched runs as user-role
    messages. They are tool runs, not the operator's words: a value in one
    is the tool's, never a figure the user stated.
    """
    from app.agents.runtime import _PREDISPATCH_PREFIX, _TOOL_RESULT_PREFIX

    head = (content or "").lstrip()
    if head.startswith(_TOOL_RESULT_PREFIX):
        rest = head[len(_TOOL_RESULT_PREFIX):]
        name, sep, body = rest.partition("):")
        if sep:
            return name.strip(), body.strip()
        return "", rest
    if head.startswith(_PREDISPATCH_PREFIX):
        match = _TOOL_NAME_RE.match(head[len(_PREDISPATCH_PREFIX):])
        return (match.group("name") if match else ""), head
    return None


def _working_document_records(text: str) -> list[EvidenceRecord]:
    """One retrieval record per file whose whole text a tool returned.

    A fetched document is read evidence the same as a retrieved chunk: a
    Source line naming it, or a figure quoted from it, is backed.
    """
    obj = _parse_dict(text)
    if obj is None:
        return []
    from app.agents.runtime import _working_document_reads

    records: list[EvidenceRecord] = []
    seen: set[str] = set()
    for name, body, doc_id in _working_document_reads([obj, {"result": obj}]):
        key = (doc_id or name).lower()
        if key in seen:
            continue
        seen.add(key)
        records.append(EvidenceRecord(
            kind="retrieval",
            text=body,
            source_name=name,
            doc_id=doc_id or None,
        ))
    return records


def _message_records(messages: Iterable[dict[str, Any]] | None) -> list[EvidenceRecord]:
    """Tool-run records (name + the arguments actually passed + the result)
    and the operator's own words."""
    records: list[EvidenceRecord] = []
    pending_args: dict[str, str] = {}
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                tid = tc.get("id") or fn.get("name") or ""
                args = fn.get("arguments")
                if tid:
                    pending_args[str(tid)] = args if isinstance(args, str) else json.dumps(args or {})
        elif role == "tool":
            name = m.get("name") or None
            tid = str(m.get("tool_call_id") or "")
            text = str(m.get("content") or "")
            records.append(EvidenceRecord(
                kind="tool_run",
                text=text,
                tool=name,
                inputs=pending_args.get(tid, "") or pending_args.get(str(name), ""),
                source_class="tool",
            ))
            records.extend(_working_document_records(text))
        elif role == "user":
            text = str(m.get("content") or "")
            bubble = _platform_bubble(text)
            if bubble is None:
                records.append(EvidenceRecord(kind="user", text=text, source_class="user"))
                continue
            tool, body = bubble
            records.append(EvidenceRecord(
                kind="tool_run", text=body, tool=tool or None, source_class="tool",
            ))
            records.extend(_working_document_records(body))
            if _MARKER_RE.search(body):
                records.extend(_retrieval_records({"content": body}))
    return records


def _tool_passage_records(messages: Iterable[dict[str, Any]] | None) -> list[EvidenceRecord]:
    """Retrieved passages that came back as a tool's result -- any tool result
    shaped ``{"result": {"results": [{"text": ...}, ...]}}`` -- one record per
    passage. Driver mode retrieves with tools, so this is where its passages
    are; the old path's arrive in the injected retrieval message instead."""
    records: list[EvidenceRecord] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        try:
            payload = json.loads(m.get("content") or "")
        except (TypeError, ValueError):
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        hits = result.get("results") if isinstance(result, dict) else None
        for hit in hits if isinstance(hits, list) else []:
            if not isinstance(hit, dict):
                continue
            # The document search returns {document_id, filename, snippet,
            # score, origin} (live; ``chunk`` on the direct path); other tools
            # may say text / content / doc_name / doc_id.
            body = str(hit.get("snippet") or hit.get("chunk") or hit.get("text")
                       or hit.get("content") or "")
            if not body.strip():
                continue
            page = hit.get("page")
            layer = str(hit.get("layer") or hit.get("origin") or "") or None
            doc_id = hit.get("document_id") or hit.get("doc_id")
            records.append(EvidenceRecord(
                kind="retrieval",
                text=body,
                source_name=str(hit.get("filename") or hit.get("doc_name") or hit.get("original_name")
                                or doc_id or ""),
                source_class="general_knowledge" if layer == "general_knowledge" else "project_corpus",
                chunk_index=hit.get("chunk_index") if isinstance(hit.get("chunk_index"), int) else None,
                doc_id=str(doc_id) if doc_id else None,
                page=page if isinstance(page, int) else None,
                layer=layer,
            ))
    return records


def build_evidence(
    rag_sys_msg: dict[str, Any] | None,
    messages: Iterable[dict[str, Any]] | None,
    *,
    tool_passages: bool = False,
) -> Evidence:
    """The turn's evidence objects. This is what a citation is rendered from.
    ``tool_passages``: also count passages returned by retrieval tools."""
    messages = list(messages or [])
    extra = _tool_passage_records(messages) if tool_passages else []
    return Evidence(records=_retrieval_records(rag_sys_msg) + extra + _message_records(messages))


# A removal leaves a hole. Marking it lets the tidy pass clean up exactly the
# lines that changed -- an answer's markdown tables are full of pipes and
# bullets, and a blanket separator sweep would eat them.
SENTINEL = "\x00"


def _tidy(text: str) -> str:
    """Clean up only the lines a strip actually touched.

    Whole-line context is what tells debris from structure: the same pipe is
    an orphan at the end of a sentence and a column boundary inside a table.
    """
    out: list[str] = []
    for ln in text.split("\n"):
        if SENTINEL not in ln:
            out.append(ln)
            continue
        table_row = bool(_TABLE_ROW_RE.match(ln))
        if table_row:
            # Leave the emptied cell in place; the row keeps its columns.
            ln = re.sub(r"[ \t]{2,}", " ", ln.replace(SENTINEL, ""))
            out.append(ln.rstrip())
            continue
        from app.lib.source_labels import tidy_removal_debris

        ln = tidy_removal_debris(ln, SENTINEL)
        ln = re.sub(r"[ \t]*\|[ \t]*$", "", ln)
        ln = re.sub(r"^([ \t]*)\|[ \t]*", r"\1", ln)
        ln = re.sub(r"[ \t]+([.,;:])", r"\1", ln)
        ln = ln.rstrip()
        if not ln.strip(" \t|-*\u2022"):
            ln = ""
        out.append(ln)
    t = "\n".join(out)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.rstrip()


def _strip_boq_attributions(text: str, ev: Evidence) -> tuple[str, list[str]]:
    """Remove a "BOQ context: ..." claim when no record carries a bill.

    This is the F2 line. ``generate_wbs`` has no BOQ field to read, and no
    chunk was retrieved, so the claim had nothing behind it at all.
    """
    if ev.boq_backed():
        return text, []
    removed = [m.group(0).strip() for m in _BOQ_ATTRIB_RE.finditer(text)]
    if not removed:
        return text, []
    return _BOQ_ATTRIB_RE.sub(SENTINEL, text), removed


def _strip_paren_ids(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove "(AB-2001-101)" when no record names that contract."""
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        if m.group(1).lower() in allowed:
            return m.group(0)
        removed.append(m.group(1))
        return SENTINEL

    return _PAREN_ID_RE.sub(repl, text), removed


def _strip_cued_ids(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Strip "per AB-2001-101" / "as set out in AB-2001-101" in running prose.

    Only reached when NOTHING read the corpus this turn: with no retrieval and
    no corpus-reading tool, an id the operator did not name has no possible
    origin but invention. The cue is required so a standards reference --
    "ISO-9001-2015" matches the contract-id shape exactly -- is never touched.
    The cue word goes with the id; a dangling "per" would read worse than the
    fabrication did.
    """
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        if m.group("id").lower() in allowed:
            return m.group(0)
        removed.append(m.group("id"))
        return SENTINEL

    return _ATTRIB_CUE_ID_RE.sub(repl, text), removed


# A chunk citation ("chunk 40", "file.pdf, chunk 40"). The whole Source
# line is one shape; an inline mention is the other. Both are the credit
# for a figure only when a calculator produced that figure.
_CHUNK_REF_RE = re.compile(r"\bchunks?\s*(\d+)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
_INLINE_CHUNK_RE = re.compile(
    r"[\[【]\s*sources?\s*:\s*[^\]】,]{1,200}?\s*,\s*chunks?\s*\d+\s*[\]】]"
    r"|\(\s*(?:sources?\s*:\s*)?[^()]{0,200}?\s*chunks?\s*\d+\s*\)"
    r"|(?<![\w.])[\w][\w.+-]{0,160}\.(?:pdf|docx?|xlsx?|txt|md|csv)"
    r"\s*,?\s*chunks?\s*\d+"
    r"|(?:\b(?:see|per|from|source|sources)\s+)?\bchunks?\s*\d+\b",
    re.IGNORECASE,
)
_CALC_INPUT_SKIP = frozenset({
    "calculation", "name", "calculator", "params", "input", "text", "formula",
    "message", "query", "project_id", "conversation_id", "user_id", "action",
})


@dataclass
class _CalculatorCredit:
    """One successful calculator envelope and the figure it actually returned."""

    tool: str
    calculation: str
    notes: list[str]
    inputs: dict[str, Any]
    result_numbers: list[float]
    default_lines: list[str] = field(default_factory=list)
    #: (currency code, default line) for a currency the formula stamped
    #: because the user gave none. Stated only when the answer shows it.
    currency_defaults: list[tuple[str, str]] = field(default_factory=list)
    #: envelope_provenance badge for this run: provenance (kernel /
    #: reference_table / improvised), the standard it states, and whether the
    #: figure is certified. Threaded onto the figure entry so the "Why this
    #: answer" panel can show WHERE the number came from, not just that a
    #: calculator produced it.
    envelope: dict[str, Any] = field(default_factory=dict)
    #: S-12a: per-input origin stamped at compose time
    #: ({name: "project_data" | "user_text" | "document:<chunk_id>"}). An
    #: operand not stamped here and not matchable to the user's words or a
    #: retrieved chunk is "unresolved" -- never a guessed "derived" (ruling A).
    input_sources: dict[str, str] = field(default_factory=dict)


def _parse_dict(raw: str) -> dict | None:
    if not raw or not str(raw).lstrip().startswith("{"):
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        _LOG.debug(
            "citation_provenance: tool payload is not JSON (%s)",
            type(exc).__name__,
        )
        return None
    return obj if isinstance(obj, dict) else None


def _calculation_payload(raw: str) -> dict | None:
    """The ``run_calculation`` envelope, or one wrapped a single level down."""
    obj = _parse_dict(raw)
    if not obj:
        return None
    if obj.get("status") not in (None, "success", "ok"):
        return None
    if obj.get("calculation") and isinstance(obj.get("result"), dict):
        return obj
    inner = obj.get("result")
    if isinstance(inner, dict) and inner.get("status") in (None, "success", "ok"):
        if inner.get("calculation") and isinstance(inner.get("result"), dict):
            return inner
    return None


def _result_numbers(result: dict) -> list[float]:
    """Numeric fields the calculator returned. Notes are prose, not targets."""
    found: list[float] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            if obj != float("inf") and obj == obj:  # not NaN
                found.append(float(obj))
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key == "notes":
                    continue
                walk(val)
            return
        if isinstance(obj, list):
            for val in obj:
                if not isinstance(val, str):
                    walk(val)

    walk(result)
    return found


def _governing_inputs(raw: str) -> dict[str, Any]:
    obj = _parse_dict(raw) or {}
    params = obj.get("params") if isinstance(obj.get("params"), dict) else None
    if params is None and isinstance(obj.get("input"), dict):
        params = obj["input"]
    if params is None:
        params = obj
    clean: dict[str, Any] = {}
    for key, val in params.items():
        if key in _CALC_INPUT_SKIP or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            clean[str(key)] = val
        elif isinstance(val, str) and val.strip():
            clean[str(key)] = val.strip()
    return clean


def _defaulted_run(calculation: str, inputs: str, ask: str) -> bool:
    """A run on signature defaults alone that the operator's ask did not request.

    Its figures are neither the user's nor a document's, so they are not
    credited as an answer to that ask.
    """
    from app.agents.runtime import _defaulted_run_refusal, _unwrap_rag_folded_operator_text

    args = _parse_dict(inputs) or {}
    operator = _unwrap_rag_folded_operator_text(ask or "")
    return _defaulted_run_refusal(calculation, args, operator) is not None


def _calculator_credits(ev: Evidence) -> list[_CalculatorCredit]:
    credits: list[_CalculatorCredit] = []
    users = [r.text for r in ev.records if r.kind == "user"]
    user_text = "\n".join(users)
    ask = users[-1] if users else ""
    for rec in ev.records:
        if rec.kind != "tool_run":
            continue
        payload = _calculation_payload(rec.text)
        if payload is None:
            continue
        result = payload["result"]
        calculation = str(payload.get("calculation") or "").strip()
        numbers = _result_numbers(result)
        if not calculation or not numbers:
            continue
        if _defaulted_run(calculation, rec.inputs, ask):
            continue
        notes_raw = result.get("notes") or []
        notes = (
            [str(note) for note in notes_raw if str(note).strip()]
            if isinstance(notes_raw, list) else []
        )
        from app.lib.source_labels import (
            calculator_currency_defaults,
            calculator_default_lines,
            stated_calculator_inputs,
        )

        try:
            from app.lib.construction_formulas import envelope_provenance
            envelope = envelope_provenance(calculation, result)
        except Exception:  # noqa: BLE001 -- a missing badge must not drop the credit
            envelope = {}

        raw_inputs = _governing_inputs(rec.inputs)
        src_map = payload.get("_input_sources")
        src_map = src_map if isinstance(src_map, dict) else {}
        credits.append(_CalculatorCredit(
            tool=rec.tool or "construction_calc",
            calculation=calculation,
            notes=notes,
            inputs=stated_calculator_inputs(calculation, raw_inputs, user_text),
            input_sources={str(k): str(v) for k, v in src_map.items()},
            result_numbers=numbers,
            default_lines=calculator_default_lines(
                calculation, result, user_text, passed=raw_inputs,
            ),
            currency_defaults=calculator_currency_defaults(
                calculation, result, user_text, passed=raw_inputs,
            ),
            envelope=envelope,
        ))
    return credits


def _numbers_in(text: str) -> list[float]:
    found: list[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        try:
            found.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return found


def _close(left: float, right: float) -> bool:
    tol = 1e-3 * max(1.0, abs(left), abs(right))
    if abs(left - right) <= tol:
        return True
    return abs(round(left, 3) - round(right, 3)) <= 1e-9


def _has_number(text: str, targets: list[float]) -> bool:
    if not targets:
        return False
    return any(
        _close(found, target)
        for found in _numbers_in(text)
        for target in targets
    )


def _answer_commits(answer: str, credit: _CalculatorCredit) -> bool:
    return _has_number(answer, credit.result_numbers)


def _record_supplies_input(text: str, inputs: dict[str, Any]) -> bool:
    """True when this chunk's words carry a value the calculator was given."""
    low = (text or "").lower()
    for key, val in inputs.items():
        if isinstance(val, str):
            if val.lower() in low:
                return True
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        label = str(key).replace("_", " ").lower()
        if label not in low and str(key).lower() not in low:
            continue
        if _has_number(text, [float(val)]):
            return True
    return False


def _record_backs_figure(rec: EvidenceRecord, credit: _CalculatorCredit) -> bool:
    text = rec.text or ""
    if _has_number(text, credit.result_numbers):
        return True
    return _record_supplies_input(text, credit.inputs)


def _records_for_citation(citation: str, ev: Evidence) -> list[EvidenceRecord]:
    low = (citation or "").lower()
    indexes = {int(n) for n in _CHUNK_REF_RE.findall(citation or "")}
    named = [
        rec for rec in ev.records
        if rec.kind == "retrieval" and rec.source_name and rec.source_name.lower() in low
    ]
    if named and indexes:
        indexed = [rec for rec in named if rec.chunk_index in indexes]
        if indexed:
            return indexed
    if named:
        return named
    if not indexes:
        return []
    return [
        rec for rec in ev.records
        if rec.kind == "retrieval" and rec.chunk_index in indexes
    ]


def _citation_is_false_chunk_credit(
    citation: str,
    ev: Evidence,
    credits: list[_CalculatorCredit],
    answer: str,
) -> bool:
    """A chunk mention crediting a figure the chunk did not produce.

    Kept when the chunk text contains the figure, or a governing input the
    calculator was actually given. No calculator, or no chunk mention, and
    this class does not apply.
    """
    if not credits or not _CHUNK_REF_RE.search(citation or ""):
        return False
    committed = [c for c in credits if _answer_commits(answer, c)]
    if not committed:
        return False
    pool = _records_for_citation(citation, ev)
    if not pool:
        return True
    for rec in pool:
        for credit in committed:
            if _record_backs_figure(rec, credit):
                return False
    return True


def _fmt_credit(credit: _CalculatorCredit) -> str:
    """The credit a user reads: the formula's declared display name and the
    inputs the user stated, the same entry the Sources panel shows. The
    calculator's result notes are the result, not its source."""
    from app.lib.source_labels import calculator_label

    return "Source: " + calculator_label(credit.calculation, credit.inputs)


_MIN_STEM = 5


def _source_name_forms(name: str) -> list[str]:
    """A read document's name as an answer may write it: whole, its base
    name, or the base name without the extension."""
    low = (name or "").strip().lower()
    if not low:
        return []
    base = re.split(r"[\\/]", low)[-1].strip()
    stem = re.sub(r"\.[a-z0-9]{1,5}$", "", base).strip()
    forms = [low, base]
    if len(stem) >= _MIN_STEM:
        forms.append(stem)
    return [f for f in dict.fromkeys(forms) if f]


def _line_names_source(line_low: str, name: str) -> bool:
    """The line names this document, or is a shortened form of its name."""
    if not line_low.strip():
        return False
    for form in _source_name_forms(name):
        if form in line_low or line_low.strip() in form:
            return True
    return False


def _credit_already_present(answer: str, credit: _CalculatorCredit) -> bool:
    return _fmt_credit(credit) in (answer or "")


def _missing_calculator_credit(
    out: str,
    credits: list[_CalculatorCredit],
    answer: str,
) -> str:
    lines: list[str] = []
    for credit in credits:
        if not _answer_commits(answer, credit):
            continue
        if _credit_already_present(out, credit):
            continue
        line = _fmt_credit(credit)
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


def _strip_inline_chunk_credits(
    text: str,
    ev: Evidence,
    credits: list[_CalculatorCredit],
    answer: str,
) -> tuple[str, list[str]]:
    """Drop an inline chunk credit that is not a whole Source line.

    ``_SOURCE_LINE_RE`` only sees a line that begins with Source. "see chunk
    40" and "(file.pdf chunk 40)" sit in the sentence and used to survive.
    """
    if not credits:
        return text, []
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        if not _citation_is_false_chunk_credit(m.group(0), ev, credits, answer):
            return m.group(0)
        removed.append(m.group(0).strip())
        return SENTINEL

    return _INLINE_CHUNK_RE.sub(repl, text), removed


def _strip_source_lines(
    text: str,
    ev: Evidence,
    credits: list[_CalculatorCredit] | None = None,
    answer: str | None = None,
) -> tuple[str, list[str]]:
    """Drop a Source line whose named sources are all unbacked.

    A line naming something the evidence does have is left alone: the job is
    to remove invention, not to delete correct attributions.

    A filename match is not enough when the line is a chunk credit for a
    figure a calculator returned and that chunk did not produce. The
    calculation's own line (it names the tool) is still kept.
    """
    allowed_ids = ev.citable_ids()
    tools = ev.tool_names()
    credits = credits or []
    judged = text if answer is None else answer
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        body = m.group("body")
        low = body.lower()
        ids = {i.group(1).lower() for i in _CONTRACT_ID_RE.finditer(body)}
        if ids & allowed_ids:
            return m.group(0)
        # Named a document rather than an id: backed when an evidence source
        # name meets it either way round (the answer may shorten the filename,
        # or quote it in full).
        #
        # When the line DOES name an identifier and none of them are allowed,
        # only a citable record's filename may rescue it. Otherwise a template
        # or reference note whose own name carries the contract id would back
        # the attribution the class exists to refuse.
        #
        # That rescue does not cover a chunk credit for a calculator figure
        # the chunk did not produce (Dewatering-p2: the proposal chunk was
        # retrieved, and the factor of safety came from the tool).
        names = ev.source_names(citable_only=bool(ids))
        name_hit = any(_line_names_source(low, n) for n in names)
        if name_hit and not _citation_is_false_chunk_credit(body, ev, credits, judged):
            return m.group(0)
        # A line naming the TOOL that ran is a RENDERED citation, not an
        # invention -- and R3 requires exactly that self-declaration from the
        # template scheduler. Never strip it. The name a user reads (the
        # tool's or the formula's declared display name) is the same citation.
        # The rendered line is the registry label. A Source line the model
        # wrote in its own words, which merely contains that display name,
        # is not the credit.
        from app.lib.source_labels import (
            calculator_label,
            formula_display_name,
            tool_display_name,
        )

        if credits:
            for c in credits:
                label = calculator_label(c.calculation, c.inputs)
                if label and label.lower() in low:
                    lead = re.match(r"[ \t]*(?:[-*•][ \t]+)?", m.group(0))
                    return (lead.group(0) if lead else "") + _fmt_credit(c)
            displays = [formula_display_name(c.calculation) for c in credits]
            shown_now = [t for t in displays if t]
            shown_now.extend(tool_display_name(t) for t in tools)
            if any(name and name.lower() in low for name in shown_now):
                # The calculator did run: the model's own wording of its
                # credit gives way to the rendered one. Nothing unbacked
                # was named, so this is not a removal.
                return SENTINEL
            if "platform calculator" in low:
                removed.append(m.group(0).strip())
                return SENTINEL

        shown = [*tools, *(tool_display_name(t) for t in tools),
                 *(formula_display_name(c.calculation) for c in credits)]
        for t in shown:
            if t and t.lower() in low:
                return m.group(0)
        if ev.any_corpus_read() and not ev.source_names() and not ids:
            # Retrieval happened in a form carrying no source names; the gate
            # cannot judge this line, so it does not touch it.
            return m.group(0)
        removed.append(m.group(0).strip())
        return SENTINEL

    return _SOURCE_LINE_RE.sub(repl, text), removed


# A sentence that points at retrieved material as the authority for a claim.
# The noun is the kind of source; the verb is the claim that it said so.
_RETRIEVAL_AUTHORITY_RE = re.compile(
    r"\b(?:retrieved|excerpts?|commentary|knowledge[-\s]?base|project documents)\b",
    re.IGNORECASE,
)
_CITE_VERB_RE = re.compile(
    r"\b(?:indicates?|shows?|states?|says|said|provides?|specifies|requires?|"
    r"suggests?|according\s+to|as\s+stated|as\s+set\s+out)\b",
    re.IGNORECASE,
)
_DENIAL_RE = re.compile(
    r"\b(?:not|no|never|isn['’]t|aren['’]t|wasn['’]t|weren['’]t|doesn['’]t|"
    r"don['’]t|cannot|can['’]t|without)\b",
    re.IGNORECASE,
)


def _sentence_names_evidence(sentence: str, ev: Evidence) -> bool:
    low = (sentence or "").lower()
    return any(_line_names_source(low, name) for name in ev.source_names())


def _unbacked_retrieval_sentence(sentence: str, ev: Evidence) -> bool:
    """A retrieval citation that names no document this turn actually read."""
    if _RETRIEVAL_AUTHORITY_RE.search(sentence or "") is None:
        return False
    if _CITE_VERB_RE.search(sentence or "") is None:
        return False
    if _DENIAL_RE.search(sentence or ""):
        return False
    return not _sentence_names_evidence(sentence, ev)


def _strip_unbacked_retrieval_prose(
    text: str,
    ev: Evidence,
    credits: list[_CalculatorCredit],
    answer: str,
) -> tuple[str, list[str]]:
    """Drop a retrieval attribution the Sources panel cannot show.

    When a calculator is the credited source of the figure, a sentence that
    cites retrieved material without naming an evidence document is an
    attribution with nothing behind it. A sentence that names a document
    the turn read stays; that document is a source beside the calculator.
    """
    if not any(_answer_commits(answer, credit) for credit in credits):
        return text, []
    removed: list[str] = []
    lines: list[str] = []
    for line in (text or "").split("\n"):
        parts = re.split(r"((?<=[.!?])\s+)", line)
        kept: list[str] = []
        for part in parts:
            if re.fullmatch(r"\s+", part or ""):
                kept.append(part)
                continue
            if _unbacked_retrieval_sentence(part, ev):
                removed.append(part.strip())
                if kept and re.fullmatch(r"\s+", kept[-1] or ""):
                    kept.pop()
                continue
            kept.append(part)
        lines.append("".join(kept).strip())
    if not removed:
        return text, []
    return "\n".join(line for line in lines if line.strip()), removed


def _missing_default_lines(
    out: str,
    credits: list[_CalculatorCredit],
    answer: str,
) -> str:
    lines: list[str] = []
    for line in _credited_default_lines(credits, answer):
        if line not in (out or "") and line not in lines:
            lines.append(line)
    return "\n".join(lines)


def _credited_default_lines(credits: list[_CalculatorCredit], answer: str) -> list[str]:
    """Every default the answer's committed figures rest on.

    A currency default counts once the answer writes that currency: the
    figure is then shown in a unit the user did not give.
    """
    lines: list[str] = []
    for credit in credits:
        if not _answer_commits(answer, credit):
            continue
        lines.extend(line for line in credit.default_lines if line)
        for code, line in credit.currency_defaults:
            if line and re.search(
                rf"(?<![A-Za-z]){re.escape(code)}(?![A-Za-z])", answer or "",
            ):
                lines.append(line)
    return lines


# A sentence that says no default (or no other default) was used: the
# negation governs "default" within a few words, either side.
_DEFAULT_DENIAL_RE = re.compile(
    r"(?:\b(?:no|not|none|never|without)\b|n['’]t\b)(?:\W+\w+){0,4}?\W+defaults?\b"
    r"|\bdefaults?\b(?:\W+\w+){0,3}?\W+(?:not|never)\b",
    re.IGNORECASE,
)


def _denied_params(sentence: str, credits: list[_CalculatorCredit]) -> tuple[bool, bool]:
    """``(names a parameter, names a defaulted parameter)`` for this sentence."""
    from app.lib.source_labels import calculator_parameter_words

    low = sentence.lower()
    named = defaulted = False
    for credit in credits:
        for key, words, is_default in calculator_parameter_words(
            credit.calculation, credit.inputs,
        ):
            if len(words) >= 3 and re.search(rf"\b{re.escape(words)}\b", low):
                named = True
                defaulted = defaulted or is_default
    return named, defaulted


def _strip_default_denials(
    text: str,
    credits: list[_CalculatorCredit],
    answer: str,
    *,
    whole_answer: bool = True,
) -> str:
    """Drop a sentence denying defaults when the answer's credit lists some.

    A denial about one input the user did state stays; a blanket one ("no
    other defaults were used") or one about a defaulted input contradicts
    the default lines beneath it. ``whole_answer=False``: ``answer`` is one
    piece of a streamed answer, so any credited run's defaults count.
    """
    committed = [c for c in credits if not whole_answer or _answer_commits(answer, c)]
    if whole_answer:
        stated = _credited_default_lines(committed, answer)
    else:
        stated = [line for credit in committed for line in credit.default_lines if line]
    if not stated:
        return text
    lines: list[str] = []
    changed = False
    for line in (text or "").split("\n"):
        parts = re.split(r"((?<=[.!?])\s+)", line)
        kept: list[str] = []
        for part in parts:
            body = part.strip()
            if (
                body
                and _DEFAULT_DENIAL_RE.search(part)
                and not any(body in dl or dl in part for dl in stated)
            ):
                named, defaulted = _denied_params(part, committed)
                if defaulted or not named:
                    changed = True
                    if kept and re.fullmatch(r"\s+", kept[-1] or ""):
                        kept.pop()
                    continue
            kept.append(part)
        rebuilt = "".join(kept)
        if line.strip() and not rebuilt.strip():
            continue
        lines.append(rebuilt.rstrip() if rebuilt != line else line)
    return "\n".join(lines) if changed else text


def gate(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    *,
    tool_passages: bool = False,
    annotate: bool = True,
) -> str:
    """Strip attributions no evidence record backs; flag the answer when any
    were removed. ``annotate=False`` strips only (one piece of a streamed
    answer; see ``closing_notes``). Content is never rewritten. Never raises -- a gate that can
    break an answer is a gate that gets switched off."""
    try:
        if not _enabled() or not text or not text.strip():
            return text
        ev = build_evidence(rag_sys_msg, messages, tool_passages=tool_passages)
        credits = _calculator_credits(ev)
        removed: list[str] = []

        out, r = _strip_boq_attributions(text, ev)
        removed += r
        out, r = _strip_source_lines(out, ev, credits, text)
        removed += r
        out, r = _strip_unbacked_retrieval_prose(out, ev, credits, text)
        removed += r
        out, r = _strip_inline_chunk_credits(out, ev, credits, text)
        removed += r

        if ev.any_corpus_read():
            out, r = _strip_paren_ids(out, ev.citable_ids())
            removed += r
        else:
            # Nothing read the corpus: every attribution in the answer is
            # unbacked by construction. The operator's own ids still stand.
            allowed = ev.user_ids()
            out, r = _strip_paren_ids(out, allowed)
            removed += r
            out, r = _strip_cued_ids(out, allowed)
            removed += r

        if SENTINEL in out:
            out = _tidy(out)
        out = _strip_default_denials(out, credits, text, whole_answer=annotate)
        if not annotate:
            # A piece of a streamed answer: strip only; the stream adds the
            # calculator credit and the note once, at its end (closing_notes).
            if removed:
                _LOG.warning(
                    "citation_provenance: removed %d unbacked attribution(s): %s",
                    len(removed), removed[:5],
                )
                if any(names_a_reference(r) for r in removed):
                    _REMOVED_IN_PIECE.set(True)
            return text if out == text else out
        credit = _missing_calculator_credit(out, credits, text)
        defaults = _missing_default_lines(out, credits, text)
        extra = "\n".join(part for part in (defaults, credit) if part)
        if not removed and not extra:
            return text if out == text else out
        if removed:
            _LOG.warning(
                "citation_provenance: removed %d unbacked attribution(s): %s",
                len(removed), removed[:5],
            )
        if extra:
            out = out.rstrip() + "\n\n" + extra
        if any(names_a_reference(r) for r in removed):
            out += UNVERIFIED_NOTE
        return out
    except Exception:  # noqa: BLE001 -- a gate must never break an answer
        _LOG.exception("citation_provenance failed; passing answer through")
        return text


import contextvars as _contextvars

# A reference a reader could go and check: a stored file name, or a code,
# volume, table, clause or similar designation carrying its number ("ACI
# 318", "Vol 6", "Table 7.3.1.1", "BS8110").
_FILE_REFERENCE_RE = re.compile(r"\b[\w-]+\.(?:pdf|docx?|xlsx?|md|txt|csv|dwg|xer)\b", re.IGNORECASE)
_DESIGNATION_RE = re.compile(r"\b[A-Z][A-Za-z]{0,11}[ .-]?\d[\d.\-]*")
# The platform's own passage keys (a chunk number, a document id) point at a
# passage too.
_PASSAGE_KEY_RE = re.compile(r"\b(?:chunk|doc[_ ]?id|src)\s*[:#=]?\s*\w*\d", re.IGNORECASE)


def names_a_reference(attribution: str) -> bool:
    """Does a removed attribution name a source a reader could look up? The
    note "named a source that nothing read" is true only for those. A line
    naming no source -- a bare "Sources:" heading, "Source: the platform's
    calculator, on your inputs", "this is general knowledge" -- is removed
    without the note."""
    text = str(attribution or "")
    return bool(_FILE_REFERENCE_RE.search(text) or _DESIGNATION_RE.search(text)
                or _PASSAGE_KEY_RE.search(text))


#: Set when a piece checked with annotate=False had an attribution removed.
_REMOVED_IN_PIECE: "_contextvars.ContextVar[bool]" = _contextvars.ContextVar(
    "citation_removed_in_piece", default=False)


def closing_notes(answer: str, messages: list[dict[str, Any]] | None, *,
                  tool_passages: bool = False, removed: bool = False) -> str:
    """What a streamed answer gets once, at its end: the calculator credit it
    does not already carry, and the note when any piece lost an attribution."""
    try:
        ev = build_evidence(None, messages, tool_passages=tool_passages)
        credits = _calculator_credits(ev)
        credit = _missing_calculator_credit(answer, credits, answer)
        defaults = _missing_default_lines(answer, credits, answer)
    except Exception:  # noqa: BLE001 -- a gate must never break an answer
        _LOG.exception("closing_notes failed; no credit appended")
        credit = ""
        defaults = ""
    extra = "\n".join(part for part in (defaults, credit) if part)
    out = ("\n\n" + extra) if extra else ""
    if removed:
        out += UNVERIFIED_NOTE
    return out


# ── Figure provenance ────────────────────────────────────────────────────────
#
# The same evidence objects, asked one more question: where did each FIGURE in
# the answer come from? Every figure (a number carrying a unit, a currency or a
# percent) gets an entry -- user input, calculator (formula and inputs),
# project document or general knowledge (document and page). A figure with no
# entry is not stated (when enforced): its clause is removed. One mechanism:
# the calculator credit, retrieval records and user words above are what back
# a figure here.

_PROV_FIGURE_RE = re.compile(
    r"(?<![\w.])(?P<pre>(?:SAR|AED|USD|EUR|GBP|QAR|KWD|OMR|BHD|\$|£|€)\s?)?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?P<post>\s?(?:%|(?:mm|cm|km|m|m2|m²|m3|m³|kg|t|kN|kN/m|kN/m2|kN/m²|MPa|kPa|Pa|"
    r"N/mm2|N/mm²|psi|ksi|days?|weeks?|months?|years?|hours?|hrs?|mins?|kW|kWh|lux|lx|"
    r"°C|L|litres?|liters?|tonnes?|tons?|nos?|sqm|cum)\b))?",
    re.IGNORECASE,
)


def _figure_mentions(text: str) -> list[tuple[re.Match, float]]:
    out: list[tuple[re.Match, float]] = []
    for m in _PROV_FIGURE_RE.finditer(text or ""):
        if not (m.group("pre") or m.group("post")):
            continue  # a bare number (clause 4.2, page 12, a year) is not a figure
        try:
            out.append((m, float(m.group("num").replace(",", ""))))
        except ValueError:
            continue
    return out


def _nums(text: str) -> list[float]:
    out = []
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _match(value: float, pool: Iterable[float]) -> bool:
    return any(_close(value, p) for p in pool)


def _rescaled(values: Iterable[float]) -> set[float]:
    vals = set(values)
    return vals | {v * 1000 for v in vals} | {v / 1000 for v in vals}


def _working(values: set[float]) -> set[float]:
    """Pairwise working on grounded numbers: a substitution line explains the
    calculator's arithmetic; it does not invent a figure."""
    out: set[float] = set()
    vals = sorted(values)[:40]
    for a in vals:
        for b in vals:
            out.add(a * b)
            out.add(a + b)
            out.add(abs(a - b))
            if b:
                out.add(a / b)
    return out


def _entry_for_record(rec: EvidenceRecord, figure: str) -> dict[str, Any]:
    from app.agents.provenance_trail import SOURCE_GENERAL, SOURCE_PROJECT

    gk = (rec.layer or "") == "general_knowledge"
    return {
        "figure": figure, "source": SOURCE_GENERAL if gk else SOURCE_PROJECT,
        "doc_id": rec.doc_id, "doc_name": rec.source_name or None, "page": rec.page,
        "chunk_index": rec.chunk_index,
    }


def _input_unit(calculation: str, name: str) -> str:
    """The declared registry unit for one formula input ('' when none/'-')."""
    try:
        from app.lib import formula_registry
        spec = formula_registry.get(calculation or "")
        param = spec.inputs.get(name) if spec is not None else None
    except Exception:  # noqa: BLE001 -- a lookup miss must not drop provenance
        _LOG.debug("input-unit lookup failed for %s.%s", calculation, name, exc_info=True)
        return ""
    unit = "" if param is None else str(getattr(param, "unit", param) or "")
    return "" if unit == "-" else unit


def _input_rows(
    calculation: str,
    inputs: dict[str, Any] | None,
    user_vals: set[float],
    retrievals: list[EvidenceRecord],
    stamped: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """S-12a: each calculator input as {name, value, unit, source}. The unit is
    the declared registry Param. The source is, in order: the origin STAMPED at
    compose time (project_data / user_text / document:<chunk_id>); else
    user_text when the operator stated the value; else document:<chunk_id> when
    it appears in a retrieved chunk; else "unresolved". An unstamped operand is
    never guessed as "derived" -- that would invent a fact (ruling A)."""
    stamped = stamped or {}
    rows: list[dict[str, Any]] = []
    scaled_user = _rescaled(user_vals)
    for name, value in (inputs or {}).items():
        source = stamped.get(name) or ""
        if not source:
            try:
                fval: float | None = float(value)
            except (TypeError, ValueError):
                fval = None
            if fval is not None and _match(fval, scaled_user):
                source = "user_text"
            elif fval is not None:
                for rec in retrievals:
                    if _match(fval, _nums(rec.text)):
                        cid = rec.doc_id or rec.source_name or "chunk"
                        source = (f"document:{cid}#{rec.chunk_index}"
                                  if rec.chunk_index is not None else f"document:{cid}")
                        break
        rows.append({"name": name, "value": value,
                     "unit": _input_unit(calculation, name),
                     "source": source or "unresolved"})
    return rows


def figure_provenance(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    *,
    enforce: bool = True,
    tool_passages: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(answer, provenance)``: every figure credited to its source --
    the user's own words, a calculator run (formula and inputs), or a
    retrieved chunk (layer, document, page). With ``enforce`` a figure that
    matches none of them is removed (its clause); without, the answer is
    returned unchanged and only the record is built. Never raises -- on
    failure the answer passes through with an empty record."""
    from app.agents import provenance_trail as pt

    try:
        # Empty/whitespace text yields no figure mentions, so the normal flow
        # already returns no record; a non-branching guard keeps None-safety
        # without an early-return `if` (which would be an equivalent mutant).
        text = text or ""
        # S-12a PR-2: a composed project-rate calc (delay_damages_daily) is
        # mirrored as a credited turn so its rate + ACA read project_data, not
        # "unresolved". Display is untouched; this only adds provenance.
        from app.agents.runtime import credit_composed_rate_turns
        credit_composed_rate_turns(messages)
        ev = build_evidence(rag_sys_msg, messages, tool_passages=tool_passages)
        user_vals: set[float] = set()
        for rec in ev.records:
            if rec.kind == "user":
                user_vals = set(_nums(rec.text))  # the latest operator turn wins
        retrievals = [r for r in ev.records if r.kind == "retrieval" and r.text]
        calc_entries: list[tuple[set[float], dict[str, Any]]] = []
        for credit in _calculator_credits(ev):
            nums = set(credit.result_numbers) | set(_nums(json.dumps(credit.inputs, default=str)))
            meta: dict[str, Any] = {
                "source": pt.SOURCE_CALCULATOR, "formula": credit.calculation,
                "inputs": _input_rows(credit.calculation, credit.inputs, user_vals,
                                      retrievals, credit.input_sources)}
            # Thread the envelope_provenance badge + the default lines onto the
            # entry so "Why this answer" can state WHERE the figure came from
            # (kernel / reference_table / improvised, the standard, certified)
            # and which inputs the formula had to default. S-12a.
            if credit.envelope:
                meta["provenance"] = credit.envelope.get("provenance")
                meta["standard"] = credit.envelope.get("standard")
                meta["certified"] = credit.envelope.get("certified")
            defaults = list(credit.default_lines) + [line for _code, line in credit.currency_defaults]
            if defaults:
                meta["defaults"] = defaults
            calc_entries.append((nums, meta))
        # A platform workflow's own result is a source too: the figures its
        # answer carries (an EVM's CV and EAC, a cash flow's month rows) were
        # produced by the run, not by the model.
        for rec in ev.records:
            if rec.kind != "tool_run":
                continue
            try:
                payload = json.loads(rec.text or "")
            except (TypeError, ValueError):
                continue
            if (isinstance(payload, dict) and payload.get("ok") and payload.get("workflow")
                    and isinstance(payload.get("answer"), str)):
                nums = set(_nums(payload["answer"]))
                if nums:
                    calc_entries.append((nums, {"source": pt.SOURCE_CALCULATOR,
                                                "formula": str(payload["workflow"]), "inputs": {}}))
        calc_pool: set[float] = set()
        for nums, _meta in calc_entries:
            calc_pool |= nums
        working = _working(_rescaled(calc_pool | user_vals)) if calc_entries else set()

        entries: list[dict[str, Any]] = []
        bad: list[re.Match] = []
        for m, value in _figure_mentions(text):
            fig = m.group(0).strip()
            entry: dict[str, Any] | None = None
            if _match(value, _rescaled(user_vals)):
                entry = {"figure": fig, "source": pt.SOURCE_USER}
            if entry is None:
                for nums, meta in calc_entries:
                    if _match(value, _rescaled(nums)):
                        entry = {"figure": fig, **meta}
                        break
            if entry is None:
                for rec in retrievals:
                    if _match(value, _nums(rec.text)):
                        entry = _entry_for_record(rec, fig)
                        break
            if entry is None and working and _match(value, working):
                entry = {"figure": fig, "source": pt.SOURCE_CALCULATOR, "working": True}
            if entry is None:
                bad.append(m)
                entries.append({"figure": fig, "source": None})
            else:
                entries.append(entry)
        if not enforce:
            return text, entries
        out = text
        for m in reversed(bad):
            start, end = _clause_span(out, m.start(), m.end())
            out = out[:start].rstrip(" ,;") + out[end:]
        if bad:
            _LOG.info("figure_provenance: removed %d figure(s) with no source", len(bad))
        return out, [e for e in entries if e.get("source")]
    except Exception:  # noqa: BLE001 -- a gate must never break an answer
        _LOG.exception("figure_provenance failed; passing answer through")
        return text, []


def _clause_span(text: str, start: int, end: int) -> tuple[int, int]:
    """The parenthetical around [start, end), else the sentence around it."""
    open_ = text.rfind("(", 0, start)
    close = text.find(")", end)
    if open_ != -1 and close != -1 and ")" not in text[open_:start] and "\n" not in text[open_:close]:
        return open_, close + 1
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    rights = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(rights) + 1 if rights else len(text)
    return (left + 1 if left != -1 else 0), right
