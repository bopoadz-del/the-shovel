"""What a driver answer shows as its sources: the passages its own searches
returned, in the shape the chat page's Sources panel reads, and a line
naming each calculation tool it ran.

The same fixed rules as the old path's panel: confidence bands by score
(High >= 0.75, Medium >= 0.5, else Low), a layer label, and names of
documents outside the user's own project scrubbed of identifiers.
Passages that back a figure in the answer come first; otherwise the best
few by score.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

#: Retrieval and navigation tools: their results are passages, not figures.
RETRIEVAL_TOOLS = frozenset({"search_project_documents", "search_general_knowledge",
                             "fetch_document", "list_project_documents"})
#: Tools whose credit is written elsewhere (the attribution gate credits
#: construction_calc) or that produce no figures.
_NO_CREDIT_LINE = frozenset({"select_hat", "construction_calc", "remember_fact", "delegate_to_agent"})

_LAYER_LABELS = {"own": "Project document", "master_corpus": "Master Corpus (fallback)",
                 "general_knowledge": "Knowledge base"}
_MAX_SOURCES = 5


def _hits(trail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in trail:
        if m.get("role") != "tool" or m.get("name") not in RETRIEVAL_TOOLS:
            continue
        try:
            payload = json.loads(m.get("content") or "")
        except (TypeError, ValueError):
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        for hit in (result or {}).get("results") or [] if isinstance(result, dict) else []:
            if isinstance(hit, dict):
                out.append(hit)
    return out


def _confidence(score: float) -> str:
    return "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"


def _page(hit: Dict[str, Any]) -> str:
    """A page or section a user can open: a positive page number or a named
    section. -1 / 0 / empty mean "no page" and are never shown."""
    page = hit.get("page")
    if isinstance(page, (int, float)) and not isinstance(page, bool):
        return str(int(page)) if page > 0 else ""
    text = str(page or hit.get("page_or_section") or "").strip()
    return "" if text.lstrip("-").isdigit() and int(text) <= 0 else text


def panel(
    trail: List[Dict[str, Any]],
    provenance: List[Dict[str, Any]],
    answer: str = "",
    calculated: bool = False,
) -> List[Dict[str, Any]]:
    """Sources for the Sources panel, one per document.

    When ``answer`` uses a working document's text (a file read on this
    turn, not a retrieved hit), that file is listed by its shown name and
    hits the answer did not use are left out. With no such use, the best
    retrieved hits stay.
    """
    backing = {str(e.get("doc_id") or "") for e in provenance if e.get("doc_id")}
    backing |= {str(e.get("doc_name") or "") for e in provenance if e.get("doc_name")}
    hits = _hits(trail)
    used_rows: List[Dict[str, Any]] = []
    if (answer or "").strip():
        from app.agents.runtime import _answer_quotes_passage, _working_document_sources

        used_rows = _working_document_sources(answer, trail, {})
        if used_rows:
            def _hit_used(hit: Dict[str, Any]) -> bool:
                doc_id = str(hit.get("document_id") or hit.get("doc_id") or "")
                name = _hit_name(hit)
                if doc_id in backing or name in backing or str(hit.get("filename") or "") in backing:
                    return True
                snippet = str(hit.get("snippet") or hit.get("text") or "")
                return bool(snippet) and _answer_quotes_passage(answer, snippet)

            hits = [hit for hit in hits if _hit_used(hit)]
    if calculated and (answer or "").strip():
        # A calculator answered: only passages the answer actually used are
        # its sources, never the best of whatever the searches returned.
        from app.agents.runtime import _answer_quotes_passage

        def _backs(hit: Dict[str, Any]) -> bool:
            doc_id = str(hit.get("document_id") or hit.get("doc_id") or "")
            name = _hit_name(hit)
            snippet = str(hit.get("snippet") or hit.get("text") or "")
            return bool({doc_id, name, str(hit.get("filename") or "")} & backing) or (bool(snippet) and _answer_quotes_passage(answer, snippet))
        hits = [hit for hit in hits if _backs(hit)]
    hits.sort(key=lambda h: (
        not ({str(h.get("document_id") or h.get("doc_id") or ""), str(h.get("filename") or "")} & backing),
        -float(h.get("score") or 0.0)))
    seen, out = set(), []
    for h in hits:
        doc_id = str(h.get("document_id") or h.get("doc_id") or "")
        name = _hit_name(h) or doc_id
        if not name or (doc_id or name) in seen:
            continue
        seen.add(doc_id or name)
        layer = str(h.get("origin") or h.get("layer") or "own")
        score = float(h.get("score") or 0.0)
        out.append({
            "doc_id": doc_id,
            "doc_name": library_title(name) if layer in LIBRARY_LAYERS else name,
            "page_or_section": _page(h),
            "score": round(score, 3),
            "confidence": _confidence(score),
            "layer": layer,
            "layer_label": _LAYER_LABELS.get(layer, "Knowledge base"),
        })
        if len(out) >= _MAX_SOURCES:
            break
    if used_rows and not calculated:
        seen_ids = {str(row.get("doc_id") or "") for row in out}
        seen_names = {str(row.get("doc_name") or "") for row in out}
        extra = [
            row for row in used_rows
            if str(row.get("doc_id") or "") not in seen_ids
            and str(row.get("doc_name") or "") not in seen_names
        ]
        out = [*extra, *out][:_MAX_SOURCES]
    return out


# Layers whose files the platform curates, not a user: their stored file
# names are storage keys ("design_values_metric.md") and read as their words.
# A user's own document keeps the name its owner gave it.
LIBRARY_LAYERS = frozenset({"general_knowledge"})
_STORED_NAME_RE = re.compile(r"^(?:.*[\\/])?([^\\/]+?)(?:\.[A-Za-z0-9]{1,5})?$")


def library_title(name: str) -> str:
    """A library file's stored name as words: no folder, no extension, no
    underscores."""
    m = _STORED_NAME_RE.match((name or "").strip())
    base = m.group(1) if m else (name or "")
    if " " in base or "_" not in base:
        return name  # already written as words ("Formwork notes.md")
    return re.sub(r"_+", " ", base).strip() or name


def library_names(trail: List[Dict[str, Any]]) -> Dict[str, str]:
    """{stored file name: title} for every library passage this turn read."""
    out: Dict[str, str] = {}
    for hit in _hits(trail):
        layer = str(hit.get("origin") or hit.get("layer") or "")
        stored = str(hit.get("filename") or hit.get("doc_name") or "").strip()
        if layer in LIBRARY_LAYERS and stored:
            out[stored] = str(hit.get("title") or "").strip() or library_title(stored)
    return out


def _hit_name(hit: Dict[str, Any]) -> str:
    """What a reader calls a passage's document: its title when the hit
    carries one, then its source label, then the stored file name."""
    for key in ("title", "source_label", "original_name", "filename", "doc_name"):
        value = str(hit.get(key) or "").strip()
        if value:
            return value
    return ""


def tool_credit_lines(calls: List[Dict[str, Any]]) -> str:
    """'Calculated with: <tool's display name> (<inputs>)' for each
    calculation tool run -- the name the tool declares, never its id."""
    from app.lib.source_labels import tool_label

    lines = []
    for name, args in calls:
        if name in RETRIEVAL_TOOLS or name in _NO_CREDIT_LINE or name.startswith("run_workflow"):
            continue
        shown = {k: v for k, v in args.items() if k not in ("message", "user_message", "text", "brief")}
        line = "Calculated with: " + tool_label(name, shown)
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)
