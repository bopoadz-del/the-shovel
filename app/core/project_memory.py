"""Project memory — durable facts accumulated across a project's documents.

Roadmap V2 · Epic 3 (Project memory).

Documents are no longer processed in isolation: when one is analysed, the
durable facts it yields (contract value, key dates, parties, BOQ totals…) are
written to the project, and later questions can be answered from that
accumulated knowledge without re-attaching the source document.
"""

import os
from typing import Any

from app.core import projects as store

# Durable fields worth remembering, mapped to a human label.
DURABLE_FIELDS: dict[str, str] = {
    "contract_value": "Contract value",
    "contract_type": "Contract type",
    "commencement_date": "Commencement date",
    "completion_date": "Completion date",
    "ld_rate": "Liquidated damages rate",
    "liquidated_damages": "Liquidated damages",
    "defects_liability_period": "Defects liability period",
    "retention_percent": "Retention percentage",
    "employer": "Employer",
    "contractor": "Contractor",
    "drawing_number": "Drawing number",
    "revision": "Revision",
    "scale": "Drawing scale",
    "boq_total": "BOQ total",
    "project_name": "Project name",
}

_JUNK = {"", "none", "null", "n/a", "unknown", "0", "0.0"}


def extract_facts(result: dict[str, Any]) -> list[dict[str, str]]:
    """Pull durable facts from an analysis result (depth-limited dict scan)."""
    found: dict[str, str] = {}

    def scan(node: Any, depth: int = 0) -> None:
        if depth > 2 or not isinstance(node, dict):
            return
        for k, v in node.items():
            if k in DURABLE_FIELDS and isinstance(v, (str, int, float, bool)):
                sv = str(v).strip()
                if sv.lower() not in _JUNK:
                    found.setdefault(k, sv)
            elif isinstance(v, dict):
                scan(v, depth + 1)

    scan(result)
    return [
        {"key": k, "value": v, "label": DURABLE_FIELDS[k]}
        for k, v in found.items()
    ]


def remember_from_result(
    project_id: str,
    result: dict[str, Any],
    source_document: str | None = None,
) -> list[dict[str, Any]]:
    """Extract durable facts from a result and persist them to project memory."""
    saved = []
    for fact in extract_facts(result):
        saved.append(
            store.set_fact(
                project_id, fact["key"], fact["value"],
                source_document=source_document,
            )
        )
    return saved


def build_project_context(project_id: str, query: str = "", limit: int = 8) -> str:
    """A combined context block with project facts + document listing.

    Returns a non-empty string when the project has facts or documents.
    Returns "" when neither exists.

    The document listing is CAPPED (``PROJECT_CONTEXT_DOC_CAP``, default 40).
    A large corpus (e.g. the 2,997-doc master corpus) previously dumped every
    filename into a system message on every turn — ~60k tokens of noise the
    model can't use (retrieval surfaces the relevant docs) that blew past
    provider rate limits. We list the newest ``cap`` docs and summarise the
    rest as a count.
    """
    facts_block = build_memory_context(project_id, query, limit)

    doc_cap = int(os.getenv("PROJECT_CONTEXT_DOC_CAP", "40"))
    total_docs = store.count_documents(project_id)
    docs = store.list_documents(project_id, limit=doc_cap, newest_first=True)

    if not facts_block and not total_docs:
        return ""

    parts: list[str] = []
    if facts_block:
        parts.append(facts_block)
    if docs:
        lines = [f"Project documents ({total_docs} total):"]
        for doc in docs:
            lines.append(
                f"- {doc['original_name']} (type: {doc['doc_type']}, role: {doc['doc_role']})"
            )
        remaining = total_docs - len(docs)
        if remaining > 0:
            lines.append(
                f"- …and {remaining} more document(s). Use document search to "
                "find specific ones — do not assume they are absent."
            )
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def build_memory_context(project_id: str, query: str = "", limit: int = 8) -> str:
    """A compact text block of relevant project facts, for injection into chat.

    Query-MATCHED facts rank first, but the remaining facts still fill the
    limit: keyword-gating used to EXCLUDE everything the message didn't
    happen to mention, which inverts the store's purpose — the fact the
    user forgot to name is exactly the one the model needs surfaced. Live
    find 2026-08-15 (F23): a take-off request matched the stored
    floor-to-ceiling height (via the word "floor") but the stored door
    schedule never reached the model, which then asked the operator for a
    value the project already carried. A project's durable facts are few
    and the limit caps the block, so including the rest costs tokens only
    when the facts exist.
    """
    all_facts = store.list_facts(project_id)
    if not all_facts:
        return ""
    if query:
        matched = store.search_facts(project_id, query) or []
        matched_keys = {f["key"] for f in matched}
        facts = matched + [f for f in all_facts if f["key"] not in matched_keys]
    else:
        facts = all_facts
    lines = ["Known facts about this project:"]
    for f in facts[:limit]:
        label = DURABLE_FIELDS.get(
            f["key"], f["key"].replace("_", " ").capitalize()
        )
        lines.append(f"- {label}: {f['value']}")
    return "\n".join(lines)
