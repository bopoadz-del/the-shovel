"""Backfill knowledge_layer + authority onto EXISTING chunks (layered RAG stage 5).

Fresh ingests are classified at index time (retriever.index_chunks -> classify).
This one-shot backfills the corpus that was indexed BEFORE the layered columns
existed: it walks every project's documents, runs the SAME classify() used at
ingest, and fills the two columns on that doc's chunks.

Idempotent + safe:
  * Only fills rows where knowledge_layer IS NULL (never overwrites a fresh tag).
  * DRY-RUN by default — prints the per-layer counts it WOULD write. Pass
    --apply to execute the UPDATEs.
  * Reads the active RAG namespace, so it targets the live table (chunks_v2).

Usage:
    python -m scripts.backfill_layers            # dry run, prints plan
    python -m scripts.backfill_layers --apply    # write the tags
"""
from __future__ import annotations

import sys
from collections import Counter

from sqlalchemy import text

from app.core.db import get_database_url, _session_factory_for_url
from app.core.models import rag_chunk_table_name
from app.core.rag import layers
from app.core.rag.vector_store import _rag_vector_namespace
from app.core import projects as store


def _is_user_upload(doc: dict) -> bool:
    return ((doc or {}).get("metadata") or {}).get("provenance") == "user_upload"


def backfill(apply: bool) -> dict:
    table = rag_chunk_table_name(_rag_vector_namespace())
    url = get_database_url()
    Session = _session_factory_for_url(url)

    planned = Counter()          # (layer, authority) -> chunk count
    per_project = Counter()      # project_id -> chunk count touched
    unlayered_before = 0

    with Session() as session:
        # Count chunks still unlayered up front (idempotency visibility).
        unlayered_before = session.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE knowledge_layer IS NULL")
        ).scalar() or 0

        projects = store.list_projects(include_hidden=True)
        for proj in projects:
            pid = proj["id"]
            for doc in store.list_documents(pid):
                did = doc["id"]
                layer, authority = layers.classify(
                    pid, doc.get("original_name") or doc.get("name") or "",
                    is_user_upload=_is_user_upload(doc))
                # How many of THIS doc's chunks are still unlayered?
                n = session.execute(
                    text(f"SELECT COUNT(*) FROM {table} "
                         f"WHERE project_id = :p AND doc_id = :d "
                         f"AND knowledge_layer IS NULL"),
                    {"p": pid, "d": did},
                ).scalar() or 0
                if not n:
                    continue
                planned[(layer, authority)] += n
                per_project[pid] += n
                if apply:
                    session.execute(
                        text(f"UPDATE {table} SET knowledge_layer = :kl, "
                             f"authority = :au WHERE project_id = :p "
                             f"AND doc_id = :d AND knowledge_layer IS NULL"),
                        {"kl": layer, "au": authority, "p": pid, "d": did},
                    )
        if apply:
            session.commit()

    return {
        "table": table,
        "unlayered_before": unlayered_before,
        "planned_total": sum(planned.values()),
        "by_layer_authority": dict(planned),
        "by_project": dict(per_project),
        "applied": apply,
    }


def main() -> int:
    apply = "--apply" in sys.argv
    report = backfill(apply)
    print(f"table               : {report['table']}")
    print(f"unlayered before    : {report['unlayered_before']}")
    print(f"chunks to tag        : {report['planned_total']}")
    print("by (layer, authority):")
    for (layer, authority), n in sorted(report["by_layer_authority"].items(),
                                        key=lambda kv: -kv[1]):
        print(f"  {layer:<15} {authority:<12} {n}")
    print("by project:")
    for pid, n in sorted(report["by_project"].items(), key=lambda kv: -kv[1]):
        print(f"  {pid:<28} {n}")
    if report["applied"]:
        print("APPLIED: tags written.")
    else:
        print("DRY RUN: no rows written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
