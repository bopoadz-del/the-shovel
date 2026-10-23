"""Stage 5 — backfill tags existing unlayered chunks via the same classify().

Inserts a project + priced-BOQ document + chunks with knowledge_layer NULL into
the suite's DB, runs the backfill, and asserts those chunks get the classify()
layer/authority. Asserts on THIS project's rows only (robust to other suite
rows). Idempotent: a second pass tags nothing new for this doc.
"""
import numpy as np
import pytest


def test_backfill_tags_unlayered_chunks(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")
    from app.core import projects as store
    from app.core.rag.vector_store import get_store
    from scripts.backfill_layers import backfill

    pid = "proj_bf_stage5"
    store.create_project("the client project", user_id="system", project_id=pid)
    doc = store.add_document(pid, "priced_boq_sewer.xlsx", size=1)

    # insert chunks WITHOUT layer (simulates a pre-migration corpus) into the
    # SAME store the backfill reads (get_store -> suite DB).
    st = get_store()
    st.upsert_chunks(pid, doc["id"], ["a chunk", "b chunk"],
                     np.ones((2, st.dim), dtype=np.float32))

    plan = backfill(apply=False)
    assert plan["by_project"].get(pid) == 2          # 2 of MY chunks pending
    assert plan["applied"] is False

    report = backfill(apply=True)
    assert report["by_project"].get(pid) == 2

    got = st.search(pid, np.ones(st.dim, dtype=np.float32), k=2)
    assert got and all(c.knowledge_layer == "project_record" for c in got)
    assert all(c.authority == "commercial" for c in got)

    # idempotent: this doc is now fully tagged
    assert backfill(apply=False)["by_project"].get(pid, 0) == 0
