"""No answer-path query may sequentially scan the chunk table.

The chunk table is the corpus (live: ~145k rows, ~1.4 GB with its TOASTed
embeddings and tsvectors). One question used to read ~0.9 GB from storage,
almost all of it from that table. A sequential scan of it is the one plan
shape that reads all of it every time, whatever the question.

This seeds a multi-project corpus large enough for the planner to prefer an
index whenever one fits, ANALYZEs it (real statistics, not ``enable_seqscan
= off``), runs the chat's retrieval entry point (``rag_inject``) for
questions of different shapes, records every statement it sends, and
EXPLAINs each one. Any ``Seq Scan`` on the chunk table fails the test and
names the statement.

PostgreSQL only: plans are a property of the PostgreSQL planner. The CI job
``test-postgres`` runs it.
"""
from __future__ import annotations

import json
import os

import pytest
from sqlalchemy import event, text

pytestmark = pytest.mark.skipif(
    os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() not in ("1", "true", "yes"),
    reason="query plans are a PostgreSQL planner property (test-postgres job)",
)

OWN = "qp-own"
EMPTY = "qp-empty"
CORPUS = "qp-corpus"
GK = "qp-gk"
FILLER = tuple(f"qp-f{i}" for i in range(6))

# Function words, a band of common filler words, the construction vocabulary
# the questions use, then a long tail. Words are drawn Zipf-like (rank r with
# probability ~1/r), which puts a domain word such as "cover" in ~3-5% of
# chunks -- about its share of the live corpus (4.9%).
_STOP = "the of and to in for shall be is with on by as or".split()
_DOMAIN = (
    "contractor works engineer contract date period sub-clause clause "
    "specification drawing concrete steel cover drainage pipe trench "
    "excavation backfill pump station rate quantity bill item amount total "
    "sum particulars data conditions completion defects notification "
    "retention payment certificate variation programme commencement access "
    "site cement aggregate reinforcement formwork curing membrane "
    "waterproofing slab beam column foundation pile manhole chamber gully "
    "kerb asphalt subgrade compaction density moisture testing sampling "
    "inspection approval submittal method statement hold point tolerance "
    "survey setting-out levels gradient invert outfall culvert headwall "
    "riprap geotextile filter granular sub-base wearing course binder"
).split()
_WORDS = (
    _STOP
    + [f"zc{i:03d}" for i in range(200)]
    + _DOMAIN
    + [f"zq{i:04x}" for i in range(4000)]
)

# name, project, chunks per document
_DOCS = (
    [(f"Contract Data Volume {i}.pdf", OWN, 40) for i in range(3)]
    + [(f"Conditions of Contract Part {i}.pdf", OWN, 40) for i in range(3)]
    + [(f"Bill of Quantities Section {i}.pdf", OWN, 40) for i in range(4)]
    + [(f"Specification Volume {i}.pdf", CORPUS, 100) for i in range(30)]
    + [(f"Drawing Register {i}.pdf", CORPUS, 100) for i in range(30)]
    + [(f"Reference Note {i}.pdf", GK, 30) for i in range(10)]
    + [(f"Archive File {p}-{i}.pdf", p, 60) for p in FILLER for i in range(25)]
)

QUESTIONS = (
    # project lookup on the active project (own corpus + general knowledge)
    (OWN, "What is the defects notification period stated in the contract data?"),
    # identifier lookup
    (OWN, "What is the quantity and rate for bill item D529.2?"),
    # standards-shaped ask on an empty project: labelled Master-Corpus fallback
    (EMPTY, "What minimum concrete cover does the drainage specification require for the manhole chamber?"),
    # calculation-shaped ask
    (OWN, "Calculate the excavation volume for a trench 40 m long, 1.2 m wide and 2.5 m deep."),
)


def _seed(engine, table: str) -> None:
    with engine.begin() as conn:
        for pid in (OWN, EMPTY, CORPUS, GK) + FILLER:
            conn.execute(text(
                "INSERT INTO projects (id, name, status, aconex_connected, user_id, created_at) "
                "VALUES (:id, :id, 'active', false, 'system', '2026-01-01T00:00:00Z') "
                "ON CONFLICT (id) DO NOTHING"
            ), {"id": pid})
        for n, (name, pid, _per) in enumerate(_DOCS):
            conn.execute(text(
                "INSERT INTO documents (id, project_id, original_name, file_path, "
                "uploaded_at, ingest_status) VALUES (:id, :p, :n, :fp, "
                "'2026-01-01T00:00:00Z', 'INDEXED')"
            ), {"id": f"qpdoc{n}", "p": pid, "n": name, "fp": f"/uploads/{pid}/{name}"})
        # Index builds after the bulk load are much faster than per-row
        # maintenance; the store recreates them below exactly as it would.
        for idx in ("embedding_hnsw", "fts_gin", "text_trgm"):
            conn.execute(text(f"DROP INDEX IF EXISTS {table}_{idx}"))
        for n, (_name, pid, per) in enumerate(_DOCS):
            conn.execute(text(f"""
                INSERT INTO {table} (chunk_id, project_id, doc_id, chunk_index, text,
                    embedding, created_at, embedding_model, embedding_dim,
                    embedding_normalized)
                SELECT :p || ':' || :d || ':' || g.i, :p, :d, g.i,
                    (SELECT string_agg(
                         w[floor(exp(random() * ln(array_length(w, 1))))::int], ' ')
                     FROM generate_series(1, 80 + (g.i % 9)) s,
                          (SELECT CAST(:words AS text[]) AS w) words),
                    (SELECT array_agg(random()::real - 0.5)
                     FROM generate_series(1, 256 + 0 * g.i))::vector,
                    '2026-01-01T00:00:00Z', 'fake', 256, true
                FROM generate_series(0, :per - 1) g(i)
            """), {"p": pid, "d": f"qpdoc{n}", "per": per, "words": list(_WORDS)})
        conn.execute(text(
            f"CREATE INDEX {table}_embedding_hnsw ON {table} "
            "USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.execute(text(
            f"CREATE INDEX {table}_fts_gin ON {table} USING GIN (text_search)"
        ))
        conn.execute(text(
            f"CREATE INDEX {table}_text_trgm ON {table} "
            "USING gin (lower(text) gin_trgm_ops)"
        ))
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"VACUUM ANALYZE {table}"))
        conn.execute(text("ANALYZE documents"))


def _seq_scans_on(plan: dict, table: str) -> list:
    found = []
    stack = [plan]
    while stack:
        node = stack.pop()
        if node.get("Node Type") == "Seq Scan" and node.get("Relation Name") == table:
            found.append(node)
        stack.extend(node.get("Plans") or [])
    return found


def _explain(engine, statement: str, params) -> dict:
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        try:
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        except Exception:  # noqa: BLE001 — pgvector < 0.8: plan without it
            raw.rollback()
        cur.execute("EXPLAIN (FORMAT JSON) " + statement, params)
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        return plan[0]["Plan"]
    finally:
        raw.rollback()
        raw.close()


def test_no_answer_path_statement_seq_scans_the_chunk_table(monkeypatch):
    from app.core.db import get_engine
    from app.core.rag import vector_store
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.inject import rag_inject
    from app.core.rag.retriever import project_is_rag_ready

    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", CORPUS)

    engine = get_engine()
    store = vector_store.get_store(dim=get_embedder().dim)
    table = store._table_name
    _seed(engine, table)

    # Warm the process the way a live worker is warm: store construction and
    # its one-off probes are not part of answering a question.
    rag_inject("warm up retrieval for the drainage works", OWN, None, None, "qp", history=[])

    statements: list = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", _record)
    try:
        for project_id, question in QUESTIONS:
            project_is_rag_ready(project_id)
            rag_inject(question, project_id, None, None, "qp", history=[])
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    on_table = [
        (s, p) for s, p in statements
        if table in s and s.lstrip().split(None, 1)[0].upper() in ("SELECT", "WITH")
    ]
    # The path really ran against the table: semantic, lexical and recall reads.
    assert len(on_table) >= 10, f"only {len(on_table)} chunk-table reads captured"
    assert any("ts_rank" in s for s, _ in on_table), "no BM25 statement captured"
    assert any("<=>" in s for s, _ in on_table), "no vector statement captured"

    offenders = []
    for statement, params in on_table:
        plan = _explain(engine, statement, params)
        if _seq_scans_on(plan, table):
            if os.getenv("QP_DEBUG"):
                print("OFFENDER", statement, params, json.dumps(plan, indent=1))
            offenders.append(" ".join(statement.split())[:300])
    assert not offenders, (
        f"{len(offenders)} answer-path statement(s) sequentially scan {table}:\n"
        + "\n".join(offenders)
    )
