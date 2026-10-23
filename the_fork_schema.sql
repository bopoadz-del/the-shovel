-- The Fork unified PostgreSQL schema (reverse-engineered from SQLite stores).
-- Source modules: users, projects, workflows, agent_memory, doc_index,
-- usage_tracker, hydration_store, rag/budget, rag/vector_store.
-- Embedding dimension: 256 (model2vec minishlab/potion-base-8M; see
-- app/core/models.py EMBEDDING_DIM and embeddings.py EMBEDDING_DIM_MODEL2VEC).

CREATE EXTENSION IF NOT EXISTS vector;

-- ── users (app/core/users.py) ───────────────────────────────────────────────

CREATE TABLE users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    salt          TEXT,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'user'
                  CHECK (role IN ('user', 'admin')),
    created_at    TEXT NOT NULL
);

INSERT INTO users (id, email, password_hash, salt, display_name, role, created_at)
VALUES (
    'system',
    'system@local',
    NULL,
    NULL,
    'System',
    'admin',
    to_char(timezone('utc', now()), 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"')
)
ON CONFLICT (id) DO NOTHING;

-- ── projects (app/core/projects.py) ─────────────────────────────────────────

CREATE TABLE projects (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    client           TEXT,
    status           TEXT NOT NULL DEFAULT 'active',
    aconex_connected BOOLEAN NOT NULL DEFAULT FALSE,
    user_id          TEXT NOT NULL DEFAULT 'system'
                     REFERENCES users (id) ON DELETE RESTRICT,
    created_at       TEXT NOT NULL
);

CREATE INDEX idx_projects_user_created ON projects (user_id, created_at DESC);

-- ── documents (app/core/projects.py) ──────────────────────────────────────

CREATE TABLE documents (
    id             TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL
                   REFERENCES projects (id) ON DELETE CASCADE,
    original_name  TEXT NOT NULL,
    stored_as      TEXT,
    file_path      TEXT,
    doc_type       TEXT NOT NULL DEFAULT 'document',
    doc_role       TEXT NOT NULL DEFAULT 'other'
                   CHECK (doc_role IN (
                       'baseline_schedule', 'daily_report', 'weekly_report', 'other'
                   )),
    size           INTEGER NOT NULL DEFAULT 0,
    uploaded_at    TEXT NOT NULL,
    content_sha256 TEXT,
    metadata       JSONB
);

CREATE INDEX idx_documents_project_uploaded ON documents (project_id, uploaded_at);
CREATE INDEX idx_documents_project_sha ON documents (project_id, content_sha256);

-- ── project_facts (app/core/projects.py) ────────────────────────────────────

CREATE TABLE project_facts (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL
                    REFERENCES projects (id) ON DELETE CASCADE,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    source_document TEXT,
    confidence      DOUBLE PRECISION,
    updated_at      TEXT NOT NULL,
    UNIQUE (project_id, key)
);

CREATE INDEX idx_project_facts_project_key ON project_facts (project_id, key);

-- ── workflows (app/core/workflows.py) ───────────────────────────────────────

CREATE TABLE workflows (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    project_id TEXT REFERENCES projects (id) ON DELETE SET NULL,
    owner_id   TEXT REFERENCES users (id) ON DELETE SET NULL,
    steps      JSONB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_workflows_owner_created ON workflows (owner_id, created_at DESC);
CREATE INDEX idx_workflows_project_created ON workflows (project_id, created_at DESC);

-- ── agent memory (app/core/agent_memory.py) ─────────────────────────────────

CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    project_id TEXT,
    owner_id   TEXT,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_conversations_agent_updated ON conversations (agent_name, updated_at DESC);
CREATE INDEX idx_conversations_project_updated ON conversations (project_id, updated_at DESC);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
                    REFERENCES conversations (id) ON DELETE CASCADE,
    role            TEXT NOT NULL
                    CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    provenance      TEXT
);

CREATE INDEX idx_messages_conv ON messages (conversation_id, created_at);

CREATE TABLE agent_facts (
    id              TEXT PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    project_id      TEXT NOT NULL DEFAULT '',
    conversation_id TEXT,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (agent_name, project_id, key)
);

CREATE INDEX idx_agent_facts_agent_project ON agent_facts (agent_name, project_id);

-- ── doc_index (app/core/doc_index.py) ───────────────────────────────────────

CREATE TABLE doc_index (
    project_id TEXT PRIMARY KEY
               REFERENCES projects (id) ON DELETE CASCADE,
    index_json JSONB NOT NULL,   -- header only: project_id, built_at (0018)
    updated_at TEXT NOT NULL
);

-- One row per document / skipped entry of a project's index (alembic 0018).
-- The entries left the doc_index blob so a single-document update moves one
-- row instead of rewriting the whole project (48 MB, live 2026-10-03).
CREATE TABLE doc_index_entries (
    project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- 'document' | 'skipped'
    seq         INTEGER NOT NULL DEFAULT 0,
    entry_json  JSONB NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (project_id, document_id)
);
CREATE INDEX idx_doc_index_entries_project ON doc_index_entries (project_id, seq);

-- ── usage tracker (app/core/usage_tracker.py) ───────────────────────────────

CREATE TABLE runs (
    id                 TEXT PRIMARY KEY,
    user_id            TEXT,
    agent_name         TEXT,
    provider           TEXT,
    model              TEXT,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    total_tokens       INTEGER,
    estimated_cost_usd DOUBLE PRECISION,
    created_at         TEXT NOT NULL
);

CREATE INDEX idx_runs_user_created ON runs (user_id, created_at);

-- ── hydration store (app/core/hydration_store.py) ───────────────────────────

CREATE TABLE hydration_runs (
    id          TEXT PRIMARY KEY,
    run_date    TEXT NOT NULL,
    scope       TEXT NOT NULL CHECK (scope IN ('project', 'global')),
    project_id  TEXT REFERENCES projects (id) ON DELETE SET NULL,
    summary_md  TEXT NOT NULL,
    facts_json  JSONB NOT NULL,
    provider    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_hydration_lookup ON hydration_runs (scope, project_id, run_date DESC);

-- ── RAG daily budget (app/core/rag/budget.py) ────────────────────────────────

CREATE TABLE rag_budget (
    day      TEXT PRIMARY KEY,
    consumed INTEGER NOT NULL DEFAULT 0
);

-- ── RAG vector store (app/core/rag/vector_store.py) ─────────────────────────
-- embedding: vector(256) per EMBEDDING_DIM in app/core/models.py (model2vec).

CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,
    -- Decoupled from the workspace row: ON DELETE SET NULL (not CASCADE) so
    -- deleting a project orphans its chunks rather than destroying the RAG.
    -- (Prod writes to the namespaced chunks_v2 table, which has no FK at all.)
    project_id  TEXT
                REFERENCES projects (id) ON DELETE SET NULL,
    doc_id      TEXT NOT NULL
                REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   vector(256) NOT NULL,
    created_at  TEXT NOT NULL,
    knowledge_layer TEXT,
    authority       TEXT,
    page            INTEGER,
    UNIQUE (project_id, doc_id, chunk_index)
);

CREATE INDEX idx_chunks_project ON chunks (project_id);
CREATE INDEX idx_chunks_doc ON chunks (project_id, doc_id);
CREATE INDEX idx_chunks_knowledge_layer ON chunks (knowledge_layer);
-- Keep the embedding in the heap row: an exact distance over a project's rows
-- must not walk the TOAST index per row (alembic 0023).
ALTER TABLE chunks ALTER COLUMN embedding SET STORAGE MAIN;
-- Keep the tsvector in the heap row too, so BM25 ranking reads no TOAST: an
-- ordinary chunk row stays whole below 4080 bytes (alembic 0026).
ALTER TABLE chunks SET (toast_tuple_target = 4080);
ALTER TABLE chunks ALTER COLUMN text_search SET STORAGE MAIN;
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_fts_gin ON chunks USING GIN (text_search);
