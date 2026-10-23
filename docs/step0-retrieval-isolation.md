# STEP 0 — Retrieval Isolation (pilot-critical)

Owner directive, 2026-07-17: project scope must be a **hard structural filter**,
not a ranking penalty; another corpus's chunks must be structurally unreachable
from a project-scoped query regardless of score. An empty/thin project may fall
back to the Master Corpus **only when disclosed** (labeled in the answer + the
sources panel). Silent cross-corpus is banned.

## What was actually true (root cause, verified live)

`store.search(project_id, …)` was **already a hard SQL `WHERE project_id = X`**
(`_search_pgvector`, `_search_numpy`, both BM25 legs, `identifier_search`). An
*arbitrary* third project was already structurally unreachable — the retriever
only ever asks the store for the active project + the configured
general-knowledge (GK) ids. So 0a at the SQL level was largely already satisfied.

The leak was elsewhere and worse than a ranking penalty. Live config:

```
RAG_GENERAL_KNOWLEDGE_PROJECTS = curated_kb,example_infra_pack,drive_archive
MASTER_CORPUS_SOURCE_PROJECT_ID = drive_archive
```

The **entire the client project client corpus** (`drive_archive` + `example_infra_pack`) was
declared as "general knowledge" and **silently merged into every other project's
retrieval**, governed only by the ranking knobs (margin/cap/lexical-fold). That
is the `example_estate → the client project` leak: a client corpus surfacing in another project's
answers through a ranking layer, exactly what the directive forbids. `curated_kb`
(FIDIC / OSHA / units / rates) is the only legitimately-general member.

## The fix (two layers, not one)

1. **General Knowledge** (`curated_kb`) — genuinely cross-project reference.
   Always eligible, but now **disclosed**: chunks are tagged and the sources
   panel labels them "Knowledge base". This preserves the calc/standards
   features (FIDIC clauses, unit tables) that *depend* on GK winning — the
   entire `RAG_AUDIT_V2` / GK-margin / lexical-fold history exists for them.

2. **Master-Corpus / client fallback** (`MASTER_CORPUS_SOURCE_PROJECT_ID`) —
   **structurally barred from the GK merge** (`_general_knowledge_project_ids`
   strips it even when a stale env still lists it), so a client corpus can
   never silently blend into another project. It surfaces **only** as the
   labeled empty/thin fallback.

### Behaviour

- **Populated, strong project** (best own chunk clears `RAG_CONFIDENCE_THRESHOLD`)
  → own chunks only (+ disclosed GK). Does **not** pull the Master Corpus even
  if the Master chunk outscores. This is the leak fix.
- **Empty / thin project** → query the Master-Corpus source, tag those chunks
  `layer="master_corpus"`, set `fallback_used=True`. The answer is prefixed
  with a visible banner ("This project has no documents of its own for this
  question — answering from the Master Corpus.") and each source is
  layer-labelled ("Master Corpus (fallback)").
- **No `MASTER_CORPUS_SOURCE_PROJECT_ID`** (CI / self-host default) → empty
  projects still return `[]`; the fallback is opt-in by config, so the
  pre-existing "unindexed project" contract is unchanged.

## Isolation signal

`Chunk.layer` ∈ {`own`, `general_knowledge`, `master_corpus`}, set by
`retrieve_with_filter`, carried into `rag_inject`'s audit record and out to the
runtime. It is **never** sent to the LLM or the API payload (`to_dict` drops it,
`compare=False` so it never affects `Chunk` equality).

## Tests (`tests/test_step0_retrieval_isolation.py`)

- **0a golden** — a third project holds a 0.99 chunk; a query scoped to the
  active project never even *asks* the store for it, and it never appears.
  Locks the invariant against a future "search every project, then rank".
- **0b** — empty → labeled Master-Corpus fallback; thin → own(weak)+Master;
  strong → own only, no Master pull (the leak fix); Master-as-active → own.
- **regression** — Master source listed in the GK env is stripped from the
  merge and cannot leak into a populated project.
- **disclosure** — `rag_inject` sets `fallback_used` + per-chunk `layer`;
  `_postprocess_answer` prepends the banner (idempotent, off by default);
  the sources panel carries `layer` + `layer_label`.

## Config change deployed with this fix

`RAG_GENERAL_KNOWLEDGE_PROJECTS` set live to `curated_kb` (dropping
`example_infra_pack` + `drive_archive`). The code guard makes the leak
structurally impossible for the Master source regardless of config; this config
change removes the remaining `example_infra_pack` GK entry so only the genuinely-
general corpus stays in the always-on layer.

## Reconciliation for the owner (one call, vetoable)

The directive's plain reading of 0b ("company-layer = fallback only") would ban
the GK merge for *populated* projects — which would regress FIDIC/standards/calc
grounding. The reconciliation: **GK (curated standards) stays always-eligible
but disclosed**; only the **client/Master corpus** is fallback-only. Disclosed ≠
silent, so "silent cross-corpus banned" holds while calc/standards keep working.
If the owner wants GK to be fallback-only too, set `RAG_GENERAL_KNOWLEDGE_PROJECTS=""`
— no code change needed.
