# Layered RAG — go-live runbook

The layered-RAG engine (4 layers + authority precedence, plus the decoupling of
knowledge from workspace rows) is **fully built and merged, but dormant**. Every
code path is gated on `RAG_LAYERED`; with the flag unset the platform behaves
byte-for-byte as it did before. This runbook is the supervised activation.

## What is already live (flag OFF, no behaviour change)

- `chunks.knowledge_layer` + `chunks.authority` columns on every chunk table
  (migrations 0011); `chunks.project_id` FK softened to `SET NULL` (0012);
  `projects.hidden_from_sidebar` column (0013). All additive, all deployed.
- `app/core/rag/layers.py` — layer/authority vocabulary, `classify()`,
  `precedence_bonus()`.
- Ingest classification wired into `retriever.index_chunks` (Stage 2).
- Authority-precedence re-rank in `retrieve_with_filter` (Stage 3).
- User uploads routed to the `user_session` layer + "Your upload" disclosure
  (Stage 4).

With `RAG_LAYERED` unset, `classify()` never runs, `precedence_bonus` is never
added, and every chunk stays `knowledge_layer = NULL` (`precedence_bonus(None,
None) == 0`). Retrieval ordering is unchanged.

## Activation — do these IN ORDER, with a person watching

### 1. Backfill the existing corpus (safe; idempotent; reversible)

Fresh ingests self-classify, but the corpus indexed before the columns existed
is still `NULL`. Tag it with the SAME `classify()` used at ingest:

```bash
# On a box with the prod DATABASE_URL exported (or run against a restore first):
python -m scripts.backfill_layers            # DRY RUN — prints the plan only
python -m scripts.backfill_layers --apply    # writes knowledge_layer + authority
```

The dry run prints, per `(layer, authority)` and per project, how many chunks it
would tag. Review it before `--apply`. Re-running is a no-op (only fills rows
where `knowledge_layer IS NULL`). This does NOT change retrieval yet — the flag
is still off.

### 2. Turn the flag on

`render.yaml` env does NOT auto-sync — set it live via the Render API (or the
dashboard) so prod picks it up:

```bash
curl -X PUT "https://api.render.com/v1/services/srv-d8hdc6ek1jcs739rq5sg/env-vars/RAG_LAYERED" \
  -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
  -d '{"value":"1"}'
```

Render redeploys. From this point the authority-precedence re-rank is active on
the (now backfilled) corpus.

### 3. Validate before walking away

Run a handful of grounded queries and confirm ordering is sensible:

- A project-record question surfaces the project doc, not a generic KB note.
- A user's uploaded file surfaces on a direct question about it, labelled
  "Your upload", but does NOT override an authoritative answer.
- Spot-check a few KSA / SBC questions still cite `curated_kb`.

Optional tuning knobs (env, no redeploy of code needed):
`RAG_AUTHORITY_WEIGHT` and `RAG_LAYER_WEIGHT` (default `0.05` each). Raise to make
precedence stronger, lower to make it more of a pure tie-breaker.

## Rollback

Instant: set `RAG_LAYERED` to `0` (or delete the env var) and redeploy. The
columns stay populated but are ignored — retrieval reverts to today's ordering.
No data is lost; the backfill tags remain for the next activation.

## Sidebar hygiene (independent of the flag)

`projects.hidden_from_sidebar` hides a RAG corpus / general-knowledge / eval
project from the sidebar WITHOUT archiving it, so it stays fully retrievable.
Use `store.set_hidden_from_sidebar(project_id, True)` (reversible) — never
archive a corpus row, which would drop it from retrieval. Candidates on the
pilot: `curated_kb`, `projects_folder`, and the two eval fixtures.
