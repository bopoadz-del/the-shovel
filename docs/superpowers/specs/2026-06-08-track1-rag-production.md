# Track 1 - RAG pipeline production-ready

Spec for shipping retrieval-augmented chat as the default behaviour for The Fork's
project workspace. Closes the gap between "the platform knows about your uploaded
documents" (true) and "the platform actually uses them when you chat" (intermittent
today, depends on the model choosing to call `search_project_documents`).

## Goal

Every chat turn against a project is grounded in that project's documents by
default. The model never invents a fact it could have looked up. When retrieval
genuinely has nothing relevant, the assistant says so before falling back to
general knowledge.

## Non-goals

- Track 2 (LoRA fine-tuning on a local 4B model). Parked until Track 1 is live
  and a regression set proves RAG is working.
- Cross-project retrieval. Each chat sees only its own project's index.
- Re-ranking model on top of cosine similarity. Top-K-by-cosine + token cap is
  v1; a learned reranker is a v2 question.
- Streaming retrieval (showing sources while the answer streams). Sources land
  with the final SSE end event.

## Architecture decision

The UI's chat composer calls `/v1/agents/project-assistant/chat/stream`, which
routes through the **agent runtime** (`app/agents/runtime.py`), not through
`ConstructionContainer.chat`. Setting a flag on the container alone would not
change UI behaviour.

**RAG pre-injection lives in the agent runtime, gated to `project-assistant`.**
Before iteration 0 of the agent loop, the runtime retrieves top-K chunks for the
user's message and inserts them as a system message ahead of the user turn. The
agent's existing `search_project_documents` tool stays available for the model
to call when it needs deeper or refined retrieval.

`ConstructionContainer.chat` ALSO gets `use_rag=True` as its hard default so
that any API caller hitting that route gets the same behaviour. Both gates flip
together.

Key consequence: the existing `search_project_documents` tool path is now a
**supplement** for follow-up queries, not the primary retrieval mechanism. The
model doesn't have to remember to call it; the runtime always serves it
context up front.

## Components

### 0. Noise document filter (new, in `app/core/rag/retriever.py`)

Some projects accumulate documents that are unrelated to construction work
(stray pptx menus, office lockfile artifacts like `~$C-201_*.docx`, etc.).
We can't mutate `doc_type` via a PATCH endpoint without adding one, and we
don't want to delete the files (the operator's instruction). Instead, the
retriever filters them out by filename pattern at query time.

Configuration:
- `RAG_NOISE_FILENAME_REGEX` env var, default
  `^(~\$|nambae-menu|SandsChina_Application)`. Operator can append more
  patterns over time without a redeploy.
- Matched docs are skipped from the candidate pool BEFORE top-K selection,
  so they cannot consume a slot a real doc would have taken.

Audit log records noise-filter activity:
- New field `noise_filtered_count: int` in the audit record, counting how
  many candidates were dropped by the noise filter for that query.
- Lets the operator monitor whether the regex is over- or under-matching.

Out of scope here: building a UI to manage noise rules; that's a v2.

### 1. RAG injector (new, in `app/agents/runtime.py`)

A helper called from `chat_stream` before iter 0:

```
def _rag_inject(user_message, project_id, agent_name) -> Tuple[Optional[Dict], Dict]:
    """Returns (system_message_or_None, audit_record).

    - If agent_name != "project-assistant" or project_id is None: returns (None, {}).
    - Calls app.core.rag.retriever.retrieve(user_message, project_id, k=K).
    - Applies the token cap (see below).
    - On low confidence (top score < THRESHOLD): returns (None, audit) with a
      flag that the chat_stream will surface as a prefix.
    - Otherwise: returns ({"role": "system", "content": "<formatted chunks>"}, audit).
    """
```

The formatted chunks system message looks like:

```
Relevant project context (top {N} of {total} matches; cosine in [{min}, {max}]):

[doc_id={a} chunk={i} score={s:.3f}] {chunk_text}

[doc_id={b} chunk={j} score={s:.3f}] {chunk_text}
...
```

### 2. Token cap (new)

```
K_DEFAULT = 5                  # int(os.getenv("RAG_K", 5))
MAX_RAG_TOKENS = 1500          # int(os.getenv("MAX_RAG_TOKENS", 1500))
THRESHOLD = 0.4                # float(os.getenv("RAG_CONFIDENCE_THRESHOLD", 0.4))
```

- Retrieve top-K by cosine.
- Estimate token count of each chunk via the active model's tokenizer if cheap,
  else `len(chunk_text) // 4` as a fast proxy.
- Drop whole chunks from the bottom (lowest cosine) until total tokens <= MAX_RAG_TOKENS.
- Never truncate mid-chunk. A chunk is included or excluded whole.
- Audit log records both `requested_k` and `injected_k`.

All three constants are env-var overridable on Render. No redeploy needed to
tune.

### 3. Confidence fallback

Defined as: top-1 cosine score across the K retrieved chunks.

- If `top_score >= THRESHOLD` (default 0.4): inject context as above. Log
  `threshold_fired=false`.
- If `top_score < THRESHOLD`: do NOT inject the chunks. Prepend a fixed prefix
  to the model's response (handled in `chat_stream` when the final assistant
  text is emitted): *"I couldn't find relevant project context, answering from
  general knowledge."* Log `threshold_fired=true` and STILL record the
  attempted retrieval (chunks, scores) so we can tune the threshold later.
- If the index is empty (no chunks at all): same fallback prefix, same audit.

### 4. Audit log (new)

Append-only JSONL at `${DATA_DIR}/logs/rag_audit.jsonl`. One record per
turn, schema:

```
{
  "timestamp": "2026-06-08T17:42:11.123Z",
  "project_id": "fb776aa2",
  "conversation_id": "ws-fb776aa2",
  "user_id": "abc123",
  "agent_name": "project-assistant",
  "user_message_preview": "first 200 chars of user message",
  "requested_k": 5,
  "injected_k": 3,
  "injected_tokens": 1247,
  "top_score": 0.71,
  "threshold_fired": false,
  "budget_remaining": 487253,
  "budget_degraded": false,
  "chunks": [
    {"doc_id": "client-bod-pdf-1", "chunk_index": 42, "score": 0.71},
    {"doc_id": "client-rfp-docx-1", "chunk_index": 17, "score": 0.62},
    {"doc_id": "client-bod-pdf-1", "chunk_index": 43, "score": 0.58}
  ]
}
```

`threshold_fired=true` rows still carry full `chunks` and `top_score` so we can
re-evaluate the threshold later from the log. `budget_remaining` and
`budget_degraded` are written on every turn regardless of the budget state
(see Component 4.5).

### 4.5 Daily token budget (Phase 2.5, non-blocking for Phase 2 go-live)

A soft daily cap on RAG injection tokens so a runaway day cannot drain the
LLM token quota. Layered on TOP of the per-turn `MAX_RAG_TOKENS` cap and the
confidence threshold, so degradation has three independent dimensions.

Configuration:
- `RAG_DAILY_TOKEN_BUDGET` env var, default `500_000` per day per Render
  service.

State:
- New SQLite table `rag_budget` in the existing audit DB (or `agent_memory.db`
  if cheaper to colocate):
  ```
  CREATE TABLE IF NOT EXISTS rag_budget (
      day        TEXT PRIMARY KEY,   -- ISO YYYY-MM-DD in UTC
      consumed   INTEGER NOT NULL DEFAULT 0
  );
  ```
- Updated atomically inside the same transaction that emits the audit record,
  so consumption can never drift from the log.

Behaviour on each turn (before RAG injection). Boundary semantics are
**inclusive** on the budget: when consumption *reaches* the budget the
next turn already degrades — it does not have to *exceed* it.

1. Compute `today = utcnow().strftime("%Y-%m-%d")`.
2. Read `consumed = rag_budget WHERE day = today` (default 0).
3. If `consumed >= RAG_DAILY_TOKEN_BUDGET`  (note: `>=`, not `>`):
   - Degrade: set `effective_k = 2` for this turn (instead of `K_DEFAULT`).
   - Apply the same MAX_RAG_TOKENS cap to the 2 chunks.
   - Set `budget_degraded = true` for the audit record.
4. Otherwise:
   - Proceed with `K_DEFAULT` and the standard token cap.
   - Set `budget_degraded = false`.
5. After deciding `injected_tokens`, atomically:
   `UPDATE rag_budget SET consumed = consumed + injected_tokens WHERE day = today`
   (with an `INSERT OR IGNORE` to seed the row).

Reset: implicit by date rollover at midnight UTC. No cron / scheduled task
required. The first turn of a new UTC day creates the row with
`consumed = 0`.

Acceptance criteria for Phase 2.5:
- `RAG_DAILY_TOKEN_BUDGET` env var present, defaults to 500,000.
- A day with budget exhausted shows `budget_degraded=true` and `injected_k <= 2`
  in the audit log.
- A day rollover (manipulated via test fixture) resets the consumed counter
  to 0.
- `budget_remaining` appears in every audit record, including non-degraded
  turns and threshold_fired turns.
- No core architecture change needed; the budget hook is a 30-50 line addition
  to `_rag_inject`.

### 5. Debug mode (new query param)

`POST /v1/agents/project-assistant/chat/stream?rag_debug=true`

When `rag_debug=true`:

- Runtime makes TWO LLM calls per user message: one with RAG context injected,
  one without.
- Final SSE event carries both responses + the rag_audit record under a
  `rag_debug` field.
- Costs ~2x tokens; opt-in. Not for default use.
- Frontend will not surface this in v1; it's a developer-tool affordance for
  prompt iteration.

### 6. Drive ingestion (incremental, in `app/routers/drive.py`)

The existing walker endpoint `POST /v1/projects/{id}/drive/index-folder` is the
hot path. Augment with SHA-256 deduping:

- Add column `content_sha256 TEXT` to the `documents` table in `doc_index.db`
  via online ALTER TABLE migration on startup.
- On walk: compute SHA-256 of file bytes after Drive download, before
  encryption.
- If a document row already exists with the same project_id + sha256: skip.
- Otherwise: encrypt, index, store with the new hash.
- Report: counts of new, skipped, re-indexed.

Initial smoke test: feed ONE file under 1MB into a project, observe the row,
verify a second run skips it. Then unlock full-folder ingestion.

### 7. Source UX (frontend, in `frontend/src/pages/ProjectWorkspace.tsx`)

Backend changes:

- `chat_stream` final SSE event currently emits `{type: "end", iterations: N}`.
  Extend to `{type: "end", iterations: N, sources: [...]}` where each source is
  `{doc_id, doc_name, page_or_section, score, confidence}`.
- Top 3 sources only, sorted by score descending.
- `page_or_section`: PDFs use the extracted page number from doc_index;
  docx/xlsx fall back to `"chunk #{chunk_index}"`.
- `confidence`: "High" if score >= 0.75, "Medium" if 0.5 <= score < 0.75,
  "Low" if score < 0.5.

Frontend changes:

- Below the assistant message bubble, render a collapsed "Sources (N)" footer
  when `sources` is non-empty.
- Expanding shows the list with name, page/section, score, confidence label.
- No footer rendered when `sources` is empty (fallback case or non-RAG turn).

## Data generation (Phase 1)

Run once, before Phase 2 ships, against project `fb776aa2`:

```
python scripts/generate_training_scenarios.py \
    --project-id fb776aa2 \
    --out data/learning/training_scenarios.jsonl \
    --questions-per-chunk 3 \
    --max-chunks 200 \
    --provider ollama
```

(The script currently has `--provider deepseek|local_lora|offline_template`.
Add `ollama` as an option that goes through `_llm_config()`.)

Validation pipeline added to the script:

- Drop rows with empty `response` or `instruction`.
- Drop rows where the response is < 30 characters.
- Compute zvec embeddings on all responses; drop any row whose response cosine
  similarity to any earlier row is >= 0.85.
- Suspicious flag (logged, not dropped): response does not contain any of the
  top 3 noun-phrases from the source chunk (heuristic via NLTK or spaCy if
  available, else skip the flag).

Report on stdout:
- Total chunks visited
- Total raw Q/A pairs generated
- Dropped: empty / short / duplicate counts
- Kept count
- Top 5 source documents by row contribution

Target: 500 rows kept. If we fall short, raise `--questions-per-chunk` or
`--max-chunks`.

## Regression test set

Five queries become the gate between phases. Each phase ships only after all
five run through the updated code path and you sign off on the outputs.

The queries reference two doc sets, both now uploaded (commit-time setup
completed 2026-06-08):

- **Anthropic project (`fb776aa2`):**
  - Example Client Request for Proposals 010101.docx
  - Example Client Performance Basis of Design.pdf (`d0c0001a`, 762 KB)
  - Example Client RFP Appendix B 010101.xlsx (`d0c0001b`, 14.5 KB)
  - PRC-921 Time Management.pdf
  - PRC-931 Request for Information.pdf
  - Plus two unrelated docs (nambae-menu pptx, SandsChina docx) and a
    `~$C-201` Office lockfile artifact — these will appear in the index but
    are noise for the regression queries.
- **the client BOQ Test project (`3f6f28b2`):**
  - the client project - Infra-1 - Demolition BOQ.pdf (`c6dae280`, 9.7 MB; auto-classified
    by the upload route as doc_type=`boq`)

Phase 1 of the implementation plan should kick off the doc_index extraction
on both projects so chunks + embeddings exist before the regression queries
run.

### The five queries

1. **What is the IT load specified in the Performance Basis of Design?**
   - Project: `fb776aa2`
   - Target retrieval: BOD section on IT load capacity
   - RAG-off failure mode: invents a number
   - RAG-on win: cites the exact figure with the doc reference

2. **Item D999.14 prices 300mm HDPE potable water pipe at SAR 1,060/m for
   depth 1 to 1.5m and D999.15 at SAR 1,288/m for depth 1.5 to 2.0m. Is the
   SAR 228/m depth increment reasonable and what scope does it cover?**
   - Project: `the client-bqa-test`
   - Target retrieval: BOQ Part 3 line items D999.14 and D999.15
   - RAG-off failure mode: generic rate commentary, no specific numbers
   - RAG-on win: anchors response on the exact rates with depth bands

3. **What cooling architecture is in the BOD?**
   - Project: `fb776aa2`
   - Target retrieval: BOD cooling section (CDUs, primary loop, redundancy)
   - RAG-off failure mode: generic data center cooling explainer
   - RAG-on win: cites BOD's specific cooling architecture choices

4. **Generate a 50-activity construction schedule based on the RFP scope.**
   - Project: `fb776aa2`
   - Target tool call: `generate_wbs(brief, target_count=50, project_type=data_center)`
   - Verifies the runtime change does NOT break tool calling; RAG context
     should not displace the tool-call mandate.
   - Pass criterion (explicit): the iter-0 LLM response has
     `finish_reason == "tool_calls"` AND the `tool_calls` array contains an
     entry with `function.name == "generate_wbs"`. The streaming UI showing
     a tool-call card is a downstream consequence, not the test. Match
     against the LLM response shape, not the rendered chat trace. Tested
     in `tests/test_rag_injection.py::test_q4_tool_call_discipline_under_rag`.

5. **The BOD specifies cooling availability at 99.99% across 100% of IT
   capacity. One CDU failure must not impact more than 4 rows. What does this
   blast radius requirement mean for MEP redundancy design and busway sizing?**
   - Project: `fb776aa2`
   - Target retrieval: BOD redundancy + blast radius requirements
   - RAG-off failure mode: hallucinates threshold values
   - RAG-on win: cites the 99.99% / 4-row figures, then reasons about MEP
     implications

After each phase: run all five, paste output to operator, operator signs off
before next phase begins.

## Sequencing and checkpoints

Phase 1 - Data generation
  -> smoke: scenario JSONL exists with >= 500 rows, validation report posted
  -> CHECKPOINT (operator reviews row count + sample rows)

Phase 2 - RAG default
  -> smoke: 5 regression queries run, sources visible (raw)
  -> CHECKPOINT (operator signs off on retrieval quality)

Phase 2.5 - Daily token budget (non-blocking; can ship after Phase 2 go-live)
  -> smoke: budget exhausted via test fixture, audit shows budget_degraded=true,
     injected_k <= 2; new day resets to 0
  -> CHECKPOINT (operator confirms budget telemetry visible in audit log)

Phase 3 - Drive ingestion incremental
  -> smoke: one file ingested, second run skips, full folder ingest succeeds
  -> CHECKPOINT (operator sees row count + skip count)

Phase 4 - Source UX
  -> smoke: 5 regression queries again, sources visible in UI
  -> CHECKPOINT (operator confirms confidence labels look right)

DONE when all five queries produce grounded answers with visible sources, and
Phase 2.5 has shipped (budget enforcement live).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 1500-token cap drops legitimate high-score chunks | Log requested vs injected K; if injected_k < requested_k routinely, raise MAX_RAG_TOKENS via env |
| 0.4 confidence threshold misfires on legitimate questions | Audit log preserves every retrieval; tune from data after one week |
| Phase 1 generation burns more tokens than expected | Cap at --max-chunks 200 initially; if 500-row target needs more, raise per-chunk count instead |
| SHA-256 ALTER TABLE on a populated doc_index.db | doc_index is small (under 100 docs typically); online ALTER is fast |
| Drive walker timeout on large folders | Walker already streams; pagination is folder-by-folder, not file-by-file |
| RAG context pollutes tool-calling discipline | Q4 in the regression set specifically tests this; the iter-0 forced tool_choice fix from yesterday handles the case where the model would otherwise drift to prose |
| Token cost balloon: ~1500 tokens injected per turn | Three layers: per-turn MAX_RAG_TOKENS cap (1500), confidence threshold (drops injection when top_score < 0.4), daily token budget (Phase 2.5: 500K/day default, degrades to K=2 when exhausted). All env-overridable on Render. |

## Out of scope for this spec

- Migrating chunked text to a vector DB other than zvec/SQLite (works fine at
  current scale)
- Showing sources inline in the streaming response (sources land with the end
  event in v1)
- Caching identical retrievals across turns (zvec retrieval is fast enough)
- Multi-project federated search

## Acceptance criteria

- All five regression queries return grounded answers with cited sources
  (queries 1, 2, 3, 5)
- Query 4 produces a real tool-call card + tool-result-cited summary
- `rag_audit.jsonl` has one row per turn, including `threshold_fired=true`
  rows
- `MAX_RAG_TOKENS`, `RAG_K`, `RAG_CONFIDENCE_THRESHOLD`, `RAG_DAILY_TOKEN_BUDGET`
  are all tunable via Render env without code change
- Drive walker on second run reports skipped count > 0 for any unchanged file
- Frontend renders a Sources footer when `sources` is non-empty, hides when
  empty, expandable, shows confidence labels per spec

## Files expected to change

- `app/agents/runtime.py`: new `_rag_inject`, hook in `chat_stream` before iter 0
- `app/containers/construction.py`: `ConstructionContainer.chat` default `use_rag=True`
- `app/routers/agents.py`: forward `?rag_debug=true` query param to chat_stream
- `app/core/rag/retriever.py`: add token-aware truncation helper
- `app/routers/drive.py`: SHA-256 dedupe in walker
- `app/core/doc_index.py` (or equivalent): `content_sha256` column + migration
- `scripts/generate_training_scenarios.py`: validation pipeline, `--provider ollama`
- `frontend/src/pages/ProjectWorkspace.tsx`: Sources footer component
- `tests/test_rag_injection.py` (new): unit tests for `_rag_inject`, token cap,
  threshold behaviour, audit log shape, noise filter, Q4 tool-call discipline
  under RAG, **and budget boundary**: explicit test that
  `consumed == RAG_DAILY_TOKEN_BUDGET` (exact-equal, not strictly-over)
  triggers degradation. Off-by-one on that boundary is a classic bug — the
  spec is "degrade when consumed >= budget", not "consumed > budget".
- `tests/test_drive_sha256_dedupe.py` (new): walker skips unchanged file on
  second run
