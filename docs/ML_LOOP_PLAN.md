# ML loop — one block, one interface (ML-0 plan)

Owner direction 13 Nov 07:00Z: the learning loop is ONE Store-shaped block so
nothing tangles with the answer path and the same block can ship to the
Block Store. ML-0 lands the block and its own code; ML-1..ML-4 are built
inside it.

## Shape

| Piece | Where |
|---|---|
| Block (manifest + `process`) | `app/blocks/ml_loop.py` (`MLLoopBlock`, registry name `ml_loop`, generic spec — loads on both profiles) |
| Settings object | `app/core/ml_loop/settings.py` (`MLLoopSettings`: `enabled`, `build_sha`, ONE `session_factory`, caps, proposal thresholds) |
| Payload schemas | `app/core/ml_loop/schemas.py` (one pydantic model per operation, `extra="forbid"`) |
| Tables + repository | `app/core/ml_loop/store.py` (own `MetaData`, `ml_` prefix, `MLLoopRepository(session_factory)`) |
| Read-only proposals | `app/core/ml_loop/proposals.py` |
| Shadow comparison | `app/core/ml_loop/shadow.py` |
| Router prefix | `/v1/ml/...` (ML-1/ML-2) |
| Log prefix | `ml_loop_` (logger `ml_loop`) |

Isolation (tested): block code imports nothing from the host except
`app.core.universal_base`; `runtime.py`, `core/loop.py`, `core/streaming.py`,
`driver/loop.py` import nothing from `app.core.ml_loop`. No `app.agents`
import. No product, project or document name in the block.

## Interface — `process({"operation": <op>, "payload": {...}})`

Every write payload carries `project_id` (required), `conversation_id`,
`message_id`, `build_sha` (optional; defaults to the settings' sha from
`GIT_SHA`/`SOURCE_VERSION`, never invented).

| Operation | Payload (beyond the keys) | Writes | Result |
|---|---|---|---|
| `record_turn` | `question`, `answer`, `verdict`, `route?`, `evidence_plan[]`, `retrieved_ids[]`, `cited_ids[]` | `ml_turns` | row id |
| `record_feedback` | `kind` ∈ correct / wrong_figure / wrong_document / missing, `correct_value?`, `unit?`, `note?` (≤200), `cited_ids[]`, `user_id?` | `ml_feedback` | row id |
| `record_correction` | `formula_id`, `predicted`, `actual`, `source` | `ml_corrections` | row id, sample count |
| `record_route_correction` | `message`, `correct_action`, `original_action?`, `source` | `ml_route_corrections` | row id |
| `propose` | `project_id?`, `window_days?` | — (read-only) | verdict rates by route, feedback by kind, route-correction pairs, most-cited ids in wrong feedback, proposals (`route_rule`, `retrieval_review`) carrying labels + evidence row ids only |
| `shadow_predict` | `predictor`, `live_label`, `shadow_label`, `confidence?` | `ml_shadow_predictions` | row id, `acted_on: false`, `decision = live_label` |
| `shadow_report` | `predictor?`, `project_id?`, `window_days?` | — | per predictor: total, agreement rate, judged/live_right/shadow_right vs route corrections on the same message |
| `status` | `project_id?` | — | enabled, store_configured, build_sha, row counts |
| `export_tables` / `import_tables` | `project_id?` / `{table: rows}` | import appends | the block's own tables only; unknown table/column refused |

No session factory → every non-status op returns `not_configured`, stores
nothing. `ML_LOOP_ENABLED=0` → writes return `disabled`.

## Tables (one Alembic series, ML-1)

All carry `id, project_id, conversation_id, message_id, build_sha, created_at`.

- `ml_turns` — question, answer, verdict, route, evidence_plan, retrieved_ids, cited_ids
- `ml_feedback` — kind, correct_value, unit, note, cited_ids, user_id
- `ml_corrections` — formula_id, predicted, actual, source
- `ml_coefficients` — formula_id, version, coefficients, mae, tier, provenance
- `ml_route_corrections` — message, correct_action, original_action, source
- `ml_shadow_predictions` — predictor, live_label, shadow_label, confidence

## What lands when

- **ML-0 (this PR):** block, settings, schemas, tables (SQLAlchemy defs),
  repository, propose, shadow, export/import, tests. No host wiring, no
  migration, no behaviour change on any existing path.
- **ML-1:** Alembic migration for the six tables; host builds
  `MLLoopSettings.from_env(session_factory=<app SessionLocal>)` and calls
  `MLLoopBlock.configure(...)` at boot; adapters —
  `dataset_log.append_dataset_record` → `record_turn` (file write only when
  no DB); `learning_engine` corrections → `record_correction`, its
  `_fit_linear`/`_compute_tier` move to `app/core/ml_loop/tuning.py`
  unchanged and write `ml_coefficients`; `feedback.py` route corrections →
  `record_route_correction`. Migration round-trip test; two projects.
- **ML-2:** UI control + `POST /v1/feedback/answer` → `record_feedback`
  (owner-only on own conversations).
- **ML-3:** weekly dispatch calling `propose`, report to FLEET_OPS; proposals
  applied only as ordinary PRs through the diff read.
- **ML-4:** route trainer / coefficient candidates via `shadow_predict`,
  judged by `shadow_report` and then the sealed set.
- **Store publish:** after ML-2 is live and certified (FACTORY/Store item).
