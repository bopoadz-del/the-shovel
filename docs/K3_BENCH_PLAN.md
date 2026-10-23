# K3 Bench Plan — un-park decision for Kimi on The Fork

**Status:** READY-TO-EXECUTE (waiting on model availability)
**Trigger:** Kimi K3 weights release **2026-07-27** → watch for a `kimi-k3*-cloud`
listing on Ollama (`ollama.com/search?q=kimi`). K2/K2.6/K2.7-code are already
listed; K3 is expected to follow the same pattern.
**Context:** Kimi K2.6 is PARKED on the G2 verdict (see `.env.example` provider
ladder). K3 is a new generation (2.8T MoE, 1M context, topped Arena Frontend
Code vs Claude Fable 5 / GPT-5.6) — the park reason does not carry over
automatically. This plan re-judges Kimi on merit, through the same gates that
parked it, against the current incumbents.

## Candidates (3-way)

| Lane | Model | Role |
|------|-------|------|
| A | `glm-5.2:cloud` | Incumbent (prod chat, per DECISIONS.md) |
| B | `gpt-oss:120b-cloud` | Incumbent alt (README lane) |
| C | `kimi-k3:cloud` (exact tag TBD on release) | Challenger |

All via Ollama Cloud — same lane as prod, so the comparison is apples-to-apples.
Do NOT bench K3 via Moonshot's native API for this decision; that conflates
provider-lane latency with model quality.

## Referees (same bars as LOCAL_MODEL_DECISION.md)

| # | Referee | Harness | Pass bar |
|---|---------|---------|----------|
| R1 | Tool-call reliability through the agent loop | `scripts/bench_local_model.py` (native `/api/chat` tools API) | ≥ 90% |
| R2 | Grounding / no-fabrication probe | `scripts/bench_local_model.py` grounding probe | ≥ 90% |
| R3 | Golden set end-to-end quality | `scripts/golden_set_gate.py` | ≥ cloud baseline − 1 |
| R4 | Calc-intact (deterministic calc unchanged) | `scripts/rag_calc_intact_eval.py` | 100% |
| R5 | Latency on the chat hot path | `scripts/smoke.sh` / `smoke.ps1` timing output | general chat ≤ incumbent + 2s |

Then the standing **10/10 smoke gate** on the live deploy before any provider
flip (same gate Groq/Scout is currently parked on).

## K3-specific watch items

1. **Always-on reasoning.** K3 thinks at max effort on every call; thinking
   tokens bill as output. Forcing a specific tool 400s on thinking models —
   keep `tool_choice=auto` (same rule already documented for kimi-k2.6 in
   render.yaml).
2. **Latency budget.** User-measured `kimi-k2:1t-cloud` at ~13–14s on
   2026-07-19. If K3 lands in that band it fails R5 for the hot path
   regardless of quality — it would then be considered for offline/batch
   lanes only (scenario generation, deliverables), never interactive chat.
3. **Context-window temptation.** 1M tokens is not a reason to skip
   retrieval — the R4 calc-intact and R2 grounding referees exist precisely
   because big contexts still fabricate.
4. **History contamination** (TODO.md open issue): WBS/BOQ hallucinated
   assistant turns must be stripped before re-sending; verify K3 doesn't
   pattern-match the hallucination the way earlier models did.

## Procedure

1. Confirm the exact Ollama tag (`ollama.com/search?q=kimi` post-2026-07-27);
   record it here before running anything.
2. Run R1+R2 locally against the tag via `bench_local_model.py`.
3. Run the 3-way pilot eval (`scripts/_eval_pilot_3way.py` pattern) on the
   pilot corpus for lanes A/B/C.
4. Run R3 golden set + R4 calc-intact against the app pointed at lane C.
5. If all pass → staging flip (`OLLAMA_MODEL` to lane C) → 10/10 smoke gate →
   prod flip. Update `.env.example` ladder: Kimi un-parked, note the verdict.
6. If any referee fails → record the numbers, keep the park, revisit at the
   next Moonshot release. No partial credit.

## Rollback

No config changes happen before every referee passes, so rollback is trivial:
`OLLAMA_MODEL=glm-5.2:cloud` stays untouched throughout. Intent routing
(`ORCHESTRATOR_INTENT_MODEL=gpt-oss:20b-cloud`) is independent of this
decision and stays as-is either way.
