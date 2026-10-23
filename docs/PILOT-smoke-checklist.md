# Pilot smoke checklist — The Shovel (theshovel.ai)

A ~10-minute human-paced pass to confirm the live box is demo-ready **before**
an operator-led pilot session. Run it on the deployed site while signed in.
Record PASS / FAIL + a note per step. Any 502 or empty/error bubble is a FAIL.

> Scope: this validates the **operator-led** pilot (you driving). It deliberately
> steers around the known broad-query precision boundary — see step 9.

## Pre-flight

| # | Step | Expect | Result |
|---|------|--------|--------|
| 0 | `GET /v1/health` (or load the site) | `status: healthy`, `blocks_failed: {}` | |
| 1 | Sign in → land on **Master Corpus** | Project opens, chat composer ready | |

## Grounded retrieval (the core value)

| # | Step | Expect | Result |
|---|------|--------|--------|
| 2 | Ask a **sharp, specific** question grounded in the corpus (e.g. a named spec value / a specific BOQ section) | A sourced answer; **Sources panel populated** with cited docs + confidence | |
| 3 | Ask the **BOQ total** question (e.g. "what is the total package value of the demolition BOQ?") | A concrete number **or** an explicit "partial" hedge — never a confident wrong number | |
| 4 | On a BOQ answer that cites a digital (xlsx/csv) BOQ, look under the answer bubble | A **"Cost BOQ (Excel)"** download appears beside "Download" (no separate toolbar button) | |
| 5 | Click "Cost BOQ (Excel)" | An `.xlsx` downloads and opens; totals are formula-linked, not static | |

## Construction knowledge base (new)

| # | Step | Expect | Result |
|---|------|--------|--------|
| 6 | Ask a KB engineering question, e.g. "mass concrete equilibrium time" or "compaction acceptance criteria" | A **cited rule** from the corpus (id + statement) with a "verify against your spec" caveat | |

## General chat + plumbing

| # | Step | Expect | Result |
|---|------|--------|--------|
| 7 | Ask a general (non-document) construction question | A sensible answer via Ollama Cloud — no error, no deepseek-fallback banner | |
| 8 | Per-message **Download** (Word) on any answer | A `.docx` downloads | |

## Known boundaries — demo around these, do not trigger live

| # | Watch-out | Why |
|---|-----------|-----|
| 9 | Avoid broad "what does package X **cover**?" phrasing | Precision boundary — broad queries can return "cannot find" even though content is indexed. Ask **sharp** questions instead. |
| 10 | Ingest BOQs as **xlsx/csv** only | A large scanned BOQ OOMs the 2 GB box. Use the local-extraction playbook for scanned ones. |
| 11 | The per-project **document shells** in the sidebar may show "Not indexed / 0 B" | The real corpus lives in the Master Corpus backing store (~142k chunks), not these partial shells. Demo questions against the Master Corpus, not a shell. |

## Smoke run — 2026-06-30 (automated, theshovel.ai / Master Corpus)

Backend is healthy and fast; the failure is in the **browser streaming UI**, not the engine.

| Probe | Result | Evidence |
|-------|--------|----------|
| `/v1/health` | PASS | 44 blocks, 0 failed |
| UI loads, signed in | PASS | 200, React SPA |
| Ollama Cloud up (general chat) | **PASS — 9.0s** | `POST /v1/chat` → `{"text":"OK","provider":"ollama","model":"gpt-oss:120b-cloud"}` |
| Grounded RAG, **non-streaming** | **PASS — 16.5s** | `POST /v1/agents/project-assistant/chat` → full sourced answer (cites risk-register template) |
| Grounded RAG, **streaming, fresh** | **PASS — ~10s** | raw SSE: route→start→tokens→end, incl. new `"exports": []` field (export wiring is live) |
| Grounded RAG via **browser UI, existing conversation** | **FAIL — hang** | 80–100s+, zero tokens, no error banner, composer locks; lock persists across reload |
| Broad-query precision | **KNOWN-LIMIT reproduced** | "main risks" → "could not find a specific list… only the risk register template" |

### Diagnosis
The engine works (9–16s, grounded, cited, export field present). The hang is **specific to
the browser streaming path when the conversation carries state** (existing `conversation_id`
with history, and/or an interrupted prior turn). A *fresh* streaming request to the same
endpoint streams fine in ~10s. An interrupted turn also leaves the assistant message marked
`streaming`, which **locks the composer even after a page reload** — a frontend
state-recovery issue.

### Next actions (ranked)
1. Reproduce the browser hang with a populated `conversation_id` (history) vs none → confirm
   whether accumulated/poisoned history drives a long tool-loop ([[the-fork-hallucination-problem]]).
2. Fix the frontend so an interrupted/stalled stream surfaces the error and **unlocks the
   composer** (today it spins indefinitely; the 90s server timeout banner did not appear in the UI).
3. Broad-query retrieval precision (pre-existing #1 demo risk).

## Smoke run #2 — 2026-06-30 (post-fix, deployed)

After the streaming + DB-index fixes:

| Check | Result |
|-------|--------|
| Grounded chat in the **browser** (deployed build) | **PASS — real cited answer in ~15s**, clean settle, no infinite spinner: *"...I could not confirm the contract completion period... (source: 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx, chunks 29,55,345,309)"* |
| Streaming stall recovery (FIX A) | Heartbeats no longer reset the 95s deadline → a stall now surfaces a banner + unlocks the composer instead of spinning forever |
| Export descriptor on the stream | `exports: []` present (empty for a `.docx`-cited answer — correct; only BOQ-cited xlsx answers get a "Cost BOQ (Excel)" offer) |
| Workspace load latency | Was ~11s warm (every call). Root cause: missing `idx_chunks_project` btree on the `chunks` table (checkfirst skipped it on the pre-existing prod table) → COUNT-by-project seq-scanned 139k rows. Fix deployed (`9e2c51c`); re-measurement pending. |

### Fixes shipped this session
- `3c9a992` — streaming stall timeout ignores heartbeats; client history bounded.
- `9e2c51c` — create `chunks.project_id` index on pre-existing tables.

### Still open (deferred, stated)
- Broad-query retrieval **precision** ("what are the main risks/cover" → cites the right family but can't surface specifics). Quality-tuning, not a reliability bug.

## Sign-off

- [ ] Steps 0–8 all PASS, no 502s
- [ ] Demo script uses sharp grounded questions (step 9 respected)
- [ ] BOQs pre-ingested as xlsx (step 10 respected)
- [ ] Pilot date / attendees: __________
