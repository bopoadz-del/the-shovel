# Discipline Hats — Architecture Principle

Status: DESIGN PRINCIPLE ONLY. DEFERRED until after the pilot is deployed.
Do NOT implement any of this during the pilot. Recorded 2026-07-14 so the
direction is locked when we pick it up.

## The principle

One hardened **base product agent** that puts on a **discipline hat** per turn —
a Planning hat, a Commercial hat, a Contracts hat, a QA/QC hat, a Procurement
hat. The base is the engine (reasoning + execution + honesty). The hat is a
**portable contract** that scopes what the agent knows, which formulas it can
run, which rules it follows, and what it has been taught.

Not N separate agents. One engine, many hats.

## Why hats beat separate agents

1. **Honesty is inherited once.** The grounding gate, cost-grounding gate,
   standards advisory, and the learning loop already live in one runtime
   (`app/agents/runtime.py`). Every hat inherits them for free. Separate agents
   would re-implement (and eventually diverge on) honesty.
2. **Cross-discipline handoff is a hat-swap, not a protocol.** A change order
   hits Contracts (entitlement), Commercial (price), and Planning (EVM/schedule
   impact) at once. The base agent wears each hat in turn within one flow — no
   brittle inter-agent messaging. The existing cross-domain engine
   (`cross_domain_reasoner` + dependency graph) sits underneath as the router.
3. **Least greenfield.** The platform is already one runtime plus swappable
   agent configs (`project-assistant`, `heavy-reasoning`). Those configs are
   proto-hats. A hat is a config upgraded with three extra dimensions: scoped
   knowledge, formula tools, and taught memory.

## What a hat is (the contract)

A hat is a vendor-neutral manifest (same shape as the dev-agent contracts under
`.claude/agents/`, but for the PRODUCT surface). Fields:

- `activation` — how the user turns the hat on/off, and what triggers it.
- `knowledge` — two layers:
  - **standard (ships)** — PRC/FIDIC procedures, document standards, and the
    discipline's formulas, seeded into the General-Knowledge RAG.
  - **taught (user/company)** — the client's own rules (retention %, IPC days,
    VO templates, naming), held in the project RAG.
  - The taught layer OUTRANKS standard via the existing `RAG_GK_SCORE_MARGIN`.
    This is the exact two-layer pattern proven for rates (company priced-BOQ >
    GK fallback) — generalised. See `the-fork-rates-in-rag`.
- `formulas` — the discipline's calculators wired as deterministic TOOLS
  (`calculate_evm`, `calculate_payment`, `evaluate_tender`, `score_risk`, ...).
  The agent CALLS them; it never does the arithmetic in prose.
- `memory` — what the agent has been taught, scoped to the user/project,
  attributable ("you told me 45-day IPC on 2026-07-12"), and overridable.
- `handoffs` — which other hats a turn should fan out to (the cross-discipline
  edges).
- `verification` / `failure_policy` — the honesty contract (no synthetic
  outputs; if the documents cannot support an answer, refuse honestly).

## Invariants (non-negotiable, or it breaks)

1. **Dev surface != product surface.** The repo-editing coding agents
   (`.claude/agents/construction-expert.md` et al.) change code. The discipline
   hats answer users (`app/agents/configs/` + runtime). Borrow the coding
   agents' contract RIGOR; never overload the coder to answer users.
2. **Learning tunes parameters, not arithmetic.** "The hat keeps developing"
   means it learns WHICH rate / lead-time / retention / procedure applies —
   calibration (the existing `ProcurementLeadTimeLearner` / `DurationCalibration`
   family). The formulas stay deterministic, verified Python. The LLM never
   rewrites the math.
3. **Taught is not invented.** A taught rule is grounded knowledge for that
   user/project, traceable and scoped. The anti-hallucination doctrine still
   holds: a hat may only answer from retrieved standard/taught knowledge or a
   computed tool result, else it refuses.
4. **Advisory, not blocking.** A deviation from standard is flagged, never
   halted — operators bend rules deliberately (matches the standards advisory
   already shipped).

## Productisation path

Each hat is a portable JSON/markdown contract, so the future CerebrumDev.ai
catalog can import the discipline set without scraping platform wrappers. Ships
standard; each deployment teaches it.

## Pilot (when we build, post-pilot-deploy)

Prove ONE hat end-to-end first: standard knowledge + teach-me + execute (real
formulas) + learn + on/off + one cross-discipline handoff stubbed in. Recommended
first hat: **Planning (EVM)** — it exercises the formulas, the schedule, and
forces the cross-discipline swap early, so we do not accidentally design a silo.
Once the vertical genuinely works, Commercial / Contracts / QA/QC / Procurement
are repetition, not invention.

## Explicitly out of scope now

Everything above. During the pilot we do not build hats, do not add discipline
agents, do not wire the calculators as tools. This document only fixes the
direction.
