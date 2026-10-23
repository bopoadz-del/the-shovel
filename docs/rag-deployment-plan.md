# RAG Deployment Plan

> Status: **authoritative plan.** The pilot runs the simple single-corpus RAG
> below. The layered RAG ("the real long-term RAG brain") is **deferred until
> client deployment** — at deployment, all current corpus content is treated as
> historical / training data and re-organised into the layers described here.
> Do NOT build the layers during the pilot.

---

## Current pilot RAG (KEEP SIMPLE)

**Master Corpus = single knowledge base.**

Flow:

> User asks → retrieve from Master Corpus → answer with sources → refuse / caveat
> if not grounded.

No layered RAG yet.

**This is ONE layer, not a frozen corpus.** Everything still connects INTO the
Master Corpus as a single piece — new uploads, project docs, BOQs, training
material all flow into the one corpus and become retrievable. What is deferred
is **splitting that content into separate layers (1 / 2A / 2B / 3) and applying
authority scoring** — not the ingestion. So during the pilot: keep connecting
content to the RAG as one undifferentiated layer; do the layer separation at
deployment.

**Confirmed working behaviour (pilot):**

- Exact references boosted (identifier-aware retrieval).
- Missing references controlled (caveat / decline instead of hallucinating).
- Sources structured and clean (Sources panel reflects cited chunks).
- Project Mode resolves the master-corpus alias to its backing corpus.
- Partial project shells hidden / not treated as main answerable corpora.

---

## Future RAG layers (AFTER deployment)

The layered RAG plan already discussed. Build at deployment, not before.

### Layer 1 — Shared Domain Knowledge
General construction knowledge.
Examples: FIDIC, construction standards, materials science,
concrete / steel / asphalt / soil knowledge, QA/QC procedures, general formulas,
planning / project-controls concepts.
**Use when:** the user asks general construction reasoning.

### Layer 2A — Company / Client Rules
Organisation-specific truth.
Examples: client procedures, company templates, approval workflows,
naming conventions, reporting rules, design-review procedures, commercial rules.
**Use when:** the answer depends on "how this company / client does things."

### Layer 2B — Live Project Record
The actual project truth.
Examples: contracts, specifications, drawings, BOQs, RFIs, NCRs, submittals,
site reports, letters, VOs, PEPs, programmes.
**This layer should DOMINATE when the user asks:**
- What does this project say?
- What is in this package?
- What did the contractor submit?
- What is the VO status?

### Layer 3 — User / Session Memory
Personal working context.
Examples: user preferences, current conversation, recently selected
project / package, previous decisions, draft notes, working assumptions.
**Rule:** never override project documents unless clearly marked as user
preference or working context.

---

## Cross-cutting authority scoring

Across all layers, attach an authority label to every piece of knowledge:

`contractual` · `design` · `commercial` · `operational` · `policy/procedure` ·
`historical` · `personal/session`

Precedence examples:

- Contract beats meeting note.
- Approved drawing beats old draft.
- Project BOQ beats generic cost benchmark.
- Company procedure beats general construction advice.

That is the real long-term RAG brain.

---

## What this means for the deferred Workstream-1 items

The earlier "indexing quality" items (per-page OCR for scanned/priced BOQs,
`boq_processor` → RAG wiring, package scoping) are part of building **Layer 2B**
(Live Project Record) and the authority scoring. They re-organise and re-index
content, so they belong to the deployment phase — NOT the pilot. The pilot's
single Master Corpus already answers grounded questions with sources today.
