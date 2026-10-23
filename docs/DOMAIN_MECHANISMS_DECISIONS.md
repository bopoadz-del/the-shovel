# Domain Mechanisms — Decide-and-Declare (Market-Readiness Audit §5)

Status: **decision memo** for the product owner. This document changes **no**
runtime or retrieval behavior. It states the current real behavior with
`file:line` evidence, lays out options, and ends each section with a single
clear RECOMMENDATION for the owner to accept or reject.

Scope covered: §5.1 Source precedence, §5.2 Revision currency, §5.3
Pre-retrieval scope refusal.

> **Flag-truth caveat (applies to 5.1).** `render.yaml` env is **not**
> auto-synced to the Render dashboard, and `RAG_LAYERED` /
> `RAG_AUTHORITY_WEIGHT` / `RAG_LAYER_WEIGHT` are **not pinned in
> `render.yaml`** (confirmed: no match in `render.yaml`). Therefore the
> version-controlled truth is the **CODE DEFAULT** and the **live dashboard
> value is UNKNOWN**. Where the live flag state matters below, it is flagged as
> "must be confirmed against the Render dashboard" and is not asserted here.

---

## 5.1 Source precedence — PARTIAL

### Current real behavior (evidence)

- **Authority order is real and construction-shaped.** The precedence order is
  a fixed tuple, strongest first:
  `AUTHORITIES = ("contractual", "design", "commercial", "operational", "policy", "historical", "personal")`
  — `app/core/rag/layers.py:19-20`. `authority_rank()` returns `0` for the
  strongest and sorts unknowns weakest (`layers.py:39-44`).
- **The precedence signal is an additive tie-break bonus, not a conflict
  resolver.** `precedence_bonus(knowledge_layer, authority)` —
  `app/core/rag/layers.py:150-157` — returns
  `w_auth * authority_weight(authority) + w_layer * layer_weight(knowledge_layer)`
  with **both weights defaulting to `0.05`** (`RAG_AUTHORITY_WEIGHT`,
  `RAG_LAYER_WEIGHT`; `layers.py:155-156`). Its own docstring states the intent:
  "the term breaks ties and nudges, **it does not override cosine**"
  (`layers.py:153-154`).
- **It combines with cosine as a small post-hoc addend.** In the retriever the
  bonus is added to the already-fused score, then the pool is re-sorted:
  `new_score = score + bonus` (`app/core/rag/retriever.py:722-725`), applied
  **after** the GK-margin gate and only to change final ordering
  (`retriever.py:713-728`). `authority_weight` scales from `1.0`
  (contractual) down to ~`1/N` (`layers.py:142-147`), so a contractual chunk's
  maximum lift is ~`0.05` — enough to reorder two near-equal-cosine chunks, not
  enough to pull a semantically weaker contract clause over a strongly-matching
  lower-authority chunk.
- **The entire mechanism is DEFAULT-OFF.** `layered_enabled()` reads
  `RAG_LAYERED` live and returns **`False` by default** (truthy only for
  `1/true/yes/on`) — `app/core/rag/layers.py:33-36`. When off, the re-rank
  loop is skipped entirely and ordering is "byte-for-byte" today's behavior
  (`retriever.py:718-728`). The go-live doc confirms: with the flag unset,
  `classify()` never runs and `precedence_bonus` is never applied
  (`docs/layered-rag-golive.md:20`).

So the audit finding is accurate: real authority order exists, but it is (a)
default-off, and (b) a tie-break nudge rather than a doc-vs-doc conflict
resolver.

### Options

1. **Leave OFF (status quo).** Zero behavior change. Authority order never
   influences retrieval on the live corpus. Cost: the product cannot claim
   "contract beats drawing beats submittal" — because at runtime it does not.
2. **Flip `RAG_LAYERED=1` in prod, keep the `0.05` nudge.** Turns on the
   tie-break. Low blast radius (only reorders near-equal-cosine neighbors), but
   also low marketing-to-reality payoff: it will rarely change a top-K answer,
   and it silently depends on the layer/authority backfill having been run
   correctly on the live corpus (`chunks.layer` / `chunks.authority` populated;
   see Alembic 0011). On rows where those columns are null, the bonus is `0.0`
   (`layers.py:138-147`) — i.e. a partially-backfilled corpus gets inconsistent,
   hard-to-explain nudges.
3. **Flip ON and raise the weights toward a real precedence tier** (e.g.
   `RAG_AUTHORITY_WEIGHT` large enough that contractual can clear a cosine gap).
   This is the only variant that makes authority *win* conflicts — but it is
   also the highest-risk: a large additive term over cosine will surface
   loosely-relevant contract text on unrelated queries and is exactly the kind
   of retrieval change the audit says must not be made autonomously on a live
   system.
4. **Build a true conflict resolver** (detect that two retrieved chunks make
   competing claims about the *same* subject, then prefer the higher-authority
   source and disclose the override). This is a genuine feature, not a flag flip;
   it needs same-subject linkage (see 5.2, which is the prerequisite for
   "same drawing / same clause") and answer-time disclosure. Largest effort.

### RECOMMENDATION — **known-limitation-for-now; do NOT autonomously flip ON.**

Keep `RAG_LAYERED` at its code default (OFF) until three preconditions are met,
because flipping it on a live corpus is a retrieval change with a real failure
mode (inconsistent nudges on partially-backfilled rows) and near-zero upside at
the current `0.05` weight:

1. Confirm the **actual live `RAG_LAYERED` value on the Render dashboard**
   (unknown today — see caveat). If it is already `1` in prod, that is an
   undocumented live-retrieval state the owner must reconcile, not a settled
   fact.
2. Verify the layer/authority **backfill is complete** on the live corpus (no
   null `layer`/`authority` on active chunks), so the bonus is uniform.
3. Decide the product claim. A `0.05` tie-break is **not** sufficient to
   truthfully advertise "contract overrides drawing." For construction
   authority ordering to be a real guarantee you need option 4 (a conflict
   resolver with disclosure), which is a scoped build — not this flag. Until
   then, describe source precedence as "authority-aware tie-breaking," not
   "authority wins."

Owner decision required: (a) keep OFF pending preconditions, or (b) schedule the
option-4 conflict-resolver build. Do not ship option 3 (weight-raise) blind on
the live corpus.

---

## 5.2 Revision currency — FAIL

### Current real behavior (evidence)

- **Classification tags document TYPE, not baseline-vs-revised.**
  `_classify_document()` returns a type string (`bim`, `schedule`, `contract`,
  `specification`, `bom`, `report`, `image`, `change_order`, `safety_audit`,
  else `drawing`) purely from filename keywords —
  `app/containers/construction/__init__.py:51-72`. Nothing in it distinguishes a
  current revision from a superseded one.
- **The only "superseded" signal is a filename keyword that maps to the weakest
  authority, and it is layered-off.** `superseded` (with `obsolete|archive|old|
  previous|deprecated`) matches the `historical` authority pattern —
  `app/core/rag/layers.py:70`. `historical` is second-weakest in the authority
  tuple (`layers.py:19-20`), so via `precedence_bonus` it earns the smallest
  bonus. But that down-weight only exists when `RAG_LAYERED` is on
  (`retriever.py:718-728`), which is default-OFF (§5.1). With the flag off there
  is **no** superseded down-weight at all.
- **Drawings DO parse a revision field — but retrieval never uses it.**
  A revision is extracted from the filename (`_extract_revision()` regex
  `[Rr][Ee]?[Vv]?\s*([A-Z0-9])`) — `construction/__init__.py:173-176` — and
  stored in the document record (`"revision": self._extract_revision(...)`,
  `app/containers/construction/documents.py:198`). The drawing-QTO title-block
  parser also derives a `revision` with multiple fallbacks
  (`app/blocks/drawing_qto.py:686-727`, `:953-956`, surfaced in metadata at
  `:1214-1222`). **However**, `revision` appears in the retriever only inside a
  comment about lexical code tokens (`app/core/rag/retriever.py:45`, `:58`) —
  there is **no** ranking logic anywhere that compares revisions of the same
  drawing and prefers the higher one. Confirmed by grep: `revision` has zero
  ranking uses in `app/core/rag/` or `app/agents/runtime.py`.

So retrieval can, today, return **Rev A of a drawing when Rev C exists** in the
same corpus, with equal footing — and (flag-off) with no superseded penalty.
The audit finding is accurate.

### What it would take to prefer the higher revision

The parsed data already exists; what is missing is **same-document identity +
a currency preference**. Concretely:

1. **Stable drawing identity.** Group chunks by drawing number
   (`drawing_number` is already extracted — `construction/__init__.py:187`,
   `drawing_qto.py:915-926`) so "Rev A" and "Rev C" of one sheet are known to be
   the same logical document. Filename/hash identity is not enough (`_doc_id`
   keys on type+hash — `construction/__init__.py:42-50`).
2. **A comparable revision order.** Normalize the parsed `revision`
   (letter A<B<C; numeric 00<01; the parser already uppercases/strips —
   `drawing_qto.py:727`) into a sortable key. Handle the messy real cases the
   parser already flags (`revision_fallback_to_filename`,
   `drawing_qto.py:722`) as low-confidence.
3. **A retrieval preference.** When two retrieved chunks share a drawing
   identity, **suppress or hard-down-rank the lower revision** and keep the
   highest. This is a stronger action than the `0.05` nudge — for stale-revision
   safety it should be a filter/cap, not an additive bonus.
4. **Answer-time disclosure.** State the revision used ("per Rev C") and, when a
   superseded rev was excluded, say so — so a user cannot be silently answered
   from stale drawings.

### RECOMMENDATION — **MUST-FIX for a construction client** (staged).

A stale-revision answer on a drawing (dimension, detail, or spec that changed in
a later rev) is a real-world safety and liability hazard, and it is exactly the
failure this product exists to prevent. "Type not currency" is acceptable for a
demo, not for a live construction client. Recommended concrete design, in the
order it de-risks fastest:

- **Phase 1 (near-term, low risk): disclosure + explicit superseded penalty
  that does NOT depend on `RAG_LAYERED`.** Today the only superseded signal
  rides on the default-off layered path (`layers.py:70` via
  `retriever.py:718-728`). Add a small, always-on down-rank for chunks whose
  document is keyword-superseded, and surface the revision string in the answer.
  This removes the worst case (confidently citing a doc whose own filename says
  "superseded") without a corpus-wide re-architecture.
- **Phase 2 (the real fix): same-drawing revision preference.** Implement steps
  1-4 above as a retrieval post-filter keyed on `drawing_number` + normalized
  `revision`, preferring the highest rev and disclosing it. Because it only acts
  among chunks that share a drawing identity, its blast radius is contained and
  auditable.

Both phases are retrieval-behavior changes and therefore **owner-approved,
separately-scoped work** — this memo recommends them, it does not implement
them. Until Phase 1 ships, revision currency should be disclosed to the client
as a **known limitation** ("always confirm the drawing revision"), not presented
as handled.

---

## 5.3 Pre-retrieval scope refusal — FAIL

### Current real behavior (evidence)

- **`MissingContextPolicy.REFUSE` is a declared enum value with no runtime
  branch.** The enum exists — `DEPENDENCY_REQUIRED | ASK_CLARIFYING |
  USE_BEST_EFFORT | REFUSE` at `app/agents/models.py:43-47` — and
  `FailurePolicy.on_missing_context` defaults to `ASK_CLARIFYING`
  (`models.py:113-114`). But the only place `failure_policy` is consumed is a
  config merge (`merged.failure_policy = hat.failure_policy`,
  `app/agents/catalog.py:146`). There is **no** runtime code that reads
  `on_missing_context` and acts on `REFUSE` (confirmed by grep across `app/`:
  the value is defined and merged, never branched on). `REFUSE` is dead config.
- **There is no pre-retrieval scope primitive.** No `out_of_scope`,
  `must_refuse`, or `never_attempt` exists anywhere in `app/` (grep: NONE
  FOUND). Nothing inspects the incoming question *before* retrieval to decide
  the query is out of the product's domain and refuse up front.
- **All existing refusals are corpus-empty or post-retrieval honest-error:**
  - **Corpus-empty (pre-LLM, but not scope-based):** the zero-chunk project
    guardrail refuses before spending LLM budget when the project has no indexed
    documents and no non-RAG context, returning `_UNINDEXED_PROJECT_MESSAGE`
    (`app/agents/runtime.py:2689-2705`; message at `runtime.py:204-208`). This
    keys on *corpus readiness*, not on the *question's* scope.
  - **Post-retrieval honest-error:** the cost-grounding gate refuses to emit a
    money/rate figure that is not grounded in retrieved context, returning
    `_CG_REFUSAL` (`app/agents/runtime.py:1300-1315`). This runs *after*
    retrieval, on the drafted answer.

The audit finding is accurate: `REFUSE` is unwired, and refusal today is either
"no corpus" or "answer wasn't grounded," never "this question is out of scope,
don't even try."

### Options

1. **Keep as-is.** Rely on the two existing gates. An out-of-scope question
   against a real project still runs retrieval; it gets low-relevance chunks and
   is then caught by grounding gates (cost-grounding, cite-required
   `Verification`, `models.py:104-109`) or answered as best-effort with
   citations. Marginal safety already exists; the miss is that a clearly
   off-domain question (e.g. medical/legal-outside-construction) still consumes
   retrieval + LLM budget before any honest-error gate fires.
2. **Implement pre-retrieval scope refusal** by wiring `REFUSE`: add an
   `out_of_scope` classifier (keyword or intent) that runs before retrieval and,
   when the active hat's `on_missing_context == REFUSE`, returns a refusal
   without retrieving. Cost: a new classifier is a **new failure surface** on a
   live system — false-positive refusals (declining a legitimate construction
   question) are a worse client experience than a well-grounded "I don't have
   that in your documents," and the classifier must be tuned against the real
   corpus. It also duplicates protection the grounding gates already provide for
   the dangerous case (ungrounded figures).

### RECOMMENDATION — **keep-as-is; do NOT implement pre-retrieval scope refusal
now.**

The marginal safety value is low and the implementation risk is real. The
genuinely dangerous outcome — a confident, ungrounded number — is already
blocked *after* retrieval by the cost-grounding gate (`runtime.py:1300-1315`)
and by `Verification.cite_required` / `refuse_unaided_figures`
(`models.py:104-109`); and a no-corpus project is already refused up front
(`runtime.py:2689-2705`). A pre-retrieval scope classifier would mainly save
compute on off-domain questions while adding a new false-refusal risk on a live
client, which is the wrong trade for a construction product whose users ask
wide-ranging but on-domain questions.

Two low-risk follow-ups instead of a full classifier:

- **Remove the dead-config smell:** either wire `MissingContextPolicy.REFUSE` to
  a real branch *or* document it as reserved/not-implemented so it cannot be
  mistaken for a live guarantee (today it silently no-ops —
  `catalog.py:146`, `models.py:47`).
- **Revisit only if** the product opens to untrusted/public queries where
  refusing off-domain input up front becomes a cost or abuse concern; at that
  point implement option 2 with the classifier tuned on the live corpus and a
  bias toward *answering-with-citations* over refusing.

---

## Summary of recommendations

| § | Mechanism | Current behavior (key evidence) | Recommendation |
|---|-----------|----------------------------------|----------------|
| 5.1 | Source precedence | Real authority order (`layers.py:19-20`) as a `0.05` additive tie-break (`layers.py:150-157`, `retriever.py:722-725`), **default-OFF** (`layers.py:33-36`); not pinned in `render.yaml` so live value UNKNOWN | **Known-limitation; do NOT autonomously flip ON.** Confirm dashboard value + backfill first; a nudge ≠ "contract wins" — schedule a conflict resolver for that claim |
| 5.2 | Revision currency | Type-not-currency classification (`construction/__init__.py:51-72`); revision parsed (`__init__.py:173-176`, `documents.py:198`, `drawing_qto.py:686-727`) but **never used in ranking**; superseded down-weight is layered-off (`layers.py:70`) | **MUST-FIX (staged).** Phase 1: always-on superseded penalty + revision disclosure; Phase 2: same-drawing highest-revision preference |
| 5.3 | Pre-retrieval scope refusal | `REFUSE` enum unwired (`models.py:47`, only merged at `catalog.py:146`); no `out_of_scope`/`must_refuse`; refusals are corpus-empty (`runtime.py:2689-2705`) or post-retrieval honest-error (`runtime.py:1300-1315`) | **Keep-as-is.** Marginal safety low vs false-refusal risk; the dangerous case is already gated post-retrieval. Document `REFUSE` as reserved |
