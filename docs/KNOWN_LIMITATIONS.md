# KNOWN_LIMITATIONS — The_Fork / The Shovel

Worst-impact first. This is the page a serious buyer should read before
trusting the vendor. Every item is sourced to a repo file or a live sweep;
where a number is uncertain, it says so. It is deliberately non-defensive —
the goal is that nothing here surprises you after you sign.

Last reviewed against `main`: 2026-07-30. Numbers below are the most recent
figures in the repo at that date; several predate the current corpus/config
and are flagged as such.

---

## 1. Feature / recall pass rate is well under half (~41%)

This is the single worst headline number and it leads on purpose.

`TODO.md:200-201` records the last measured sweep:

> "TASK H GK knobs + RAG_AUDIT_V3 - PR #149: cfg7 recommended, calc-intact
> 3/3, embedder upgrade now ON the pre-pilot list (**doc recall 41% < 50% at
> best**)."

The adjacent orchestrator feature sweep is comparable:

> `TODO.md:197` — "TASK G feature sweep - PR #147: **23/54 PASS** through the
> orchestrator" (≈43%), with 20 prompts routing at confidence 0.0 (a
> routing-keyword-dictionary gap deferred post-pilot).

So the honest statement is: **on the last full measurement, document recall
sat at ~41% (below the 50% bar the team set), and end-to-end feature routing
passed ~23/54.** These two numbers measure different things (retrieval recall
vs. orchestrator feature pass) and the repo does not publish a single reconciled
"feature-matrix %"; treat ~41% as the floor to anchor on, not a precise
composite.

**2026-08-02 re-measurement and diagnosis** (full record:
`docs/rag-reranker.md`; artifacts in `data/learning/rag_audit/`): the 41% was
re-measured LIVE on the current BGE-384 corpus — it is current, not stale.
Two remedies were then measured and ruled out with data:

- **Re-embedding is a dead lever.** The potion-256 -> bge-small-384 migration
  already happened (2026-07-12) and left recall@5 at ~41%. The "embedder
  upgrade" note above is therefore SPENT — do not plan another corpus
  re-embed to move this number.
- **Off-the-shelf cross-encoder reranking measured NEGATIVE.** Reranking live
  top-50 with ms-marco-MiniLM dropped doc-recall@5 from 47% to 43% (n=60).
  The flag-gated hook ships DORMANT (`RAG_RERANKER`, default off); do not
  enable it with the default model.

What the floor actually decomposes into (never-found@50, n=60): **~15% of
ground-truth docs no longer exist in the live corpus** (the stalled 2026-07-12
backfill / June-corpus drift — no retrieval change can answer these; this is
missing client data and the highest-leverage fix), a vocabulary-mismatch slice
(chunks present, questions never reach them), and a near-duplicate
drawing-sheet slice (hundreds of sheets share title-block text; needs
metadata-aware retrieval, not ranking). Priority order and evidence:
`docs/rag-reranker.md`.

**2026-08-02 corpus restore:** the 9 eval-verified missing documents were
located on Drive (3 had moved folders), re-uploaded through the app API
(pilot-verified retrievable before batching), and confirmed live-answerable.
drawing-package chunks verified present. The remaining known
chunk gap concentrates in the vol 3 drawings block of the executed contract — restoring
it needs the owner-gated direct-DB re-encode (see PLATFORM_HEALTH_REPORT
addendum, owner action 6). Note: restored uploads carry NEW doc ids, so the
seed-42 recall metric still scores those 9 queries as ID-misses even though
the content now answers — the metric understates user-facing recall.

---

## 2. Source-precedence / authority ordering ships DEFAULT-OFF, and is only a tie-break nudge

Layered-RAG — the feature that would let a contract outrank a meeting note, or
an approved drawing outrank a draft — is gated behind `RAG_LAYERED`, which
**defaults OFF** (`app/core/rag/layers.py:33-36`: `os.getenv("RAG_LAYERED", "")`,
truthy only on `1/true/yes/on`). With the flag off, retrieval ordering is
"byte-for-byte today's ordering" (`app/core/rag/retriever.py:719`).

Even when enabled, it is a **small additive bonus applied after fusion**, not a
document-vs-document conflict resolver. From `retriever.py:715-717`: it adds "a
small term so a higher-authority / higher-layer chunk … outranks a
*comparably-relevant* low-authority one." It re-orders chunks that already
scored close together; it does not detect that two documents disagree and
suppress the wrong one. If a low-authority doc is simply more lexically
relevant, it can still win.

---

## 3. Revision currency: implemented (always-on), with narrow residual limits

Retrieval DOES prefer the newer drawing revision, always-on and independent of
`RAG_LAYERED` (PR #292, extended 2026-07-30). Two filename-derived mechanisms in
`app/core/rag/revision.py`, applied by the retriever:
- **Superseded down-rank** — a doc whose name matches
  `superseded|obsolete|archive|\bold\b|previous|deprecated` takes a firm score
  penalty (not deletion — it survives as a flagged last resort).
- **Highest-revision preference** — among retrieved chunks sharing a drawing
  number, a lower revision is suppressed when a strictly-higher, same-kind one
  is also retrieved. `Rev 10` correctly outranks `Rev 9` (multi-digit), and
  `Rev AA` outranks `Rev Z` (base-26). The model is also told which revision the
  evidence is from.

Residual limits (all fail SAFE — they keep both revisions, never suppress the
current one):
- Numeric and letter revisions of the same sheet are left un-ordered (a mixed
  `Rev 2` vs `Rev B` is not guessed).
- A digit run longer than 3, or a revision marker not preceded by a boundary
  (e.g. inside a run-together name), yields no revision signal.
- Multi-character alphabetic revisions beyond two letters (`Rev AAA`) are not
  ordered.

---

## 4. No pre-retrieval scope refusal

The system does not decide "this question is out of scope" before searching.
Refusals come from two places, both after or independent of retrieval:
- **Corpus-empty / no-context** paths (nothing was retrieved), and
- **Post-retrieval honest-error gates** — e.g. the cost-grounding gate refuses a
  cost figure that cannot be traced to a rate-semantic chunk
  (`CONSTRUCTION_LEDGER.md:96-100`, `agents/runtime.py` `_cost_grounding_gate`),
  and the fabrication-kill gates return requirement-naming errors instead of
  invented facts (`CONSTRUCTION_LEDGER.md:19-24`, F1–F4).

There is no classifier that turns away an off-domain or out-of-project question
up front. An off-scope question is answered from whatever the retriever returns
(or from a no-context fallback), not rejected as off-scope.

---

## 5. "Cites real documents … in EVERY answer" is overstated

README's product blurb says every answer is grounded and cited. The
cost-grounding gate is real but **narrowly scoped to cost/rate figures**
(`CONSTRUCTION_LEDGER.md:96-100`): it grounds a *figure plus its semantics*, not
every sentence. Critically, the **degraded / no-context paths carry no
citations** — when retrieval returns nothing, the answer takes a fallback prefix
(`app/core/rag/inject.py`, confidence threshold 0.4, "fallback prefix on miss"
per README's Retrieval section) with no source list.

So the accurate claim is: *grounded answers cite their sources, and cost figures
are gated against fabrication; but not every answer is a cited answer* — the
no-context and degraded paths are uncited by construction. README's "cites real
documents with confidence scores in every answer" should be read with that
caveat.

---

## 6. The 28-question golden set was authored WITH sight of the corpus

The golden / eval set (28 questions, per `PILOT_READINESS.md:33` T6 "28 golden";
results in `golden_set_results.jsonl`) had its ground-truth expectations written
by someone who had already seen the corpus. That makes it a useful regression
harness, **not a blind evaluation bar**. No blind eval (questions and answer keys
authored without corpus sight) has been passed yet. Recent golden runs report
27/29 PASS under prod config (`PLATFORM_HEALTH_REPORT.md:108`) — but against this
same non-blind set, and with 2 known failures (a Groq TPM billing gate and a real
RFP-generation bug). Read the pass rate as "our own regression suite is green,"
not "an independent evaluator scored us."

---

## 7. BIM family generation needs a BIM model — parked

Anything that depends on an actual BIM/IFC model is parked, not shipped:
- `track_progress` (photo-vs-BIM) and `generate_construction_report` are
  **PARKED-BY-DESIGN** pending multi-file/photo param resolution
  (`CONSTRUCTION_LEDGER.md:41`, W1).
- Photo-BIM geo-anchoring and the ResNet-18 CNN head are listed
  **PARKED-BY-DESIGN (verified dormant, no code)** at
  `CONSTRUCTION_LEDGER.md:137`.
- `digital_twin_sync` **prepares** platform payloads but never pushes to a live
  twin — `sync_status: "prepared_not_pushed"` (`CONSTRUCTION_LEDGER.md:42`, W2).

These features are honestly gated: absent the required model/inputs they return
empty or honest errors rather than fabricating.

---

## 8. 17 skip/xfail test files — "green" is not full coverage at ~41%

CI green does not mean full functional coverage. Contributing factors:
- The audit cites **17 skip/xfail test files**; a broader repo scan for
  skip/xfail/skipif markers currently hits ~31 test files (`grep` over `tests/`
  on 2026-07-30) — either way, a non-trivial slice of the suite is not asserting
  live behavior. Known quarantines include a win32 concurrency skip and 2
  GK-crowding xfails pending a config pick (`TODO.md:208-210`).
- The CI **coverage floor is 25%** (`README.md` Tests section,
  `.github/workflows/test.yml`); `diff-cover` ratchets new code but does not
  raise the floor.
- Green sits on top of the ~41% recall / 23-of-54 feature reality in §1.

Interpretation: passing CI proves the asserted paths still work and nothing
regressed against a partial, corpus-aware suite. It does not certify the whole
designed surface.

---

## A note on staleness (read this too)

The repo's own docs disagree in places because prod moved faster than the
docs, and this file does not hide that:
- **Provider ladder.** `DECISIONS.md` / `docs/archive/PROGRESS.md` still
  describe OpenAI-primary or Groq-Scout-primary + Ollama-fallback;
  live prod is **Kimi-primary + Groq-fallback** since 2026-07-24
  (`PLATFORM_HEALTH_REPORT.md:22` F2, `render.yaml:82-104`). README's LLM claim
  was corrected in this same change set.
- **Groq free-tier ceiling.** The Groq fallback 413s payloads above ~12k tokens
  on the current free account, so it catches normal turns but not large grounded
  ones — a billing gate, not a code bug (`PLATFORM_HEALTH_REPORT.md:20-21` F1a,
  parked gate #1).
- **Embedding default vs. prod.** Shipped code default is
  `minishlab/potion-base-8M` (model2vec, 256-dim,
  `app/core/rag/embeddings.py:33`); prod overrides `RAG_EMBEDDING_MODEL` to
  `BAAI/bge-small-en-v1.5` (384-dim, `PLATFORM_HEALTH_REPORT.md:67`).
- **Retrieval fusion wording.** `CONSTRUCTION_LEDGER.md:138` flags README's "RRF
  fused" description as stale (the RRF seam changed); not corrected in this pass.
- **Corpus size.** The canonical `chunks_v2` store was 53 docs / 10,502 chunks
  on 2026-07-12 (`PILOT_READINESS.md:14`); the older README "~142k chunks" figure
  is not sourced anywhere in the repo and has been removed.

If any number here matters to your decision, ask for it to be re-run live —
several of the worst figures (§1, §6) predate the current corpus and provider
ladder, and the honest answer today is "measured then, not re-measured since."

## Decisions recorded 2026-08-15 (operator-directed residual sweep)

- **RAG_LAYERED stays OFF for the pilot, by design.** Operator decision
  ("no per-project isolation at this period"): the pilot runs one corpus;
  the layered system is merged, tested, and reserved for per-client
  deployments.
- **DWG is not read (owner ruling, 2026-10).** DWG take-off is removed:
  take-off (`drawing_qto`) and BIM (`bim`) answer a DWG with 415 and
  "DWG is not supported. Export DXF or PDF from your CAD software."
  (`app/core/cad_formats.py`). The image carries no DWG converter. DXF and
  PDF take-off are unchanged.
- **Compute tier stays `standard` (2 GB) for now, measured.** Both
  2026-08-15 OOMs were code defects, since bounded (pixel cap, page-size
  table gate, geometry skip). Neon database usage is 43 MB of the 512 MB
  free tier. Upgrade compute only when throughput measurably demands it.
