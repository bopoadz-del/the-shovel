# RAG Quality + Training/Evaluation Questions Audit

**Audited:** 2026-06-23  
**Environment:** production `https://the-fork.onrender.com`, commit `221ac51`  
**Scope:** inventory the current corpus, inventory existing training/evaluation questions, evaluate whether RAG answers are grounded, source-supported and construction-useful, and deliver a RAG quality verdict.  
**Constraints respected:** no code changes, no commits, no reindex/rebuild, no Drive imports, no data deletion.

---

## 1. Executive verdict

**RAG quality is functional for broad, high-level questions but unreliable for precise, document-specific construction queries.**

The chat surface can produce fluent, source-cited summaries of well-represented topics (e.g. the the client project Project Execution Plan, the MV Culvert Diversion drawing). However, objective measurement against the project's own training questions shows that the retriever fails to surface the exact source document in **~69 % of cases** and fails to surface the exact ground-truth chunk in **~80 % of cases** (doc recall@5 = 31 %, chunk recall@5 = 20 % on a 100-question sample).

High semantic similarity scores (avg top score ≈ 0.69) are therefore misleading: the model returns chunks that are *topically* close but often from the wrong drawing, the wrong BOQ sheet, or a generic procedure, which leads to:

* answers that are plausible but sourced from the wrong document;
* answers that say "I cannot find this" when the information is in the corpus;
* occasional hallucinated/contradictory answers on critical rules (e.g. whether `APPROVED` is allowed on design documents);
* degenerate outputs such as raw tool-call JSON or empty assistant bubbles.

The training/evaluation corpus is large, mostly well-mapped to the indexed chunks, and covers the right construction disciplines, but it is noisy: duplicated instructions, version sprawl, and contradictory labels for the same procedure (PRC-951 design-review statuses).

**Recommendation:** do not rely on RAG for fine-grained document lookup until retrieval is upgraded. The highest-impact fixes are (1) move from the current 256-dim `model2vec` embedder to a stronger sentence-transformer or domain embedding model, (2) verify hybrid BM25 is actually active and tuned, (3) deduplicate and reconcile training labels, and (4) add a continuous recall@K evaluation gate.

---

## 2. Corpus inventory

### 2.1 Production collections (`GET /v1/admin/corpus/collections`)

| project_id | documents | chunks | notes |
|---|---|---|---|
| `projects_folder` | 2 713 | 110 379 | backing corpus for the `master_corpus` master-corpus alias |
| `training_material` | 241 | 10 907 | cross-project general-knowledge corpus (procedures, scanned references) |
| `example_estate` | 62 | 383 | user project; own corpus exists |
| `fb776aa2` | 15 | 57 | small user project |
| `c0ac2b2d` | 23 | 33 | small user project |
| `77dd3f5d` | 8 | 23 | small user project |
| `3f6f28b2` | 3 | 8 | small user project |
| `8f73170f` | 2 | 6 | small user project |
| `example_infra_pack_2` | 1 | 3 | small user project |
| `bb00878f` | 8 | 0 | **documents present, no indexed chunks** |
| `df28d3c0` | 8 | 0 | **documents present, no indexed chunks** |
| `e483b574` | 2 | 0 | **documents present, no indexed chunks** |
| **total** | **3 086** | **121 799** | across 12 project ids |

*Drive import is currently disconnected on production.*

### 2.2 Content composition (local `data/rag/vectors.db` is representative)

A 100 000-chunk sample of the `drive_archive` / `projects_folder` corpus shows the content is overwhelmingly the the client project project:

* **99 650 / 100 000 chunks** are under `X:\Example Drive\Master Folder\the client project\...`.
* The remaining ~350 chunks are scattered personal files, CVs, certificates and unrelated spreadsheets.
* Average chunk length ≈ 650 characters (max 994, min 101). Chunking exceeds the nominal 512-char window in places, likely because the `[source: ...]` prefix is included in the stored text.

This concentration is good for domain focus but creates a retrieval confounder: many drawings/BOQs share identical boilerplate ("GEODETIC DATUM: WGS-84", "28W LED fixture for 4.5M pole"), so a small embedding model cannot reliably distinguish one drawing from another.

### 2.3 Master-corpus alias behaviour

The public project id `master_corpus` is an alias; the chat route resolves it to `projects_folder`. The standalone `POST /v1/rag/search` route **does not resolve the alias**, so direct RAG debugging must use `projects_folder`. This was confirmed by the retrieval recall test.

---

## 3. Training / evaluation question inventory

### 3.1 Files found

All files are under `data/learning/`.

| file | rows | type | key schema |
|---|---|---|---|
| `training_scenarios.jsonl` | 26 245 | instruction-tuning (drive_archive corpus) | `instruction`, `context`, `response`, `source`, `discipline` |
| `training_scenarios_drive_archive.jsonl` | 498 | curated drive_archive sample | as above + `source_doc_path` |
| `training_scenarios_drive_archive_clean.jsonl` | 498 | cleaned curated sample | as above |
| `training_scenarios_drive_archive_v2.jsonl` | 1 430 | expanded curated sample | as above |
| `training_scenarios_merged.jsonl` | 1 349 | mixed knowledge + docs | `instruction`, `response`, `source`, `source_detail` |
| `training_scenarios_rag_grounded.jsonl` | 1 349 | RAG/no-RAG mixed | includes `context` retrieved chunks; 471 rows explicitly `no_rag_hit` |
| `training_scenarios_v[3-8]_shard_*.jsonl` | ~17 000+ | versioned shards | same schema as main file |
| `expert_scenarios.jsonl` | 435 | construction expert prompts | `instruction`, `response`, `source` = `construction_expert.txt` |
| `knowledge_scenarios.jsonl` | 201 | construction knowledge prompts | source = `construction_knowledge.py` |
| `evm_scenarios.jsonl` | 590 | earned-value management prompts | source = `construction_evm.md` |
| `high_density_facts.jsonl` | 48 | adversarial fact drills | source tags like `prc501_approved_forbidden` |
| `the client_negatives.jsonl` | 25 | negative/correction examples | source tags like `the client_negatives` |
| `adapters/2026*/..._eval.md` | 15+ | adapter evaluation reports | human/LLM-judged pass/fail |
| `rag_baseline_eval.md` | 1 report | RAG-only baseline on `globalkb` | 7/10 pass |

Total on-disk training examples: **~59 000+ rows**.

### 3.2 Discipline coverage (main `training_scenarios.jsonl`)

| discipline | count |
|---|---|
| contract | 6 864 |
| drawings | 5 233 |
| schedule | 4 992 |
| other | 3 473 |
| report | 2 556 |
| mep | 650 |
| lighting | 638 |
| hse | 569 |
| structural | 550 |
| spec | 298 |
| boq | ~200+ |

Coverage is well balanced across construction domains.

### 3.3 Source mapping to the indexed corpus

| training file | unique drive_archive doc_ids | % of those doc_ids present in local chunks |
|---|---|---|
| `training_scenarios.jsonl` | 2 778 | **99.8 %** (2 772 / 2 778) |
| `training_scenarios_drive_archive_v2.jsonl` | 976 | **100 %** |

Source-chunk presence is also very high:

| training file | source chunks present in local chunks |
|---|---|
| `training_scenarios.jsonl` | 25 477 / 26 245 = **97.1 %** |
| `training_scenarios_drive_archive_v2.jsonl` | 1 430 / 1 430 = **100 %** |

**Conclusion:** the training/evaluation set is not missing from the corpus. The problem is that the retriever cannot reliably pull the right chunk when asked the training question.

### 3.4 Quality issues in the question corpus

* **Duplicate instructions:** `training_scenarios.jsonl` has **1 556 duplicate instructions** out of 26 245 rows (~6 %). `drive_archive_v2` has 80 duplicates out of 1 430 (~6 %).
* **Version sprawl:** multiple versions of the drive_archive training set (`v2`, `v3` shards, `v4`-`v8` shards, `clean`, `merged`, `rag_grounded`) make it unclear which is the canonical eval set.
* **Contradictory labels for PRC-951:**
  * `app/prompts/construction_expert.txt` and `app/core/construction_knowledge.py` say valid design statuses are `FOR_COMMENT`, `ACCEPTANCE`, `BUY_OFF` (and `APPROVED` is forbidden).
  * `data/learning/high_density_facts.jsonl` says valid statuses are `REVIEWED`, `COMMENTS INCORPORATED`, `REJECTED`.
  * `data/learning/expert_scenarios.jsonl` says valid statuses are `FOR COMMENT`, `ACCEPTANCE`, `BUY-OFF`.
  * This inconsistency makes it impossible to train or evaluate a reliable PRC-951 classifier without first reconciling the source of truth.

---

## 4. RAG retrieval quality

### 4.1 Objective recall measurement

Method: sample questions from `training_scenarios_drive_archive_v2.jsonl` (each has a known `drive_archive:<doc_id>:<chunk_index>` source), call `POST /v1/rag/search` with `project_id=projects_folder` and `k=5`, and check whether the ground-truth doc/chunk appears in the top 5.

| metric | value |
|---|---|
| sample size | 100 |
| **doc recall@5** | **31 %** |
| **exact-chunk recall@5** | **20 %** |
| average top-1 semantic score | **0.693** |

Observations:

* The retriever almost always returns 5 chunks (only a handful returned fewer).
* Top scores are consistently high (>0.5 and often >0.8), but the returned chunks are frequently from the *wrong* drawing/BOQ/procedure.
* This pattern is characteristic of a small, general-domain embedding model collapsing fine-grained distinctions in a domain with repetitive boilerplate.

### 4.2 Search spot-checks

| query | project | result | issue |
|---|---|---|---|
| `concrete` | `example_estate` | returned generic `training_material` chunks about prestressed/reinforced concrete | active project corpus ignored for a generic term; GK corpus dominates |
| `rebar` | `master_corpus` (alias) | only 1 chunk, score 0.06, garbled OCR | very poor retrieval |
| `the client project project execution plan` | `master_corpus` | **0 chunks** | high-value document not retrieved |
| `bill of quantities` | `master_corpus` | 3 relevant contract-template chunks, scores ~0.57-0.58 | good |
| `FIDIC clause` | `master_corpus` | 2 near-duplicate chunks, score ~1.4e-7 | effectively no signal |
| `lighting fixture` | `master_corpus` | generic scanned high-rise building chunks, not the relevant the client project drawing | wrong source |

The master corpus of 110k+ chunks is large enough that generic queries drown specific ones, and the 256-dim `model2vec` embeddings do not preserve enough discriminative signal.

### 4.3 Hybrid search status

`RAG_HYBRID_SEARCH` defaults to `true` and the retriever passes `query_text` into the vector store, but the chat path and the standalone RAG search both exhibited the high-score/wrong-source behaviour above. The BM25 leg does not appear to rescue document-specific lookups. Without query logs we cannot confirm it is active in production; the operator should add an observability log or test with `RAG_HYBRID_SEARCH=false` to compare.

---

## 5. Chat answer quality

Method: call `POST /v1/agents/project-assistant/chat` with `project_id=master_corpus` and inspect the returned answer, source citations and `iterations`.

| query | status | verdict | notes |
|---|---|---|---|
| "What is the the client project project execution plan about?" | success, well-cited | **Good** | Accurate table of PEP sections; cites the correct source PDF and chunk list |
| "What modifications were made to the intersection design in the MV Culvert Diversion?" | success, cited | **Good** | Matches ground-truth answer; cites the correct drawing |
| "What was the status of VO Ref: 31 and when was it closed?" | success, **wrong** | **Fail** | Ground truth exists in corpus (closed 2024-02-12), but chat says it cannot locate it |
| "What type of lighting fixture is specified for the 4.5M pole...?" | success, no citation | **Mixed** | Correct generic fact (28W LED) but no source cited; cannot verify which drawing |
| "Is APPROVED a valid status on design documents per PRC-951?" | success, **wrong** | **Fail** | Answers "Yes — APPROVED is valid", contradicting `construction_expert.txt`, `construction_knowledge.py`, and `high_density_facts.jsonl` |
| "What are the valid statuses for a design package under PRC-951?" | success, malformed | **Fail** | Returned raw `search_project_documents` tool-call JSON instead of an answer |
| "What is the contract rule about 'no approved on design'?" | success, long | **Mixed** | Grounded in contract templates but tangential; misses the simple critical rule |
| "What is the required commencement date for the maintenance works under the the client project PEP?" | success, empty | **Fail** | Answer string is empty after 12 iterations |

**Overall:** 2 good answers, 3 mixed, 3 clear failures in a small sample. Failures correlate with the retrieval failures above (VO Ref 31, PRC-951, empty PEP detail).

---

## 6. Training/evaluation data verdict

| dimension | verdict |
|---|---|
| **Coverage** | Strong — tens of thousands of examples across contract/drawings/schedule/BOQ/MEP/HSE/structural/spec |
| **Corpus grounding** | Strong — ~98-100 % of source doc_ids/chunks exist in the indexed corpus |
| **Construction usefulness** | Good — questions are realistic document lookups, BOQ extractions, drawing Q&A, and procedure checks |
| **Cleanliness** | Weak — 6 % duplicate instructions, multiple overlapping versions, no clear canonical eval set |
| **Consistency** | **Critical** — PRC-951 design-status labels are contradictory across files, which poisons both training and evaluation |

---

## 7. RAG quality verdict

| dimension | score | explanation |
|---|---|---|
| **Retrieval recall** | 🔴 Poor | 31 % doc recall@5, 20 % chunk recall@5 on in-distribution questions |
| **Retrieval precision** | 🟡 Mediocre | High scores but frequently the wrong document; generic boilerplate displaces specific answers |
| **Answer groundedness** | 🟡 Mediocre | Good for broad topics; fails on document-specific lookups and critical rules |
| **Source citation** | 🟡 Mediocre | Often present, but sometimes missing or citing the wrong source |
| **Construction usefulness** | 🟢 Good | When retrieval succeeds, answers are actionable (quantities, dates, drawing numbers, procedures) |
| **Reliability** | 🔴 Poor | Empty answers, tool-call JSON, and contradictory answers on safety/compliance questions |

**Overall RAG quality: ⚠️ Not production-trusted for precise construction document lookup.**

It is acceptable for exploratory, high-level questions ("What is the PEP about?", "Summarise the MV Culvert changes") but should not be relied upon for contractually sensitive queries such as VO status, design-approval rules, or exact drawing quantities without human verification.

---

## 8. Recommendations (prioritised)

1. **Upgrade the embedding model** — move from `minishlab/potion-base-8M` (256 dims) to `sentence-transformers/all-MiniLM-L6-v2` (384 dims) or a construction/domain fine-tuned model. This is the single biggest lever for recall.
2. **Verify and tune hybrid retrieval** — add runtime logging of the BM25 leg (hits, fusion scores) and A/B test `RAG_HYBRID_SEARCH` on/off against the in-distribution eval set.
3. **Add a recall@K gate** — promote `training_scenarios_drive_archive_v2.jsonl` (or similar) to a canonical eval set and fail CI when recall@5 drops below a threshold.
4. **Deduplicate and version the training corpus** — pick one canonical `training_scenarios.jsonl`, archive the shard/version sprawl, and remove duplicate instructions.
5. **Reconcile PRC-951 labels** — decide the authoritative source of truth (procedure DB + `construction_knowledge.py` is recommended) and rewrite the contradictory `high_density_facts.jsonl` entries.
6. **Improve chunking and OCR quality** — chunks >512 chars and mangled Unicode/OCR text reduce retrieval signal; review the indexer chunker and the OCR pipeline for construction drawings.
7. **Address empty/malformed chat outputs** — the raw tool-call JSON and empty-string answers need guardrails in the agent loop.
8. **Re-index the 0-chunk projects** — `bb00878f`, `df28d3c0`, `e483b574` have documents but no chunks; users querying those projects will get only general-knowledge results.

---

## 9. Audit artifacts (no code changes)

The following read-only artifacts were produced by this audit and are left in the repo for reference:

* `RAG_AUDIT.md` — this report
* `data/learning/rag_audit/sample_retrieval_recall_projects_folder.json` — 30-question recall sample vs `projects_folder`
* `data/learning/rag_audit/retrieval_recall_100.json` — 100-question recall summary
* `data/learning/rag_audit/chat_quality_sample_1.json` — chat answer sample (drawing + VO questions)
* `data/learning/rag_audit/chat_quality_sample_2.json` — chat answer sample (PRC-951 + PEP questions)

No source files were modified and no production data was mutated.
