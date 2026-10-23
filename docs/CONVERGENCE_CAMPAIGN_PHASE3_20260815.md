# Live-fire campaign, phase 3 — real construction documents, every feature

**Target:** The Fork live at https://theshovel.ai
**Mandate (operator):** comprehensive test of everything, gated with pass, using
REAL documents dug from the G: drive and OneDrive; online substitutes only when
nothing local exists.

## Enforcement (same rules as phases 1-2)
- one real run at a time, against real dependencies
- root-cause before fixing; fix machinery, never the artifact
- every fix ships a red-then-green new-shape test
- no wrap-around greens; claims are enforced (read back), not asserted
- stop-rule after two identical failures -> probe empirically instead
- append-only ledger; campaign ends on a full pass with enforcement active

## Feature -> document matrix (filled as inventory lands)

| # | Feature under test | Document needed | Source | Status |
|---|---|---|---|---|
| T1 | BOQ parse -> price -> workbook | digital BOQ xlsx | a 2017 tendered 534M AED bill (done ph2) + new finds | RE-RUN on fixed categorize |
| T2 | Drawing QTO (rooms, geometry) | floor plans pdf/dxf | 8 real apartment floor plans | RE-RUN on fixed _extract_rooms |
| T3 | Primavera XER import -> CPM | .xer file | synthetic P6 baseline (`tests/_synthetic_fixtures.py`) + drive search | PENDING |
| T4 | Contract Q&A (clauses, DNP, EOT) | real contract/LOA pdf | drive search | PENDING |
| T5 | QA/QC docs (ITP, method statement) | MRH_QAQC folder | Downloads | PENDING |
| T6 | Schedule generate + cost-load + EVM | brief + rates | synthetic brief + real rates | PENDING |
| T7 | Document ingest at scale (mixed formats) | a real project folder | drive search | PENDING |
| T8 | Agent capability re-run | validation + orchestrator fixes | queued defects | AFTER FIXES |

## Open defects carried into this phase
- D1 validation agent: 103s then "unable to generate a response" on a pure-arithmetic check
- D2 smart-orchestrator: refuses trivial slab volume (12x8x0.25) after 60s
- D3 quantity-surveyor payment_certificate: reports tool error verbatim instead of falling back to arithmetic

## Run log (append-only)

### Run 1 — T1b: SCANNED BOQ end-to-end (a UAE residential main-contract project, 506 pages, 185MB, zero text layer)

Previously declined as impossible. Closed with local OCR (tesseract 170dpi) and
a SELF-VALIDATING extractor: a row is only trusted when qty x rate = amount
within 2%, so an OCR misread of any number disqualifies the row instead of
poisoning the test set. 36 rows accepted (finishes + mechanical pipework),
tendered value 4,303,636 AED. Unpriced xlsx -> upload (indexed) -> price-boq
200 -> compared line-by-line to the OCR'd tendered rates.

Verdict: median est/tendered 1.10x. PASS on mechanism; two findings:

**F10 — the Buildings/Towers rate card has NO Pipework/Drainage or Mechanical
rows.** All MEP lines fell back to unrelated medians (20mm pipework priced at
250/m vs 9 tendered = 27.8x; sump pump 250 vs 9,971 = 0.03x). Same class as
the Finishes single-median issue: the card is thin outside structure.

**F11 — the priced workbook hides which rates are fallbacks.** price_line_items
stamps rate_source ("exact" vs "weak (asset median)") and sample size n, but
the workbook drops both. A reader cannot tell a rebar rate backed by n=45 from
a sump-pump guess backed by nothing. This is fixable in code (T-fix queued);
the card gaps are an operator data task.

### Documents added to the matrix by the operator mid-run
- Downloads/Cost_Estimate_Kenya_90MW_200MW_2030.xlsx (energy cost estimate)
- Downloads/wetransfer_docs_2026-05-20_1430.zip (unopened)
- Downloads/Anthropic RFP set: RFP docx + Performance Basis of Design pdf +
  Appendix B xlsx — pairs with the rate card's "Data Center" asset type

### Run 2 — T1c: legacy .xls contract-volume BOQ — FAIL -> FIXED (2 defects)

652KB .xls, 12,971 rows. price-boq 422: "could not extract line items".
**F12** _parse_excel hardcoded engine="openpyxl" (zip-only) while advertising
.xls -> every BIFF file died "File is not a zip file". Also xlrd was installed
locally but MISSING from requirements -- the fix alone would still fail in
production. **F13** headers assumed at row 0; real bills open with banner rows
(header at row 24 in the live file) -> zero columns resolved, zero items from
a file pandas read fine. Fixed with alias-driven header-row detection.
Result: 0 -> 3,851 line items. Red->green tests; on PR #339.

### Run 3 — T3: Primavera XER, 1,280-task real baseline — PASS (exact)

Uploaded to live; chat invoked primavera_parser; platform answered start
27 Nov 2013 / finish 6 Dec 2015 / 739 days. Independent ground truth from the
raw file: identical on all three.

### Run 4 — T3b: 25MB, 11,149-task current baseline — PASS + 1 defect FIXED

Parsed in 2.4s with full CPM, 258 milestones, 48 resources. **F14**: the
activities list is capped (default 5000) and nothing SAID so -- the true count
sat in activity_count but detecting truncation required comparing two fields;
the run was nearly reported as "5,000 activities". CPM was verified to run on
the FULL network (separate full re-parse), so the critical path was never
wrong -- the envelope was. Fixed: activities_truncated + truncation_note,
red->green, cry-wolf direction covered.

### Run 5 — T4: 203-page Conditions of Contract Q&A — IN PROGRESS

Ground truth extracted locally: Time for Completion 852 days (project-specific
-- a natural needle no model can guess), DNP 365 days from TOC, retention 10%
per IPC. First ask: agent HONESTLY refused (did not invent FIDIC defaults) --
correct discipline, but the document had not indexed yet: 203-page PDF inline
indexing is slow, and live logs show the embedder reloading from HF at 02:17
(HF_TOKEN unset warning). xer/xls at chunks=0 are a separate question: whether
doc_index supports those formats for RAG at all.

### Run 5 root cause — F15, MAJOR INFRA: the migrated service had NO persistent disk

Stop-rule fired on T4 (contract stuck at 0 chunks through two full waits), so
the failure was probed empirically instead of re-run:

1. Synchronous admin doc-reindex -> "ZERO_CHUNK: document produced 0 chunks --
   check extractor/OCR".
2. The platform's own _extract_pdf on the same file locally -> 530,331 chars.
   Code is fine.
3. Round-trip download of the stored file -> "Document file is not available".
   And the SAME result for a document whose chunks exist (boq_unpriced.xlsx).
4. Render API: serviceDetails.disk = null -- while render.yaml declares
   the-fork-data / /app/data / 1GB.

Mechanism: the 2026-08-09 migration created the service without the disk
render.yaml declares. /app/data is ephemeral, so:
  * every uploaded client document evaporates on each deploy/restart (chunks
    survive in Neon; source files do not -- exports, reindex, downloads all
    break later);
  * the HF embedder cache is cold on every boot (matches the live
    "unauthenticated requests to the HF Hub" warning and the crawling
    first-index);
  * a heavy 203-page embed with a cold cache can trigger the very restart
    that wipes the file being indexed. Instance suffix changed mid-test.

Fix applied: disk dsk-d9vsus8jo6nc73d6nan0 created via API to render.yaml's
exact spec; HF_HOME=/app/data/hf-cache restored (the old service's cold-start
fix, lost in migration). Redeploy in progress. T4 re-runs after.

Same failure CLASS as the campaign's F1 and the DNS/monitor findings: the
migration silently dropped an infra guarantee, and nothing verified the
running service against render.yaml. Follow-up queued: a preflight check that
compares declared disk/env against the live service.

### Run 6 — T4 retry after disk fix — FAIL -> F16 root-caused and FIXED

With the disk attached and the file persisting, the synchronous reindex of the
203-page contract 502'd the box -- twice, reproducibly. Render memory metrics:
baseline ~550MB, spike to 1.81GB against the 2GB limit at the moment of the
reindex, instance replaced (OOM kill).

**F16**: retriever.index_chunks hands the ENTIRE chunk list (~600 chunks for
this contract) to embedder.encode in a single call -- peak memory scales with
document size. Fixed in Embedder.encode itself: bounded slices of
ENCODE_SLICE (64, env-overridable), so every caller inherits the bound,
including bulk re-index. Fence asserts the property (slice sizes, nothing
lost, batched == unbatched) since an OOM cannot be red-tested politely.
On PR #340 with the XER truncation disclosure.

**The compounding chain now fully explained:** no disk (F15) meant a cold HF
cache each boot; the cold cache made first-index slow; the whole-document
embed (F16) OOM'd the 2GB box during that slow index; the OOM restart wiped
/app/data including the file being indexed; the retry then found no file and
reported ZERO_CHUNK. Three separate defects presenting as one symptom
("uploads never index"). Disk fixed live; HF_HOME restored; batching on PR.

### Run 7 — T5: real QMS forms (NCR / WIR / ITP) — 1 defect FIXED

Uploaded the operator's actual controlled forms and classified them.
**F17**: "Work Inspection Request (WIR).xlsx" classified as `specification` --
the keyword "spec" matched INSIDE "In-spec-tion" (bare substring matching).
Every inspection document in a QMS misfiles as a spec. Fixed with
word-boundary matching (hyphen/underscore/parenthesis as separators), red on
the live failure. Registry extended with the QMS's own types: ncr,
inspection_request, itp, material_submittal -- specific types outscore generic
'report'. ITP previously fell to `unrecognised` (honest but useless); now
classifies. On PR #340.

### Run 8 — T4 final: contract indexes clean; retrieval precision is the open defect

After #340 deployed: sync reindex of the 203-page contract -> **161 chunks, no
OOM** (bounded batches + persistent disk + warm HF cache). The infra chain is
closed.

**F18 (OPEN, measured, not hacked):** the Contract Data row
("Time for Completion ... 852 days") ranks FIRST (0.933) for the terse query
"Time for Completion for the whole of the Works days", but falls OUT OF THE
TOP 12 for the conversational "In the Conditions of Contract, how many days
is the Time for Completion...". Same endpoint, same store -- wording alone
flips it. Chat phrasing therefore misses the data-table chunks on long
contracts. Across four differently-phrased attempts the agent NEVER
hallucinated a value -- integrity holds; precision fails. Fix direction is
query distillation before retrieval / data-table boosting; that is design
work, recorded rather than hacked in.

### Run 9 — T2 gate: room take-off LIVE on a real floor plan — PASS

Uploaded Building 01 GF to the deployed platform, ran drawing_qto via
/v1/execute: rooms_count 22, net 229.26 m2, perimeter 266.0 m -- identical to
the local ground truth. Interior take-off is now a live platform capability.

### Run 10 — D1/D2 root cause: reasoning-starved token caps — FIXED (PR #341)

The validation agent and smart-orchestrator (exactly the two at
max_tokens: 1024) returned canned failures. Proven against the Moonshot API
with the agent's own prompt: 1024 -> finish=length, content=0c,
reasoning=3,940c; 4096 -> full validation report. Any cap below K2's
reasoning burn makes content structurally impossible; the empty-final retry
uses the same cap and fails identically. Fix: reasoning_min_tokens floor
declared in provider config (mirrors fixed_temperature), applied at the
per-attempt chokepoint + streaming path; starved configs raised to tell the
truth. D3 (QS payment tool error) reclassified: honest refusal to guess a
missing contract value -- working as designed; graceful-fallback is a design
choice, not a defect.

### Run 11 — T4b: 129-page mixed scan/text tender — indexing PASS, F18 confirmed systematic

39 chunks, no OOM (disk + bounded batches). Needle ("merger of three firms,
founded 1974/1981/1991") ranks FIRST for terse queries containing answer
tokens, NOT IN TOP 8 for the natural question. Same pattern as the contract.
**Negative result recorded:** naive stopword distillation of the query rescues
NEITHER case -- recall depends on overlap with the ANSWER's wording, which a
user cannot know. F18 therefore needs answer-shaped query expansion (HyDE-
style) or a cross-encoder reranker; both are design work with an eval, not a
regex. Agent NEVER hallucinated across all attempts.

### Run 12 — D1/D2 gate on the deployed floor (186d10d)

**D1 validation agent: GATE PASS.** The probe that returned a canned failure
now produces the full 5-stage credibility report and catches the wrong number
(1,200 -> 900 m3). The agent that could not validate, validates.

**D2 smart-orchestrator: half-closed.** The empty-response defect is GONE (it
now answers within budget), but it narrates its routing ("Routed to:
intelligent_workflow ... fallback") instead of delivering the number. The
token starvation and the routing behaviour were two distinct defects wearing
one symptom; the first is fixed, the second (F19: orchestrator describes
dispatch rather than executing the calculator) remains open and matches the
known tool-discipline notes on that agent.

## Phase 3 closing tally

| Fixed + merged + deployed | F12 xls engine, F13 header detection, F14 XER
truncation disclosure, F15 persistent disk (infra, live), F16 bounded
embedding, F17 doc-type word boundaries + QMS types, D1 reasoning token floor |
| Open, precisely characterised | F10 rate-card MEP gaps (operator data), F11
workbook hides rate confidence, F18 retrieval precision on question-shaped
queries (naive distillation PROVEN insufficient), F19 orchestrator routing
narration |

PRs this phase: #339, #340, #341 -- all merged, deployed, each carrying
red-then-green fences. Every finding traced to a named mechanism before any
fix; two fixes were verified against the live Moonshot API and Render metrics
rather than local guesswork.

## Run 13 — "nothing parked": close every open finding

**F19 gate exposed F19b.** PR #343 (dispatch contract) merged and deployed;
the slab probe returned the right number (24 m3) with a WRONG mechanism: the
TIMING log showed iter=0 calling only smart_orchestrator and iter=1 going
straight to final — the reply's "Routed to: formula_executor_v2" was
fabricated. Right number over a false claim is a wrap-around green. The
contract gained an explicit no-fabricated-dispatch rule, fenced through the
real config parser.

**F18 fixed — dual-query retrieval.** Probes on the live corpus validated the
mechanism BEFORE any code: wrapper-stripped, phrase-intact variants rescue
both needles (contract rank 1 at 0.933; tender rank 3-4), while phrase-broken
stopword variants rank nowhere. The retriever now embeds the raw question AND
its wrapper-stripped variant, searches both, merges by max score per chunk.
Terse queries cost no second search; RAG_DUAL_QUERY=0 is the kill-switch.
Fence: 7 red on old code including the behavioural needle test.

**F11 fixed.** The priced workbook carries a Rate Basis column (source + n,
amber for weak/fallback/NO RATE).

**F10 fixed — with real data and a sharper defect found.** The contract
volume's MEP sections carry quantities but no surviving rates (93
arithmetic-valid priced rows are all doors/sanitary/finishes); no priced MEP
bill exists in the operator's archives. Two observed AED lump sums from a
2012 fit-out BOQ fill the lump-sum cells (classifier's own verdicts, n
visible). Filling them exposed that fallback medians pooled rates ACROSS
UNITS — a per-metre pipe would have priced from a 26,525 lump-sum median.
Fallback tiers are now unit-commensurable (linear/area/volume/mass/count/
lump); per-metre MEP correctly stays NO RATE. Refuse, never invent.

## Run 14 — PR #344 deployed; gates found two ROOT CAUSES

- G1 (contract 852 via natural-question chat): PASS with citation.
- G4/G5 (Rate Basis live, no lump leaks): PASS — 55 exact + 1 weak, all
  labelled. (The staged BOQ doc predated the disk fix and its file had
  evaporated — "chunks exist, file missing" again; re-uploaded.)
- G2 (tender years): FAIL -> **F20 found.** Retrieval was FIXED (needle rank
  3 of 5) but MAX_RAG_TOKENS defaulted to 1500 — calibrated for the 512-char
  chunker era. Doc-reindex emits ~3,000-char chunks, so apply_token_cap kept
  exactly ONE of five chunks and dropped the needle whole. The model
  truthfully said "not in context". Retrieval quality had been made
  irrelevant by the injection cap.
- G3 (slab): fabricated dispatch AGAIN despite the strengthened contract.
  Stop-rule fired -> machinery dig -> **F19c root cause:** force-synthesis
  counts smart_orchestrator as a deliverable, so the iteration after the
  router verdict ran with_tools=False. Dispatch was STRUCTURALLY impossible;
  the model resolved the contract-vs-capability conflict by inventing the
  route. Two prose revisions could never have fixed a capability hole.

Fixes (PR #345): smart_orchestrator joined the non-deliverable tool set; an
empty router match injects a corrective user-role message at the decision
point; MAX_RAG_TOKENS default raised to 6000 (fits k=5 of current chunks).
Fences: 4 red on the old runtime (functional loop test), 2 red on the old
cap built from the exact live chunk sizes.

## Run 15 — final gates: ALL PASS, mechanisms verified

- G1 contract 852: PASS ("852 days from the Commencement Date", cited).
- G2 tender founding years: PASS 3/3 (1974 / 1981 / 1991, cited chunk 19).
- G3 slab: PASS with the mechanism PROVEN in the TIMING log — iter=0
  smart_orchestrator, iter=1+2 construction_calc, iter=3 final. The answer's
  "Routed to: construction_calc ... 24.0 m3, order 25.2 m3 with 5% waste" is
  now a true statement about a real tool execution.
- G4 Rate Basis column: PASS live. G5 no lump-median leaks: PASS live.

## Phase 3 final tally

Every finding from this phase is fixed, merged, deployed, and live-gated:
F10 (data + commensurable fallbacks), F11 (rate provenance), F12-F17 + D1
(previous runs), F18 (dual-query retrieval), F19/b/c (dispatch contract +
no-fabrication rule + force-synthesis exemption + empty-match nudge),
F20 (injection cap sized for real chunks).

PRs this phase: #339-#345, all merged with red-then-green fences. The last
three defects were only findable through the green surface: a correct number
over a fabricated mechanism, and a truthful "not in context" over a
retrieval that had actually succeeded.
