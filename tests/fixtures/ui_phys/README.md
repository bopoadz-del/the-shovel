# UI-PHYS A–H battery — the two sources, and which one may change

The battery has **two** sources of record and they are not the same file.
Confusing them is what produced a scored run whose questions had drifted
from the instrument, and it is the reason this note exists.

| | source of record | may it change? |
| --- | --- | --- |
| **Questions** | `UI-PHYS_QP2_results.xlsx`, column **"Question (ask exactly)"** (owner's Drive) | **No.** Never reworded, never paraphrased, never "tidied". |
| **Expectations** | a **dated revision log** under `FLEET_OPS/artifacts/`, currently `GROUND_TRUTH_REVISIONS_2026-09-01.md` | Yes — only by appending a new dated entry. |

## Questions are frozen

A question is the measurement. Changing its wording changes what is being
measured, and the change is invisible in a pass/fail column.

This is not a style preference; it was measured. **F-PHRASE-1** in
`gate_battery_13b2bf7_2026-08-31.md` found the routing to be
phrasing-sensitive, and that **the sheet's phrasing is the one that fails**:

* E4 — `"Concrete volume for a raft 30x20x1.5 m including your documented
  waste factor."` → refusal. A conversational paraphrase of the same
  question fired `construction_calc` and returned 945 m³ an hour earlier.
* C2 — `"Which specification document covers the Variation Procedure, and
  what is its number?"` → wrong document. The paraphrase returned the right
  one, an hour earlier.

> **A prior "PASS" obtained on a paraphrase is not evidence about the
> battery question.**

`test_ui_phys_fixture_catalog.py` pins a digest over every `(id, ask)` pair.
If it fails, a question string moved. The fix is never to update the digest
to match — it is to restore the wording. A genuinely new question gets a
**new id**; an existing id's wording is never edited.

## Expectations change only through the log

Ground truth is allowed to be wrong, and it has been: F1 asked for a WBS
"derived from this project's BOQ" from a capability that is a deterministic
template scheduler with no BOQ input at all. Silently rescoring that would
have hidden both the gap and the correction.

So an expectation changes **only** by appending to the dated revision log,
which records, per revision:

1. the case id and the **old** expectation, preserved verbatim;
2. the **new** expectation;
3. why it changed — a capability that does not exist, a spec gap, a
   mis-transcription — and what was built or deferred as a result;
4. the date and the SHA the decision was taken against.

The spreadsheet is **not** edited in place. Its `Ground Truth / Expected`
column stays as originally written; the log is what the scorer reads. A
reader can therefore always reconstruct what was expected on any given run,
which an in-place edit destroys.

## This directory

`questions.json` plus `S1`–`S6` are the **sanitized** stand-in: the same case
ids and the same `ask` strings as the confidential Drive pack, with
fixture-only figures, names and document titles. Live client figures must
never land in git — that is a hard stop, and
`test_ui_phys_fixture_catalog.py` is the CI gate for it.

Expected values here are the FIXTURE's own (S1 says Delay Damages are 0.2%
and Time for Completion is 420 days; the live pack says otherwise). The
revision log governs the live pack's expectations, not these.
