# ORDER: NO HARDWIRING (standing rule) + The_Fork retriever surgery

Owner's order, 2026-10-02. Read every line. This order executes under one
rule. Violating it is not a mistake to be fixed later; it is a rejected
deliverable.

## 1. THE RULE (opens every order; applies to every repo, no exceptions)

> NO HARDWIRING. A failing case is never fixed by name. No per-case branches,
> no literal document strings, no test-case IDs, no rescue functions, no
> per-case switches. Find the general cause and fix the mechanism so every
> case of that kind passes. If you cannot find a general fix, say so and stop
> -- a pass patch is a failure, not progress.

## 2. THE GATE (what gives the rule teeth)

> A fix is accepted only if it raises the score on questions and corpora the
> builder has never seen. CI rejects any diff that adds `_rescue_*`,
> `*_NEEDLES`, `*_RESCUE` knobs, or a probe ID inside product code.

- The CI tripwire exists: `scripts/scan_hardwiring.py`, a blocking step in
  `.github/workflows/lint.yml`. It rejects any NEW `_rescue_*` function,
  `*_NEEDLES` list, `RAG_*_RESCUE` / `*_BONUS` / `*_EXTRA_K` knob, or probe ID
  under `app/`. Exit 1 is a rejection, not a warning.
- Pre-doctrine forms are grandfathered in `scripts/hardwiring_baseline.json`.
  **You never add to the baseline.** It may only shrink. If the gate fails
  your diff, your diff is the problem -- not the gate.
- The acceptance is NOT the lint and NOT the seen probe sets. It is the
  owner-SEALED unseen set, scored by the owner. You will never see those
  questions. Do not ask for them. Do not write them. A case you have read is
  no longer unseen.

## 3. THE FOUR TESTS: variable by design, or just looks that way?

1. **No named cases in production code.** If R18, LIVE_VETCARE_*, or any
   probe ID appears in the shipped path, it is not variable -- it is a
   fixed-point with a variable-shaped wrapper.
2. **It passes on inputs it has never seen. By construction, not by luck.**
   If the only cases it clears are ones the code was tuned on, variability is
   an illusion.
3. **Every fix is general, or it is rejected.** A patch that makes one case
   pass and cannot be justified as a class-level fix does not ship. Not
   "advisory" -- rejected. This is the test that kills the 170th rescue before
   it is written.
4. **Contracts, not answers.** A block declares what it accepts and what shape
   it produces. It never declares the specific values it expects. Shape is
   enforced; values flow through.

## 4. THE JOB: The_Fork retriever surgery

Context. `app/core/rag/retriever.py` (10,872 lines) holds 44 `_rescue_*`
functions, 7 `_NEEDLES` lists, 22 per-case knobs, and probe IDs photographed
into production (`E1` x64 in retriever and x31 in runtime; `A3` x28; an
older OLD-pack / F-BAT-D G1-G6 series; wave-1 A3/A5/A9). Each is one failed
probe turned into a special case. None is a retrieval improvement. Most are
the same defect -- "a specific table or row lost the k-cut" -- wearing
different names.

Method, in this order. Do not skip a step.

a. **FREEZE.** From now, no new per-case form. The gate enforces it.

b. **CLASSIFY** every one of the grandfathered forms into exactly one bucket,
   with the evidence, and report the table BEFORE writing any fix:
   - **GENERAL DEFECT** -- the mechanism is wrong for a whole class (chunking
     loses tables; k is too small for table-bearing queries; ranking ignores
     document structure; a row is split across chunks).
   - **CHECKER ARTIFACT** -- the probe's checker was wrong, not the product
     (N3 / T11 / T20 / A3 were exactly this).
   - **CORPUS GAP** -- the fact is genuinely absent; the correct answer is a
     refusal, and a refusal is not a defect.

c. **FIX THE MECHANISM, once per class.** One general change -- e.g.
   table-aware chunking plus structure-aware ranking -- that makes every case
   of that class pass, not 44 named ones. Contracts, not answers: the fix
   declares what SHAPE it handles, never which VALUES.

d. **DELETE** what the general fix makes redundant. Each deleted rescue,
   needle list, or knob shrinks the baseline
   (`python scripts/scan_hardwiring.py --write-baseline` after each batch).
   Never regenerate the baseline to admit a new form.

e. **MEASURE** on the owner-sealed unseen set. The owner scores; you report
   only what you were given. No regression on the seen sets (SET4 / SET5 /
   golden) is the floor, not the bar.

f. **STOP CONDITION.** If a bucket has no general fix, write "no general fix
   found for <class>: <why>" and stop. That is a valid, complete deliverable.
   A pass patch is not.

Deliverables: the classification table; the general mechanism (one diff per
class); the baseline shrink count (before -> after); the unseen-set score as
reported by the owner; the gate green.

## 5. THE DESIGN (the RAG layer work that follows; same rule applies)

- **Three layers.** L1 domain general knowledge / L2 company core + projects,
  ONE admin-only tier with projects isolated inside it by project / L3
  per-user personal (`user_<uid>`), private, weakest, never overrides L2.
- **Authority** (contractual > design > commercial > operational > policy >
  historical) is the intrinsic ordering of L2 documents and stays there. It
  is not a cross-cutting scale. `personal` is simply L3.
- **Company docs** (policies, procedures, codes of conduct, HR) are L2
  company-wide. There is nothing to design.
- **Sandbox.** Any file type in; it is parsed; **only the `.txt` persists, for
  the session only; the binary is ejected.** It WORKS in native format during
  the session (BOQ structure, formulas intact); it PERSISTS as text.
- **Save.** The user says "save it": the session `.txt` goes to **L3**, never
  L2. **L2 writes are admin-only.**
- Everything behind `RAG_LAYERED` (default OFF) until the owner flips it.

## 6. WHAT GETS YOUR DELIVERABLE REJECTED

- A `_rescue_*`, needle list, per-case knob, or probe ID anywhere in `app/`.
- Adding an entry to `scripts/hardwiring_baseline.json`.
- A fix justified by a seen case ("R18 now passes") instead of by a class.
- Asking for, guessing at, or writing the unseen set.
- Declaring READY or green from seen-probe scores. Only the owner declares,
  from the unseen set.
- Treating a per-case patch as "advisory". It is rejected, not noted.
- Reporting a round as done while a classification bucket is unfilled.
