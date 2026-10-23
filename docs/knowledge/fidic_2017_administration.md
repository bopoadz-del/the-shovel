# FIDIC 2017 Contract Administration — Practice Layer (claims, time, payment, disputes)

Structural / mechanism reference for the platform's knowledge base. Captures the
*practical administration* layer of the FIDIC 2017 Second Edition suite — document
hierarchy, notice/response deadlines, the claims and determination procedure, and
the DAAB — as a complement to the structural note (`fidic_contracts_red_white.md`)
and the drafting guard-rails (`fidic_golden_principles.md`). NOT a reproduction of
copyrighted FIDIC text; the official clause wording lives only in the published
FIDIC books.

Sources: FIDIC 2017 Conditions of Contract, Second Edition (mechanism-level
paraphrase); distilled from a practitioner guide based on the FIDIC Contracts
Management Course by Husni Madi (FCIArb, FIDIC Contracts Committee). Educational /
reference use — for legal matters consult the official FIDIC publications.

Cross-refs: `fidic_contracts_red_white.md` (clause structure), `fidic_golden_principles.md` (GP1-GP5).

---

## What changed 1999 → 2017 (why the practice differs)

- **DAAB replaces DAB** — a *standing* board appointed at commencement, tasked with
  dispute *avoidance* (site visits, informal opinions), not just adjudication.
- **Stricter, more detailed claims procedure** with hard **time-bars**.
- **Enhanced Engineer role** — explicit impartiality + a formal *determination*
  path (Sub-Clause 3.7).
- **Early-warning / collaborative** provisions; clarified **termination**;
  **Review / No-objection** replaces "approval"; Quality Management added.
- Express references to the **Employer's Requirements** grew 24 → 32.

---

## Document hierarchy (descending precedence)

1. Contract Agreement → 2. Letter of Acceptance → 3. Letter of Tender →
4. **Particular Conditions** → 5. **General Conditions** → 6. Employer's
Requirements → 7. Contractor's Proposal → 8. Schedules (BoQ, Programme, etc.).

Key practical point: **Particular Conditions override General Conditions** — so
amended PCs can silently re-allocate risk (this is exactly what the Golden
Principles police, see GP3).

---

## Employer's Requirements + the three-phase error check (Yellow/Silver)

The ER is the primary channel for the Employer's needs; must be project-specific,
concise, non-overlapping with the Conditions, and GP-compliant. For Contractor-
designed books the risk of ER errors is allocated in three phases:

| Phase | When | Who bears an error found here |
|---|---|---|
| 1 | Site/data inspection **before tender** | Contractor (own cost) |
| 2 | Scrutiny within the Contract-Data period **after Commencement** (SC 5.1) | Contractor (own cost) if an experienced contractor exercising due care should have found it |
| 3 | Discovered later, **not** reasonably discoverable earlier | Employer → Contractor gets **EOT + Cost** (SC 1.9) |

**Financial arrangements (SC 2.4):** Employer must give reasonable evidence of
funding on request; failure → Contractor may suspend / reduce rate after **21 days'
notice**. MDB version makes it a condition precedent to commencement.

---

## The Engineer (Clause 3) and determinations (SC 3.7)

Engineer acts **impartially/fairly**, issues instructions, variations, certificates,
and — when the parties cannot agree (EOT, cost, variation value) — makes a
**determination**: consult both parties, seek agreement, else determine on the
contract + evidence. Binding unless revised by DAAB/arbitration. ~40+ sub-clauses
route to a 3.7 determination (1.9, 2.5, 4.12, 8.4, 12.3, 13.3, 20.1, etc.).

---

## Review vs No-objection (SC 5.2.2)

- **Review** = Engineer examines a Contractor's Document for compliance.
- **No-objection** = may be used for the Works. **Silence within the Review Period
  = deemed No-objection** (prevents approval-by-delay).
- Construction of a part must **not** start until No-objection (actual or deemed).
- Review **never** relieves the Contractor of responsibility (why 2017 dropped
  "approval" — an approval could shift liability back to the Employer).

---

## Programme & time (Clause 8) — ties to the platform's schedule engine

| Item | Rule |
|---|---|
| Initial detailed Programme | Submit **within 28 days** of Commencement (SC 8.3) |
| Engineer review of initial Programme | **21 days** → silence = deemed No-objection (SC 1.1.66) |
| Engineer review of a revised Programme | **14 days** → silence = deemed accepted |
| EOT grounds (SC 8.4) | Variation/qty change; delay giving Cost entitlement; exceptionally adverse climate; Unforeseeable shortages (epidemic/govt); any Employer-attributable delay |
| EOT notice | **28 days** from awareness (SC 20.1) or risk **time-bar** |
| Delay Damages | Pre-agreed rate; **capped** in Contract Data → hitting the cap can trigger termination (SC 15.2) |

The **Programme is the baseline** for delay analysis — un-updated programme
undermines an EOT claim.

---

## Tests on Completion, Taking-Over, Defects (Clauses 9-11)

- **Fail Tests on Completion (SC 9.4):** Engineer may accept with reduced
  performance (→ Performance Damages / price reduction), reject, or re-test
  (no repeat limit).
- **Taking-Over Certificate (SC 10.1):** issued when *substantially* complete;
  Engineer has **28 days** or the TOC is **deemed issued**. Effects: care transfers
  to Employer; **DNP starts**; **50% retention released**; sets Date of Completion.
- **Parts/early use (10.2/10.3):** using a part = deemed taken over; outstanding
  tests can run into the DNP (see WWTP case below).
- **Defects Notification Period:** typically 12 months from TOC; extendable to a
  max **2 years** if the Works can't be used (SC 11.3). Contractor remedies own-
  risk defects at own cost; Variation/Employer-risk defects → Cost.
- **Performance Certificate (SC 11.9):** the *only* acceptance of the Works;
  issued **28 days** after last DNP expiry → returns Performance Security + releases
  2nd 50% retention. Watch mandatory civil-law warranties that outlive the DNP.

---

## Measurement, valuation & payment (Clauses 12 & 14)

- **Measure net actual quantity** of permanent works (SC 12.3); method per ER/
  Schedules. **BoQ-vs-General-Conditions clashes** are a classic dispute — a BoQ
  lump-sum/provisional item against the GC's re-measurement rule; the pricing model
  must be explicit in the BoQ or remeasurement disputes follow. *(Directly relevant
  to the platform's BOQ handling.)*
- **Interim payment:** monthly Statement → IPC **within 28 days** → Employer pays
  **within 56 days**. Late/withheld IPC → Contractor may suspend (SC 16.1).
- **Final:** Final Statement + Written Discharge → Final Payment Certificate
  **within 28 days** → generally extinguishes further claims (except disputes
  already referred).

---

## Securities (Clause 4 / 14)

| Instrument | Rule |
|---|---|
| **Performance Security** | On-demand guarantee, **5-10%** of price, within **28 days** of Contract Agreement; returned **28 days** after Performance Certificate. On-demand = callable without proving breach (abusive-call risk). |
| **Retention** | ~5% withheld per IPC; **50% at TOC**, **50% at Performance Certificate**. |
| **Advance Payment** | Paid within 42 days of Agreement (or 21 days of the Advance Payment Guarantee); recovered ~25% per IPC. |

---

## Suspension, termination, force majeure (Clauses 15, 16, 19)

- **Employer termination (SC 15.2):** failure to correct/perform, abandonment,
  no-consent whole-Works subcontract, insolvency, bribery. **2017 added:** ignoring
  a binding DAAB decision; exceeding the delay-damages cap; JV-member default.
  Prefer a **Notice to Correct (15.1)** first — termination is the nuclear option.
- **Contractor suspension/termination (16.1/16.2):** non-payment, no financial
  evidence, prolonged suspension. **2017 added:** Employer ignoring a DAAB decision,
  no Commencement notice, Employer corruption (immediate). Otherwise **14 days'
  notice**; suspend on **21 days' notice**.
- **Force Majeure (Clause 19):** exceptional, beyond control, unavoidable, not
  attributable. Notice **within 14 days** (SC 19.2), mitigate, detailed claim
  within 28 days. If it continues **> 84 days**, either party may terminate on
  **28 days' notice**.

---

## Claims procedure (Clause 20) — the hard time-bars

| Step | Deadline |
|---|---|
| **Notice of Claim** | **28 days** from awareness — miss it and the claim is **time-barred** |
| Fully detailed claim | **84 days** from the event (interim detailed claims every 28 days if the event is continuing) |
| Engineer initial response | 14 days |
| Engineer agreement/determination | proceed under SC 3.7; determine within **84 days** of the claim |

**Employer's claims (SC 2.5)** follow the same discipline — notice + particulars +
3.7 determination; the Employer **cannot** simply deduct from IPCs.

---

## DAAB (Clause 21) — standing dispute board

- **Standing** board (1 or 3 members), appointed at commencement; **avoids** as well
  as adjudicates. Members must be independent, impartial, conflict-free, not
  previously engaged by either party, no ex-parte contact.
- **A DAAB decision is a condition precedent to arbitration** (this is Golden
  Principle GP5) and is **binding unless/until** revised by arbitration.
- **84-day procedural calendar** (illustrative, SC 13.7 price-adjustment dispute):
  Day 0 case-management call · Day 1 Referral · Day 21 Defence · Day 31 Reply ·
  Day 41 Rejoinder · Day 55 hearing · Day 66 further submissions · **Day 84
  Decision**.
- Final resort: **arbitration** (Clause 21), typically ICC or agreed rules.

---

## Key deadline cheat-sheet

| Event | Deadline | Owner |
|---|---|---|
| Deliver Performance Security | ≤ 28 days after Contract Agreement | Contractor |
| Initial Programme | ≤ 28 days after Commencement | Contractor |
| Engineer review — initial Programme | 21 days | Engineer |
| Engineer review — revised Programme | 14 days | Engineer |
| Notice of Claim | ≤ 28 days from awareness | Claiming party |
| Fully detailed claim | ≤ 84 days from event | Claiming party |
| Issue IPC | ≤ 28 days of Statement | Engineer |
| Pay certified amount | ≤ 56 days of IPC | Employer |
| Issue Taking-Over Certificate | ≤ 28 days of application (else deemed) | Engineer |
| Performance Certificate | ≤ 28 days after last DNP expiry | Engineer |
| Return Performance Security | ≤ 28 days after Performance Certificate | Employer |
| DAAB decision | ≤ 84 days from referral | DAAB |

---

## Worked scenarios (how the clauses interact)

- **WWTP / no inflow at Tests on Completion (Yellow Book):** Engineer's 4 options —
  do nothing (Contractor claims 20.1, may move to terminate 16.2), instruct wait
  (suspension → Cost/profit/EOT under 8.9, a Variation), issue TOC at end of Time
  for Completion (no Variation, no certainty), or issue TOC + tests in the DNP
  (10.3, contract-provided). Each shifts time/cost/risk differently.
- **COVID-19 Force Majeure (Apr 2020):** even a pandemic runs through the *contract*
  — 19.2 notice within 14 days, mitigate, 20.1 detailed claim; a Contractor can't
  claim "all disadvantages" without following procedure. "There is always an
  exception" — but found in the contract or governing law, not wishful thinking.
