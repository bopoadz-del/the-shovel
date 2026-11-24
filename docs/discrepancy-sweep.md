# Discrepancy sweep (STREAM A2, PR 2a) — design note

Owner 06:01Z (19 Nov UTC): "the product goes and looks for the other value."
When the drafted answer states a value for what was asked, the exit takes the
DISTINCTIVE IDENTIFIERS of that statement and runs ONE narrow lexical search
over the project for passages that carry the same identifier together with a
typed value of the same kind whose value DIFFERS. Every such passage is added
to the turn's evidence, and the existing conflict statement then does the
rest (every value, every origin with its standing, "the documents differ",
the user decides).

This PR carries the **pure detection half** only, so it can land independently
and be tested against recorded strings (the same discipline as the
clarifying-question exit). The caller-owned integration (the single
`bm25_search` call, the evidence merge, and the digest-only trace) is the next
commit; it needs a read-only replay on the real data, which is the acceptance
and is owner-gated.

## What the detector decides

`app.agents.discrepancy_sweep.sweep_differences(stated_text, candidates)`:

- reads the stated sentence's figures (`_figure_mentions` / `_figure_unit`)
  and its distinctive identifiers (`_DESIGNATION_RE` — a clause / code /
  standard / table designation such as "ACI 318", "Vol 6", "Table 7.3.1.1");
- for each candidate passage the caller retrieved, a hit is a passage that
  shares at least one identifier AND states a figure that differs from a
  stated figure in the same unit (`_competing_figure_sentences`, which already
  applies the product's own close-number / rounding tolerance, so a repeated
  figure or a rounding is never a difference).

Readers are reused from `citation_provenance` — the same typed readers the
conflict statement uses — not re-implemented. No case, document, standard,
value, item or row is named anywhere.

## Bounds (each one a test)

1. Only passages that share a distinctive identifier are recalled.
2. Only a *different* figure (same unit) is recalled; a repeated figure is
   not.
3. A rounding to the same figure is not recalled (`_close`).
4. A figure in another unit of the SAME kind is compared after conversion,
   with the conflict statement's own reader and equality
   (`evidence_conflicts.extract` / `same`), so recall and judgement agree on
   what one value is: "40 mm" against "0.05 m" is recalled, "50 mm" against
   "0.05 m" is the same value and is not. The close-number tolerance applies
   at the scale the draft wrote the value. A unit of another kind (a length
   against a mass) is never compared.
5. A bare number is never a figure (the shared reader already excludes it).
6. Only a figure stated FOR the shared identifier is compared
   (`_identifier_scope`): one in a sentence that names it, or in the body
   under a heading line led by it (no figure, not ending as a sentence) down
   to the next line led by a designation. A passage that names the
   identifier in one sentence and gives an unrelated figure in another is no
   difference.

## Known limits (stated, not hidden)

- A bare clause number ("4.2.1 Minimum cover ...", "see 4.2.1") is read
  by the clarifying-question exit's reference reader (`clarify_cases`),
  kept only in the clause shape (three or more dotted numbers; a decimal is
  never one; a part of four digits is a dotted date and is skipped). A bare
  number names the same thing as a labelled designation ending in it
  ("Clause 4.2.1"); two different labels on one number ("Table 4.2.1",
  "Clause 4.2.1") never do. Limit: a bare heading cannot say which
  structure it heads, so "Table 4.2.1" in a draft recalls a bare "4.2.1"
  heading too; the conflict statement still judges the recalled passage.
- Only measured values (a quantity in a unit, money in one currency) are
  compared across units; a date, a grade or a labelled phrase is not swept.
- A value in a passage that was never handed cannot be reported (the 05:55Z
  known limit); the sweep only widens recall within one narrow search, it is
  not a re-retrieval of the whole project.

## Acceptance (next commit, owner-gated)

A read-only replay on the real data, counts only: the conflict row — the sweep
finds the other value and the answer states both (5 of 5); the tester's last
two full sets — how many plain rows gain a "differ" line, each judged (a true
discrepancy is a gain, a false one a fail); added time per turn, median and
worst. The time budget and clean skip (answer goes out as-is, trace says
"skipped") are the caller's.
