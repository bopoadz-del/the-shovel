# Formula-drop integration — status + Slice 4 reconciliation decision

Integrating the operator's SPEC-C900 formula/engine upgrade
(`~/Downloads/Kimi_Agent_Fork Repo Upgrade Plan/`). Recorded 2026-07-14.

## Slices

| # | What | Status |
|---|---|---|
| 1 | `app/lib/construction_formulas.py` — 40+ deterministic engineering calculators, verified (hand-computed + smoke) | DONE (#227) |
| 2 | `construction_calc` agent tool — the model runs the calculators; rates from RAG, errors honestly | DONE (#228) |
| 3 | `app/lib/construction_catalog.py` + generated GK note → catalog rates retrievable/grounded in RAG | DONE (#229) |
| 4 | Reconcile the divergent `construction_learning` / `cross_domain_reasoner` rewrites | DONE — decision below |

## Slice 4 decision: keep the wired repo versions, do NOT merge the drop's rewrites

The drop's `construction_learning.py` and `cross_domain_reasoner.py` share names with
live repo files but are **parallel reimplementations with different APIs**. Verdict after a
method-level comparison: **nothing critical to reconcile — the repo versions win.**

- **`construction_learning`** — both have the same four learning surfaces (duration,
  procurement lead-time, defects, templates). The repo version is the WIRED, superior base:
  `shared_instance()` singleton into the hydration scheduler / learned router / feedback
  endpoint, `min_samples` gating, P80 lead-times, state persistence; passes
  `comprehensive_engine_test` 93/93. The drop's version is a record-based reimplementation of
  the same core.

- **`cross_domain_reasoner`** — different PURPOSES. Repo = chat-turn enrichment
  (`TemplateMatcher` / `CrossDomainIntentDetector` / `MultiDomainPlanBuilder` /
  `SystemPromptInjector`), wired into runtime/orchestrator/action_router. Drop = an
  intent→calculator router (`detect_intent` → `ExecutionPlan`) that was scaffolding for the
  drop's standalone `engine.py`. That routing job is now done by the LLM + the `construction_calc`
  tool (Slice 2), so the drop's reasoner is **superseded, not graftable**.

Overwriting either repo file would clobber the wired 93/93 engine — the landmine flagged at the
start of this integration. So Slice 4 keeps the wired versions unchanged.

### Banked for later (only genuinely-new bits; skipped now — no consumer)

Three ANALYSIS methods exist only in the drop's `construction_learning.py` and are **not
called by anything**. Skipped now (shipping them = dead code); grafted only IF a future
feature needs them:

- `get_supplier_reliability()` — per-supplier delivery-reliability scoring
- `get_root_cause_analysis()` — groups defects by root cause
- `get_ranked_templates()` — ranks workflow templates by success rate

Source of the banked code: `~/Downloads/Kimi_Agent_Fork Repo Upgrade Plan/construction_learning.py`.

## Also skipped from the drop
- `construction_calc_tricks.py` — older duplicate of `construction_calculations.py` (Slice 1).
- `engine.py` / `engine_test.py` / the drop's `comprehensive_engine_test.py` — the standalone
  orchestrator; superseded by the wired runtime + `construction_calc` tool.

## Net
Formula-drop integration COMPLETE. The genuinely-new value (engineering calculators + KSA
catalog rates) is integrated, grounded, and tested. The wired reasoning/learning engine is
untouched. Follow-up (optional): wire the catalog `classify`/`lookup` as a classification tool
(must return classification/tier, not quote rates — rates come from RAG).
