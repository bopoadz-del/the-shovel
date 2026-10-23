# Agent packages by hat (F-DRIVER Phase A)

Each tool and each formula belongs to exactly one owner: `base` (every hat) or
one hat. A hat is a package `app/agents/hats/<hat>/` holding its `tools.py`
and its formula modules; a new hat is a new package and needs no core edit
(`tests/test_agent_packages.py`). Formula ownership is declared per formula
with `@formula(owner=...)` (`app/lib/formula_registry.py`).

## Tools

| Owner | Tools | Where |
|---|---|---|
| base | construction_calc, delegate_to_agent, fetch_document, list_project_documents, remember_fact, search_project_documents; block adapter: construction; block without adapter: sympy_reasoning | `app/agents/base/tools.py`, `app/agents/base/blocks.py` |
| commercial | cash_flow_forecast, evm_calculate, payment_certificate | `hats/commercial/tools.py` |
| contracts | rfi_generator | `hats/contracts/tools.py` |
| planning | generate_wbs, look_ahead, resource_histogram | `hats/planning/tools.py` |
| procurement | procurement_list_generator | `hats/procurement/tools.py` |
| qaqc | commissioning_checklist; block adapter: validation_pipeline | `hats/qaqc/tools.py`, `hats/qaqc/blocks.py` |
| design, quantities, safety | none of their own (formulas only) | — |

Nothing goes to a legacy package: `construction`, `validation_pipeline` and
`sympy_reasoning` are each listed in agent configs today (8, 3 and 8
configs), so each moved to its owner. Blocks without an adapter run through
the generic block dispatch unchanged.

## Formula modules (formulas per owner)

Single owner: the module moves whole into that owner's package.

| Module | Owner (count) |
|---|---|
| construction_formulas_beam_analysis | design 4 |
| construction_formulas_columns | design 1 |
| construction_formulas_loads | design 3 |
| construction_formulas_masonry | design 1 |
| construction_formulas_structural_rc | design 4 |
| construction_formulas_structural_steel | design 3 |
| construction_formulas_qc | qaqc 3 |
| construction_formulas_safety | safety 3 |
| construction_formulas_general | base 1 |

Mixed: split by owner, one file per owner.

| Module | Owners (count) |
|---|---|
| construction_formulas | design 16, qaqc 6, quantities 5, planning 3, procurement 1, base 1, safety 1 |
| construction_formulas_commercial | base 3, contracts 1, commercial 1 |
| construction_formulas_earthwork | base 2, qaqc 1, quantities 1, design 1 |
| construction_formulas_planning | planning 3, quantities 2, base 1 |
| construction_formulas_quantities | quantities 4, base 1 |
| construction_formulas_reference_tables | qaqc 2, design 1, commercial 1 |
| construction_formulas_additions | commercial 1, safety 1 |
| app.core.construction_knowledge | commercial 2, procurement 1, contracts 1 |
