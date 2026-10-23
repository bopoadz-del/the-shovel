# Construction Container Refactor — delegate to blocks, kill demo data

**Goal:** `ConstructionContainer` (`app/containers/construction.py`, ~5,885 lines)
must delegate to the standalone construction blocks instead of reimplementing
them inline — the "lego" principle. Delete confirmed duplicates. Remove every
demo-mode / fabricated-data path: bad/empty input must error, never invent data.

**Delegation mechanism:** inside the container, call
`block = self.get_dep("<block_name>")` → `await block.process(input_data, params)`.
`ConstructionContainer.requires` already lists all 8 blocks, so `get_dep` returns
live wired instances. Guard for `None`; fall back to `BLOCK_REGISTRY.get(name)()`
only if `get_dep` is None (e.g. unit tests bypassing the assembler). Use
`.process()` not `.execute()` (avoids the `{block,result,...}` wrapper).

**Source of truth:** the duplication audit (in-session). Honest-uncertainty
flags from it are reproduced per task.

**Sequencing:** T2–T8 all edit `construction.py` → STRICTLY SEQUENTIAL.
T1 edits only `boq_processor.py` → independent. T9 is the test triage, last.

---

## T1 — Remove demo data from `boq_processor`
`app/blocks/boq_processor.py`: delete `_demo_boq` (lines ~79–115); change line ~58
so a missing/nonexistent `file_path` (or inline-only input) returns
`{"status": "error", "error": "..."}` — matching the other 7 blocks. No
`demo_mode`, no invented line items.

## T2 — Dead-code removal in `construction.py` (do first; lowest risk)
Delete: the FIRST `route` method + `_status` (~102–124, shadowed by the real
`route` at ~5761); the duplicate stub method block (~456–519, shadowed by the
real implementations at ~2600–2821); the fabricating BIM internals
`_parse_ifc_geometries`, `_detect_model_clashes`, `_detect_internal_clashes`
(~2892–2924). Verify each deletion target is genuinely shadowed/unused first.

## T3 — BIM delegates to `bim_extractor`
Rewrite `bim_analysis` (~2437) and `bim_clash_detection` (~2823) to delegate to
`self.get_dep("bim_extractor")`. Remove their demo-mode branches (~2448, ~2839).
Remap block output to the response shape callers expect. Keep clash-shaping
helpers as a thin layer only if they add value over the block's `clash_report`.

## T4 — Cost benchmarking delegates to `historical_benchmark`
Replace `_lookup_unit_cost` (~1313) and `_get_rsmeans_data` (~1225) with
delegation to `historical_benchmark` (`action: "lookup"`/`"batch"`). KEEP
`generate_cost_estimate`'s overhead/profit/contingency markup aggregation —
container-only. Bridge: block returns `rates.adjusted_usd`; the aggregation
wants a float.

## T5 — Schedule: `_parse_xer_file` delegates to `primavera_parser`
Replace `_parse_xer_file` (~740) body with delegation to `primavera_parser`.
Add an adapter remapping the block's nested `activities`/`schedule_data` to the
FLAT shape `_calculate_cpm`/`_analyze_delays` need (`total_float` ←
`total_float_days`, etc.) — this adapter is real work, not a delete. KEEP
`_calculate_cpm`, `_analyze_delays`, `_parse_xml_schedule`, milestones, recovery.
Remove `resource_histogram`'s synthetic-data fallback (~3404–3411).

## T6 — Spec: delegate extraction to `spec_analyzer`
`process_specification_full` (~994): delegate grade/material/compliance
extraction to `spec_analyzer`; KEEP the CSI MasterFormat division-splitting
(container-only); map block output into the `spec_items` shape. Remove the
demo-mode branch (~1004–1028) → error on no input. PARTIAL duplicate — verify
the block's regex set covers the container's before deleting
`_extract_specs_advanced`/`_extract_materials`; keep what the block lacks.

## T7 — Purge remaining demo/fake data in `construction.py`
Make these error on missing input instead of fabricating:
`payment_certificate` (~1493), `safety_compliance_audit` (~1799),
`forensic_delay_analysis` synthetic path (~3984–4004), `extract_measurements`
mock quantities (~5618–5635).

## T8 — Wrapper cleanup
Update the 13 existing delegation wrappers (~5663–5757) to the `get_dep`-first
form with `None`-guards, for consistency.

## T9 — Un-skip & triage construction tests
Remove the blanket class-skip on `TestConstructionBlocks` in `tests/test_e2e.py`
(keep the 11 already-passing tests live). Triage the 4 drifted tests against the
now-refactored behaviour (no demo mode): fix block/container or rewrite the test,
per-case. Same for `TestContainers::test_construction_container`. Full suite must
stay green.
