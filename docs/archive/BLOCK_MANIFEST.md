# Block Manifest — kit recurrence guard (2026-07-12)

Prevents the T1 recurrence (44→17 collapse from an unset `CEREBRUM_DOMAIN_KITS`). This is the committed source of truth for the expected loaded-block set; a boot assertion fails loud on shortfall.

## Expected set: 40 blocks (with `CEREBRUM_DOMAIN_KITS=construction`, default VIRGIN)
```
async_processor bim bim_extractor boq_processor cache_manager chat code
construction construction_advisor cpm_engine document_engine
drawing_qto fasttrack_analyzer file_hasher formula_executor_v2 historical_benchmark
image learning_engine manpower_planner ocr orchestrator pdf primavera_parser
project_dashboard project_reasoner recommendation_template schedule_excel_writer
schedule_generator scope_extractor search smart_orchestrator spec_analyzer
sympy_reasoning translate validation_pipeline vector_search voice web zvec
```
- **17 base** (no kit) + **23 construction-kit** blocks = 40.
- Historical 44 → 40: **4 deletions** since the old set (e.g. `historical_benchmark`-style consolidations, `formula_executor`→`formula_executor_v2`, `ocr`/`pdf` v2 folded). List the 4 explicitly once the pre-rebuild 44-list is recovered from git history of `app/blocks/__init__.py`.

## Guard wiring (2 parts)
**1. render.yaml — pin the kit** (IaC consistency; env is already set on prod via API):
```yaml
      - key: CEREBRUM_DOMAIN_KITS
        value: construction
```
**2. Boot assertion in `app/main.py` lifespan (fail-loud on shortfall):**
```python
    # Kit recurrence guard: fail loud if the loaded block set fell short of the
    # committed manifest (the T1 44->17 collapse: an unset CEREBRUM_DOMAIN_KITS
    # silently dropped the construction kit). Absent blocks != failing blocks.
    _EXPECTED_MIN_BLOCKS = 40
    from app.blocks import BLOCK_REGISTRY as _BR
    if os.getenv("ENV", "").strip().lower() in ("prod", "production") and len(_BR) < _EXPECTED_MIN_BLOCKS:
        raise RuntimeError(
            f"Block shortfall: {len(_BR)} loaded, expected >= {_EXPECTED_MIN_BLOCKS}. "
            f"Construction kit likely inactive (check CEREBRUM_DOMAIN_KITS=construction). "
            f"Loaded: {sorted(_BR)}"
        )
```

## Status
- **Manifest: committed (this file).**
- **render.yaml pin + boot assertion: PARKED** — `app/main.py` is under concurrent edit by a parallel session (PR #191 conflict; a `__pycache__`-clearing block was added by another actor). Applying the assertion now risks a merge war. Apply in the next window when `main.py` is not contended (code above is ready to paste). Prod is already protected functionally: `CEREBRUM_DOMAIN_KITS=construction` is set on the service and verified (40 blocks live).
