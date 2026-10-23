# The Fork — Platform Audit & Recommendations

**Audited:** 2026-06-10  
**Scope:** Construction container kit, chat/domain coupling, store publish path  
**Runtime role:** Live production — do not wholesale-migrate into Cerebrum-Blocks  
**Store role:** Kits published to [Cerebrum-Blocks](https://github.com/bopoadz-del/Cerebrum-Blocks) for discovery/install

---

## Executive summary

The Fork is the battle-tested runtime: real construction data, Masterise pitch, fine-tuned adapter pipeline, and a **7,330-line** `ConstructionContainer` that routes **47 actions** across **15 required blocks**. Domain knowledge is correctly split into `construction_knowledge.py`, prompts, and procedure DB — but was leaking into generic `ChatBlock` via a default prompt (fixed in `fa76e0a`).

**Cerebrum-Blocks is the store layer**, not a replacement runtime. Publish from Fork → install on consumer instances.

---

## Recent change (committed)

| Commit | File | Change |
|--------|------|--------|
| `fa76e0a` | `app/blocks/chat.py` | Removed 6-line auto-inject of `construction_expert.txt` when no caller prompt |

**Merge rule:** Generic `ChatBlock` must not pick a domain prompt. `ConstructionContainer.chat()` already sets `construction_evm.md` — that is the correct injection point.

**Not changed:** RAG, local LLM (Ollama / llama.cpp), fine-tuned adapter path, TypedBlock schemas, `_resolve_system_prompt()`, streaming, cloud fallback chain.

---

## Architecture findings

### 1. `app/containers/construction.py` (v3.1)

| Metric | Value |
|--------|-------|
| Lines | ~7,330 |
| Public actions | 47 (`route()` / `get_actions()`) |
| Methods | ~262 |
| Top-level app imports | `UniversalContainer`, `construction_types` only |
| `construction_knowledge` imports | **0** — knowledge used elsewhere, not in monolith |

**Block delegation** (via `_resolve_block` / `BLOCK_REGISTRY`):

| Block | Usage |
|-------|-------|
| `bim_extractor` | 6× |
| `primavera_parser`, `spec_analyzer`, `historical_benchmark` | 4× each |
| `boq_processor`, `chat` | 2× / 1× |
| Week 1–4 intelligence blocks | 1× each |

**Required blocks:** `pdf`, `ocr`, `image`, `boq_processor`, `spec_analyzer`, `sympy_reasoning`, `drawing_qto`, `primavera_parser`, `smart_orchestrator`, `jetson_gateway`, `formula_executor`, `bim_extractor`, `learning_engine`, `recommendation_template`

**Recommendation:** **Do not split the monolith** until a failing test forces it. It is working production code; split is migration-path debt, not urgent refactor.

---

### 2. Domain layering (correct vs fixed)

| Layer | Mechanism | Status |
|-------|-----------|--------|
| Container policy | `ConstructionContainer.chat()` → `construction_evm.md`, `use_rag=True` | ✓ Correct |
| Generic chat | ~~Default `construction_expert.txt`~~ | ✓ Fixed (`fa76e0a`) |
| Block rules | `construction_v2` → `ConstructionKnowledge` (PRC validation) | ✓ Correct |
| RAG retriever | Domain-agnostic; context via prompts + tools | ✓ Correct |

**Dual prompt note:** `construction_evm.md` (container chat) and `construction_expert.txt` (expert persona for blocks/tests) serve different roles. Only the **generic chat default** was wrong; both prompt files should remain in the kit.

---

### 3. `app/core/construction_knowledge.py`

508 lines. Loads `app/data/procedures/procedures_db.json` and `app/prompts/construction_expert.txt`.

**Construction-specific (keep in kit):**

- `CRITICAL_RULES`, `enforce_critical_rules()`
- `generate_doc_number()` — RFI/NCR/VO/DD numbering
- PRC-951 design review: `validate_design_status()`, `check_review_timeline()`
- PRC-942 NCR: `validate_ncr_disposition()`, `next_ncr_status()`
- `get_procedure()`, `ConstructionKnowledge` facade

**Methodology-shaped (could extract later, not urgent):**

- `score_risk()` — 1–5 matrix (PRC-932)
- `calculate_payment()` — retention math (PRC-965)
- `calculate_evm()` — PV/EV/AC metrics
- `evaluate_tender()` — weighted scoring (PRC-963)

**Consumers:** `app/blocks/construction_v2.py`, `scripts/generate_knowledge_scenarios.py`, `tests/test_knowledge_scenarios.py`

---

### 4. Supporting artifacts (publish with construction kit)

| Path | Purpose |
|------|---------|
| `app/core/construction_types.py` | Shared `Measurement`, `SpecItem`, `RiskItem` |
| `app/core/construction_constants.py` | Grade tables, default numeric constants |
| `app/prompts/construction_evm.md` | Container chat default |
| `app/prompts/construction_expert.txt` | Expert system prompt for knowledge/blocks |
| `app/data/procedures/procedures_db.json` | PRC procedure definitions |
| `app/knowledge/construction_kb.json` | Static KB snippets |
| `app/blocks/construction_v2.py` | TypedBlock construction analysis |

**Recommendation:** Keep `construction_types` in Fork at `app/core/construction_types.py`. CB store publishes it as-is in the kit bundle — do not relocate to a CB-only path.

---

## Cerebrum-Blocks store integration

```
The Fork (main)
    │  python scripts/publish_construction_kit.py  (run from CB repo)
    ▼
block_store/kits/construction/bundle/
    │  GET  /store/containers
    │  POST /store/containers/construction/install
    ▼
Consumer instance app/ tree
```

**Publish script:** `Cerebrum-Blocks/scripts/publish_construction_kit.py`  
**Kit manifest:** `block_store/kits/construction/manifest.json` (9 artifacts)

**Recommendation:** Add CI on Fork tag/release → trigger kit republish in CB so store bundle stays in sync with production.

---

## Recommendations (prioritized)

### Do now (Fork-side)

| # | Item | Notes |
|---|------|-------|
| 1 | ~~Remove chat prompt hardcode~~ | Done — `fa76e0a` |
| 2 | Push `fa76e0a` to GitHub | Unblocks team / Render deploy |
| 3 | Verify `ConstructionContainer.chat()` still sets `construction_evm.md` in staging | Regression check after chat fix |

### Do soon (Fork-side, no monolith surgery)

| # | Item | Notes |
|---|------|-------|
| 4 | Deprecate `historical_benchmark` delegate | Comment in container says `learning_engine` replaces it; 4 `_resolve_block` refs remain |
| 5 | Document publish cadence | When to run CB `publish_construction_kit.py` after Fork merges |
| 6 | Cherry-pick `_safe_float` from CB if needed | CB `construction.py` has audit-hardening CB-only; evaluate diff vs Fork, don't replace monolith |

### Defer

| # | Item | Trigger |
|---|------|---------|
| 7 | Split `construction.py` into mixins | Failing test or unmaintainable merge conflict |
| 8 | Extract generic math from `construction_knowledge` | Second domain kit (legal/medical) needs shared EVM/payment helpers |
| 9 | Replace container with `construction_v2` only | TypedBlock migration complete + parity tests |

### Do not do

| Item | Why |
|------|-----|
| Replace Fork `chat.py` with Cerebrum-Blocks chat | CB block is ~310 lines, no RAG/local LLM/TypedBlock — massive regression |
| Wholesale migrate Fork → CB | Risk with no upside while Masterise + adapter are live |
| Move runtime to CB store repo | CB is discovery/install only |

---

## Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Store bundle drift vs Fork `main` | High | Tag-driven republish; manifest `version` matches container `3.1` |
| Direct `ChatBlock` calls without prompt | Medium | Document that domain apps must use container `chat()` or pass `system_prompt_file` |
| `historical_benchmark` vs `learning_engine` | Low | Track deprecation; remove delegate when parity confirmed |
| 7k-line monolith merge conflicts | Medium | Defer split; enforce Fork as sole runtime editor |
| Missing `construction_knowledge` on install | High | CB kit bundle includes all 9 artifacts; run publish after Fork changes |

---

## Audit tooling

From Cerebrum-Blocks repo (optional local clone of Fork):

```bash
python scripts/audit_fork_container.py
```

GitHub reference SHA (2026-06-10): `app/containers/construction.py` — 363,558 bytes on `main`.

Full migration map: `Cerebrum-Blocks/docs/container_migration_manifest.md`

---

## Virgin Fork strip checklist

Goal: boot with **~17 generic blocks** + **`DomainContainer` host** — no construction in default registry.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Remove `construction_expert.txt` auto-inject from `ChatBlock` | Done | `fa76e0a` |
| 2 | Add `app/containers/base.py` (`DomainContainer`) | Done | Kit entry point; `chat()` injects `system_prompt_file` |
| 3 | Gate `ConstructionContainer` from default `BLOCK_REGISTRY` | Done | Virgin default; enable via `CEREBRUM_DOMAIN_KITS=construction` or store install |
| 4 | Gate construction domain blocks (`boq_processor`, `construction_v2`, …) | Done | Load only with construction kit |
| 5 | Lazy `app/containers/__init__` — no `ConstructionContainer` at import | Done | `__getattr__` loads on demand |
| 6 | `data/domain_kit_registry.json` written on store install | Done | CB `container_kit_store._register_kit_on_target` |
| 7 | `domain_kit_loader` merges kit blocks at boot | Done | `app/core/domain_kit_loader.py` |
| 8 | Document 17 generic blocks | Done | `Cerebrum-Blocks/docs/generic_blocks.md` |
| 9 | Platform charter | Done | `Cerebrum-Blocks/docs/platform_charter.md` |
| 10 | Production deploy env | **Operator** | Set `CEREBRUM_VIRGIN=false` + `CEREBRUM_DOMAIN_KITS=construction` on live Fork |
| 11 | Republish construction bundle after Fork merges | Pending | `python scripts/publish_construction_kit.py` from CB |
| 12 | Verify `ConstructionContainer.chat()` still sets `construction_evm.md` | Pending | Staging regression |

### Boot env reference

| Variable | Default | Production Fork |
|----------|---------|-----------------|
| `CEREBRUM_VIRGIN` | `true` | `false` |
| `CEREBRUM_DOMAIN_KITS` | *(empty)* | `construction` |

Virgin boot: `construction` ∉ `BLOCK_REGISTRY`. Kit-enabled boot: construction container + 15 kit blocks register automatically.

---

## Change policy (post-audit)

**The Fork:** Runtime fixes only. No structural refactors until a test forces them. Store publishing happens from CB.

**Cerebrum-Blocks:** Store API, kit manifests, publish scripts, store UI.

---

*Last updated: 2026-06-10 — virgin Fork platform vision: DomainContainer host, construction gated from default registry.*
