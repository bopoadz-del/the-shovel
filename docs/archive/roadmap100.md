# Cerebrum — Roadmap to 100% Operational

> **ARCHIVED 2026-06-21.** This document describes the pre-platform
> Cerebrum Blocks fork (28 blocks, no React frontend, no hosted
> deploy, no agents, no RAG). The platform has since grown into
> The Fork — see [README.md](README.md) for current state. Kept
> here for historical reference; do not use as a current spec.

_Last updated: 2026-05-04 (now archived)_

---

## What the platform does

Upload any construction document → get structured intelligence → act on it.

| Input | Output |
|---|---|
| PDF drawing / spec / report | Quantities, measurements, materials |
| Excel / CSV BOQ | Line items, cost breakdown, procurement list |
| Contract PDF | Clauses, obligations, risk register |
| P6 Schedule (XER / XML) | CPM, critical path, delay analysis |
| BIM / IFC file | Element counts, quantities |
| Site photo / scanned drawing | OCR text, measurements |
| Any document | AI chat Q&A |

---

## Current State (as of 2026-05-04)

**28 active blocks** — trimmed from 82 (removed generic platform noise).

### ✅ Working
- PDF / image text extraction (pdf, ocr, claude vision fallback)
- BOQ Excel/CSV parsing → cost breakdown (boq_processor)
- Contract analysis → clause extraction, risk register
- Schedule XER/XML → CPM, critical path, delay analysis
- Spec analysis → material list, submittal log
- Auto-pipeline: upload file → document type detection → panels rendered
- Quantities panel: area/volume now extracted from direct mentions (m2, m3)
- Procurement list: rendered as proper table with lead times
- Cost estimate → payment certificate
- AI chat with document context (DeepSeek + Claude fallback)
- Redis caching (Upstash)
- Arabic/multilingual translation
- Local drive file browsing

### ❌ Not working / incomplete

#### Priority 1 — Core accuracy
- [ ] **Quantities still fragile for complex PDFs** — regex extraction misses
  structured tables, annotated drawings, multi-column layouts. Needs
  LLM-assisted extraction (pass text to Claude, ask for structured quantities).
- [ ] **BOQ auto-detection from drawing PDFs** — currently only works if user
  uploads an Excel/CSV. Should extract a BOQ from a PDF drawing automatically.

#### Priority 2 — File integrations
- [ ] **Google Drive OAuth** — block exists, needs `GOOGLE_ACCESS_TOKEN` env var
- [ ] **OneDrive OAuth** — block exists, needs `ONEDRIVE_ACCESS_TOKEN` env var

#### Priority 3 — Specialized file types
- [ ] **BIM/IFC** — `bim_extractor` registered, not tested with real IFC file
- [ ] **DXF drawings** — `drawing_qto` registered, needs real DXF to validate
- [ ] **P6 XER** — `primavera_parser` works, needs real project XER to validate end-to-end

#### Priority 4 — UI / UX
- [ ] **Project persistence** — each upload is stateless. No way to open a
  previous analysis. Needs a project list backed by Redis or SQLite.
- [ ] **Chat streaming** — responses appear all at once. Should stream tokens.
- [ ] **Procurement panel in auto-pipeline** — currently only appears after
  clicking "Generate Procurement List". Should auto-run when quantities found.

---

## Block Registry (31 blocks)

```
Document Extraction:  pdf, pdf_v2, ocr, ocr_v2, image, document_engine  (6)
AI / Language:        chat, translate, voice, web                         (4)
Construction:         construction, construction_v2, boq_processor,
                      bim, bim_extractor, drawing_qto, primavera_parser,
                      spec_analyzer, formula_executor, sympy_reasoning,
                      historical_benchmark, smart_orchestrator           (12)
File Access:          local_drive, google_drive, onedrive                 (3)
Search / Memory:      vector_search, zvec, cache_manager                  (3)
                                                               Total:    28
```

---

## Done

- [x] Phase 1–4: construction engine, CPM scheduler, BIM, QTO, Primavera, spec,
  formula, smart_orchestrator, voice, translate, drive blocks, vector search,
  cache_manager, document_engine, boq_processor, historical_benchmark
- [x] InputAdapter bug — all blocks handle text/input key fallback
- [x] Chat — DeepSeek primary, Claude haiku fallback
- [x] OCR — Claude Vision fallback when tesseract missing
- [x] Redis — Upstash connected via REDIS_URL
- [x] Auth hardening + CORS
- [x] Quantities: add direct m2/m3 extraction patterns (fixes zeros)
- [x] Quantities: remove nested item_counts object (fixes [object Object])
- [x] Procurement: proper panel renderer with lead times + priorities
- [x] Procurement: passes real project quantities from auto_pipeline result
- [x] Registry trimmed from 82 → 31 blocks (removed generic platform noise)
