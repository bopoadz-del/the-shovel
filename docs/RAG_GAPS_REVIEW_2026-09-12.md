# RAG Review — everything discussed, what's in, what's missing — 2026-09-12

Master record of the full RAG review. Verified against Neon (`ep-fragrant-river-a6dizxhm`)
and live `theshovel.ai`. Final retrievable chunk count: **124,166**.
This note lives in `docs/` (NOT `docs/knowledge/`) so it is not itself ingested as knowledge.

## A. Standards / codes knowledge to AUTHOR then ingest into curated_kb

The full copyrighted standard PDFs are NOT and will not be ingested (licensed; can't serve).
Legitimate path = **authored summaries** (your own words, cite by clause, no verbatim text) placed
in `docs/knowledge/` → ingested into `curated_kb` → served via the general-knowledge merge.
This is the single list — no separate stub files; author against these here.

**Saudi Arabia** — SBC 201 (admin), 301 (loads), 302/303 (soils/foundations), 304 (concrete),
305 (masonry), 306 (steel), 401 (electrical), 501 (mechanical), 601 (energy), 701 (sanitary),
801 (fire), 901 (existing). *A filled SBC summary already exists and is ingested:*
`docs/knowledge/ksa_saudi_building_code.md` — extend it rather than duplicate.

**UAE** — Dubai Building Code (DBC), Al Sa'fat green regs, Abu Dhabi International Building Code
(ADIBC), Estidama Pearl Rating (PBRS/PCRS), UAE Fire & Life Safety Code; authority NOC workflow
(DM, DCD, DMT, ADCD).

**US — residential / commercial / industrial** — IBC, IRC, IECC, IFC, IPC, IMC, IFGC; ASCE 7,
AISC 360, ACI 318, ASTM; ASHRAE 90.1 & 62.1; NFPA 101/13/72/70(NEC)/30/497/499; ADA / ICC A117.1;
OSHA 1910; FM Global, API (process/industrial).

**International reference (cite, don't reproduce):** BS EN / Eurocodes, ISO.

Already in curated_kb and reachable: FIDIC (×3), OSHA 1926 (×6), CESMM4, WBDG (×8 — US design
guidance), rates (KSA/Gulf 2025), EVM, productivity norms. To add the above: author summaries,
drop as `.md` in `docs/knowledge/`, ingest into `curated_kb`.

## B. Owned documents MISSING from the RAG (can ingest — yours, no license issue)

| item | count | where | note |
|---|---|---|---|
| Governance docs | 5 | Downloads (staged local) | Governance manuals (PM / document-control / project-reporting procedures), construction submittal form, operational guidance |
| Transmittal message files | 3 | Master Folder root (online-only in Drive) | Transmittal messages (multi-part), separate from the submittal-package parts |
| Drive-pack text docs never ingested | ~134 | infrastructure pack (Drive) | mostly large drawing PDFs (ingest THIN — geometry, low value); the ~15-20 genuinely valuable are an executed contract, design reports, board outcomes, and phasing / site-initiative decks |

To ingest B: the worker (`the-fork-ingest`, has the bge embedder) into `drive_archive`, or a local
path once `sentence-transformers` is installed. Route: client docs → `drive_archive`; standards
summaries → `curated_kb`.

## C. Discussed but NOT owned / NOT available

- **Consultant rate schedule** — not in the RAG and no rate schedule found on the drive. Consultant
  correspondence is in (1992 chunks). A rate schedule would need the actual file if it exists.

## D. Confirmed IN the RAG and reachable (verified live, 12/12 retrieval test)

Submittal-package parts 1–4 · the 4 Method Statements · governance-manual family · Project Controls
(49 indexed / 174 thin) · consultant correspondence · risk register · baseline/critical-path · and the
curated standards summaries (SBC/FIDIC/OSHA/CESMM4/WBDG surface via general-knowledge merge).

## E. Excluded BY DESIGN — never ingested, not gaps

CAD (`.dwg/.dxf`), images (`.jpg/.png`), Google Earth (`.kmz/.kml`), video, GIS internals
(`.gdbtable/.spx/…`), fonts, and Arabic-named files. Only text formats produce retrievable chunks.

## F. Known limitations (not fixable by ingest)

- **1,710 thin docs** — drawing/CAD PDFs with no extractable text. No ingest thickens them.
- **curated_kb ranking** — 252 chunks vs drive_archive's 119,813; general-knowledge answers can be
  out-ranked unless the query is clearly general. Fix = boost GK weight, not re-ingest.

## Next actions (owner's call, none blocking)
1. Fill the US + UAE stubs → ingest into `curated_kb`.
2. Ingest the ~20 valuable owned text docs (governance + contracts + reports) via the worker.
3. Optionally retire the redundant duplicate infrastructure-pack project.

*Reviewed 2026-09-12. Nothing was lost; the gaps above were never ingested, not deleted.*
