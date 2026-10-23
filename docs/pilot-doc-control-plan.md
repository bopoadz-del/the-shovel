# Pilot Document-Control & Project-Structure Plan (AGREED, DEFERRED)

**Status:** Agreed 2026-07-16. **Not built.** Prepare-only now; define exactly and
implement when we move to pilot (soon). This file is the notice of what we agreed.

## Decision (operator, 2026-07-16)

The full document-control / RAG-taxonomy build is **NOT needed at this stage**. It
**is** needed at pilot. So: prepare for it now, no filing action now, define the
exact structure later. **No permission gating now. Global template, later.**

For testing right now: **one project on the left panel that can retrieve ALL
documents**, so testing isn't fragmented across projects. Everything set up
properly when we deploy the pilot.

## Test-mode setup (now)

- Keep a **single** project visible in the sidebar; it must retrieve the full
  document set. The full corpus is `drive_archive` (2,737 docs / 123,405 chunks).
- Hide the other projects via the **Hide** button (soft-archive — hides from
  listings; does NOT delete RAG chunks; reversible by setting status back to
  active). Hidden ≠ deleted.
- No gating, no per-doc approval, no folder taxonomy in test mode.

## Target model for PILOT (deferred — to finalize then)

Real-contractor document control:

- **Global layer, inherited by every project:** Standards & Codes (OSHA / FIDIC /
  local authority / QPCL / international) + General Specifications.
- **Per-project departments:** Contracts, Design, Development, Delivery.
  - Under **Delivery:** Cost Control, QS, Planning, Procurement, Engineering,
    Site / Construction.
- **Cross-cutting, surfaced in every project:** unpriced BOQ, Drawings, Specs.
- **Restricted classes (RBAC — pilot):** Priced BOQ (authorized commercial only)
  and Contract documents (Contracts dept only). Everything else open.
- **Semi-auto indexing:** upload -> auto-classify (suggest dept/type/sensitivity)
  -> **admin approves each document per project** -> index on approve.
- Source the exact filing taxonomy from a real big-contractor document-control
  procedure (may already be in `drive_archive` under "document control"; if not,
  operator will supply one).

## Locked decisions

- Prepare, don't build, now. | No gating now. | Global template, later. | the client project /
  broad corpus for testing now.
- **NEVER hard-delete a project** — schema is `ON DELETE CASCADE` on chunks;
  hard delete destroys the RAG. UI removal = soft-archive only.

## Separate bug track (surfaced from the 2026-07-16 admin/agent review)

1. Empty-project leakage: `example_estate` (0 docs) answered from the client project content —
   an empty project must say "no documents", not borrow another corpus.
2. Intent bug: "what tools do you have" RAG-retrieved literal power tools instead
   of listing the agent's software tools.
3. Duplicate / mislabeled projects: `master_corpus` and `example_infra_pack`
   both 39/9,861 (same corpus, two names); `drive_archive` mislabeled
   "the client project".
