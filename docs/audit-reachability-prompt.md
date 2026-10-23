# Audit prompt — reachability, wiring and formula integrity

Repo-agnostic prompt for any coding agent. It reproduces the 2026-07-24
"registered-but-dead" sweep generically: paste it at any repo, kit, formula set
or feature list. Checks 3 and 12 are the ones that catch the seam class of bug
(`bim_extractor` emitted, `bim_extract` implemented — routable, hinted,
permanently undeliverable, every per-file test green). The permanent in-repo
guard for this codebase is `tests/test_capability_seams.py`.

```
AUDIT PROMPT — Reachability, Wiring & Formula Integrity (run on any repo)

You are auditing this repo for the "registered-but-dead" trap: functions,
formulas, actions, or features that exist and pass unit tests, yet are
unreachable, hollow, or unverified in production. Do NOT modify code. Produce a
table classifying every unit as LIVE / DEGRADED / STUB / UNVERIFIED with
file:line evidence. A unit is LIVE only if it clears ALL applicable checks
below; otherwise say which check it fails and the one change that fixes it.

For every ACTION / FEATURE / ENDPOINT:
1. REGISTERED: it's declared in the dispatch/registry/router map. Give file:line.
2. REACHABLE: trace the full path from a user input (keyword/route/intent/URL)
   to this handler. If any gate (allowlist, intent set, permission, feature
   flag) excludes it, it is NOT reachable — flag it.
3. SEAM CHECK: the name used to route/emit it is IDENTICAL to the handler key.
   Cross-reference every routing/intent/hint/allowlist string against the actual
   handler names. Any string that resolves to no handler is a dead seam — list it.
4. HONEST OUTPUT: on valid input it returns a non-empty, populated result. On
   invalid/missing input it returns an explicit error/degraded status — NEVER a
   silent success over empty ([], 0, None, {}) internals. Grep for handlers that
   return status:"success" while sub-extractors return empty; list them.
5. TESTED END-TO-END: at least one test drives input->route->dispatch->output and
   asserts the DELIVERABLE (not just status=="success"). Layer-only or
   existence-only tests do NOT count. Name the test or say "none".

For every FORMULA / CALCULATION:
6. SOURCED: it cites an authoritative reference (standard/spec/paper/worked
   example). Uncited = UNVERIFIED regardless of tests.
7. UNITS & DOMAIN: input units and valid ranges are declared and guarded.
8. GOLDEN VECTOR: a test asserts output against a worked example FROM the source,
   to a stated tolerance. No golden vector = UNVERIFIED.
9. NO ORPHAN CONSTANTS: every magic number traces to the cited source, not
   inlined ad hoc.
10. SINGLE SOURCE: one implementation, imported everywhere (no drifting copies).

For the TEST SUITE itself:
11. SILENT SKIPS: list every skip/skipif/xfail and whether it hides a real
    feature or fixture. In CI, skip==green — treat fixture-gated skips as
    UNVERIFIED, not passing.
12. INVARIANTS PRESENT: is there a meta-test asserting "every gated/deliverable
    name resolves to a real handler"? If not, that's the top-priority gap.

OUTPUT: (a) the classification table; (b) a ranked "dead/degraded" list with the
single fix each needs; (c) the exact meta-test to add so this class of bug fails
the build forever. Assert nothing you haven't traced to file:line.
```
