# CLAUDE.md

Claude Code reads this file; Cursor, Codex and other agents read `AGENTS.md`.
`AGENTS.md` holds the full standing orders and run notes; read it next.

## STANDING ORDER -- small core first (owner, 2026-11-26)

> Every new build and every project: start with a small core only - RAG of 30 pages at most of different document types, a few formulas, a couple of skills, 1 kernel, a couple of encoded-knowledge items. Nothing else is loaded until that core is all tested and working, with no hardwiring, passing every live UI test. Only then add the full content. Never load everything at once and then start trial and error.

## STANDING ORDER -- no hardwiring (owner, 2026-10-02)

> NO HARDWIRING. A failing case is never fixed by name. The full order is the
> first section of `AGENTS.md` and `docs/ORDER-no-hardwiring.md`.
