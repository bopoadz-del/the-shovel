# Load baseline — 2026-08-05

First measured load numbers for the production web service
(`srv-d8hdc6ek1jcs739rq5sg`, Render starter: 0.5 CPU / 512 MB,
`UVICORN_WORKERS=2`). Before this run, capacity was a known unknown — the
2026-08-04 PRR listed "no load test exists" as a should-fix.

## Method

10 concurrent HTTP clients, 60 seconds, GET-only, alternating `/health` and
`/ready` (both execute a real `SELECT 1` DB probe per hit — see
`app/routers/health.py`). No LLM/chat endpoints: those bill per call and the
box serves a live client desk. Run from a residential connection (Windows,
httpx async, connection reuse). Script: session scratchpad
`load_baseline.py` (method recorded here; numbers are the artefact).

## Results

| metric | /health | /ready |
|---|---|---|
| requests (60 s) | 463 | 462 |
| non-200 | 0 | 0 |
| p50 | 562 ms | 562 ms |
| p95 | 813 ms | 797 ms |
| p99 | 4,953 ms | 5,203 ms |
| max | 6,250 ms | 5,828 ms |

Throughput: **15.4 req/s aggregate, zero errors.**

## Findings

1. **The box holds 10 concurrent probe clients** with a healthy median
   (~560 ms including TLS + Oregon round-trip from the test location).
2. **The tail collapses under exactly this load**: p99 stretches to 5–6 s.
   During the 60-second window, the external uptime monitor for
   `theshovel.ai` (5-min interval checks) recorded the site as DOWN for
   ~1m24s — its check landed inside the test window and timed out. The
   monitor page the operator saw was this test, not an outage; both domains
   returned 200 immediately after.
3. Practical ceiling for the starter box is therefore **below** 10
   sustained concurrent request streams. For the single-client pilot desk
   this is ample; for multi-user demos, either keep concurrent users ≤ ~5
   or bump the instance plan first.

## What this does NOT cover

Chat/RAG latency under load (deliberately untested — costs money and
serves a live desk), sustained load beyond 60 s, and cold-start behaviour.
If a multi-user rollout is planned, rerun with a staging instance and a
realistic chat mix before committing to a plan size.
