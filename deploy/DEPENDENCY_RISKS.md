# Accepted dependency risks (Dependabot / OSV)

Review date: **2026-08-02** (was 2026-06-13).

> **CORRECTION 2026-08-02 — the previous header was factually wrong.** It read
> *"Production Render image installs `requirements.txt` only"*. It does not.
> `Dockerfile` also installs, by explicit pin:
> `torch==2.5.1`, `torchvision==0.20.1`, `ultralytics==8.4.75`,
> `onnxruntime==1.27.0`, `opencv-python-headless==4.10.0.84`,
> `sentence-transformers==5.5.1`.
>
> That error propagated into the torch rows below as
> *"Production exposure: No — not in `Dockerfile`"*, which inverted the
> actual situation: **torch ships in the production image**, and at `2.5.1`
> — *older* than the `2.12.0` in the optional lockfiles and still inside the
> advisory's affected range. An accepted-risk register that understates
> production exposure is worse than no register, so the rows are corrected
> rather than re-dated.
>
> The risk acceptance still holds, but for a different and verifiable
> reason — reachability, not absence. See Alert 2.

---

## Alert 1 — `deep-translator` — **RESOLVED** (2026-06-13)

| Field | Value |
|-------|--------|
| **Was** | Direct dep in `requirements.txt` — PYSEC-2022-252 |
| **Fix** | Removed `deep-translator`; `app/blocks/translate.py` calls the public Google Translate HTTP endpoint via `requests` (same `translate.googleapis.com` client=gtx path). |
| **Production** | `requirements.txt` no longer lists `deep-translator`. |

---

## Alert 2 — `torch` (CORRECTED 2026-08-02)

| Field | Value |
|-------|--------|
| **Manifests** | `requirements-cv.txt` / `-ml.txt` / `-rag.txt` pin `torch==2.12.0` (optional tiers, **not installed by the image**) |
| **Production image** | **`torch==2.5.1`, pinned directly in `Dockerfile`** — CPU wheels, alongside `torchvision`/`ultralytics`/`onnxruntime` for the YOLO ONNX path |
| **Advisory** | [GHSA-rrmf-rvhw-rf47](https://osv.dev/GHSA-rrmf-rvhw-rf47) — **LOW**, memory corruption via `torch.jit.script` |
| **Production exposure** | **YES — torch is in the image.** (The previous "No" was wrong.) |
| **Reachability** | **Nil in our code.** `torch.jit.script` / `jit.script` has **zero call sites** across `app/` and `scripts/` — verified 2026-08-02. The advisory requires calling it, on attacker-influenced input. |
| **Patched version** | `2.13.0` **now exists** (the old note "no patched release" is stale). |

**Why still accepted, and what it would take to close:** the risk is LOW and
unreachable in our code paths, while closing it is not a one-line bump — the
image pins a coordinated CPU-wheel stack (`torch` + `torchvision` +
`ultralytics` + `onnxruntime`) and moving torch 2.5.1 → 2.13.0 drags
`torchvision` and re-qualifies the baked YOLO-World ONNX pipeline. That is a
deliberate piece of work with its own verification, not a dependency bump.

**Version drift worth knowing:** the image runs `2.5.1` while the optional
lockfiles say `2.12.0`. Nothing consumes both at once, so it is not a runtime
hazard — but do not "reconcile" them by editing one number; the Dockerfile pin
is chosen for CPU-wheel + ultralytics compatibility.

**Dismiss comment:** `Accepted: LOW severity, torch.jit.script has zero call sites in app/ or scripts/ (verified 2026-08-02). torch IS present in the production image at 2.5.1 — accepted on reachability, not absence. Closing requires a coordinated torch/torchvision/ultralytics/onnxruntime bump + ONNX pipeline re-qualification.`

---

## Alert 3 — `torch` in `requirements-ml.txt` / `requirements-rag.txt` (optional)

Same advisory **GHSA-rrmf-rvhw-rf47**, same disposition as Alert 2 — accepted on
**reachability** (zero `torch.jit.script` call sites), not on absence from the
image. Locks recompiled 2026-06-13.

**Dismiss comment:** `Accepted: LOW, torch.jit.script unused in this codebase (verified 2026-08-02). See Alert 2 — torch does ship in the production image at 2.5.1; the earlier "not shipped on Render" wording was incorrect.`

---

## Alert 4 — `react-router` RSC-mode CSRF (2026-08-02)

| Field | Value |
|-------|--------|
| **Manifest** | `frontend/package-lock.json` — `react-router` / `react-router-dom` `7.18.1` |
| **Advisory** | [GHSA-qwww-vcr4-c8h2](https://osv.dev/GHSA-qwww-vcr4-c8h2) — **HIGH**, RSC-mode CSRF bypass; affected `>=7.12.0, <8.3.0` |
| **Production exposure** | Present in the built SPA bundle |
| **Reachability** | **Nil.** The advisory requires React Router **RSC mode** with server actions. `frontend/` is a client-side Vite SPA using `BrowserRouter`; verified 2026-08-02 — zero RSC markers, no `use server`, no `react-server` imports, no server-side router. |

**Why not bump:** the fix is `8.3.0`, a **major** version. A breaking router
migration on a live client application is not justified to clear an
unreachable advisory.

**This alert stays OPEN and will keep appearing in the API** — that is
expected, not an oversight. See `docs/CLIENT_DESK_READINESS_20260802.md` §3.

**Dismiss comment:** `Accepted: RSC-mode-only advisory; this frontend is a client-side BrowserRouter SPA with no RSC/server actions (verified 2026-08-02). Fix requires the v8 major migration — not justified for an unreachable path on a live client app.`

---

## Production manifest status

Measured 2026-08-02 after PR #299 (`gh api .../dependabot/alerts`):
**1 high, 1 medium, 4 low open.**

| Manifest | Status |
|----------|--------|
| `requirements.txt` | **Clean of HIGH** — pyasn1 bumped to 0.6.4 in #299 (3 HIGHs closed). Carries the pytest MEDIUM; see below. |
| `frontend/package-lock.json` | 1 HIGH open — react-router RSC (Alert 4, unreachable) |
| `Dockerfile` (direct pins) | torch 2.5.1 — LOW, unreachable (Alert 2) |
| `requirements-cv.txt` | torch GHSA (optional tier) |
| `requirements-ml.txt` | torch GHSA (optional tier) |
| `requirements-rag.txt` | torch GHSA (optional tier) |

**pytest MEDIUM (`<9.0.3`)** — `requirements.txt` pins `pytest==8.4.2`, so the
test framework ships in the production image but is never invoked there, and
the advisory (tmpdir handling) requires *running* pytest. `requirements.in`
bounds it at `<9.0` for pytest-asyncio 0.x compatibility. An upgrade
experiment on 2026-08-02 was **inconclusive** — see
`docs/CLIENT_DESK_READINESS_20260802.md` §3 and §4b. The durable fix is to
stop installing test dependencies into the runtime image at all.

---

## Revisit triggers

1. **torch:** PyTorch publishes a version outside GHSA-rrmf-rvhw-rf47 — re-run `scripts/compile-requirements.sh`.
2. **Quarterly:** scan `requirements.txt` with osv-scanner on main.
