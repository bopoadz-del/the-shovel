# DEPLOYMENT_PROFILE=onprem: the zero-egress profile (part 1)

Status 2026-11-15: part 1 of the restored on-premise program is in code. Part 2
(a local model provider and the boxed stack) is not built; it waits on the
client's hardware.

One setting, `DEPLOYMENT_PROFILE`, names the profile. It is read in one place,
`app/core/deployment_profile.py:deployment_profile()`:

- **`hosted`** (unset, `hosted` or `cloud`): today's behaviour. No check runs.
- **`onprem`** (`onprem`, `on-prem`, `on_premise`, `on-premise`): the service
  must not call outside the box, and it **refuses to start** while any setting
  would.
- Any other value refuses to start. A mistyped profile must not quietly run as
  hosted.

The check runs in `app/main.py` lifespan (`enforce_deployment_profile()`),
right after the startup env validation.

## Design boundary

Network isolation itself is the deployment's job. This check proves only that
the configuration does not need egress. It installs no socket blocker.

## What the boot check refuses, and why

Rules, not a list of services. The refusal names every offending setting at
once and never prints a value.

| Stage | Rule | Example offender |
|---|---|---|
| model provider | `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` is resolved through the runtime's own provider table. A name the runtime does not support is refused: the runtime would route it to its default cloud provider. A supported provider is refused when its endpoint is not inside the box. With no provider named and no key for the default one, no model is configured and nothing is called. | `LLM_PROVIDER=ollama`, `LLM_PROVIDER=deepseek`, unset provider with `DEEPSEEK_API_KEY` set |
| embedder | A model-hub id needs `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` truthy, with the weights pre-seeded. A local directory or `fake` needs neither. Unset uses the served default, which is a hub id. | hub id without the flags |
| credential | Any non-empty `*_API_KEY`, `*_ACCESS_TOKEN` or `*_DSN` is a credential for a remote service. | `SENTRY_DSN`, `RESEND_API_KEY`, `GOOGLE_ACCESS_TOKEN` |
| endpoint | Any non-empty `*_URL` with a host must be inside the box: loopback, private or link-local address, a single-label (compose service) name, or a name that resolves only to such addresses. A name that does not resolve is refused. The service's own inbound addresses (`FRONTEND_URL`, `PUBLIC_BASE_URL`, `RENDER_EXTERNAL_URL`) are not dialled and are not checked. | `DATABASE_URL` on a hosted database |

A refusal line is the setting's name and a fixed verdict. It never prints the
value, nor anything read from it: not the provider the setting names, not the
model id, not the host it dials. Read the setting itself to see which host.

### Internal names are judged by resolving them, once, at start

A dotted internal name (for example `db.plant.internal`) passes only if, **at
start**, it resolves to inside addresses (loopback, private or link-local),
and only to such addresses. A name that does not resolve is refused. So the
box's resolver must already answer for every internal name when the service
starts.

**A later change of that name's address is NOT caught.** The check runs once
at boot. If the name is re-pointed at an outside address after start, the
running service follows it and nothing here notices. Network isolation (the
deployment's job, see the design boundary above) is what stops that, not this
check. The re-check happens on the next start.

Why resolution, and not a list of internal suffixes taken from settings: a
suffix list trusts a name by how it is spelt. A name under a trusted suffix
that resolves to a public address (a split-horizon mistake, a stale record)
would pass, and that is exactly the egress this check exists to refuse.
Resolution judges where the name actually points at start. A suffix list
would also be one more setting that must itself be trusted and kept per site.
Both rules share the boot-time limit above, so a suffix list would add no
cover against a later change.

The offline model flags are asserted, never set from Python: `transformers`
reads them at import time. They live in the deployment env
(`deploy/onprem/onprem.env.example`, `Dockerfile.onprem`).

## Not in this repository (part 2, or never restored)

An earlier version of this file described these. They are **not** in code:

- a local model provider (Ollama). `LLM_PROVIDER=ollama` is refused at boot;
- per-feature runtime gates (weather, Drive, OneDrive, web fetch, web search,
  translate, outbound webhook, MCP consumer) returning an `onprem_unavailable`
  payload. Under `onprem` those features are switched off by leaving their
  credentials and URLs unset, which the boot check enforces; a call to one
  fails with its normal network error;
- a boot manifest and a disk canary.

## Running on-prem locally

1. Copy `deploy/onprem/onprem.env.example` to `onprem.env` and fill the local
   secrets. Leave every model provider setting unset.
2. Pre-seed the embedder weights (the on-prem image bakes them) or, for a smoke,
   set `RAG_EMBEDDING_MODEL=fake`.
3. `set -a; . onprem.env; set +a; uvicorn app.main:app`. The boot check either
   starts the service or names each setting that would call outside.

## Tests

`tests/test_deployment_profile_onprem.py`: the profile switch (hosted, onprem,
unknown), the hosted profile untouched, a clean box that starts, and one
must-pass and one must-fail setting for each stage above. Host names go
through a fake resolver, so the tests never use DNS.
