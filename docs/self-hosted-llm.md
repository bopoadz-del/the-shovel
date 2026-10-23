# Self-hosted LLM (Ollama) — escape Groq's 30K TPM cap

> **Current (2026-07-18):** Ollama is now the LLM path for the full **on-prem /
> air-gapped deployment profile** (`DEPLOYMENT_PROFILE=onprem`), not just a
> cost/rate escape hatch — see `docs/onprem-profile.md` and
> `deploy/onprem/`. This note still applies for using Ollama on the cloud
> service; the boot-gated on-prem profile is the productionised version.

The agent runtime supports Ollama as a first-class provider. Set two
environment variables on Render and every chat in the UI is served by
your own machine instead of Groq. Zero per-token cost. No TPM rate limits.
Bounded only by your hardware.

## TL;DR

```
LLM_PROVIDER=ollama
OLLAMA_URL=https://<your-tunnel-url>
OLLAMA_MODEL=qwen2.5:7b-instruct   # optional; default qwen2.5:7b-instruct
```

The runtime appends `/v1/chat/completions` automatically if you pass a
bare host or a `/v1` suffix, so any of these works:

- `http://my-pc.tunnel.cf`
- `http://my-pc.tunnel.cf/v1`
- `http://my-pc.tunnel.cf/v1/chat/completions`

## Setup

### Option A: Ollama on your PC + Cloudflare Tunnel (free)

Best fit when you have a desktop or laptop that stays on, and you want
zero hosting cost.

1. Install Ollama: <https://ollama.com/download>
2. Pull a model that fits your hardware:
   - CPU only (16 GB RAM): `ollama pull qwen2.5:7b-instruct`
   - 8 GB GPU: `ollama pull qwen2.5:14b-instruct` or `llama3.2:11b`
   - 24 GB GPU: `ollama pull llama3.3:70b-instruct-q4_K_M`
3. Start the server (runs on `http://localhost:11434` by default):
   `ollama serve`
4. Install Cloudflare Tunnel (`cloudflared`): <https://github.com/cloudflare/cloudflared/releases>
5. Create a tunnel:
   `cloudflared tunnel --url http://localhost:11434`
   Cloudflare prints a URL like `https://random-words-1234.trycloudflare.com`.
6. On Render Dashboard -> The_Fork -> Environment, add:
   - `LLM_PROVIDER=ollama`
   - `OLLAMA_URL=https://random-words-1234.trycloudflare.com`
   - `OLLAMA_MODEL=qwen2.5:7b-instruct` (or whichever you pulled)
7. Render redeploys automatically. Test by chatting in the UI.

A named tunnel (instead of the throwaway `--url` form) survives
reboots. Cloudflare's "Connect an application" walkthrough takes about
5 minutes.

### Option B: Ollama on a cheap GPU VPS

Best fit when you don't want a PC-on-24/7 dependency.

- Hetzner GEX44 (RTX 4000 SFF, 20 GB): ~60 EUR/month, runs llama3.3:70b
  at usable speed
- Runpod / Vast.ai (spot GPU): ~$0.20-0.40/hour on-demand
- A vanilla 8 GB VPS will only fit ~7B-Q4 models — fine for chat but
  slower than Groq

Same wiring: install Ollama, expose port 11434 (firewall it to Render's
egress IPs only), set the same env vars.

### Option C: Ollama on the Render box itself

NOT recommended on the starter plan. Render starter is 512 MB RAM with
no GPU. Even Qwen2.5:0.5B would be slow and unstable. Upgrade to a Pro
plan (4 GB) and a 3B model is borderline tolerable for short answers,
but you'll regret it for anything involving long tool calls.

## Falling back to Groq when local is unreachable

The runtime does NOT auto-failover. If `LLM_PROVIDER=ollama` is set and
your tunnel goes down, chats return errors until you either restart
the tunnel or flip the env var. Two options if you want resilience:

1. Keep `GROQ_API_KEY` set and leave `LLM_PROVIDER=ollama`. When the
   tunnel is up, Ollama serves. When it's down, manually unset
   `LLM_PROVIDER` and Render's redeploy picks Groq automatically (since
   precedence is: explicit env var -> presence of GROQ_API_KEY ->
   DeepSeek).
2. Future work: a runtime-level "try ollama then fall back to groq"
   chain. Not built yet — file a request if you want it.

## Tool calling considerations

Ollama's tool-calling support varies by model. As of early 2026:

- Qwen2.5 7B+: tool calls work
- Llama 3.3 70B: tool calls work
- DeepSeek-R1: tool calls work but verbose
- Smaller models (<7B): tool calls unreliable; chat OK but `generate_wbs`
  / `boq_processor` calls often malformed

If you're using project-assistant (the UI agent), pick a model that
supports tool calls — otherwise the agent can only chat, never call
the construction backend.

## Why this matters

The Groq free tier shows a token-per-minute cap of 30,000 across all
models. With the construction expert prompt auto-injected (~3,500
tokens), two back-to-back chats blow through TPM and the user sees
HTTP 429. Self-hosting eliminates that cap entirely — the only limit
is how fast your hardware can decode tokens.
