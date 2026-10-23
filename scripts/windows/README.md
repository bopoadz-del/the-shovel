# Windows tunnel helpers for The Fork

Durable Ollama bridge for the operator's Windows PC. Keeps the platform's
chat path (Render -> Cloudflared tunnel -> local Ollama -> Ollama Cloud)
alive across reboots and PC sleeps without paying for a domain or a
Cloudflare account.

## Files

| File | What it does |
|---|---|
| `fork-tunnel-start.ps1` | Kills any prior `cloudflared`, starts a fresh quick-tunnel pointed at `http://localhost:11434`, writes its stderr to `%LOCALAPPDATA%\fork-tunnel\cloudflared.log`. Returns immediately. |
| `fork-tunnel-update.ps1` | Reads the tunnel log, grabs the `*.trycloudflare.com` URL, PUTs it to Render's `OLLAMA_URL` env var, triggers a redeploy. Idempotent: if the URL hasn't changed since the last run, no Render calls. |
| `fork-tunnel-up.cmd` | One-click "bring chat online" — runs start, waits 10s, runs update. Leaves a console window open with the output so you can see success/failure. |
| `fork-tunnel-autostart.cmd` | Silent version of `fork-tunnel-up.cmd` for the Startup folder. No console window, no pause. |

## Install for auto-start at login

Copy the four `.ps1`/`.cmd` files into `C:\Users\<you>\.local\bin\` (or any
stable path), then drop a copy of `fork-tunnel-autostart.cmd` into:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
```

(or use shell:startup in Run dialog). It runs silently at every login,
~12 seconds later you're back on the air.

## Manual recovery if the tunnel dies mid-day

Double-click `fork-tunnel-up.cmd`. Output appears on screen. Press a key
to dismiss. Chat is back ~2 minutes later (after Render redeploys with
the new URL).

## Logs and state

All under `%LOCALAPPDATA%\fork-tunnel\`:

- `cloudflared.log` — raw cloudflared stderr (URL + connectivity checks)
- `update.log` — audit trail of every Render env-var push
- `last-url.txt` — current tunnel URL, used for change detection

## Hard-coded values

The scripts hard-code the Render API key and service id. They're already
in the operator's environment elsewhere (memory, prior commits) so this
isn't new exposure. Treat the scripts the same as `~/.env`.

To rotate or move services, edit `fork-tunnel-update.ps1`:

```
$RenderApiKey   = 'rnd_...'
$RenderService  = 'srv-...'
```

## Why not a named Cloudflare tunnel?

Named tunnels need either a custom domain in Cloudflare DNS or a Cloudflare
Zero Trust subscription. This operator has neither. Quick tunnels are free
and unlimited; the only cost is the URL changes on every restart, which
the update script papers over by pushing the new URL to Render
automatically.
