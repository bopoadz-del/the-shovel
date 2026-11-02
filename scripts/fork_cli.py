#!/usr/bin/env python3
"""fork_cli.py — thin terminal client for The Fork.

Pure client: hits the same /v1/agents/{name}/chat/stream endpoint the browser
does, so every turn still passes through the smart_orchestrator routing gate
(PR #78). Nothing is bypassed. What you get that the browser hides: every SSE
event (route / start / tool_call / tool_result / token / heartbeat / end /
error) printed raw with wall-clock deltas — client-side timing that doesn't
depend on Render's log levels.

Auth (either):
  --token <JWT>            or env FORK_TOKEN     (from /v1/users/login)
  --api-key <key>          or env FORK_API_KEY   (CEREBRUM_MASTER_KEY, or
                                                  the development key in dev)
  --email/--password       or env FORK_EMAIL / FORK_PASSWORD  (logs in for you)

Usage:
  python fork_cli.py health
  python fork_cli.py projects
  python fork_cli.py agents
  python fork_cli.py chat "generate a commissioning checklist for the MV substation" \
      --project <project_id> --events
  python fork_cli.py chat --repl --project <project_id>
  python fork_cli.py chat "..." --raw          # untouched SSE lines
  python fork_cli.py chat "..." --rag-debug    # A/B: with vs without RAG context

Only dependency: httpx  (pip install httpx)
"""

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")
DEFAULT_AGENT = os.getenv("FORK_AGENT", "project-assistant")

# Windows consoles default to cp1252; force utf-8 so event traces with unicode
# source content don't crash the client before the summary line is printed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── auth ─────────────────────────────────────────────────────────────────────

def _accepts(base: str, cred: str) -> bool:
    """True when the live API ACCEPTS this credential for the user-scoped calls
    the clients make (GET /v1/projects). A 401/403 means fall through to the
    next credential; a transport error is treated as not-accepted."""
    try:
        r = httpx.get(f"{base}/v1/projects",
                      headers={"Authorization": f"Bearer {cred}"}, timeout=15)
    except httpx.HTTPError as exc:
        print(f"[auth] probe transport error, treating as not-accepted: "
              f"{type(exc).__name__}", file=sys.stderr)
        return False
    return r.status_code not in (401, 403)


def _login(base: str, email: str, password: str) -> str | None:
    """A session token from /v1/users/login, or None when the login is rejected."""
    r = httpx.post(f"{base}/v1/users/login",
                   json={"email": email, "password": password}, timeout=30)
    if r.status_code in (401, 403):
        return None
    r.raise_for_status()
    return r.json().get("token")


def resolve_auth(args) -> str:
    """A Bearer credential the live API ACCEPTS, tried in precedence order and
    VERIFIED with one probe each (GET /v1/projects); a credential the API
    rejects (401/403) falls through to the next. A stale FORK_API_KEY therefore
    no longer wedges the clients when a valid FORK_EMAIL+FORK_PASSWORD is also
    set. Only the credential LABELS tried are ever named -- never the values."""
    base = args.base
    tried: list[str] = []
    token = args.token or os.getenv("FORK_TOKEN")
    if token:
        tried.append("FORK_TOKEN")
        if _accepts(base, token):
            return token
    api_key = args.api_key or os.getenv("FORK_API_KEY")
    if api_key:
        tried.append("FORK_API_KEY")
        if _accepts(base, api_key):
            return api_key
    email = args.email or os.getenv("FORK_EMAIL")
    password = args.password or os.getenv("FORK_PASSWORD")
    if email and password:
        tried.append("FORK_EMAIL+FORK_PASSWORD")
        cred = _login(base, email, password)
        if cred and _accepts(base, cred):
            print(f"[auth] logged in as {email}", file=sys.stderr)
            return cred
    sys.exit(
        f"::error::No accepted credential (tried: {', '.join(tried) or 'none'}). "
        "Set a valid FORK_TOKEN / FORK_API_KEY / FORK_EMAIL+FORK_PASSWORD "
        "(values are never shown)."
    )


def headers(cred: str) -> dict:
    return {"Authorization": f"Bearer {cred}"}


# ── simple GET commands ──────────────────────────────────────────────────────

def cmd_get(args, path: str):
    cred = resolve_auth(args)
    r = httpx.get(f"{args.base}{path}", headers=headers(cred), timeout=30)
    print(f"HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2)[:8000])
    except Exception:
        print(r.text[:2000])


# ── chat / SSE ───────────────────────────────────────────────────────────────

def fmt_delta(t0: float) -> str:
    return f"{time.monotonic() - t0:7.2f}s"


def stream_turn(client: httpx.Client, args, cred: str, message: str,
                conversation_id: str | None) -> str | None:
    """Run one chat turn, print events + timing, return conversation_id."""
    url = f"{args.base}/v1/agents/{args.agent}/chat/stream"
    if args.rag_debug:
        url += "?rag_debug=1"
    body = {"message": message}
    if args.project:
        body["project_id"] = args.project
        # workspace conversation binds history to the project like the UI does
        conversation_id = conversation_id or f"ws-{args.project}"
    if conversation_id:
        body["conversation_id"] = conversation_id
    if args.model:
        body["model"] = args.model

    t0 = time.monotonic()
    first_token_at = None
    tool_timings: list[tuple[str, float]] = []
    pending_tool: tuple[str, float] | None = None
    answer_parts: list[str] = []
    served_model: str | None = None  # from the `end` event — configured vs fallback
    n_events = 0

    with client.stream("POST", url, json=body, headers=headers(cred),
                       timeout=httpx.Timeout(10, read=600)) as r:
        if r.status_code != 200:
            r.read()
            print(f"HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
            print(f"turn summary: total={time.monotonic() - t0:.2f}s  "
                  f"first_token=-  events=0  answer_chars=0  model=?")
            return conversation_id
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            n_events += 1
            payload = line[5:].strip()
            if args.raw:
                print(f"[{fmt_delta(t0)}] {payload}")
                continue
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                print(f"[{fmt_delta(t0)}] ?? unparseable: {payload[:200]}")
                continue
            etype = evt.get("type", "?")

            if etype == "token":
                if first_token_at is None:
                    first_token_at = time.monotonic() - t0
                    if args.events:
                        print(f"[{fmt_delta(t0)}] FIRST TOKEN")
                answer_parts.append(evt.get("content", ""))
                if not args.events:
                    print(evt.get("content", ""), end="", flush=True)
            elif etype == "tool_call":
                name = evt.get("name") or evt.get("tool") or "?"
                pending_tool = (name, time.monotonic())
                argstr = json.dumps(evt.get("arguments") or evt.get("args") or {})
                print(f"\n[{fmt_delta(t0)}] TOOL_CALL   {name}  args={argstr[:300]}")
            elif etype == "tool_result":
                name = evt.get("name") or evt.get("tool") or "?"
                dur = None
                if pending_tool and pending_tool[0] == name:
                    dur = time.monotonic() - pending_tool[1]
                    tool_timings.append((name, dur))
                    pending_tool = None
                size = len(json.dumps(evt, default=str))
                status = evt.get("status", "?")
                dtxt = f" ({dur:.2f}s)" if dur is not None else ""
                print(f"[{fmt_delta(t0)}] TOOL_RESULT {name}{dtxt}  status={status}  ~{size}B")
            elif etype == "heartbeat":
                if args.events:
                    print(f"[{fmt_delta(t0)}] heartbeat")
            elif etype in ("route", "start", "sources", "end", "error"):
                if etype == "end":
                    served_model = evt.get("model") or served_model
                keep = {k: v for k, v in evt.items() if k != "type"}
                brief = json.dumps(keep, default=str)
                print(f"\n[{fmt_delta(t0)}] {etype.upper():11s} {brief[:600]}")
                if etype == "end":
                    break
                if etype == "error":
                    break
            else:
                print(f"[{fmt_delta(t0)}] {etype}: {json.dumps(evt, default=str)[:300]}")

    total = time.monotonic() - t0
    print(f"\n{'-'*60}")
    print(f"turn summary: total={total:.2f}s  "
          f"first_token={'-' if first_token_at is None else f'{first_token_at:.2f}s'}  "
          f"events={n_events}  answer_chars={sum(len(p) for p in answer_parts)}  "
          f"model={served_model or '?'}")
    for name, dur in tool_timings:
        print(f"  tool {name}: {dur:.2f}s")
    if first_token_at is None and total > 5:
        print("  !! no tokens streamed — this is the empty-bubble/hang shape")
    return conversation_id


def cmd_chat(args):
    cred = resolve_auth(args)
    conversation_id = args.conversation
    with httpx.Client() as client:
        if args.message:
            stream_turn(client, args, cred, args.message, conversation_id)
        if args.repl:
            print("REPL — Ctrl-D or 'exit' to quit.")
            while True:
                try:
                    msg = input("\nfork> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not msg or msg.lower() in ("exit", "quit"):
                    break
                conversation_id = stream_turn(client, args, cred, msg, conversation_id)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Thin CLI for The Fork")
    p.add_argument("--base", default=DEFAULT_BASE, help=f"API base URL (default {DEFAULT_BASE})")
    p.add_argument("--token", help="JWT from /v1/users/login")
    p.add_argument("--api-key", help="API key (CEREBRUM_MASTER_KEY / the development key from CEREBRUM_DEV_KEY)")
    p.add_argument("--email")
    p.add_argument("--password")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("projects")
    sub.add_parser("agents")

    c = sub.add_parser("chat", help="chat with full SSE event + timing dump")
    c.add_argument("message", nargs="?", help="one-shot message (omit with --repl)")
    c.add_argument("--agent", default=DEFAULT_AGENT)
    c.add_argument("--project", help="project_id (uses ws-{id} conversation like the UI)")
    c.add_argument("--conversation", help="explicit conversation_id")
    c.add_argument("--model", help="model override")
    c.add_argument("--events", action="store_true",
                   help="event-trace mode: show heartbeats/first-token marks, suppress token text")
    c.add_argument("--raw", action="store_true", help="dump raw SSE data lines")
    c.add_argument("--rag-debug", action="store_true", help="server A/B with/without RAG context")
    c.add_argument("--repl", action="store_true", help="interactive loop, keeps conversation")

    args = p.parse_args()
    if args.cmd == "health":
        cmd_get(args, "/v1/health")
    elif args.cmd == "projects":
        cmd_get(args, "/v1/projects")
    elif args.cmd == "agents":
        cmd_get(args, "/v1/agents")
    elif args.cmd == "chat":
        if not args.message and not args.repl:
            p.error("chat needs a message or --repl")
        cmd_chat(args)


if __name__ == "__main__":
    main()
