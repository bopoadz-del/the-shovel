#!/usr/bin/env python
"""Production-shaped probe against Moonshot (Kimi K2).

WHY THIS EXISTS
---------------
The earlier ad-hoc canary POSTed a hand-built minimal payload to Moonshot and
PASSED — while production 400'd on every deliverable. The canary tested a
FRIENDLIER request than the real one: it omitted ``temperature`` (K2 rejects
anything but 1) and the real tool/system-prompt shape. A probe that is kinder
than production manufactures false confidence; it is worse than no probe.

This probe drives the REAL production payload. It instantiates the actual
``project-assistant`` Agent and calls ``Agent._call_llm`` exactly as
``chat_stream`` does, so every field prod sends is exercised through the same
code: the system prompt, the real tool definitions, the agent's own
``temperature`` (0.3), the outbound sanitiser, the per-provider payload shaping,
and ``tool_choice``. If the live config would 400 in production, it 400s here.

It forces ``LLM_PROVIDER=kimi`` and DISABLES the fallback
(``LLM_FALLBACK_PROVIDER=""``) so a failure surfaces as Kimi's real error rather
than being masked by an Ollama fallback success.

USAGE
-----
    KIMI_API_KEY=... python scripts/kimi_probe.py [--runs N]
    KIMI_API_KEY=... python scripts/kimi_probe.py --mode deliverable \
        --runs 3 --samples K2_QUALITY_SAMPLES.md

Modes:
  payload      (default) call _call_llm once per run with the real payload;
               assert HTTP 200 + a tool call, print served model + temperature.
               This is the config GATE — fast, no RAG, isolates the payload shape.
  deliverable  run the FULL turn via Agent.chat (tool loop + synthesis) to
               capture real deliverable prose; with --samples the verbatim output
               is appended to that file for a human to judge (Task 2e).

Exit code 0 iff every run passed (payload mode: 200 + tool call; deliverable
mode: a non-empty answer). Nonzero otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Make `app` importable when run as a standalone script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MESSAGE = "generate a commissioning checklist for the MV substation"
DEFAULT_PROJECT = "master_corpus"


def _prime_env(model: str | None) -> None:
    """Pin the provider to kimi and strip the fallback BEFORE importing runtime
    (so _llm_config reads these). No fallback => Kimi's real error is visible."""
    os.environ["LLM_PROVIDER"] = "kimi"
    os.environ["LLM_FALLBACK_PROVIDER"] = ""  # never mask a Kimi failure
    if model:
        os.environ["KIMI_MODEL"] = model


async def _run_payload(agent, api_key: str, message: str, project: str, with_tools: bool):
    """One _call_llm round-trip with the exact production payload shape."""
    messages = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": message},
    ]
    resp = await agent._call_llm(
        messages, api_key=api_key, project_id=project,
        with_tools=with_tools, user_id=None,
    )
    return resp


def _summarise(resp: dict) -> tuple[bool, str]:
    if resp.get("status") != "success":
        return False, f"ERROR {resp.get('error', '')[:300]}"
    msg = (resp.get("choice") or {}).get("message") or {}
    served = (resp.get("raw") or {}).get("model") or "?"
    tcs = [((tc.get("function") or {}).get("name")) for tc in (msg.get("tool_calls") or [])]
    chars = len(msg.get("content") or "")
    return True, f"200  model={served}  tool_calls={tcs or '-'}  content_chars={chars}"


async def main() -> int:
    ap = argparse.ArgumentParser(description="Production-shaped Kimi K2 probe.")
    ap.add_argument("--runs", type=int, default=3,
                    help="payload mode: number of round-trips. Ignored in "
                         "deliverable mode (one sample per --message).")
    ap.add_argument("--mode", choices=["payload", "deliverable"], default="payload")
    ap.add_argument("--message", nargs="+", default=[DEFAULT_MESSAGE],
                    help="deliverable mode: one sample is generated per message.")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--model", default=os.getenv("KIMI_MODEL"))
    ap.add_argument("--samples", help="deliverable mode: append verbatim output here")
    args = ap.parse_args()

    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        print("FAIL: KIMI_API_KEY not set in env.")
        return 2

    _prime_env(args.model)

    # Import AFTER _prime_env so _llm_config picks up LLM_PROVIDER=kimi.
    from app.agents.runtime import (  # noqa: E402
        load_agents, get_agent, _llm_config, _provider_temperature,
    )

    load_agents()
    agent = get_agent("project-assistant")
    if agent is None:
        print("FAIL: project-assistant agent not found.")
        return 2

    cfg = _llm_config()
    shaped = _provider_temperature(cfg, agent.temperature)
    print(f"provider={cfg['provider']}  model={cfg['default_model']}  "
          f"agent.temperature={agent.temperature} -> shaped_temperature={shaped}  "
          f"fallback={os.environ.get('LLM_FALLBACK_PROVIDER') or '(none)'}")
    # payload mode: repeat the first message --runs times. deliverable mode:
    # one sample per --message (so 3 prompts => 3 distinct samples for 2e).
    if args.mode == "payload":
        items = [(i, args.message[0]) for i in range(1, args.runs + 1)]
    else:
        items = list(enumerate(args.message, 1))
    total = len(items)
    print(f"mode={args.mode}  count={total}  project={args.project}")
    print("=" * 78)

    # Write the samples header up front and APPEND each sample as it completes,
    # so a mid-run kill preserves finished samples instead of losing them all.
    if args.mode == "deliverable" and args.samples:
        with open(args.samples, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Kimi K2 quality samples (Task 2e)\n\n"
                f"Provider={cfg['provider']} model={cfg['default_model']} "
                f"shaped_temperature={shaped}. Verbatim deliverable output for "
                f"human judgement — generated by scripts/kimi_probe.py "
                f"--mode deliverable.\n"
            )

    passed = 0
    samples: list[str] = []
    for i, message in items:
        if args.mode == "payload":
            # Test the tool-calling payload (the one that 400'd) AND the
            # forced-synthesis payload (with_tools=False) — both send temperature.
            ok_tool, line_tool = _summarise(await _run_payload(agent, api_key, message, args.project, True))
            ok_syn, line_syn = _summarise(await _run_payload(agent, api_key, message, args.project, False))
            # Pass requires: the tool payload accepted (200) AND it actually
            # called a tool, plus the synthesis payload accepted (200).
            run_ok = ok_tool and ok_syn and ("tool_calls=-" not in line_tool)
            if run_ok:
                passed += 1
            status = "OK  " if run_ok else "FAIL"
            print(f"run{i:>2}  {status}  tools:{line_tool}")
            print(f"        synth:{line_syn}")
        else:
            result = await agent.chat(
                user_message=message, history=[],
                project_id=args.project, conversation_id=None,
            )
            answer = (result or {}).get("answer") or ""
            n_tools = len((result or {}).get("tool_calls") or [])
            run_ok = bool(answer.strip())
            if run_ok:
                passed += 1
            status = "OK  " if run_ok else "FAIL"
            print(f"run{i:>2}  {status}  answer_chars={len(answer)}  tool_calls={n_tools}")
            block = (
                f"\n## Sample {i} (tool_calls={n_tools}, chars={len(answer)})\n"
                f"**Prompt:** {message}\n\n{answer}\n"
            )
            samples.append(block)
            if args.samples:
                with open(args.samples, "a", encoding="utf-8") as fh:
                    fh.write(block)

    print("=" * 78)
    verdict = "PASS" if passed == total else "FAIL"
    print(f"VERDICT: {verdict}  ({passed}/{total} ok)")
    if args.mode == "deliverable" and args.samples and samples:
        print(f"wrote {len(samples)} samples -> {args.samples}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
