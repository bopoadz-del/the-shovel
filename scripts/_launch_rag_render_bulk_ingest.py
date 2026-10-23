#!/usr/bin/env python3
"""Detached launcher for rag_render_bulk_ingest — avoids PowerShell stderr kills."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "bulk_render_ingest.log"
ERR = REPO / "bulk_render_ingest.err.log"
PID = REPO / "bulk_render_ingest.pid"


def main() -> int:
    if not os.environ.get("RENDER_API_KEY"):
        print("RENDER_API_KEY missing", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.setdefault("TORCH_NUM_THREADS", "12")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["RENDER_MCP_TOKEN"] = env.get("RENDER_MCP_TOKEN") or env["RENDER_API_KEY"]

    args = [
        str(REPO / ".venv" / "Scripts" / "python.exe"),
        "-u",
        str(REPO / "scripts" / "rag_render_bulk_ingest.py"),
        *sys.argv[1:],
    ]
    # append mode for babysitting long runs
    mode = "w"
    with LOG.open(mode, encoding="utf-8") as out, ERR.open(mode, encoding="utf-8") as err:
        proc = subprocess.Popen(
            args,
            cwd=str(REPO),
            env=env,
            stdout=out,
            stderr=err,
        )
        PID.write_text(str(proc.pid), encoding="utf-8")
        print(f"started pid={proc.pid} log={LOG}", flush=True)
        rc = proc.wait()
        with LOG.open("a", encoding="utf-8") as out2:
            out2.write(f"\nEXIT={rc}\n")
        print(f"exit={rc}", flush=True)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
