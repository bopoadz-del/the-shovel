#!/usr/bin/env python3
"""Smoke driver for The Fork (Cerebrum Blocks).

Launches the FastAPI app on a local port, waits for it to become healthy,
exercises the core endpoints (health, landing page, block registry, a real
block execution), prints a PASS/FAIL summary, and tears the server down.

Run from the repo root:

    Windows:  .venv\\Scripts\\python.exe .claude\\skills\\run-the-fork\\driver.py
    POSIX:    .venv/bin/python      .claude/skills/run-the-fork/driver.py

Exit code 0 if every check passes, 1 otherwise. The driver launches and
stops its own uvicorn process, so nothing else may hold PORT (default 8000).
Set PORT to use another port.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# .claude/skills/run-the-fork/driver.py -> repo root is 3 levels up
REPO = Path(__file__).resolve().parents[3]
PORT = int(os.environ.get("PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
DEV_KEY = os.environ.get("CEREBRUM_DEV_KEY", "")  # the app accepts it only when ENV=development


def _req(method, path, body=None, auth=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", f"Bearer {DEV_KEY}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, r.read().decode("utf-8", "replace")


def main():
    venv_py = REPO / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")
    py = str(venv_py) if venv_py.exists() else sys.executable
    env = {**os.environ, "ENV": "development", "PYTHONIOENCODING": "utf-8",
           "DATA_DIR": str(REPO / "data")}
    (REPO / "data").mkdir(exist_ok=True)

    workers = os.environ.get("UVICORN_WORKERS", "1")
    print(f"[driver] launching uvicorn on :{PORT} workers={workers} (cwd={REPO}) ...")
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "warning", "--workers", workers],
        cwd=str(REPO), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):  # up to 30s
            if proc.poll() is not None:
                print("[driver] FAIL: server process exited during startup")
                return 1
            try:
                if _req("GET", "/v1/health")[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        else:
            print("[driver] FAIL: server did not become healthy in 30s")
            return 1

        checks = []

        st, body = _req("GET", "/v1/health")
        h = json.loads(body)
        checks.append(("GET /v1/health", st == 200 and h.get("status") == "healthy",
                       f"{h.get('blocks_loaded')} blocks loaded"))

        st, body = _req("GET", "/")
        checks.append(("GET /  (landing page)", st == 200 and "<html" in body.lower(),
                       f"HTTP {st}, {len(body)} bytes"))

        st, body = _req("GET", "/v1/blocks", auth=True)
        d = json.loads(body)
        blocks = d.get("blocks", d)
        checks.append(("GET /v1/blocks", st == 200 and len(blocks) > 0,
                       f"{len(blocks)} blocks"))

        # Smoke test of /v1/execute — uses `translate` (deterministic, no LLM,
        # no historical-benchmark dependency since that block was removed).
        st, body = _req("POST", "/v1/execute",
                        {"block": "translate",
                         "input": {"text": "hello"},
                         "params": {"target": "es"}}, auth=True)
        d = json.loads(body)
        ok = (st == 200 and d.get("status") == "success"
              and d.get("result", {}).get("status") == "success")
        translated = d.get("result", {}).get("translated", "")
        checks.append(("POST /v1/execute", ok, f"translate -> {translated!r}"))

        print()
        passed = sum(1 for _n, ok, _d in checks if ok)
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        print(f"\n[driver] {passed}/{len(checks)} checks passed")
        return 0 if passed == len(checks) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[driver] server stopped")


if __name__ == "__main__":
    sys.exit(main())
