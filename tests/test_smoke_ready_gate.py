"""scripts/smoke.sh asks /ready before it sends any chat turn.

A build that answers /ready with 503 (no database, a blocked embedder, no
serving LLM) fails every turn for that one reason. The smoke must stop at
the gate and name it, and must not stop a ready build. Each test serves
its own /ready statuses from a local stub; nothing reaches a live host.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_SH = _ROOT / "scripts" / "smoke.sh"


def _bash() -> str:
    """Git's bash on Windows (System32's bash.exe is WSL), else PATH's."""
    git = shutil.which("git")
    if os.name == "nt" and git:
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("bash")
    assert found, "bash is required to run scripts/smoke.sh"
    return found


def run_smoke(ready_statuses: list[int]) -> tuple[int, str, list[str]]:
    """Run smoke.sh (one chat run) against a stub that answers /ready with
    ``ready_statuses`` in order and 404 everywhere else. Returns the exit
    code, the output and every path the stub was asked for."""
    statuses, seen = list(ready_statuses), []

    class Stub(BaseHTTPRequestHandler):
        def _answer(self):
            seen.append(self.path)
            code = statuses.pop(0) if self.path == "/ready" and statuses else 404
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ready": %s}' % (b"true" if code == 200 else b"false"))

        do_GET = do_POST = _answer

        def log_message(self, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "FORK_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
        "FORK_API_KEY": "stub-key",
        "SMOKE_RUNS": "1",
        "SMOKE_READY_RETRY_S": "0",
        "PYTHON": sys.executable,
    }
    for key in ("FORK_TOKEN", "FORK_EMAIL", "FORK_PASSWORD"):
        env.pop(key, None)
    try:
        proc = subprocess.run([_bash(), str(_SMOKE_SH), "stub_project"], env=env,
                              capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
    return proc.returncode, proc.stdout + proc.stderr, seen


@pytest.mark.parametrize("statuses,asked", [([503, 503], 2), ([500, 503], 2)])
def test_a_build_that_is_not_ready_stops_the_smoke_before_any_turn(statuses, asked):
    rc, out, seen = run_smoke(statuses)
    assert rc == 1, out
    assert "READY: FAIL" in out and "VERDICT: FAIL  (/ready not 200)" in out, out
    assert seen == ["/ready"] * asked, seen
    assert "run 1" not in out, out


@pytest.mark.parametrize("statuses,asked", [([200], 1), ([503, 200], 2)])
def test_a_ready_build_goes_on_to_the_chat_turns(statuses, asked):
    rc, out, seen = run_smoke(statuses)
    assert "READY: OK" in out and "READY: FAIL" not in out, out
    assert seen[:asked] == ["/ready"] * asked, seen
    # The turn reached the stub (which refuses it), so the smoke ran past the
    # gate; its own verdict, not the gate's, decides the exit.
    assert len(seen) > asked, seen
    assert "VERDICT: FAIL  (/ready not 200)" not in out, out
