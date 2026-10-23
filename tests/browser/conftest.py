"""Session fixtures for the UI-PHYS Playwright nightly."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "ui_phys"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float = 90.0) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 — poll until up
            last = exc
        time.sleep(0.4)
    raise RuntimeError(f"server did not become ready at {url}: {last}")


@pytest.fixture(scope="session")
def ui_phys_catalog() -> dict:
    import json
    return json.loads((FIXTURE_DIR / "questions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def live_app(tmp_path_factory):
    """uvicorn + built React UI, stubbed LLM, fake embedder."""
    dist = REPO / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        pytest.skip(
            "frontend/dist is missing — run `npm --prefix frontend run build` "
            "before the Playwright nightly"
        )

    data_dir = tmp_path_factory.mktemp("ui-phys-data")
    port = _free_port()
    env = os.environ.copy()
    env.update({
        "ENV": "development",
        "DATA_DIR": str(data_dir),
        "PYTHONIOENCODING": "utf-8",
        "RAG_EMBEDDING_MODEL": "fake",
        "CEREBRUM_VIRGIN": "false",
        "CEREBRUM_DOMAIN_KITS": "construction",
        "CEREBRUM_UI_PHYS_STUB": "1",
        "PUBLIC_BASE_URL": "https://theshovel.ai",
        "PYTHONPATH": str(REPO),
    })
    # Pass the chat_stream env-key gate; the stub never dials a provider.
    from tests.browser._llm_stub import stub_provider_env
    env.update(stub_provider_env())
    # Dev registration auto-verifies; no email provider needed.
    env.pop("RESEND_API_KEY", None)

    boot = (
        "from tests.browser._llm_stub import install_stub; "
        "install_stub(); "
        "import uvicorn; "
        f"uvicorn.run('app.main:app', host='127.0.0.1', port={port}, log_level='warning')"
    )
    log_path = Path("/tmp/ui_phys_uvicorn.log")
    log_fh = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-c", boot],
        cwd=str(REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    try:
        try:
            _wait_http(f"http://127.0.0.1:{port}/v1/health")
        except RuntimeError:
            log_fh.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"uvicorn failed to start; log tail:\n{tail}") from None
        yield {"base": f"http://127.0.0.1:{port}", "port": port, "data_dir": data_dir}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    extra = dict(browser_context_args)
    extra["accept_downloads"] = True
    extra["viewport"] = {"width": 1440, "height": 900}
    return extra
