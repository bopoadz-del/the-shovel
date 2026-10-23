"""Scheduled workflows must not name the retired host or the /health route.

/health is the always-up liveness shape. The DB-free probe is /livez, and
the public host is https://theshovel.ai.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _workflow_hits() -> list[str]:
    hits: list[str] = []
    for path in sorted(_WORKFLOWS.iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "onrender.com" in line or "/health" in line:
                hits.append(f"{path.name}:{number}:{line.strip()}")
    return hits


def test_no_workflow_references_onrender_or_slash_health():
    hits = _workflow_hits()
    assert hits == [], "\n".join(hits)


def test_no_scheduled_workflow_probes_ready_or_health():
    """A schedule may call /livez only. /ready and /health wake the database."""
    hits: list[str] = []
    for path in sorted(_WORKFLOWS.iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if not any(line.strip() == "schedule:" for line in lines):
            continue
        for number, line in enumerate(lines, 1):
            if "/ready" in line or "/health" in line:
                hits.append(f"{path.name}:{number}:{line.strip()}")
    assert hits == [], "\n".join(hits)


def test_health_watch_probes_https_livez_without_following_redirects():
    text = (_WORKFLOWS / "health-watch.yml").read_text(encoding="utf-8")
    assert "https://theshovel.ai" in text
    assert "BASE_URL: https://theshovel.ai" in text
    assert "/livez" in text
    assert "/health" not in text
    assert "onrender.com" not in text
    assert 'cron: "*/15 * * * *"' in text
    assert 'cron: "7 */6 * * *"' not in text
    assert "/ready" not in text
    assert "-L" not in text
    assert "--proto '=https'" in text
    assert "$BASE_URL/livez" in text
    assert "$PUBLIC_URL/livez" in text
    assert "$BASE_URL/ready" not in text
