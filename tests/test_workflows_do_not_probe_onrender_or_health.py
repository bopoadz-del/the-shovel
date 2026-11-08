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


_READY_CONTRACT = (
    "$BASE_URL/ready",
    "for attempt in 1 2",
    "sleep 60",
    "scripts/ready_probe.py",
    "--status",
    "--artifact",
    "--proto '=https'",
    "-o /dev/null",
    "name: BLOCKER",
    "actions/upload-artifact@v4",
)


def scheduled_probe_violations(text: str) -> list[str]:
    """Scheduled /health is banned. Scheduled /ready must be the two-attempt page.

    A /ready probe is the dependency check, so it is allowed only when the
    file shows two attempts, discards the body, asks the shared decider, and
    uploads the sanitized BLOCKER artifact. The filename is not an exemption.
    """
    lines = text.splitlines()
    if not any(line.strip() == "schedule:" for line in lines):
        return []
    problems: list[str] = []
    for number, line in enumerate(lines, 1):
        if "/health" in line or "onrender.com" in line:
            problems.append(f"{number}:{line.strip()}")
    if "/ready" in text and not all(part in text for part in _READY_CONTRACT):
        problems.append("scheduled /ready probe lacks the two-attempt 503 page contract")
    if "/ready" in text and "-L" in text:
        problems.append("scheduled /ready probe follows redirects")
    return problems


def test_no_scheduled_workflow_probes_ready_or_health():
    """/health stays banned. /ready is only the two-attempt 503 page."""
    hits: list[str] = []
    for path in sorted(_WORKFLOWS.iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        hits.extend(
            f"{path.name}:{item}"
            for item in scheduled_probe_violations(path.read_text(encoding="utf-8"))
        )
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
    assert "-L" not in text
    assert "--proto '=https'" in text
    assert "$BASE_URL/livez" in text
    assert "$PUBLIC_URL/livez" in text
    assert "$BASE_URL/ready" in text
    assert text.count("for attempt in 1 2") == 3
    assert text.count("sleep 60") == 3
    assert "-o /dev/null" in text
    assert "scripts/ready_probe.py" in text
    assert "name: BLOCKER" in text
    assert scheduled_probe_violations(text) == []
