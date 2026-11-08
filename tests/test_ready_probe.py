"""The /ready watch pages only after two attempts, and the BLOCKER is status-only.

Two real status windows per behaviour. No network, no mail. The notify
callable is the stand-in for GitHub's workflow-failure email.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ready_probe import decide, evaluate, render_blocker
from tests.test_workflows_do_not_probe_onrender_or_health import (
    scheduled_probe_violations,
)

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([503, 200], "ok"),
        ([200, 503], "ok"),
        ([503], "retry"),
        ([500], "retry"),
        ([503, 503], "page"),
        ([502, 503], "page"),
        ([500, 502], "fail"),
        ([0, 0], "fail"),
    ],
)
def test_decide_two_attempt_window(statuses, expected):
    assert decide(statuses) == expected


@pytest.mark.parametrize("statuses", [[503, 503], [500, 503]])
def test_page_writes_sanitized_blocker_and_notifies(tmp_path, statuses):
    seen: list[dict] = []
    path = tmp_path / "BLOCKER.txt"
    code = evaluate(statuses, artifact=path, notify=seen.append)
    text = path.read_text(encoding="utf-8")
    assert code == 1
    assert seen == [
        {"channel": "workflow-failure-email", "http_status": 503, "attempts": 2}
    ]
    assert text.startswith("BLOCKER\n")
    assert "http_status=503" in text
    assert "page=workflow-failure-email" in text
    assert "@" not in text
    assert "Bearer" not in text
    assert all(ch.isalnum() or ch in "=/_,\n-" for ch in text)


@pytest.mark.parametrize("statuses", [[503, 200], [200]])
def test_success_does_not_page_or_write(tmp_path, statuses):
    seen: list[dict] = []
    path = tmp_path / "BLOCKER.txt"
    assert evaluate(statuses, artifact=path, notify=seen.append) == 0
    assert seen == []
    assert not path.exists()


@pytest.mark.parametrize("statuses", [[500, 502], [0, 0]])
def test_other_failure_is_red_without_blocker(tmp_path, statuses):
    seen: list[dict] = []
    path = tmp_path / "BLOCKER.txt"
    assert evaluate(statuses, artifact=path, notify=seen.append) == 1
    assert seen == []
    assert not path.exists()


@pytest.mark.parametrize("bad", ["Bearer sekret-value", "person@example.test"])
def test_blocker_rejects_non_numeric_status(bad):
    with pytest.raises(ValueError):
        render_blocker([bad])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "argv,exit_code,written",
    [
        (["--status", "503", "--status", "200"], 0, False),
        (["--status", "503", "--status", "503"], 1, True),
    ],
)
def test_cli_two_status_windows(tmp_path, capsys, argv, exit_code, written):
    from scripts.ready_probe import main

    path = tmp_path / "BLOCKER.txt"
    assert main([*argv, "--artifact", str(path)]) == exit_code
    err = capsys.readouterr().err
    assert path.exists() is written
    if written:
        assert "paging via the workflow failure email" in err
        assert "sekret" not in err
    else:
        assert err == ""


_ONE_SHOT = """
on:
  schedule:
    - cron: "0 * * * *"
jobs:
  probe:
    steps:
      - run: curl https://example.test/ready
"""

_CONTRACT = """
on:
  schedule:
    - cron: "*/15 * * * *"
jobs:
  readiness:
    steps:
      - run: |
          url="$BASE_URL/ready"
          for attempt in 1 2; do
            curl --proto '=https' -sS -o /dev/null "$url" || true
            [ "$attempt" = "1" ] && sleep 60
          done
          python3 scripts/ready_probe.py --status 1 --status 2 --artifact out
      - uses: actions/upload-artifact@v4
        with:
          name: BLOCKER
"""


@pytest.mark.parametrize(
    "text,ok",
    [
        (_ONE_SHOT, False),
        (_CONTRACT, True),
    ],
)
def test_scheduled_ready_contract_is_not_a_filename_exemption(text, ok):
    problems = scheduled_probe_violations(text)
    assert (problems == []) is ok


def test_health_watch_file_meets_the_contract_and_bans_health():
    text = (_ROOT / ".github" / "workflows" / "health-watch.yml").read_text(encoding="utf-8")
    assert scheduled_probe_violations(text) == []
    banned = "on:\n  schedule:\n    - cron: '0 * * * *'\njobs:\n  x:\n    run: curl /health\n"
    assert scheduled_probe_violations(banned)
