#!/usr/bin/env python3
"""Decide whether a /ready watch pages, and write a sanitized BLOCKER.

The watch takes two HTTP statuses. A 200 in that window is success. Fewer
than two statuses is a retry, not a page, so one blip does not alert. When
the window is full, contains no 200, and contains the readiness failure
status (503), this writes a BLOCKER artifact made only of fixed labels and
integers, then exits 1.

The page is that non-zero exit. GitHub emails repository watchers with its
existing workflow-failure notification. This module does not open a socket
and does not call a mail API.

Rollback: stop invoking this script from the readiness job (delete that job).
The liveness probes do not use it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ATTEMPTS = 2
READY_PAGE_STATUS = 503


def decide(
    statuses: list[int],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    page_status: int = READY_PAGE_STATUS,
) -> str:
    """Return ok, retry, page, or fail for a status window.

    ok: a 200 arrived inside the attempt budget.
    retry: the budget is not used up yet, and nothing has succeeded.
    page: the budget is used, nothing succeeded, and page_status occurred.
    fail: the budget is used, nothing succeeded, and page_status did not occur.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    window = [int(code) for code in statuses[:attempts]]
    if any(code == 200 for code in window):
        return "ok"
    if len(window) < attempts:
        return "retry"
    if int(page_status) in window:
        return "page"
    return "fail"


def render_blocker(
    statuses: list[int],
    *,
    page_status: int = READY_PAGE_STATUS,
) -> str:
    """Status-only BLOCKER text. Non-integers raise; nothing else is copied."""
    observed = ",".join(str(int(code)) for code in statuses)
    lines = (
        "BLOCKER",
        "probe=/ready",
        f"attempts={len(statuses)}",
        f"http_status={int(page_status)}",
        f"observed={observed}",
        "page=workflow-failure-email",
    )
    return "\n".join(lines) + "\n"


def workflow_failure_email(event: dict) -> None:
    """Stand-in for the workflow failure email. Does not send mail.

    The process exit is what GitHub turns into the watcher email. The line
    below is the annotation that email quotes. Only the integer status and
    the attempt count are included.
    """
    status = int(event["http_status"])
    attempts = int(event["attempts"])
    print(
        f"::error::/ready returned {status} across {attempts} attempts; "
        "paging via the workflow failure email",
        file=sys.stderr,
    )


def evaluate(
    statuses: list[int],
    *,
    artifact: Path | None = None,
    notify=workflow_failure_email,
    attempts: int = DEFAULT_ATTEMPTS,
    page_status: int = READY_PAGE_STATUS,
) -> int:
    """Exit 0 when the watch should stay green, else 1.

    A page writes the sanitized artifact before notify runs. ok and retry
    do not notify and do not write. fail is non-zero so the job is red,
    without the 503 BLOCKER record.
    """
    decision = decide(statuses, attempts=attempts, page_status=page_status)
    if decision in ("ok", "retry"):
        return 0
    if decision == "page":
        text = render_blocker(statuses[:attempts], page_status=page_status)
        if artifact is not None:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(text, encoding="utf-8")
        if notify is not None:
            notify(
                {
                    "channel": "workflow-failure-email",
                    "http_status": int(page_status),
                    "attempts": min(len(statuses), attempts),
                }
            )
        return 1
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Page a /ready watch from status codes")
    parser.add_argument("--status", type=int, action="append", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--page-status", type=int, default=READY_PAGE_STATUS)
    args = parser.parse_args(argv)
    return evaluate(
        args.status,
        artifact=args.artifact,
        attempts=args.attempts,
        page_status=args.page_status,
    )


if __name__ == "__main__":
    raise SystemExit(main())
