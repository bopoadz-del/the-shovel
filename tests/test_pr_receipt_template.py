"""The PR template must exist at the GitHub-canonical path and carry RECEIPT."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
LEGACY = REPO_ROOT / ".github" / "pull_request_template.md"

RECEIPT_LABELS = (
    "tests passed/failed",
    "mutants run/survivors",
    "retries",
    "tools used",
    "rollback SHA",
    "deploy SHA",
)


def test_canonical_pull_request_template_exists():
    assert CANONICAL.is_file(), (
        "GitHub raw URL is case-sensitive; "
        ".github/PULL_REQUEST_TEMPLATE.md must exist"
    )


def test_receipt_section_has_exact_labels():
    text = CANONICAL.read_text(encoding="utf-8")
    assert "## RECEIPT" in text
    for label in RECEIPT_LABELS:
        assert f"- {label}:" in text, f"missing RECEIPT label: {label}"


def test_mutants_default_unproduced_and_inventing_is_forbidden():
    text = CANONICAL.read_text(encoding="utf-8")
    assert "- mutants run/survivors: UNPRODUCED" in text
    assert "Inventing mutant or survivor counts is forbidden" in text


def test_legacy_lowercase_template_matches_canonical():
    """pr-quality.yml still reads the lowercase path; keep both identical."""
    assert LEGACY.is_file()
    assert LEGACY.read_text(encoding="utf-8") == CANONICAL.read_text(
        encoding="utf-8"
    )
