"""Tests for build_project_context in app/core/project_memory.py.

Phase C4 · Stream C.
"""

import pytest

from app.core import projects as store
from app.core.project_memory import build_project_context, build_memory_context


def test_build_project_context_includes_fact_and_document():
    """A project with a fact and a document should include both in the context."""
    proj = store.create_project("Context Test Project")
    pid = proj["id"]

    # Add a durable fact
    store.set_fact(pid, "contract_value", "9500000")

    # Add a document
    store.add_document(
        pid,
        original_name="baseline_schedule.xer",
        stored_as="baseline_schedule.xer",
        size=1024,
    )

    ctx = build_project_context(pid)

    # Facts block present
    assert "9500000" in ctx

    # Document listing present
    assert "baseline_schedule.xer" in ctx
    assert "type:" in ctx
    assert "role:" in ctx


def test_build_project_context_empty_for_bare_project():
    """A project with no facts and no documents should return empty string."""
    proj = store.create_project("Bare Context Project")
    ctx = build_project_context(proj["id"])
    assert ctx == ""


def test_build_project_context_facts_only():
    """A project with facts but no documents should still return a non-empty context."""
    proj = store.create_project("Facts Only Project")
    pid = proj["id"]
    store.set_fact(pid, "employer", "the client project Authority")

    ctx = build_project_context(pid)
    assert "the client project Authority" in ctx
    # No document section
    assert "Project documents" not in ctx


def test_build_project_context_documents_only():
    """A project with documents but no facts should still return a non-empty context."""
    proj = store.create_project("Docs Only Project")
    pid = proj["id"]
    store.add_document(pid, original_name="weekly_report.pdf", size=512)

    ctx = build_project_context(pid)
    assert "weekly_report.pdf" in ctx
    assert "Project documents" in ctx
    # No facts section
    assert "Known facts" not in ctx
