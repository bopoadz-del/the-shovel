"""look_ahead with an embedded synthetic programme must not ask which XER.

Live exit on 78bd9ca (FIXTURE-d with several uploaded .xer copies): look_ahead
routed correctly but _resolve_predefined_file_params returned the multi-file
ask-which message, so the tool never ran on the synthetic activity list in
the operator message. When the ask already carries durations / activity
codes, skip schedule_file auto-resolution (leave unset / do not ask-which).
"""
from __future__ import annotations
import pytest

from app.routers.chat import _resolve_predefined_file_params


SYNTHETIC_LOOK_AHEAD = (
    "Use the look_ahead tool on the SYNTHETIC activity list embedded in this "
    "message ONLY. Do NOT open uploaded XER files. "
    "Window starting 2026-09-19 for 14 days. Activities with durations (wd): "
    "A1 Mobilisation 2, A2 Temporary works 3, A3 Excavation 4, A4 Blinding 2, "
    "A5 Footings 5, A6 Ground slab 4. Show which activities fall in the window."
)



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

class _Doc(dict):
    pass


def test_synthetic_look_ahead_does_not_ask_which_of_many_xers(monkeypatch):
    docs = [
        {
            "original_name": "agent-d-valid-fixture.xer",
            "file_path": "/app/data/a.xer",
        },
        {
            "original_name": "fixture_programme_d.xer",
            "file_path": "/app/data/b.xer",
        },
    ]

    def _fake_list(_project_id):
        return docs

    monkeypatch.setattr(
        "app.core.projects.list_documents",
        _fake_list,
        raising=False,
    )
    # projects may be imported inside the resolver — patch the store module
    import app.core.projects as projects_store

    monkeypatch.setattr(projects_store, "list_documents", _fake_list)

    params, err = _resolve_predefined_file_params(
        "look_ahead",
        "0f1e2d3c",
        {},
        user_message=SYNTHETIC_LOOK_AHEAD,
    )
    assert err is None, err
    # Leave schedule_file unset so look_ahead uses the embedded programme.
    assert "schedule_file" not in params


def test_named_xer_still_resolves_when_user_names_file(monkeypatch):
    docs = [
        {"original_name": "alpha.xer", "file_path": "/app/data/alpha.xer"},
        {"original_name": "beta.xer", "file_path": "/app/data/beta.xer"},
    ]
    import app.core.projects as projects_store

    monkeypatch.setattr(projects_store, "list_documents", lambda _p: docs)
    params, err = _resolve_predefined_file_params(
        "look_ahead",
        "0f1e2d3c",
        {},
        user_message="Run look_ahead on alpha.xer for the next 14 days",
    )
    assert err is None
    assert params.get("schedule_file") == "/app/data/alpha.xer"
