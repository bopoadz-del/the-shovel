"""Phase-2 — deterministic file-param resolution for predefined dispatch.

An NL turn ("build the manpower histogram") routes to a container action but
carries no params; the calculators then honest-error for want of a file. These
tests cover the resolver that fills an action's FILE param from the project's
own uploaded documents (resource_histogram -> schedule_file from the project's
.xer), plus the honesty rules: caller-supplied wins, no match leaves the slot
empty (action honest-errors), 2+ matches -> ask-which, never silently picked.
"""

from __future__ import annotations

import json

import pytest

import app.routers.chat as chat



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

class TestResolvePredefinedFileParams:
    def _patch_docs(self, monkeypatch, docs):
        import app.core.projects as projects
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: docs)

    def test_single_xer_resolves_schedule_file(self, monkeypatch):
        self._patch_docs(monkeypatch, [
            {"original_name": "the client project Baseline.xer", "file_path": "/data/client.xer"}])
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram", "proj", {})
        assert err is None
        assert params["schedule_file"] == "/data/client.xer"

    def test_no_xer_leaves_slot_empty(self, monkeypatch):
        # No .xer -> slot stays unset so the ACTION returns its own honest error.
        self._patch_docs(monkeypatch, [
            {"original_name": "spec.pdf", "file_path": "/data/spec.pdf"}])
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram", "proj", {})
        assert err is None
        assert "schedule_file" not in params

    def test_two_xer_asks_which_never_picks(self, monkeypatch):
        self._patch_docs(monkeypatch, [
            {"original_name": "BaselineA.xer", "file_path": "/data/a.xer"},
            {"original_name": "BaselineB.xer", "file_path": "/data/b.xer"}])
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram", "proj", {})
        assert err is not None
        assert "BaselineA.xer" in err and "BaselineB.xer" in err
        # never silently resolved to one of them
        assert "schedule_file" not in params

    def test_two_xer_named_file_wins(self, monkeypatch):
        self._patch_docs(monkeypatch, [
            {"original_name": "baseline_programme.xer", "file_path": "/data/a.xer"},
            {"original_name": "resource_loaded.xer", "file_path": "/data/b.xer"}])
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram",
            "proj",
            {},
            "produce a manpower histogram from resource_loaded.xer",
        )
        assert err is None
        assert params["schedule_file"] == "/data/b.xer"

    def test_caller_supplied_param_wins(self, monkeypatch):
        self._patch_docs(monkeypatch, [
            {"original_name": "Baseline.xer", "file_path": "/data/proj.xer"}])
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram", "proj", {"schedule_file": "/user/given.xer"})
        assert err is None
        assert params["schedule_file"] == "/user/given.xer"

    def test_non_file_action_untouched(self, monkeypatch):
        self._patch_docs(monkeypatch, [
            {"original_name": "Baseline.xer", "file_path": "/data/proj.xer"}])
        params, err = chat._resolve_predefined_file_params(
            "esg_sustainability_report", "proj", {"foo": 1})
        assert err is None
        assert params == {"foo": 1}

    def test_no_project_id_noop(self, monkeypatch):
        params, err = chat._resolve_predefined_file_params(
            "resource_histogram", None, {})
        assert err is None
        assert params == {}


class TestStreamWiring:
    """The resolved file param must reach run_workflow's context, and an
    ask-which must be streamed as the answer (not fed to the action)."""

    @pytest.mark.asyncio
    async def test_resolved_schedule_file_reaches_run_workflow(self, monkeypatch):
        import app.core.projects as projects
        monkeypatch.setattr(projects, "get_project",
                            lambda pid, user_id=None, **kw: {"id": pid, "name": "P"})
        monkeypatch.setattr(projects, "list_documents",
                            lambda pid, **k: [{"original_name": "b.xer", "file_path": "/data/b.xer"}])

        captured = {}

        async def fake_run_workflow(action, context, session):
            captured["context"] = context
            return {"handled": True, "answer": "ok", "plan_steps": [action], "exports": []}

        import app.core.predefined_reasoning as pr
        monkeypatch.setattr(pr, "run_workflow", fake_run_workflow)

        async for _ in chat._stream_from_predefined(
            action="resource_histogram", user_message="build the histogram",
            project_id="proj", user_id="u", session_id="s",
            document_ids=[], params={}, emit_start=False,
        ):
            pass

        assert captured["context"]["params"]["schedule_file"] == "/data/b.xer"

    @pytest.mark.asyncio
    async def test_daily_site_report_location_from_project_metadata(self, monkeypatch):
        # F4: daily_site_report's weather location comes from the project's
        # confirmed metadata, never from the message.
        import app.core.projects as projects
        import app.core.predefined_reasoning as pr
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: [])
        captured = {}

        async def fake_run_workflow(action, context, session):
            captured["params"] = dict(context["params"])
            return {"handled": True, "answer": "ok", "plan_steps": [action], "exports": []}
        monkeypatch.setattr(pr, "run_workflow", fake_run_workflow)

        async def run(loc, caller_params):
            monkeypatch.setattr(projects, "get_project",
                                lambda pid, user_id=None, **kw: {"id": pid, "name": "P", "location": loc})
            captured.clear()
            async for _ in chat._stream_from_predefined(
                action="daily_site_report", user_message="daily site report",
                project_id="p1", user_id="u", session_id="s",
                document_ids=[], params=caller_params, emit_start=False):
                pass
            return captured["params"].get("location")

        assert await run("Riyadh", {}) == "Riyadh"          # from project metadata
        assert await run(None, {}) is None                   # unset -> honest, no inject
        assert await run("Riyadh", {"location": "Jeddah"}) == "Jeddah"  # caller wins

    @pytest.mark.asyncio
    async def test_non_daily_action_gets_no_location(self, monkeypatch):
        import app.core.projects as projects
        import app.core.predefined_reasoning as pr
        monkeypatch.setattr(projects, "get_project",
                            lambda pid, user_id=None, **kw: {"id": pid, "name": "P", "location": "Riyadh"})
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: [])
        captured = {}

        async def fake_run_workflow(action, context, session):
            captured["params"] = dict(context["params"])
            return {"handled": True, "answer": "ok"}
        monkeypatch.setattr(pr, "run_workflow", fake_run_workflow)

        async for _ in chat._stream_from_predefined(
            action="resource_histogram", user_message="x", project_id="p1",
            user_id="u", session_id="s", document_ids=[], params={}, emit_start=False):
            pass
        assert "location" not in captured["params"]

    @pytest.mark.asyncio
    async def test_two_xer_streams_ask_which_without_running_action(self, monkeypatch):
        import app.core.projects as projects
        monkeypatch.setattr(projects, "get_project",
                            lambda pid, user_id=None, **kw: {"id": pid, "name": "P"})
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: [
            {"original_name": "A.xer", "file_path": "/data/a.xer"},
            {"original_name": "B.xer", "file_path": "/data/b.xer"}])

        ran = {"called": False}

        async def fake_run_workflow(action, context, session):
            ran["called"] = True
            return {"handled": True, "answer": "should-not-run"}

        import app.core.predefined_reasoning as pr
        monkeypatch.setattr(pr, "run_workflow", fake_run_workflow)

        tokens = []
        async for evt in chat._stream_from_predefined(
            action="resource_histogram", user_message="build the histogram",
            project_id="proj", user_id="u", session_id="s",
            document_ids=[], params={}, emit_start=False,
        ):
            d = json.loads(evt[len("data: "):].strip())
            if d.get("type") == "token":
                tokens.append(d["content"])

        answer = "".join(tokens)
        assert "A.xer" in answer and "B.xer" in answer
        assert ran["called"] is False  # ask-which short-circuits before the action
