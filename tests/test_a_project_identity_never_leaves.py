"""A project's identity and its file names never leave a turn (fresh #977).

Order 2210Z (LANE4 name flag) and 0735Z item 3a. Each test names the grader
finding on parked #977 it answers: one exit, a name split across stream
tokens, failing closed, no literal word lists, whole-word matching (units,
currencies and standards untouched), the set stored rather than re-derived
per turn, library titles kept, exact labels asserted.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib

import pytest

from app.core import project_identity as pi
from app.routers import identity_exit

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: The platform's vocabulary for these tests (stands in for the registries
#: and the library corpus, which the real is_generic reads).
GENERIC = {"tower", "development", "phase", "fidic", "red", "book", "sar", "bs", "boq", "rev", "pdf",
           "building", "contract", "general", "specification"}
PROJECT = {"name": "Harbour Tower Development Phase 2", "client": "Qalam Estates", "location": "Lusail",
           "profile": {"contract form": "FIDIC Red Book", "discipline": "building", "tender no": "QH-2291"}}
DOCS = [{"id": "a1b2c3d4e5", "original_name": "QHT_BOQ_Rev3.xlsx", "doc_type": "boq"},
        {"id": "f6e5d4c3b2", "original_name": "QHT-ST-0042 General Arrangement.pdf", "doc_type": "drawing"}]


@pytest.fixture(autouse=True)
def vocabulary(monkeypatch):
    monkeypatch.setattr(pi, "is_generic", lambda word: word.lower() in GENERIC)


def _redactor(closed=False):
    members = [] if closed else pi.derive(PROJECT, [d["original_name"] for d in DOCS])
    return pi.Redactor("p1234567", members, [] if closed else DOCS, closed=closed)


def test_the_set_is_derived_from_the_record_and_documents_without_word_lists():
    members = set(pi.derive(PROJECT, [d["original_name"] for d in DOCS]))
    assert {"Harbour Tower Development Phase 2", "Harbour", "Qalam Estates", "Lusail", "QH-2291",
            "QHT", "QE"} - members <= {"Harbour", "QE"}, members
    assert "QHT" in members and "Qalam Estates" in members and "QH-2291" in members
    assert "FIDIC Red Book" not in members and "building" not in members, "generic profile values are not identity"


def test_no_literal_word_list_in_the_module():
    tree = ast.parse((ROOT / "app" / "core" / "project_identity.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Call)) and isinstance(node, ast.Call):
            continue
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            words = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            assert len(words) < 4, f"a literal word list in project_identity.py: {words}"


def test_names_codes_and_file_names_are_replaced_whole_words_only():
    red = _redactor()
    out = red.text("Qalam Estates' Harbour Tower Development Phase 2 per QHT_BOQ_Rev3.xlsx, "
                   "drawing QHT-ST-0042 General Arrangement.pdf, tender QH-2291.")
    assert "Qalam" not in out and "Harbour" not in out and "QH-2291" not in out
    assert "Boq a1b2c3" in out and "Drawing f6e5d4" in out, out
    assert "PRJ-p12345" in out


@pytest.mark.parametrize("text", ["308.83 SAR/m3", "N/mm2", "BS 8110", "FIDIC 14.6", "the tower crane"])
def test_units_currencies_and_standards_pass_untouched(text):
    assert _redactor().text(text) == text


def test_a_word_containing_a_member_is_not_cut():
    red = pi.Redactor("p1", ["Lusail"])
    assert red.text("Lusailian") == "Lusailian"
    assert red.text("at Lusail.") == "at PRJ-p1."


def test_an_unreadable_set_fails_closed(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    from app.core import projects

    monkeypatch.setattr(projects, "SessionLocal", boom)
    red = pi.for_project("p1234567")
    assert red is not None and red.closed
    out = red.text("See QHT_BOQ_Rev3.xlsx from QHT, at 308.83 SAR/m3.")
    assert "QHT" not in out and ".xlsx" not in out and "SAR" in out, out


def test_a_turn_reads_the_stored_set_and_never_derives(monkeypatch):
    from app.core import projects

    calls = []
    monkeypatch.setattr(pi, "derive", lambda *a, **k: calls.append(1) or [])

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *_a, **_k):
            class _R:
                @staticmethod
                def scalar_one_or_none():
                    return json.dumps({"derived": ["Qalam Estates"], "added": []})
            return _R()

    monkeypatch.setattr(projects, "SessionLocal", _Session)
    monkeypatch.setattr(projects, "list_document_names", lambda pid: DOCS)
    red = pi.for_project("p1234567")
    assert not calls, "a turn re-derived the identity set"
    assert red.text("Qalam Estates") == "PRJ-p12345"


def test_library_documents_keep_their_titles():
    red = _redactor()
    assert red.text("FIDIC Red Book clause 14.6") == "FIDIC Red Book clause 14.6"


async def _drain(agen):
    return [f async for f in agen]


def _tok(text):
    return f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"


def _texts(frames):
    out = []
    for f in frames:
        body = json.loads(f.split("data:", 1)[1])
        out.append(body)
    return out


@pytest.mark.asyncio
async def test_a_name_split_across_stream_tokens_is_caught():
    src = [_tok("The client is Qalam"), _tok(" Est"), _tok("ates; see QHT_BO"), _tok("Q_Rev3.xlsx now."),
           f"data: {json.dumps({'type': 'end', 'content': 'Qalam Estates done', 'sources': [{'title': 'QHT_BOQ_Rev3.xlsx'}]})}\n\n"]
    out = _texts(await _drain(identity_exit.redact_frames(_gen(src), _redactor())))
    streamed = "".join(b["content"] for b in out if b["type"] == "token")
    assert "Qalam" not in streamed and "QHT" not in streamed and "Boq a1b2c3" in streamed, streamed
    end = out[-1]
    assert end["type"] == "end" and "Qalam" not in end["content"]
    assert end["sources"] == [{"title": "Boq a1b2c3"}]


@pytest.mark.asyncio
async def test_without_a_project_the_stream_is_untouched():
    src = [_tok("Qalam Estates")]
    assert await _drain(identity_exit.redact_frames(_gen(src), None)) == src


async def _gen(items):
    for item in items:
        await asyncio.sleep(0)
        yield item


def test_both_streaming_routes_go_through_the_one_exit():
    source = (ROOT / "app" / "routers" / "chat.py").read_text(encoding="utf-8")
    assert source.count("redact_frames(") == 2 and source.count("guarantee_terminal(") == 2
    for chunk in source.split("redact_frames(")[1:]:
        assert chunk.lstrip().startswith("guarantee_terminal("), "the exit must wrap the terminal guard"


def test_every_write_path_refreshes_the_stored_set():
    tree = ast.parse((ROOT / "app" / "core" / "projects.py").read_text(encoding="utf-8"))
    callers = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_identity_changed" for c in ast.walk(n))}
    assert {"create_project", "set_project_profile", "set_project_location", "add_document", "delete_document"} <= callers


def test_a_change_marks_the_set_stale_in_its_own_thread_keeping_added_members(monkeypatch):
    """CI (#1002 first head): a refresh thread started on add_document could
    be mid-lock when an ingest child forked, and the child hung. A change now
    only marks the stored set stale, in the writing thread."""
    from app.core import projects

    class _Row:
        identity = json.dumps({"derived": ["Qalam Estates"], "added": ["Site X"]})

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *_a):
            return _Row

        def commit(self):
            pass

    monkeypatch.setattr(projects, "SessionLocal", _Session)
    import threading

    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    projects._identity_changed("p1234567")
    assert not started, "a write path started a thread"
    assert json.loads(_Row.identity) == {"stale": True, "added": ["Site X"]}


def test_a_stale_set_is_derived_once_on_the_next_read(monkeypatch):
    from app.core import projects

    state = {"identity": json.dumps({"stale": True, "added": []})}
    derived = []

    def fake_refresh(pid):
        derived.append(pid)
        state["identity"] = json.dumps({"derived": ["Qalam Estates"], "added": []})
        return ["Qalam Estates"]

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *_a, **_k):
            class _R:
                @staticmethod
                def scalar_one_or_none():
                    return state["identity"]
            return _R()

    monkeypatch.setattr(projects, "SessionLocal", _Session)
    monkeypatch.setattr(projects, "list_document_names", lambda pid: DOCS)
    monkeypatch.setattr(pi, "refresh", fake_refresh)
    assert pi.for_project("p1234567").text("Qalam Estates") == "PRJ-p12345"
    assert pi.for_project("p1234567").text("Qalam Estates") == "PRJ-p12345"
    assert derived == ["p1234567"], "derived more than once after one change"
