"""No function reads a project's documents or chunks without the session scope.

A chat session's file is that session's, not the project's
(``projects.document_in_scope``). Every function under ``app/`` and
``scripts/`` that reads the documents table or the chunk table filtered by a
project id must do one of:

* apply the scope clause (``_in_scope_clause`` / ``document_in_scope``);
* subtract the session-hidden set (``session_hidden_document_ids``,
  ``VectorStore._session_hidden_ids`` / ``_session_hidden_sql`` /
  ``_hidden_doc_ids`` / ``_hidden_doc_ids_among`` / ``_session_visible``);
* take or pass ``all_sessions`` (the explicit, visible opt-out);

or be named in ``EXEMPT`` below with a one-line reason. The walk is by
syntax tree, so a new reader that does none of these fails here by name.
``test_the_walk_flags_a_planted_unscoped_reader`` proves the walk is live.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A raw-SQL filter on a project id: ``project_id = :p``, ``project_id IN (``,
# ``project_id = ANY(:ids)``, ``project_id = %s``, ``project_id = ?``.
_SQL_PROJECT_FILTER = re.compile(
    r"project_id\s*(?:=|\bIN\b)\s*(?:ANY\s*\()?\s*[:%(?]", re.IGNORECASE)
_SQL_SELECT = re.compile(r"^\s*(?:WITH\b.*?)?SELECT\b", re.IGNORECASE | re.DOTALL)

SCOPE_MARKERS = frozenset({
    "_in_scope_clause", "document_in_scope", "session_hidden_document_ids",
    "_session_hidden_ids", "_session_hidden_sql", "_hidden_doc_ids",
    "_hidden_doc_ids_among", "_session_visible", "all_sessions",
})

# path::function -> why it may read every session's rows. Maintenance,
# operator and admin tools act on every stored row by design; the others are
# named below one by one.
EXEMPT = {
    # pass-through SQL builders: the caller builds the scope clause into ``vis``
    "app/core/rag/vector_store.py::_bm25_select": "builds SQL; callers pass the scope in vis",
    "app/core/rag/vector_store.py::_term_chunk_count": "one term's count; term_chunk_counts passes the scope in vis",
    # one named document, already resolved and access-checked by the caller
    "app/core/rag/vector_store.py::doc_chunk_texts": "reads the doc ids the caller named",
    "app/core/doc_typing_backfill.py::opening_samples": "reads the hit doc ids a session-scoped search returned",
    "app/core/chunk_bounds_plan.py::document_chunk_bounds": "one named document's chunk lengths",
    "app/core/projects.py::_project_chunk_total": "exclusion plan of one named document (maintenance)",
    # tables that hold no chat-session file
    "app/core/rag/vector_store.py::bm25_search_photos": "photo_chunks table, not documents or chunks",
    "app/core/project_identity.py::_in_library": "general-knowledge library projects only",
    # session's own listing: matches session AND uploader in the query
    "app/core/projects.py::list_session_documents": "is the session's own listing",
    # ingest de-duplication and supersede: must see every stored copy
    "app/core/projects.py::supersede_older_duplicates": "maintenance: retire older copies",
    "app/core/projects.py::audit_document_sizes": "maintenance: size audit of every row",
    # read-only maintenance plans and censuses (RQ-1..RQ-4, S10)
    "app/core/chunk_bounds_plan.py::plan_chunk_bounds": "maintenance plan over every chunk",
    "app/core/chunk_resplit.py::plan_resplit": "maintenance plan over every chunk",
    "app/core/doc_typing_backfill.py::content_samples": "maintenance: doc typing backfill",
    "app/core/doc_typing_backfill.py::plan_doc_typing": "maintenance: doc typing backfill",
    "app/core/doc_standing_backfill.py::plan_doc_standing": "maintenance: doc standing backfill",
    "app/core/noise_chunk_purge.py::plan_noise_chunks": "maintenance plan over every chunk",
    "app/core/ingest_reconcile.py::office_extraction_census": "maintenance: ingest census",
    "app/core/ingest_reconcile.py::coverage_truth": "maintenance: ingest coverage ledger",
    "app/core/ingest_reconcile.py::_load_p3_docs_and_chunk_counts": "maintenance: ingest reconcile",
    "app/core/ingest_reconcile.py::plan_legacy_guard_chunks": "maintenance plan over every chunk",
    # admin-only corpus endpoints
    "app/routers/admin.py::admin_corpus_collections": "admin corpus inventory",
    "app/routers/admin.py::admin_corpus_reconcile": "admin repair of misfiled chunks",
    # operator scripts
    "scripts/extraction_census.py::run_census": "operator census",
    "scripts/pdf_page_coverage.py::_stored_pages_sql": "operator diagnosis of named documents",
    "scripts/pdf_page_coverage.py::_pageless_sql": "operator census of one project's pageless chunks",
    "scripts/rag_cleanup_unsupported.py::cleanup": "operator cleanup",
    "scripts/rag_render_bulk_ingest.py::fetch_existing": "bulk ingest resume",
    "scripts/rag_render_bulk_ingest.py::count_chunks": "bulk ingest tally",
    "scripts/rag_render_bulk_ingest.py::count_documents": "bulk ingest tally",
    "scripts/rag_render_bulk_ingest.py::acceptance_report": "bulk ingest tally",
    "scripts/reconcile_drive_delta.py::_local_docs_by_drive_file_id": "Drive reconcile by file id",
    "scripts/stamp_ingest_status.py::main": "operator status backfill",
    "scripts/scrub_verify.py::occurrences": "operator scrub check of every row",
    "scripts/scrub_verify.py::repair": "operator scrub repair of every row",
    "scripts/drawings_pilot_report.py::fetch_first_chunk_and_titleblock": "operator report",
    "scripts/_build_drive_archive_eval_set.py::_load_chunk_text": "offline eval-set builder",
    "scripts/_probe_hybrid_vs_vector.py::main": "offline probe",
    "scripts/_probe_hybrid_vs_vector.py::doc_name": "offline probe",
    "scripts/_probe_retrieval_full.py::doc_name": "offline probe",
}


def _strings(fn: ast.AST) -> Iterator[str]:
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _identifiers(fn: ast.AST) -> Iterator[str]:
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg
        elif isinstance(node, ast.arg):
            yield node.arg


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _orm_project_filter(fn: ast.AST) -> bool:
    """``Document.project_id`` / ``<chunk class>.project_id``, or
    ``filter_by(project_id=...)``."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "project_id":
            owner = ast.unparse(node.value)
            if owner == "Document" or "chunk" in owner.lower():
                return True
        if (isinstance(node, ast.Call) and _call_name(node) == "filter_by"
                and any(k.arg == "project_id" for k in node.keywords)):
            return True
    return False


def _reads(fn: ast.AST) -> bool:
    if any(isinstance(n, ast.Call) and _call_name(n) in ("select", "query")
           for n in ast.walk(fn)):
        return True
    return any(_SQL_SELECT.match(s) for s in _strings(fn))


def unscoped_readers(root: Path, dirs=("app", "scripts")) -> List[str]:
    """``path::function`` of every reader that applies no session scope."""
    found: List[str] = []
    for top in dirs:
        for path in sorted((root / top).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(root).as_posix()
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                by_project = _orm_project_filter(fn) or any(
                    _SQL_PROJECT_FILTER.search(s) for s in _strings(fn))
                if not by_project or not _reads(fn):
                    continue
                if SCOPE_MARKERS & set(_identifiers(fn)):
                    continue
                found.append(f"{rel}::{fn.name}")
    return found


@pytest.mark.parametrize("top", ["app", "scripts"])
def test_every_project_reader_applies_the_session_scope(top):
    unscoped = [f for f in unscoped_readers(ROOT, dirs=(top,)) if f not in EXEMPT]
    assert unscoped == [], (
        "these functions read a project's documents or chunks and apply no "
        "session scope (see the module docstring): " + ", ".join(unscoped))


@pytest.mark.parametrize("top", ["app", "scripts"])
def test_every_exemption_still_names_an_unscoped_reader(top):
    """An exemption whose function is gone or now scoped is removed, so the
    list only shrinks and never hides a later reader of the same name."""
    found = set(unscoped_readers(ROOT, dirs=(top,)))
    stale = sorted(e for e in set(EXEMPT) - found if e.startswith(top + "/"))
    assert stale == [], stale


_PLANTED = '''
from sqlalchemy import select
from app.core.models import Document

def planted_reader(session, project_id):
    return session.execute(
        select(Document.id).where(Document.project_id == project_id)).all()

def planted_raw_reader(session, pid):
    return session.execute(
        "SELECT doc_id FROM chunks WHERE project_id = :pid", {"pid": pid}).all()

def planted_scoped_reader(session, project_id, scope):
    return session.execute(
        select(Document.id).where(Document.project_id == project_id)
        .where(_in_scope_clause(scope))).all()

def planted_maintenance(session, project_id, all_sessions=True):
    return session.execute(
        select(Document.id).where(Document.project_id == project_id)).all()
'''


@pytest.mark.parametrize("top", ["app", "scripts"])
def test_the_walk_flags_a_planted_unscoped_reader(tmp_path, top):
    for d in ("app", "scripts"):
        (tmp_path / d).mkdir()
    pkg = tmp_path / top / "planted"
    pkg.mkdir()
    (pkg / "readers.py").write_text(_PLANTED, encoding="utf-8")
    assert unscoped_readers(tmp_path) == [
        f"{top}/planted/readers.py::planted_reader",
        f"{top}/planted/readers.py::planted_raw_reader",
    ]


@pytest.mark.parametrize("reader", [
    "list_document_names", "documents_matching_filename_terms",
    "documents_matching_title_phrase", "compute_readiness",
])
def test_the_walk_flags_a_real_reader_with_its_scope_removed(tmp_path, reader):
    """Each real reader, its scope helpers swapped for look-alikes, is
    flagged; unchanged, it is not. The rule reads the product code as it is
    written."""
    src = (ROOT / "app" / "core" / "projects.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == reader)
    body = ast.get_source_segment(src, fn)
    stripped = re.sub(r"\b(_in_scope_clause|document_in_scope)\b", r"not\1", body)
    assert stripped != body
    (tmp_path / "app").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "app" / "projects.py").write_text(
        "from app.core.projects import *\n" + body, encoding="utf-8")
    (tmp_path / "app" / "stripped.py").write_text(
        "from app.core.projects import *\n" + stripped, encoding="utf-8")
    assert unscoped_readers(tmp_path) == [f"app/stripped.py::{reader}"]



def _takes_vis(fn: ast.AST) -> bool:
    """A pass-through SQL builder: it takes the caller's scope SQL as ``vis``."""
    a = fn.args
    return any(arg.arg == "vis" for arg in a.posonlyargs + a.args + a.kwonlyargs)


def unscoped_pass_through_callers(root: Path, dirs=("app", "scripts")) -> List[str]:
    """Callers of a pass-through builder (a function taking ``vis``) that
    neither build the scope nor take ``vis`` themselves: the pass-through
    exemptions above hold only while every caller up the chain scopes."""
    fns = []
    for top in dirs:
        for path in sorted((root / top).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(root).as_posix()
            fns += [(rel, fn) for fn in ast.walk(tree)
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))]
    pass_through = {fn.name for _rel, fn in fns if _takes_vis(fn)}
    found: List[str] = []
    for rel, fn in fns:
        if _takes_vis(fn):
            continue
        calls = {_call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        if calls & pass_through and not SCOPE_MARKERS & set(_identifiers(fn)):
            found.append(f"{rel}::{fn.name}")
    return found


@pytest.mark.parametrize("top", ["app", "scripts"])
def test_every_caller_of_a_pass_through_builder_scopes(top):
    assert unscoped_pass_through_callers(ROOT, dirs=(top,)) == []


@pytest.mark.parametrize("scoping", [
    "vis = self._session_hidden_sql(s, pid, 'c.doc_id', p)",
    "vis = self._hidden_sql(s, pid, all_sessions=True)",
])
def test_the_caller_check_flags_a_planted_unscoped_caller(tmp_path, scoping):
    (tmp_path / "app").mkdir()
    (tmp_path / "scripts").mkdir()
    planted = (
        "def _count(self, s, pid, term, vis):",
        "    return s.execute('SELECT 1 FROM c WHERE project_id = :p' + vis)",
        "def _relay(self, s, pid, term, vis):",
        "    return self._count(s, pid, term, vis)",
        "def leg(self, s, pid, term):",
        "    vis = self._hidden_doc_sql('c')",
        "    return self._relay(s, pid, term, vis)",
        "def scoped_leg(self, s, pid, term, p):",
        "    " + scoping,
        "    return self._relay(s, pid, term, vis)",
    )
    (tmp_path / "app" / "legs.py").write_text("\n".join(planted) + "\n", encoding="utf-8")
    assert unscoped_pass_through_callers(tmp_path) == ["app/legs.py::leg"]
