"""Tests for PR 25 — recursive Drive walker + training-scenario generator.

Both surfaces are integration-flavored; we mock the external boundaries
(Drive API, chat block) and assert the in-process behavior.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Chunk text is read from the vector store; the fake embedder lets the
    # scenario tests seed it without a model download.
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core.rag import vector_store as _vs
    _vs.reset_store_cache()
    from app.core import agent_memory as _am, projects as _proj
    if hasattr(_am, "_initialized"):
        _am._initialized = False
    if hasattr(_proj, "_initialized"):
        _proj._initialized = False
    yield tmp_path


# ── Recursive Drive walker ────────────────────────────────────────────────


def _fake_drive_tree(monkeypatch, tree: Dict[str, List[Dict]]):
    """Replace gdrive_service.list_folder_files with a stub backed by ``tree``.

    ``tree`` is a dict of folder_id → list of file metadata dicts. Each file
    dict has at least ``id``, ``name``, and ``mimeType``. Folders use
    mimeType ``application/vnd.google-apps.folder`` and their ``id`` is the
    key into ``tree`` for their children.
    """
    from app.core import gdrive_service as _gds

    def fake_list(folder_id, page_size=100):
        if folder_id not in tree:
            return [], f"not found: {folder_id}"
        return list(tree[folder_id]), None

    monkeypatch.setattr(_gds, "list_folder_files", fake_list)


def test_walk_folder_recursive_descends_subtree(isolated_data_dir, monkeypatch):
    """The walker pulls every file at every depth, not just the root level."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"
    tree = {
        "root": [
            {"id": "f1", "name": "readme.md", "mimeType": "text/markdown"},
            {"id": "sub1", "name": "200-Project Controls", "mimeType": FOLDER_MIME},
            {"id": "sub2", "name": "600-Procurement", "mimeType": FOLDER_MIME},
        ],
        "sub1": [
            {"id": "f2", "name": "risk_register.pdf", "mimeType": "application/pdf"},
            {"id": "deep1", "name": "2.3 Risk", "mimeType": FOLDER_MIME},
        ],
        "deep1": [
            {"id": "f3", "name": "2.3.1 Register.xlsx", "mimeType": "application/vnd.openxmlformats"},
        ],
        "sub2": [
            {"id": "f4", "name": "subcontract.docx", "mimeType": "application/msword"},
        ],
    }
    _fake_drive_tree(monkeypatch, tree)

    files, errors = _gds.walk_folder("root")
    assert errors == []
    ids = [f["id"] for f in files]
    assert sorted(ids) == ["f1", "f2", "f3", "f4"], (
        f"walker should hit every leaf file; got {ids}"
    )


def test_walk_folder_annotates_drive_path(isolated_data_dir, monkeypatch):
    """Each file gets a _drive_path so callers can attribute it to the SOP
    tree position. Joins names with forward slashes from the root."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"
    tree = {
        "root": [{"id": "sub", "name": "200-Project Controls", "mimeType": FOLDER_MIME}],
        "sub": [{"id": "deep", "name": "2.3 Risk", "mimeType": FOLDER_MIME}],
        "deep": [{"id": "f", "name": "register.pdf", "mimeType": "application/pdf"}],
    }
    _fake_drive_tree(monkeypatch, tree)

    files, _ = _gds.walk_folder("root")
    assert len(files) == 1
    assert files[0]["_drive_path"] == "200-Project Controls/2.3 Risk/register.pdf"


def test_walk_folder_respects_max_depth(isolated_data_dir, monkeypatch):
    """max_depth caps recursion and reports the cap in errors. Not a hard
    failure — partial results still ship."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"
    # 5 levels deep, walker capped at 2
    tree = {
        "L0": [{"id": "L1", "name": "lvl1", "mimeType": FOLDER_MIME}],
        "L1": [{"id": "L2", "name": "lvl2", "mimeType": FOLDER_MIME}],
        "L2": [{"id": "L3", "name": "lvl3", "mimeType": FOLDER_MIME}],
        "L3": [{"id": "f", "name": "deep.pdf", "mimeType": "application/pdf"}],
    }
    _fake_drive_tree(monkeypatch, tree)
    files, errors = _gds.walk_folder("L0", max_depth=2)
    # We descend L0 (d=0) → L1 (d=1) → L2 (d=2); at L3 (d=3) we stop
    assert any("max_depth" in e for e in errors)
    # No file collected because the file lives at L3
    assert files == []


def test_walk_folder_max_depth_reports_skipped_count(isolated_data_dir, monkeypatch):
    """PR #25 review fix #2: when max_depth trips, the error must include
    the COUNT of subfolders that went unexplored, so operators can tell
    whether one folder or fifty was skipped under that branch."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"
    # Three sibling subfolders under one branch, all past the cap
    tree = {
        "root": [{"id": "L1", "name": "deep_section", "mimeType": FOLDER_MIME}],
        "L1": [
            {"id": "L2a", "name": "a", "mimeType": FOLDER_MIME},
            {"id": "L2b", "name": "b", "mimeType": FOLDER_MIME},
            {"id": "L2c", "name": "c", "mimeType": FOLDER_MIME},
        ],
        # L2a/b/c each contain a folder that pushes beyond the cap
        "L2a": [{"id": "L3a", "name": "extra", "mimeType": FOLDER_MIME}],
        "L2b": [{"id": "L3b", "name": "extra", "mimeType": FOLDER_MIME}],
        "L2c": [{"id": "L3c", "name": "extra", "mimeType": FOLDER_MIME}],
    }
    _fake_drive_tree(monkeypatch, tree)

    _files, errors = _gds.walk_folder("root", max_depth=2)
    depth_errors = [e for e in errors if "max_depth" in e]
    assert depth_errors, "expected a depth-cap error"
    # The walker emits one tally PER truncated branch (preserving which
    # branch was hit, not just an opaque global count). Three branches
    # hit the cap → three errors, each naming its parent path and count.
    assert len(depth_errors) == 3, (
        f"expected one error per truncated branch; got: {depth_errors}"
    )
    # The actionable subfolder count appears in EACH message
    for err in depth_errors:
        assert "subfolder" in err and "skipped" in err, err
    # And the branch path is identifiable in each tally
    branches_named = {e for e in depth_errors if any(b in e for b in ("/a", "/b", "/c"))}
    assert len(branches_named) == 3, depth_errors


def test_walk_folder_emits_no_double_prefix(isolated_data_dir, monkeypatch):
    """PR #25 review fix #1: the walker prefixes its own errors with
    'gdrive walk(...):'. Callers (hydration) used to add another
    'gdrive walk:' on top, producing 'gdrive walk: gdrive walk(path): err'.
    This test pins the walker's contract — its errors must already carry
    the gdrive walk prefix so callers can pass through verbatim."""
    from app.core import gdrive_service as _gds

    def fake_list(folder_id, page_size=100):
        if folder_id == "root":
            return [], "Drive list returned 403: permission denied"
        return [], "unknown folder"

    monkeypatch.setattr(_gds, "list_folder_files", fake_list)
    _files, errors = _gds.walk_folder("root")
    assert any(e.startswith("gdrive walk(") for e in errors), errors
    # And NEVER doubles the prefix
    assert not any("gdrive walk: gdrive walk" in e for e in errors), errors


def test_walk_folder_continues_past_subtree_error(isolated_data_dir, monkeypatch):
    """A 403 on one branch doesn't abort the whole walk. The error is
    captured; other branches still yield their files."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"

    def fake_list(folder_id, page_size=100):
        if folder_id == "root":
            return [
                {"id": "ok", "name": "ok_branch", "mimeType": FOLDER_MIME},
                {"id": "denied", "name": "denied_branch", "mimeType": FOLDER_MIME},
            ], None
        if folder_id == "ok":
            return [{"id": "f", "name": "found.pdf", "mimeType": "application/pdf"}], None
        if folder_id == "denied":
            return [], "Drive list returned 403: permission denied"
        return [], "unknown folder"

    from app.core import gdrive_service as gds
    monkeypatch.setattr(gds, "list_folder_files", fake_list)

    files, errors = gds.walk_folder("root")
    assert len(files) == 1
    assert files[0]["id"] == "f"
    assert any("403" in e for e in errors), f"expected 403 in errors, got: {errors}"


def test_walk_folder_handles_cycles(isolated_data_dir, monkeypatch):
    """If Drive ever returns a cycle (shouldn't, but defensive code), the
    walker doesn't loop. visited-set guards each folder_id."""
    from app.core import gdrive_service as _gds

    FOLDER_MIME = "application/vnd.google-apps.folder"
    tree = {
        "a": [
            {"id": "b", "name": "B", "mimeType": FOLDER_MIME},
            {"id": "f1", "name": "in_a.pdf", "mimeType": "application/pdf"},
        ],
        "b": [
            {"id": "a", "name": "back_to_A", "mimeType": FOLDER_MIME},  # cycle
            {"id": "f2", "name": "in_b.pdf", "mimeType": "application/pdf"},
        ],
    }
    _fake_drive_tree(monkeypatch, tree)
    files, errors = _gds.walk_folder("a")
    ids = sorted(f["id"] for f in files)
    assert ids == ["f1", "f2"]  # both leaf files reached, no infinite loop


# ── Training scenario generator ───────────────────────────────────────────


def _seed_doc_index(project_id: str, doc_id: str, filename: str, chunks: List[str]):
    """Write an index entry (names the document) and its chunk text (the
    vector store, its only home) so iter_chunks_for_project finds both."""
    from app.core import doc_index
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    doc_index._upsert_index_entry(project_id, doc_id, "document", {
        "document_id": doc_id,
        "filename": filename,
        "fingerprint": "test",
        "chunk_count": len(chunks),
    })
    embedder = get_embedder()
    get_store(dim=embedder.dim).upsert_chunks(
        project_id, doc_id, chunks, embedder.encode(chunks),
    )


def test_iter_chunks_filters_short_chunks(isolated_data_dir):
    """Below the min_chars threshold → skipped. This keeps trivial chunks
    (file headers, single-word lines from OCR) out of the LLM input."""
    from scripts.generate_training_scenarios import iter_chunks_for_project

    long_chunk = "A" * 300
    short_chunk = "tiny"
    _seed_doc_index("p1", "doc1", "spec.pdf", [long_chunk, short_chunk, long_chunk])

    chunks = list(iter_chunks_for_project("p1", min_chars=200))
    assert len(chunks) == 2  # short_chunk filtered out
    for c in chunks:
        assert len(c["text"]) >= 200
        assert c["source"] == "spec.pdf"


def test_iter_chunks_respects_max_chunks(isolated_data_dir):
    """max_chunks cap stops the iterator. Lets operators sample large
    corpora without paying for full LLM passes during testing."""
    from scripts.generate_training_scenarios import iter_chunks_for_project

    long_chunks = ["A" * 300] * 50
    _seed_doc_index("p1", "doc1", "spec.pdf", long_chunks)
    chunks = list(iter_chunks_for_project("p1", min_chars=200, max_chunks=10))
    assert len(chunks) == 10


def test_iter_chunks_empty_project_yields_nothing(isolated_data_dir):
    """No index = no chunks. Doesn't raise."""
    from scripts.generate_training_scenarios import iter_chunks_for_project
    assert list(iter_chunks_for_project("nonexistent")) == []


@pytest.mark.asyncio
async def test_generate_for_chunk_parses_jsonl(isolated_data_dir, monkeypatch):
    """When the chat block returns well-formed JSONL, we extract the Q&A."""
    from scripts import generate_training_scenarios as _gen
    from app.blocks import BLOCK_REGISTRY

    # Stub the chat block — return a JSONL response shaped like what the
    # prompt asks for
    class _FakeChat:
        async def execute(self, *args, **kwargs):
            response = (
                '{"instruction": "what is the minimum slab cover in moderate exposure?", '
                '"response": "Per ACI 318, 30mm minimum for slabs in moderate exposure conditions."}\n'
                '{"instruction": "when is high-early-strength cement required?", '
                '"response": "When formwork must be stripped within 24-48 hours, typically for accelerated construction schedules."}'
            )
            return {
                "status": "success",
                "result": {"response": response, "provider": "deepseek"},
            }

    monkeypatch.setitem(BLOCK_REGISTRY, "chat", lambda: _FakeChat())

    chunk = {"text": "some excerpt about concrete cover and cement types", "source": "spec.pdf"}
    pairs = await _gen._generate_for_chunk(chunk, questions_per_chunk=2, provider_hint="any")

    assert len(pairs) == 2
    assert all("instruction" in p and "response" in p for p in pairs)
    assert "30mm" in pairs[0]["response"]
    assert all(p["source"] == "spec.pdf" for p in pairs)


@pytest.mark.asyncio
async def test_generate_for_chunk_rejects_offline_template(isolated_data_dir, monkeypatch):
    """When the chat block falls back to the offline template (no LLM),
    we get zero pairs rather than nonsense. Critical: keeps garbage out
    of the training set."""
    from scripts import generate_training_scenarios as _gen
    from app.blocks import BLOCK_REGISTRY

    class _OfflineChat:
        async def execute(self, *args, **kwargs):
            return {
                "status": "success",
                "result": {
                    "response": '{"instruction": "fake", "response": "fake response that is long enough"}',
                    "provider": "offline_template",  # ← this is the rejection signal
                },
            }
    monkeypatch.setitem(BLOCK_REGISTRY, "chat", lambda: _OfflineChat())

    pairs = await _gen._generate_for_chunk(
        {"text": "x", "source": "f"}, questions_per_chunk=1, provider_hint="any",
    )
    assert pairs == []


@pytest.mark.asyncio
async def test_generate_for_chunk_filters_refusals(isolated_data_dir, monkeypatch):
    """Filter rows with too-short instructions or responses. LLMs
    occasionally output partial JSON ({"instruction": "ok"}) — we drop
    those rather than write them out."""
    from scripts import generate_training_scenarios as _gen
    from app.blocks import BLOCK_REGISTRY

    class _PartialChat:
        async def execute(self, *args, **kwargs):
            return {
                "status": "success",
                "result": {
                    "response": (
                        '{"instruction": "tiny", "response": "also tiny"}\n'
                        '{"instruction": "this is a real and properly-worded question?", '
                        '"response": "this is a real, properly-detailed answer with enough words to count as a useful training row"}\n'
                    ),
                    "provider": "deepseek",
                },
            }
    monkeypatch.setitem(BLOCK_REGISTRY, "chat", lambda: _PartialChat())

    pairs = await _gen._generate_for_chunk(
        {"text": "x", "source": "f"}, questions_per_chunk=2, provider_hint="any",
    )
    assert len(pairs) == 1
    assert pairs[0]["response"].startswith("this is a real")


@pytest.mark.asyncio
async def test_generate_for_chunk_strips_code_fences(isolated_data_dir, monkeypatch):
    """LLMs often wrap JSONL output in ``` fences despite the prompt.
    We skip fence lines rather than failing to parse them."""
    from scripts import generate_training_scenarios as _gen
    from app.blocks import BLOCK_REGISTRY

    class _FencedChat:
        async def execute(self, *args, **kwargs):
            return {
                "status": "success",
                "result": {
                    "response": (
                        "```jsonl\n"
                        '{"instruction": "what materials need certificates of conformance?", '
                        '"response": "Structural steel, reinforcement bars, cement, and waterproofing membranes typically require CoCs per most project specs."}\n'
                        "```\n"
                    ),
                    "provider": "deepseek",
                },
            }
    monkeypatch.setitem(BLOCK_REGISTRY, "chat", lambda: _FencedChat())

    pairs = await _gen._generate_for_chunk(
        {"text": "x", "source": "f"}, questions_per_chunk=1, provider_hint="any",
    )
    assert len(pairs) == 1
    assert "certificates of conformance" in pairs[0]["instruction"]


@pytest.mark.asyncio
async def test_generate_provider_hint_filters(isolated_data_dir, monkeypatch):
    """When the operator requires --provider deepseek but the chat block
    returned a local fallback, we skip rather than mix providers in the
    training data."""
    from scripts import generate_training_scenarios as _gen
    from app.blocks import BLOCK_REGISTRY

    class _LocalChat:
        async def execute(self, *args, **kwargs):
            return {
                "status": "success",
                "result": {
                    "response": '{"instruction": "x?", "response": "y" * 50}',
                    "provider": "local_ollama",
                },
            }
    monkeypatch.setitem(BLOCK_REGISTRY, "chat", lambda: _LocalChat())

    pairs = await _gen._generate_for_chunk(
        {"text": "x", "source": "f"}, questions_per_chunk=1, provider_hint="deepseek",
    )
    assert pairs == []  # forced provider mismatch → skipped


def test_script_runs_end_to_end_smoke(isolated_data_dir):
    """Exit code 1 with empty index — the operator gets a clear "no chunks"
    error rather than an empty file masking misconfig."""
    out = isolated_data_dir / "out.jsonl"
    result = subprocess.run([
        sys.executable, "scripts/generate_training_scenarios.py",
        "--project-id", "no_such_project", "--out", str(out),
    ], capture_output=True, text=True, cwd=Path.cwd(),
       env={**os.environ, "DATA_DIR": str(isolated_data_dir)})
    assert result.returncode == 1
    assert "no chunks" in (result.stderr + result.stdout).lower()
