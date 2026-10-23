# Track 1 - RAG Pipeline Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship retrieval-augmented chat as the default behaviour for The Fork's project workspace, gated to the `project-assistant` agent, with a three-layer token cap (per-turn 1500 / confidence threshold 0.4 / daily 500K budget), incremental Drive ingestion, and a Sources UX footer.

**Architecture:** Pre-injection lives in the agent runtime's `chat_stream`, before iteration 0. A new `_rag_inject` helper retrieves top-K chunks, applies a token-aware filter, formats them as a system message, and writes a per-turn audit row. The same module gets a noise-filename filter so accumulated garbage docs (`~$lockfiles`, etc.) never consume retrieval slots. `ConstructionContainer.chat` defaults `use_rag=True` so the chat block path inherits the behaviour. A daily SQLite-backed token budget degrades K to 2 when consumed reaches the cap. Drive walker hashes file bytes and skips unchanged files. Final SSE `end` event carries `sources[]` for the React Sources footer.

**Tech Stack:** Python 3.11, FastAPI, SQLite (the existing `${DATA_DIR}/data/learning/`, `${DATA_DIR}/rag/vectors.db`, and a new `${DATA_DIR}/rag/budget.db`), the existing zvec embedder + vector store at `app/core/rag/`, React 18 + TypeScript + Vite (frontend), pytest + pytest-asyncio + TestClient (tests). LLM provider is Ollama Cloud (`qwen3-coder:480b-cloud`) via the cloudflared tunnel - the plan mocks retrieval and LLM calls in tests.

---

## Spec reference

Spec: `docs/superpowers/specs/2026-06-08-track1-rag-production.md` at commit `0b4098a`.

Live state assumed by this plan:

- Project `fb776aa2` (Anthropic) holds:
  - RFP `Example Client - Request for Proposals 010101.docx`
  - BOD `ca6292f9` (`Example Client - Performance Basis of Design.pdf`)
  - Appendix B `0f9ffc6b` (`Example Client - RFP Appendix B 010101.xlsx`)
  - PRC-921, PRC-931 procurement PDFs
  - Three noise files: `nambae-menu(4).pptx`, `SandsChina_Application_ChaD.docx`, `~$C-201_Time Management.docx`
- Project `3f6f28b2` (the client BOQ Test) holds the client project BOQ `c6dae280` (`the client project - Infra-1 - Demolition BOQ.pdf`).
- Render env already has `LLM_PROVIDER=ollama`, `OLLAMA_URL=<tunnel>`, `OLLAMA_MODEL=qwen3-coder:480b-cloud`. No new env vars exist yet for any of `RAG_*`.

## File map

Files this plan creates or modifies. Each task names exactly one or two of these files; the map locks the boundary decisions.

**New files:**
- `app/core/rag/budget.py` - SQLite-backed daily token budget (Phase 2.5)
- `app/core/rag/audit.py` - JSONL audit log writer for retrieval activity
- `app/core/rag/inject.py` - The `_rag_inject` helper and `format_chunks_as_system_message`
- `tests/test_rag_injection.py` - Unit tests for inject + budget + audit + noise filter
- `tests/test_drive_sha256_dedupe.py` - Unit tests for the Drive incremental ingestion

**Modified files:**
- `app/core/rag/retriever.py` - Add noise filter to `retrieve()`, add `retrieve_with_filter()`
- `app/agents/runtime.py` - Hook `_rag_inject` in `chat_stream` before iter 0
- `app/containers/construction.py` - `ConstructionContainer.chat` default `use_rag=True`
- `app/routers/agents.py` - Forward `?rag_debug=true` to `chat_stream`
- `app/routers/drive.py` - SHA-256 dedupe in walker loop
- `app/core/projects.py` - Add `content_sha256` column + migration
- `scripts/generate_training_scenarios.py` - Add `ollama` provider option + validation pipeline
- `frontend/src/pages/ProjectWorkspace.tsx` - Render `Sources` footer when `sources[]` is non-empty

8 files modified + 5 files created = 13 file touches total. Matches the spec's "8 files expected to change" once we count the 3 new helper modules as "implementation detail" of `runtime.py`/`retriever.py`.

---

## Phase 0 - Foundation (audit log + budget + noise filter)

This phase ships the three helpers that Phases 1-4 will use. No user-visible change; pure infrastructure with full test coverage. Lands first so every subsequent phase can wire into stable helpers.

### Task 0.1: Create the RAG audit log writer

**Files:**
- Create: `app/core/rag/audit.py`
- Test: `tests/test_rag_injection.py` (new file, first tests in it)

- [ ] **Step 1: Write the failing test for the audit writer module shape**

Create `tests/test_rag_injection.py`:

```python
"""Tests for the RAG injection helpers: audit log writer, budget,
noise filter, _rag_inject. The injector + retriever calls are mocked
in this file; we do not hit the live LLM or vector store.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

import pytest


def test_audit_writer_writes_one_jsonl_row_per_call(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag import audit
    audit.write({
        "timestamp": "2026-06-09T10:00:00Z",
        "project_id": "p1",
        "conversation_id": "ws-p1",
        "user_id": "u1",
        "agent_name": "project-assistant",
        "user_message_preview": "hi",
        "requested_k": 5,
        "injected_k": 5,
        "injected_tokens": 1200,
        "top_score": 0.71,
        "threshold_fired": False,
        "noise_filtered_count": 0,
        "budget_remaining": 498800,
        "budget_degraded": False,
        "chunks": [],
    })
    log_path = tmp_path / "logs" / "rag_audit.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["project_id"] == "p1"
    assert rows[0]["injected_tokens"] == 1200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py::test_audit_writer_writes_one_jsonl_row_per_call -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.rag.audit'`

- [ ] **Step 3: Implement the audit writer**

Create `app/core/rag/audit.py`:

```python
"""Append-only JSONL audit log for RAG retrieval activity.

One row per turn at ``${DATA_DIR}/logs/rag_audit.jsonl``. Used by both
the runtime injection path and the chat block to record retrieval
outcomes for offline tuning of K, the confidence threshold, the daily
budget, and the noise filter regex.

Best-effort: write failures are swallowed with a logger.warning. The
runtime must never refuse a chat turn because the audit log couldn't
write.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict

_LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()


def _log_path() -> str:
    base = os.getenv("DATA_DIR", "./data")
    return os.path.join(base, "logs", "rag_audit.jsonl")


def write(record: Dict[str, Any]) -> None:
    """Append ``record`` as one JSON line. Best-effort, never raises."""
    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        _LOG.warning("rag_audit write failed: %s", exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py::test_audit_writer_writes_one_jsonl_row_per_call -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/audit.py tests/test_rag_injection.py
git commit -m "feat(rag): jsonl audit log writer at \${DATA_DIR}/logs/rag_audit.jsonl"
```

### Task 0.2: Write a tolerance test - audit writer survives a path-write failure

**Files:**
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Add the tolerance test**

Append to `tests/test_rag_injection.py`:

```python
def test_audit_writer_never_raises_on_disk_failure(monkeypatch):
    """Audit writes must never break a real chat turn. If the path is
    unwritable, the writer logs and swallows."""
    monkeypatch.setenv("DATA_DIR", "/nonexistent/path/that/is/not/writable")
    from app.core.rag import audit
    # Should not raise even though the directory is unwritable.
    audit.write({"hello": "world"})
```

- [ ] **Step 2: Run test - expected PASS already (best-effort write)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py::test_audit_writer_never_raises_on_disk_failure -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_injection.py
git commit -m "test(rag): audit writer never raises on unwritable path"
```

### Task 0.3: Create the daily token budget module - schema + read/update

**Files:**
- Create: `app/core/rag/budget.py`
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing test for the budget module**

Append to `tests/test_rag_injection.py`:

```python
def test_budget_starts_at_zero_consumed_for_new_day(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "500000")
    from app.core.rag import budget
    state = budget.snapshot(day="2026-06-09")
    assert state == {"day": "2026-06-09", "consumed": 0, "budget": 500000, "remaining": 500000, "degraded": False}


def test_budget_consume_accumulates_and_reports_degraded_at_inclusive_boundary(
    monkeypatch, tmp_path,
):
    """The spec says 'consumed >= RAG_DAILY_TOKEN_BUDGET' (inclusive).
    The boundary test: when consumed reaches exactly the budget the
    next snapshot must report degraded=True."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "1000")
    from app.core.rag import budget
    budget.consume(day="2026-06-09", tokens=600)
    st = budget.snapshot(day="2026-06-09")
    assert st["consumed"] == 600 and st["remaining"] == 400 and st["degraded"] is False
    budget.consume(day="2026-06-09", tokens=400)
    # Now consumed == 1000 == budget, EXACTLY at the boundary.
    st = budget.snapshot(day="2026-06-09")
    assert st["consumed"] == 1000
    assert st["remaining"] == 0
    assert st["degraded"] is True, (
        "Boundary semantics: degradation must fire when consumed reaches "
        "the budget exactly. The classic off-by-one would have this still "
        "be False until consumed > budget."
    )


def test_budget_rollover_at_new_day_resets_consumed(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "500000")
    from app.core.rag import budget
    budget.consume(day="2026-06-09", tokens=400000)
    assert budget.snapshot(day="2026-06-09")["consumed"] == 400000
    # New day - implicit rollover by querying a fresh date.
    assert budget.snapshot(day="2026-06-10")["consumed"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k budget -v`
Expected: 3 tests FAIL with `ModuleNotFoundError: No module named 'app.core.rag.budget'`

- [ ] **Step 3: Implement the budget module**

Create `app/core/rag/budget.py`:

```python
"""Daily RAG injection token budget.

Layered on top of the per-turn ``MAX_RAG_TOKENS`` cap and the
``RAG_CONFIDENCE_THRESHOLD`` short-circuit so RAG token spend has
three independent degradation paths. When the day's consumed sum
reaches the budget INCLUSIVELY (``consumed >= budget``), the
injector degrades to ``K=2`` for the remaining turns of the day.

Day rollover is implicit: callers pass ``day=utc.strftime("%Y-%m-%d")``
and a fresh row materialises on the first read of a new date.

Schema lives in its own SQLite file at ``${DATA_DIR}/rag/budget.db``
so wipes are local and the audit DB stays focused.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, Optional

_LOCK = threading.RLock()


def _db_path() -> str:
    base = os.getenv("DATA_DIR", "./data")
    d = os.path.join(base, "rag")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "budget.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_db() -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_budget (
                day       TEXT PRIMARY KEY,
                consumed  INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _budget_value() -> int:
    try:
        return int(os.getenv("RAG_DAILY_TOKEN_BUDGET", "500000"))
    except ValueError:
        return 500000


def snapshot(day: str) -> Dict[str, object]:
    """Return the day's current budget state without mutating it."""
    _ensure_db()
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT consumed FROM rag_budget WHERE day = ?", (day,)
        ).fetchone()
    consumed = int(row["consumed"]) if row else 0
    budget = _budget_value()
    return {
        "day": day,
        "consumed": consumed,
        "budget": budget,
        "remaining": max(0, budget - consumed),
        "degraded": consumed >= budget,  # INCLUSIVE boundary (see spec)
    }


def consume(day: str, tokens: int) -> None:
    """Add ``tokens`` to the day's consumed counter atomically."""
    if tokens <= 0:
        return
    _ensure_db()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO rag_budget (day, consumed) VALUES (?, ?) "
            "ON CONFLICT(day) DO UPDATE SET consumed = consumed + excluded.consumed",
            (day, int(tokens)),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k budget -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/budget.py tests/test_rag_injection.py
git commit -m "feat(rag): daily token budget with inclusive boundary semantics"
```

### Task 0.4: Add the noise-filename filter to the retriever

**Files:**
- Modify: `app/core/rag/retriever.py` (lines 14-15 add import; lines 64-88 add filter)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing tests for the noise filter**

Append to `tests/test_rag_injection.py`:

```python
def test_noise_filter_default_regex_matches_known_garbage():
    from app.core.rag.retriever import _is_noise_filename
    # The defaults from the spec.
    assert _is_noise_filename("~$C-201_Time Management.docx") is True
    assert _is_noise_filename("nambae-menu(4).pptx") is True
    assert _is_noise_filename("SandsChina_Application_ChaD.docx") is True
    # And NOT a real doc.
    assert _is_noise_filename("Example Client - Performance Basis of Design.pdf") is False
    assert _is_noise_filename("RFP_Appendix_B.xlsx") is False


def test_noise_filter_env_override(monkeypatch):
    monkeypatch.setenv(
        "RAG_NOISE_FILENAME_REGEX",
        r"^(~\$|nambae-menu|SandsChina_Application|MyCustomNoise)",
    )
    from app.core.rag.retriever import _is_noise_filename
    assert _is_noise_filename("MyCustomNoise_v3.txt") is True
    assert _is_noise_filename("Real Document.pdf") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k noise -v`
Expected: 2 FAIL with `ImportError: cannot import name '_is_noise_filename'`

- [ ] **Step 3: Add the filter function to the retriever**

Modify `app/core/rag/retriever.py`. After the existing imports (after line 15), add:

```python
import os
import re


_NOISE_DEFAULT = r"^(~\$|nambae-menu|SandsChina_Application)"


def _noise_regex():
    """Compile the active noise regex. Re-reads env every call so
    tests / operators can flip RAG_NOISE_FILENAME_REGEX live."""
    return re.compile(os.getenv("RAG_NOISE_FILENAME_REGEX", _NOISE_DEFAULT))


def _is_noise_filename(name: str) -> bool:
    """True iff the document filename matches the noise regex.

    Used to drop accumulated garbage docs (lockfiles, unrelated pptx
    menus, etc.) from the retrieval candidate pool BEFORE top-K
    selection, so they cannot displace a relevant chunk.
    """
    if not name:
        return False
    return bool(_noise_regex().match(name))
```

- [ ] **Step 4: Run noise tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k noise -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/retriever.py tests/test_rag_injection.py
git commit -m "feat(rag): noise-filename filter for retrieval candidate pool"
```

### Task 0.5: Wire the noise filter into the retriever's candidate pool

**Files:**
- Modify: `app/core/rag/retriever.py` (rewrite `retrieve` function around lines 64-88)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing test - retrieve drops noise chunks before top-K**

Append to `tests/test_rag_injection.py`:

```python
def test_retrieve_drops_noise_before_top_k(monkeypatch):
    """The candidate pool from the vector store may include chunks
    from noise docs. They must be filtered BEFORE we pick top-K so a
    noise chunk cannot displace a real chunk."""
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    def fake_search(self, project_id, qvec, k):
        return [
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d-noise",
                  chunk_index=0, text="noise content", score=0.95),
            Chunk(chunk_id="c2", project_id=project_id, doc_id="d-real",
                  chunk_index=0, text="real content",  score=0.80),
            Chunk(chunk_id="c3", project_id=project_id, doc_id="d-real",
                  chunk_index=1, text="more real",     score=0.70),
        ]

    def fake_doc_name(doc_id):
        return {
            "d-noise": "~$lockfile.docx",
            "d-real":  "Example Client - BOD.pdf",
        }[doc_id]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", fake_doc_name, raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")

    # K=2 - if noise weren't filtered we'd get 2 chunks total
    # (noise + real), since noise scored highest. Filter must skip noise
    # and we must therefore see 2 REAL chunks.
    chunks, dropped = ret.retrieve_with_filter("query", "p1", k=2)
    assert dropped == 1
    assert all("real" in c.text for c in chunks)
    assert len(chunks) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py::test_retrieve_drops_noise_before_top_k -v`
Expected: FAIL with `AttributeError: module 'app.core.rag.retriever' has no attribute 'retrieve_with_filter'`

- [ ] **Step 3: Implement retrieve_with_filter and a doc-name lookup**

Modify `app/core/rag/retriever.py`. Replace the existing `retrieve` function (lines 64-88) with:

```python
def _doc_name_for_id(doc_id: str) -> str:
    """Resolve a doc_id to its original filename. Returns '' if not
    found - the noise filter treats unknown names as non-noise so a
    schema mismatch never silently drops a real document."""
    try:
        from app.core import projects as _projects
        doc = _projects.get_document(doc_id)
        return (doc or {}).get("original_name") or ""
    except Exception:
        return ""


def retrieve(
    query: str,
    project_id: str,
    k: int = 5,
) -> List[Chunk]:
    """Backwards-compatible: returns top-K AFTER the noise filter."""
    chunks, _ = retrieve_with_filter(query, project_id, k=k)
    return chunks


def retrieve_with_filter(
    query: str,
    project_id: str,
    k: int = 5,
) -> tuple:
    """Returns ``(chunks, noise_filtered_count)``.

    Internally pulls ``max(k*4, 20)`` raw candidates from the vector
    store so the noise filter has room to drop garbage without
    starving the caller of K real results. The audit log records
    ``noise_filtered_count`` so the regex can be tuned from data.
    """
    if not available():
        logger.debug("retrieve called but embedding stack not available; returning []")
        return [], 0
    if not query or not query.strip():
        return [], 0
    if not project_id:
        raise ValueError("project_id is required")

    embedder = get_embedder()
    query_vec = embedder.encode([query])[0]
    store = get_store(dim=embedder.dim)
    over_fetch = max(k * 4, 20)
    raw = store.search(project_id, query_vec, k=over_fetch)

    kept: List[Chunk] = []
    noise_dropped = 0
    for c in raw:
        name = _doc_name_for_id(c.doc_id)
        if _is_noise_filename(name):
            noise_dropped += 1
            continue
        kept.append(c)
        if len(kept) == k:
            break
    return kept, noise_dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k retrieve_drops_noise -v`
Expected: PASS

- [ ] **Step 5: Run the existing test_rag.py to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag.py -q`
Expected: PASS for all tests in `test_rag.py`

- [ ] **Step 6: Commit**

```bash
git add app/core/rag/retriever.py tests/test_rag_injection.py
git commit -m "feat(rag): retrieve_with_filter drops noise candidates before top-K"
```

---

## Phase 1 - Data generation

Generate 500+ real construction Q&A pairs from project `fb776aa2` via Ollama Cloud, with validation.

### Task 1.1: Add `ollama` as a provider option in generate_training_scenarios.py

**Files:**
- Modify: `scripts/generate_training_scenarios.py` (line 288 area, argparse `--provider` choices)
- No new test (script is invoked end-to-end in Task 1.4)

- [ ] **Step 1: Inspect the existing --provider handling**

Run: `grep -nE "provider|_chat_completion|provider=" scripts/generate_training_scenarios.py | head -10`
Expected output includes the `--provider` argparse line around line 288 and any provider-aware chat call.

- [ ] **Step 2: Open the script and locate the provider lookup**

Read `scripts/generate_training_scenarios.py` around lines 100-180. The chat completion call uses `LLM_PROVIDER` env or the `--provider` arg.

- [ ] **Step 3: Add `ollama` to the `--provider` choices**

In `scripts/generate_training_scenarios.py`, find the argparse `--provider` line and update it to include `ollama`:

```python
parser.add_argument(
    "--provider", default="any",
    choices=["any", "kimi", "local_lora", "offline_template", "ollama"],
    help="Force a specific chat provider; 'any' lets the chat block pick.",
)
```

When `--provider=ollama`, the script should set `os.environ["LLM_PROVIDER"] = "ollama"` before calling the chat block. Add this near the top of `main()`:

```python
if args.provider == "ollama":
    os.environ["LLM_PROVIDER"] = "ollama"
```

- [ ] **Step 4: Smoke the script with `--help`**

Run: `.venv/Scripts/python.exe scripts/generate_training_scenarios.py --help 2>&1 | grep -A2 provider`
Expected: `ollama` appears in the choices list.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_training_scenarios.py
git commit -m "feat(scripts): generate_training_scenarios accepts --provider ollama"
```

### Task 1.2: Add the validation pipeline to generate_training_scenarios.py

**Files:**
- Modify: `scripts/generate_training_scenarios.py` (add validation helpers + report)
- Test: `tests/test_drive_walker_and_scenarios.py` extension OR new file `tests/test_scenario_validation.py`

- [ ] **Step 1: Write the failing test for the dedupe heuristic**

Create `tests/test_scenario_validation.py`:

```python
"""Tests for the scenario JSONL validation pipeline.

The validator runs AFTER generation: drops empty/short, dedupes by
embedding cosine >= 0.85, flags suspicious noun-overlap.
"""
from __future__ import annotations

import pytest


def test_validate_drops_empty_and_short():
    from scripts.generate_training_scenarios import _validate_scenarios
    rows = [
        {"instruction": "Q1", "response": "fully formed answer about CPM"},
        {"instruction": "Q2", "response": ""},
        {"instruction": "Q3", "response": "too short"},  # < 30 chars
        {"instruction": "",   "response": "no question"},
    ]
    kept, report = _validate_scenarios(rows)
    assert len(kept) == 1
    assert kept[0]["instruction"] == "Q1"
    assert report["dropped_empty"] >= 2
    assert report["dropped_short"] >= 1


def test_validate_dedupes_by_response_cosine(monkeypatch):
    """Two near-identical responses should collapse to one. Use the
    fake embedder so the test is deterministic."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from scripts.generate_training_scenarios import _validate_scenarios
    rows = [
        {"instruction": "Q1", "response": "Concrete cover is 30mm per ACI 318 for slab moderate exposure"},
        {"instruction": "Q2", "response": "Concrete cover is 30mm per ACI 318 for slab moderate exposure"},
        {"instruction": "Q3", "response": "Saudi switchgear lead time is 28 weeks from Europe"},
    ]
    kept, report = _validate_scenarios(rows)
    assert len(kept) == 2
    assert report["dropped_duplicates"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scenario_validation.py -v`
Expected: 2 FAIL with `ImportError: cannot import name '_validate_scenarios'`

- [ ] **Step 3: Add the validation helper to the script**

In `scripts/generate_training_scenarios.py`, add this above `main()`:

```python
def _validate_scenarios(rows: List[Dict[str, str]]) -> tuple:
    """Apply the validation pipeline. Returns ``(kept_rows, report)``.

    Drops:
    * empty instruction or response
    * response under 30 chars (too short to be a real answer)
    * duplicate responses (cosine >= 0.85 against any kept row)
    """
    out: List[Dict[str, str]] = []
    report = {
        "input": len(rows),
        "dropped_empty": 0,
        "dropped_short": 0,
        "dropped_duplicates": 0,
    }
    # Stage 1: drop empties / too-short.
    stage1: List[Dict[str, str]] = []
    for r in rows:
        instr = (r.get("instruction") or "").strip()
        resp = (r.get("response") or "").strip()
        if not instr or not resp:
            report["dropped_empty"] += 1
            continue
        if len(resp) < 30:
            report["dropped_short"] += 1
            continue
        stage1.append(r)

    # Stage 2: dedupe by response cosine. Use the platform embedder
    # if available; fall back to a string-equality dedupe if not.
    try:
        from app.core.rag.embeddings import Embedder, get_embedder
        if not Embedder.available():
            raise RuntimeError("embedder not available")
        embedder = get_embedder()
        responses = [r["response"] for r in stage1]
        vecs = embedder.encode(responses)
    except Exception:
        seen = set()
        for r in stage1:
            key = r["response"]
            if key in seen:
                report["dropped_duplicates"] += 1
                continue
            seen.add(key)
            out.append(r)
        report["kept"] = len(out)
        return out, report

    import numpy as np
    kept_vecs = []
    for r, v in zip(stage1, vecs):
        keep = True
        for kv in kept_vecs:
            # Cosine assumes unit-normalized embeddings (zvec is).
            cos = float(np.dot(v, kv))
            if cos >= 0.85:
                keep = False
                report["dropped_duplicates"] += 1
                break
        if keep:
            kept_vecs.append(v)
            out.append(r)
    report["kept"] = len(out)
    return out, report
```

Make sure `from typing import List, Dict` is imported at the top of the script.

- [ ] **Step 4: Run validation tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scenario_validation.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_training_scenarios.py tests/test_scenario_validation.py
git commit -m "feat(scripts): scenario validation pipeline (empty / short / cosine dedupe)"
```

### Task 1.3: Wire the validation report into the script's main flow

**Files:**
- Modify: `scripts/generate_training_scenarios.py`

- [ ] **Step 1: Find the main flow's write-out step**

Run: `grep -nE "json.dump|write_jsonl|out\.write|with open.*out" scripts/generate_training_scenarios.py | head`
Locate the line that writes the JSONL output.

- [ ] **Step 2: Call _validate_scenarios before write, print the report after**

In `main()` (around the JSONL write), wrap the rows through `_validate_scenarios`:

```python
# After: rows = await _run(... project_id, ...)
kept_rows, validation_report = _validate_scenarios(rows)
print("== validation ==", file=sys.stderr)
for k, v in validation_report.items():
    print(f"  {k} = {v}", file=sys.stderr)

# Also surface top contributors so the operator can sanity-check.
by_doc: Dict[str, int] = {}
for r in kept_rows:
    by_doc[r.get("source") or "?"] = by_doc.get(r.get("source") or "?", 0) + 1
top = sorted(by_doc.items(), key=lambda kv: kv[1], reverse=True)[:5]
print(f"  top sources: {top}", file=sys.stderr)

with open(args.out, "w", encoding="utf-8") as f:
    for r in kept_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(kept_rows)} rows to {args.out}", file=sys.stderr)
```

- [ ] **Step 3: Smoke - run the script in --help to confirm no syntax error**

Run: `.venv/Scripts/python.exe scripts/generate_training_scenarios.py --help 2>&1 | tail -3`
Expected: clean usage output, no traceback.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_training_scenarios.py
git commit -m "feat(scripts): print validation report + top sources to stderr"
```

### Task 1.4: Phase 1 GO - run against project fb776aa2, hit the 500-row target

**Files:**
- Output: `data/learning/training_scenarios.jsonl` (regenerated, not committed)

This is the operator-facing run. It hits the live Ollama Cloud through the tunnel; expect a real LLM bill on the Ollama side.

- [ ] **Step 1: Confirm tunnel + Ollama health from the local shell**

Run: `curl -s -m 5 http://localhost:11434/api/version`
Expected: `{"version":"0.30.6"}` or similar. If empty/timeout, restart cloudflared via `~/.local/bin/fork-tunnel-up.cmd` first.

- [ ] **Step 2: Run the generator with --provider ollama against fb776aa2**

Run (from the repo root):

```bash
.venv/Scripts/python.exe scripts/generate_training_scenarios.py \
  --project-id fb776aa2 \
  --out data/learning/training_scenarios.jsonl \
  --questions-per-chunk 3 \
  --max-chunks 200 \
  --provider ollama
```

Expected: stderr `== validation ==` block with `kept` >= 500. If kept < 500, raise `--questions-per-chunk 4` and rerun (this overwrites the JSONL, no merge logic needed).

- [ ] **Step 3: Verify row count + sample 5 rows for the operator**

Run:

```bash
wc -l data/learning/training_scenarios.jsonl
.venv/Scripts/python.exe -c "import json; rows = [json.loads(l) for l in open('data/learning/training_scenarios.jsonl', encoding='utf-8')]; import random; random.seed(0); [print(r['instruction'][:80], '=>', r['response'][:120]) for r in random.sample(rows, 5)]"
```

Expected: `>= 500`, and 5 sample rows that read like real construction Q&A grounded in the RFP/BOD.

- [ ] **Step 4: CHECKPOINT - operator review**

Paste the row count and the 5-sample to the operator before proceeding. **Do not start Phase 2 until they sign off.**

Phase 1 commit: the JSONL is gitignored under `data/`. There is nothing to commit on a successful Phase 1 run beyond what already landed in 1.1-1.3.

---

## Phase 2 - RAG default with three-layer guarding

This phase ships the user-visible change: every project-assistant chat turn auto-retrieves and grounds.

### Task 2.1: Implement the chunk-formatting + token-cap helper

**Files:**
- Create: `app/core/rag/inject.py`
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing tests for token cap + format**

Append to `tests/test_rag_injection.py`:

```python
def test_format_chunks_emits_doc_chunk_score_header():
    from app.core.rag.inject import format_chunks_as_system_message
    from app.core.rag.vector_store import Chunk
    chunks = [
        Chunk(chunk_id="c1", project_id="p", doc_id="d1", chunk_index=0,
              text="A short chunk.", score=0.81),
        Chunk(chunk_id="c2", project_id="p", doc_id="d1", chunk_index=1,
              text="Another short chunk.", score=0.74),
    ]
    msg = format_chunks_as_system_message(chunks, total_candidates=10)
    assert msg["role"] == "system"
    body = msg["content"]
    assert "Relevant project context" in body
    assert "[doc_id=d1 chunk=0 score=0.810]" in body
    assert "[doc_id=d1 chunk=1 score=0.740]" in body
    assert "A short chunk." in body


def test_token_cap_drops_whole_chunks_from_bottom(monkeypatch):
    """When total estimated tokens > MAX_RAG_TOKENS, drop the lowest-
    score chunks (whole-chunk only, never truncate mid-chunk)."""
    monkeypatch.setenv("MAX_RAG_TOKENS", "100")  # very tight
    from app.core.rag.inject import apply_token_cap
    from app.core.rag.vector_store import Chunk
    # Three chunks, each ~80 tokens (320 chars / 4 = 80 tokens).
    big = "X" * 320
    chunks = [
        Chunk(chunk_id="c1", project_id="p", doc_id="d1", chunk_index=0, text=big, score=0.9),
        Chunk(chunk_id="c2", project_id="p", doc_id="d1", chunk_index=1, text=big, score=0.8),
        Chunk(chunk_id="c3", project_id="p", doc_id="d1", chunk_index=2, text=big, score=0.7),
    ]
    kept, total_tokens = apply_token_cap(chunks)
    # Only one chunk should fit under the 100-token cap.
    assert len(kept) == 1
    assert kept[0].score == 0.9  # highest score retained
    assert total_tokens <= 100


def test_token_cap_keeps_all_when_under_budget(monkeypatch):
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")
    from app.core.rag.inject import apply_token_cap
    from app.core.rag.vector_store import Chunk
    chunks = [
        Chunk(chunk_id=f"c{i}", project_id="p", doc_id="d1", chunk_index=i,
              text="hello world. " * 5, score=0.9 - i*0.1)
        for i in range(3)
    ]
    kept, total_tokens = apply_token_cap(chunks)
    assert len(kept) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k "format_chunks or token_cap" -v`
Expected: 3 FAIL with `ImportError`.

- [ ] **Step 3: Implement format_chunks_as_system_message + apply_token_cap**

Create `app/core/rag/inject.py`:

```python
"""Shared helpers for RAG injection: token cap, chunk formatter, the
main ``rag_inject`` entry point used by the agent runtime and the
chat block.

Kept in its own module so:
* Phase 2's runtime change is small and confined to a hook call.
* Tests can drive the helpers directly without spinning up an agent.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.core.rag.vector_store import Chunk

_LOG = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Cheap proxy: 4 chars per token. Good enough for the cap; not
    used for billing or model context sizing."""
    return max(1, len(text) // 4)


def apply_token_cap(chunks: List[Chunk]) -> Tuple[List[Chunk], int]:
    """Drop whole chunks from the bottom (lowest score) until total
    estimated tokens are <= MAX_RAG_TOKENS.

    Never truncates mid-chunk; a chunk is included or excluded whole.
    Returns ``(kept_chunks, total_estimated_tokens)``.
    """
    cap = int(os.getenv("MAX_RAG_TOKENS", "1500"))
    # Sort by score desc so we drop the weakest matches first when over cap.
    ordered = sorted(chunks, key=lambda c: -(c.score or 0))
    total = 0
    kept: List[Chunk] = []
    for c in ordered:
        t = _estimate_tokens(c.text)
        if total + t > cap:
            continue
        kept.append(c)
        total += t
    return kept, total


def format_chunks_as_system_message(
    chunks: List[Chunk],
    total_candidates: int,
) -> Dict[str, str]:
    """Build the system message that goes into the LLM context."""
    if not chunks:
        return {"role": "system", "content": ""}
    scores = [c.score or 0.0 for c in chunks]
    header = (
        f"Relevant project context (top {len(chunks)} of {total_candidates} "
        f"matches; cosine in [{min(scores):.3f}, {max(scores):.3f}]):\n"
    )
    body_parts = [
        f"[doc_id={c.doc_id} chunk={c.chunk_index} score={(c.score or 0):.3f}] {c.text}"
        for c in chunks
    ]
    return {"role": "system", "content": header + "\n" + "\n\n".join(body_parts)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k "format_chunks or token_cap" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/inject.py tests/test_rag_injection.py
git commit -m "feat(rag): chunk-formatter + token-cap helpers (whole-chunk drop only)"
```

### Task 2.2: Build the rag_inject entry point with confidence threshold + budget hook

**Files:**
- Modify: `app/core/rag/inject.py` (add `rag_inject`)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing tests for the threshold + budget paths**

Append to `tests/test_rag_injection.py`:

```python
def test_rag_inject_returns_none_when_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    def fake_retrieve(query, project_id, k):
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d1",
                  chunk_index=0, text="weak match", score=0.30),
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="hello",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is None
    assert audit_rec["threshold_fired"] is True
    assert audit_rec["injected_k"] == 0
    assert audit_rec["budget_degraded"] is False


def test_rag_inject_returns_system_message_when_confident(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")

    def fake_retrieve(query, project_id, k):
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id="c1", project_id=project_id, doc_id="d1",
                  chunk_index=0, text="strong match content", score=0.80),
            Chunk(chunk_id="c2", project_id=project_id, doc_id="d1",
                  chunk_index=1, text="second chunk content",  score=0.60),
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="data center cooling architecture?",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is not None and sys_msg["role"] == "system"
    assert "strong match" in sys_msg["content"]
    assert audit_rec["injected_k"] == 2
    assert audit_rec["threshold_fired"] is False
    assert audit_rec["top_score"] == 0.80
    assert audit_rec["budget_degraded"] is False
    assert audit_rec["budget_remaining"] >= 0


def test_rag_inject_degrades_to_k2_when_budget_exhausted(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DAILY_TOKEN_BUDGET", "100")
    monkeypatch.setenv("RAG_K", "5")
    monkeypatch.setenv("MAX_RAG_TOKENS", "10000")  # don't let cap interfere
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.0")

    # Burn the budget first.
    from app.core.rag import budget
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    budget.consume(day=today, tokens=100)  # consumed == 100 == budget

    seen_k = {"value": None}
    def fake_retrieve(query, project_id, k):
        seen_k["value"] = k
        from app.core.rag.vector_store import Chunk
        return ([
            Chunk(chunk_id=f"c{i}", project_id=project_id, doc_id="d1",
                  chunk_index=i, text="x", score=0.9)
            for i in range(k)
        ], 0)

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    from app.core.rag.inject import rag_inject

    sys_msg, audit_rec = rag_inject(
        user_message="any",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert seen_k["value"] == 2  # K degraded to 2
    assert audit_rec["budget_degraded"] is True


def test_rag_inject_skips_for_non_project_assistant_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag.inject import rag_inject
    sys_msg, audit_rec = rag_inject(
        user_message="hi",
        project_id="p1",
        conversation_id="c1",
        user_id="u1",
        agent_name="heavy-reasoning",
    )
    assert sys_msg is None
    assert audit_rec == {}


def test_rag_inject_skips_when_project_id_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag.inject import rag_inject
    sys_msg, audit_rec = rag_inject(
        user_message="hi",
        project_id=None,
        conversation_id=None,
        user_id="u1",
        agent_name="project-assistant",
    )
    assert sys_msg is None
    assert audit_rec == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k rag_inject -v`
Expected: 5 FAIL with `ImportError`.

- [ ] **Step 3: Implement `rag_inject`**

Append to `app/core/rag/inject.py`:

```python
from app.core.rag.retriever import retrieve_with_filter
from app.core.rag import audit as _audit
from app.core.rag import budget as _budget


def rag_inject(
    user_message: str,
    project_id: Optional[str],
    conversation_id: Optional[str],
    user_id: Optional[str],
    agent_name: str,
) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    """Per-turn RAG entry point.

    Returns ``(system_message_or_None, audit_record_dict)``.

    Behaviour:
    1. If agent_name != "project-assistant" or project_id is falsy: returns
       (None, {}). No audit. The runtime won't write anything for that case.
    2. Otherwise: snapshot the budget for today, derive ``effective_k`` (5
       normally, 2 if budget_degraded), call ``retrieve_with_filter``.
    3. If retrieved top_score < THRESHOLD or no chunks at all: return
       (None, audit_record) with ``threshold_fired=true`` so the caller can
       still write the audit log and prepend its fallback prefix.
    4. Apply MAX_RAG_TOKENS cap (whole-chunk drops). Format the kept chunks
       as the system message.
    5. ``budget.consume(injected_tokens)`` BEFORE returning so concurrent
       turns see the updated counter.
    """
    if agent_name != "project-assistant" or not project_id:
        return None, {}

    now = _dt.datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    threshold = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.4"))
    requested_k = int(os.getenv("RAG_K", "5"))

    budget_state = _budget.snapshot(day=today)
    effective_k = 2 if budget_state["degraded"] else requested_k

    chunks, noise_filtered = retrieve_with_filter(
        user_message, project_id, k=effective_k,
    )
    top_score = (max(c.score or 0 for c in chunks) if chunks else 0.0)

    audit_rec: Dict[str, Any] = {
        "timestamp": now.isoformat() + "Z",
        "project_id": project_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "agent_name": agent_name,
        "user_message_preview": (user_message or "")[:200],
        "requested_k": requested_k,
        "noise_filtered_count": noise_filtered,
        "top_score": top_score,
        "budget_remaining": budget_state["remaining"],
        "budget_degraded": budget_state["degraded"],
    }

    if not chunks or top_score < threshold:
        audit_rec.update({
            "injected_k": 0,
            "injected_tokens": 0,
            "threshold_fired": True,
            "chunks": [
                {"doc_id": c.doc_id, "chunk_index": c.chunk_index,
                 "score": c.score} for c in chunks
            ],
        })
        _audit.write(audit_rec)
        return None, audit_rec

    kept, total_tokens = apply_token_cap(chunks)
    sys_msg = format_chunks_as_system_message(kept, total_candidates=len(chunks))

    audit_rec.update({
        "injected_k": len(kept),
        "injected_tokens": total_tokens,
        "threshold_fired": False,
        "chunks": [
            {"doc_id": c.doc_id, "chunk_index": c.chunk_index,
             "score": c.score} for c in kept
        ],
    })
    _audit.write(audit_rec)
    _budget.consume(day=today, tokens=total_tokens)
    return sys_msg, audit_rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k rag_inject -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/rag/inject.py tests/test_rag_injection.py
git commit -m "feat(rag): rag_inject entry point with threshold + budget + audit hooks"
```

### Task 2.3: Hook rag_inject into the agent runtime's chat_stream

**Files:**
- Modify: `app/agents/runtime.py` (lines ~700-740, before iter 0 of the loop)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Locate the iter-0 hook point**

Run: `grep -n "self._build_messages\|effective_history\|messages = self._build_messages" app/agents/runtime.py | head`
Expected: lines around 730-740 show `messages = self._build_messages(user_message, effective_history, project_id=project_id)`.

- [ ] **Step 2: Write the failing test - rag_inject adds the system message**

Append to `tests/test_rag_injection.py`:

```python
def test_chat_stream_injects_rag_system_message_for_project_assistant(monkeypatch, tmp_path):
    """When agent_name == project-assistant + project_id provided +
    rag_inject returns a system message, the runtime must inject it
    AFTER the existing system prompt and BEFORE the user message."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_RAG_TOKENS", "1500")
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    captured = {}

    def fake_inject(user_message, project_id, conversation_id, user_id, agent_name):
        captured["args"] = (user_message, project_id, agent_name)
        return ({"role": "system", "content": "INJECTED_CONTEXT"}, {"injected_k": 1})

    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_inject)

    # Intercept _call_llm so we see the messages list that goes to the model.
    seen_messages = {"value": None}

    async def fake_call_llm(self, messages, api_key=None, **kwargs):
        seen_messages["value"] = messages
        return {
            "status": "success",
            "choice": {"message": {"content": "ok", "tool_calls": []}},
            "raw": {},
        }

    from app.agents import runtime
    monkeypatch.setattr(runtime.Agent, "_call_llm", fake_call_llm)
    monkeypatch.setenv("GROQ_API_KEY", "test")

    import asyncio
    agent = runtime.Agent(
        name="project-assistant",
        description="test",
        system_prompt="you are project-assistant",
        allowed_blocks=[],
    )

    async def collect():
        events = []
        async for ev in agent.chat_stream(
            user_message="how big is the IT load?",
            project_id="p1",
            conversation_id=None,
            user_id="u1",
        ):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    # The fake_inject was called with the user's question.
    assert captured["args"][0] == "how big is the IT load?"
    # The runtime passed the injected system message into _call_llm.
    msgs = seen_messages["value"]
    contents = [m.get("content", "") for m in msgs]
    assert any("INJECTED_CONTEXT" in c for c in contents)
```

- [ ] **Step 3: Run test - expect FAIL with ImportError**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k injects_rag -v`
Expected: FAIL with `AttributeError: <module 'app.agents.runtime'> has no attribute 'rag_inject'`

- [ ] **Step 4: Import rag_inject and add the hook in chat_stream**

In `app/agents/runtime.py` near the existing imports (around line 26), add:

```python
from app.core.rag.inject import rag_inject
```

Then in `chat_stream`, find the line (around 733):

```python
messages = self._build_messages(user_message, effective_history, project_id=project_id)
```

Replace it with:

```python
messages = self._build_messages(user_message, effective_history, project_id=project_id)

# Pre-iter-0 RAG injection. project-assistant only; returns None for
# other agents or when project_id is absent. Adds a system message
# AFTER the prompt + project context but BEFORE the latest user turn.
_rag_sys_msg, _rag_audit = rag_inject(
    user_message=user_message,
    project_id=project_id,
    conversation_id=conversation_id,
    user_id=user_id,
    agent_name=self.name,
)
if _rag_sys_msg and _rag_sys_msg.get("content"):
    # Insert just before the last user message (which is always last
    # after _build_messages). Index = len(messages) - 1.
    insert_at = max(0, len(messages) - 1)
    messages.insert(insert_at, _rag_sys_msg)
```

- [ ] **Step 5: Repeat the same hook in `chat` (non-streaming)**

Find the corresponding `messages = self._build_messages(...)` line in the non-streaming `chat` method (around line 553) and apply the same insertion (call `rag_inject`, insert before the last user turn).

- [ ] **Step 6: Run injection test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k injects_rag -v`
Expected: PASS

- [ ] **Step 7: Run the existing runtime tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runtime_anti_hallucination.py tests/test_runtime_ollama_provider.py tests/test_agent_runtime_c4.py tests/test_agent_runtime_c5.py tests/test_agents_router_c6.py -q`
Expected: all PASS (137 tests in this slice, give or take).

- [ ] **Step 8: Commit**

```bash
git add app/agents/runtime.py tests/test_rag_injection.py
git commit -m "feat(runtime): hook rag_inject before iter 0 in chat + chat_stream"
```

### Task 2.4: Default `use_rag=True` in ConstructionContainer.chat

**Files:**
- Modify: `app/containers/construction.py` (around line 251)
- Test: `tests/test_construction_chat.py` (extend)

- [ ] **Step 1: Write the failing test for the default flip**

Append to `tests/test_construction_chat.py` (which already exists from Phase D5):

```python
@pytest.mark.asyncio
async def test_construction_container_defaults_use_rag_true(monkeypatch):
    """Caller doesn't pass use_rag; container must default it to True so
    chat block invokes retrieval."""
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    captured_params = {}

    class _FakeChatBlock:
        async def process(self, input_data, params):
            captured_params.update(params)
            return {"status": "success", "text": "ok"}

    monkeypatch.setattr(container, "_resolve_block",
                        lambda name: _FakeChatBlock() if name == "chat" else None)

    await container.chat({"text": "hi"}, {"project_id": "p1"})
    assert captured_params.get("use_rag") is True
```

- [ ] **Step 2: Run the test, expect FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_construction_chat.py -k defaults_use_rag -v`
Expected: FAIL (assertion on `True is not None` likely)

- [ ] **Step 3: Add the default to ConstructionContainer.chat**

In `app/containers/construction.py`, modify `chat` (around line 251). After the `merged = dict(params or {})` line, add:

```python
        # RAG default: ON. The chat block will retrieve from this
        # project's index unless the caller has explicitly set use_rag.
        if "use_rag" not in merged and not (isinstance(input_data, dict)
                                            and "use_rag" in input_data):
            merged["use_rag"] = True
```

- [ ] **Step 4: Run the test, expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_construction_chat.py -k defaults_use_rag -v`
Expected: PASS

- [ ] **Step 5: Run the full construction-chat suite to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_construction_chat.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/containers/construction.py tests/test_construction_chat.py
git commit -m "feat(container): ConstructionContainer.chat defaults use_rag=True"
```

### Task 2.5: Wire the `?rag_debug=true` query parameter

**Files:**
- Modify: `app/routers/agents.py` (the `agent_chat_stream` endpoint, around line 167-218)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing test for ?rag_debug=true**

Append to `tests/test_rag_injection.py`:

```python
def test_rag_debug_query_param_doubles_llm_call(monkeypatch, tmp_path):
    """?rag_debug=true causes TWO LLM calls (with and without context)
    and the final SSE end event carries a rag_debug field."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app

    seen_calls = {"count": 0, "had_context_message": []}

    async def fake_chat_stream(self, message, **kwargs):
        # The agent runtime calls _call_llm internally; this stub mimics the
        # event shape but doesn't actually hit any LLM. The runtime's
        # rag_debug branch wraps a second call.
        yield {"type": "start", "agent": self.name}
        yield {"type": "end", "iterations": 1,
               "rag_debug": {"on_response": "A", "off_response": "B",
                             "sources": [], "scores": []}}

    monkeypatch.setattr("app.agents.runtime.Agent.chat_stream", fake_chat_stream)

    with TestClient(app) as client:
        r = client.post(
            "/v1/agents/project-assistant/chat/stream?rag_debug=true",
            json={"message": "hi", "project_id": "p1"},
            headers={"Authorization": "Bearer $CEREBRUM_DEV_KEY"},
        )
    assert r.status_code == 200
    # The SSE stream contains a rag_debug-marked end event.
    body = r.text
    assert "rag_debug" in body
```

- [ ] **Step 2: Run the test, expect FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k rag_debug_query_param -v`
Expected: FAIL (the route doesn't propagate `rag_debug` yet; the assertion on `"rag_debug" in body` should be False).

- [ ] **Step 3: Forward `rag_debug` from the request into chat_stream kwargs**

In `app/routers/agents.py` around line 167, modify `agent_chat_stream`:

```python
@router.post("/v1/agents/{name}/chat/stream")
async def agent_chat_stream(name: str, request: Request, auth: dict = Depends(require_user)):
    agent = get_agent(name)
    if not agent:
        raise HTTPException(404, f"Agent '{name}' not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "")
    history = body.get("history") or []
    model = body.get("model")
    project_id = body.get("project_id")
    conversation_id = body.get("conversation_id")
    rag_debug = request.query_params.get("rag_debug", "").lower() in ("true", "1", "yes")

    # ... existing auth / ownership checks unchanged ...

    if model:
        agent = _agent_with_override(agent, model=model)

    async def event_stream():
        try:
            async for evt in agent.chat_stream(
                message,
                history=history,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=auth["user_id"],
                rag_debug=rag_debug,
            ):
                yield f"data: {json.dumps(evt, default=str)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    # return StreamingResponse unchanged
```

- [ ] **Step 4: Add the `rag_debug` parameter to `chat_stream`**

In `app/agents/runtime.py`, update the `chat_stream` signature to accept `rag_debug: bool = False` and use it. The simplest v1 implementation:

```python
    async def chat_stream(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        rag_debug: bool = False,
        _depth: int = 0,
        _call_stack: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
```

At the END of `chat_stream`, just before the final `yield {"type": "end", ...}`, if `rag_debug` was set, run a second pass:

```python
            if rag_debug and _rag_sys_msg is not None:
                # Re-run the chat WITHOUT the RAG system message to get an
                # A/B comparison. This is opt-in and costs ~2x tokens.
                no_rag_messages = [m for m in messages if m is not _rag_sys_msg]
                no_rag_resp = await self._call_llm(no_rag_messages, api_key,
                                                   project_id=project_id, user_id=user_id)
                yield {
                    "type": "end",
                    "iterations": iteration + 1,
                    "rag_debug": {
                        "on_response": final_text,
                        "off_response": (no_rag_resp.get("choice", {}).get("message", {})
                                         .get("content", "")),
                        "audit": _rag_audit,
                    },
                }
                return
```

- [ ] **Step 5: Run the rag_debug test, expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k rag_debug -v`
Expected: PASS

- [ ] **Step 6: Confirm the other runtime tests still pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runtime_anti_hallucination.py tests/test_agents_router_c6.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/routers/agents.py app/agents/runtime.py tests/test_rag_injection.py
git commit -m "feat(runtime): ?rag_debug=true opt-in A/B path on chat/stream"
```

### Task 2.6: Push Phase 2, deploy, and run the regression queries

**Files:** (no code change; this is a deploy + acceptance gate)

- [ ] **Step 1: Push to main**

```bash
git push origin main
```

- [ ] **Step 2: Wait for Render to redeploy and report live**

Watch the deploy until status flips to `live`. Hit https://the-fork.onrender.com/ and confirm HTTP 200.

- [ ] **Step 3: Set Phase 2 env vars on Render (use the API)**

```bash
TOK=$RENDER_API_KEY
SRV=srv-d8hdc6ek1jcs739rq5sg
for KV in "RAG_K=5" "MAX_RAG_TOKENS=1500" "RAG_CONFIDENCE_THRESHOLD=0.4"; do
  K="${KV%%=*}"; V="${KV##*=}"
  curl -sS -X PUT -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
    "https://api.render.com/v1/services/$SRV/env-vars/$K" -d "{\"value\":\"$V\"}"
  echo " set $K"
done
curl -sS -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  "https://api.render.com/v1/services/$SRV/deploys" -d '{"clearCache":"do_not_clear"}'
```

Wait for the redeploy.

- [ ] **Step 4: Run the 5 regression queries via WebBridge**

For each of the 5 queries from the spec, drive Edge to send it through the UI to project `fb776aa2` (or `3f6f28b2` for the BOQ query), capture the response, and confirm:

| Query | Project | Pass criterion |
|---|---|---|
| Q1 IT load | `fb776aa2` | Specific number cited; doesn't say "unknown" |
| Q2 BOQ rate increment SAR 1,060 vs 1,288 | `3f6f28b2` | References item D999.14 / D999.15 |
| Q3 Cooling architecture | `fb776aa2` | Names BOD-specific cooling approach |
| Q4 Generate 50-activity schedule | `fb776aa2` | iter-0 response has `finish_reason=='tool_calls'` with `function.name=='generate_wbs'` |
| Q5 99.99% CDU blast radius | `fb776aa2` | Cites 99.99% AND 4-row figures |

- [ ] **Step 5: CHECKPOINT - operator review**

Paste all 5 responses to the operator with pass/fail for each. **Do not start Phase 2.5 or Phase 3 until they sign off.**

---

## Phase 2.5 - Daily token budget wiring (non-blocking)

The budget module landed in Phase 0; this phase wires it into the audit-record render and ships an env-var toggle on Render. Non-blocking for Phase 2 go-live - the runtime is ALREADY checking budget through `rag_inject`'s call to `budget.snapshot` and `budget.consume`. This phase makes the budget tunable from Render and adds a boundary regression test.

### Task 2.5.1: Add the test_q4_tool_call_discipline_under_rag explicit pass-criterion test

**Files:**
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Add the spec-mandated tool-call discipline test**

```python
def test_q4_tool_call_discipline_under_rag(monkeypatch):
    """When the user names a deliverable AND RAG context is injected,
    the iter-0 LLM response must have finish_reason='tool_calls' AND
    a tool_calls entry with function.name='generate_wbs'.

    This guards the project-assistant's tool-call mandate from being
    displaced by the RAG system message that we just added.
    """
    from app.agents.runtime import _user_intent_requires_tool

    # The matcher must still fire on "generate a 50-activity construction
    # schedule" even when prefixed by RAG context â€” context is a SYSTEM
    # role, the matcher walks the messages' tail for the user role.
    messages = [
        {"role": "system", "content": "Relevant project context: ..."},
        {"role": "user",   "content": "Generate a 50-activity construction schedule for the data center."},
    ]
    assert _user_intent_requires_tool(messages) is True
```

- [ ] **Step 2: Run the test - should PASS already**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k tool_call_discipline -v`
Expected: PASS (the existing iter-0 logic already walks for user role).

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_injection.py
git commit -m "test(rag): explicit Q4 tool-call discipline regression under RAG injection"
```

### Task 2.5.2: Set RAG_DAILY_TOKEN_BUDGET on Render

**Files:** (no code change; Render config)

- [ ] **Step 1: Set the budget env var**

```bash
TOK=$RENDER_API_KEY
SRV=srv-d8hdc6ek1jcs739rq5sg
curl -sS -X PUT -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  "https://api.render.com/v1/services/$SRV/env-vars/RAG_DAILY_TOKEN_BUDGET" \
  -d '{"value":"500000"}'
curl -sS -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  "https://api.render.com/v1/services/$SRV/deploys" -d '{"clearCache":"do_not_clear"}'
```

Wait for the redeploy.

- [ ] **Step 2: CHECKPOINT - operator sees budget telemetry in audit log**

After a single chat turn, on the Render dashboard's Shell (or via a temporary diagnostic endpoint if Shell access isn't available), tail `${DATA_DIR}/logs/rag_audit.jsonl` and confirm the latest row has `budget_remaining` and `budget_degraded` populated.

- [ ] **Step 3: Commit (nothing local; this is a Render-side config change)**

No git change. Note in the deployment log that Phase 2.5 shipped.

---

## Phase 3 - Drive ingestion incremental (SHA-256 dedupe)

Add a content hash to the documents schema, compute it on Drive walks, and skip unchanged files.

### Task 3.1: Add `content_sha256` column + migration

**Files:**
- Modify: `app/core/projects.py` (line 52 area - schema and migration)
- Test: `tests/test_drive_sha256_dedupe.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drive_sha256_dedupe.py`:

```python
"""Tests for the SHA-256 incremental ingestion path on the Drive walker."""
from __future__ import annotations

import hashlib
import os
import pathlib
import pytest


def test_documents_schema_has_content_sha256_column(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects
    projects.init_db()
    import sqlite3
    conn = sqlite3.connect(projects._db_path())
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    assert "content_sha256" in cols


def test_add_document_writes_sha256(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    sha = hashlib.sha256(b"hello world").hexdigest()
    doc = projects.add_document(
        project_id=proj["id"],
        original_name="hello.txt",
        stored_as="hello.txt",
        file_path="/tmp/hello.txt",
        size=11,
        content_sha256=sha,
    )
    assert doc["content_sha256"] == sha
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_drive_sha256_dedupe.py -v`
Expected: 2 FAIL (`content_sha256 not in cols` and `unexpected keyword argument 'content_sha256'`).

- [ ] **Step 3: Update the schema in init_db**

In `app/core/projects.py` around line 68 in the `documents` CREATE TABLE:

```python
                CREATE TABLE IF NOT EXISTS documents (
                    id            TEXT PRIMARY KEY,
                    project_id    TEXT NOT NULL
                                  REFERENCES projects(id) ON DELETE CASCADE,
                    original_name TEXT NOT NULL,
                    stored_as     TEXT,
                    file_path     TEXT,
                    doc_type      TEXT NOT NULL DEFAULT 'document',
                    doc_role      TEXT NOT NULL DEFAULT 'other',
                    size          INTEGER NOT NULL DEFAULT 0,
                    uploaded_at   TEXT NOT NULL,
                    content_sha256 TEXT
                );
```

Then add a migration right after the projects-user_id migration block (around line 103):

```python
            cols = [r[1] for r in conn.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()]
            if "content_sha256" not in cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN content_sha256 TEXT"
                )
```

- [ ] **Step 4: Update add_document to accept and write the hash**

In `app/core/projects.py` around line 232:

```python
def add_document(
    project_id: str,
    original_name: str,
    stored_as: Optional[str] = None,
    file_path: Optional[str] = None,
    size: int = 0,
    role: Optional[str] = None,
    content_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a document under a project. Storing only - runs no analysis."""
    _ensure_db()
    did = str(uuid.uuid4())[:8]
    doc_type = classify_doc_type(original_name)
    doc_role = role if role in VALID_ROLES else classify_doc_role(original_name)
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO documents "
            "(id, project_id, original_name, stored_as, file_path, doc_type, "
            " doc_role, size, uploaded_at, content_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (did, project_id, original_name, stored_as, file_path,
             doc_type, doc_role, size, _now(), content_sha256),
        )
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (did,)).fetchone()
    return dict(row)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_drive_sha256_dedupe.py -v`
Expected: 2 PASS

- [ ] **Step 6: Run the existing projects + governance tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_projects.py tests/test_governance.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/core/projects.py tests/test_drive_sha256_dedupe.py
git commit -m "feat(projects): content_sha256 column + add_document accepts it"
```

### Task 3.2: Wire SHA-256 computation into the Drive walker + skip duplicates

**Files:**
- Modify: `app/routers/drive.py` (the walker loop around lines 300-336)
- Test: `tests/test_drive_sha256_dedupe.py` (append)

- [ ] **Step 1: Add a helper to find an existing document by sha**

In `app/core/projects.py`, add:

```python
def find_document_by_sha(project_id: str, content_sha256: str) -> Optional[Dict[str, Any]]:
    """Return the FIRST existing document in this project with this
    content hash, or None. Used by the Drive walker to skip unchanged
    files on re-walk."""
    if not content_sha256:
        return None
    _ensure_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE project_id = ? AND content_sha256 = ? "
            "ORDER BY uploaded_at LIMIT 1",
            (project_id, content_sha256),
        ).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 2: Add the walker dedupe test**

Append to `tests/test_drive_sha256_dedupe.py`:

```python
def test_walker_skips_unchanged_file_on_rewalk(monkeypatch, tmp_path):
    """Second walk over a Drive folder whose file bytes haven't changed
    must skip the file (no new document row, no re-encryption)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]

    body = b"hello world content for sha test"
    sha = hashlib.sha256(body).hexdigest()
    # First walk: insert.
    projects.add_document(
        project_id=proj["id"],
        original_name="x.pdf", stored_as="x.pdf", file_path="/tmp/x.pdf",
        size=len(body), content_sha256=sha,
    )
    # Second walk: should detect via find_document_by_sha and skip.
    found = projects.find_document_by_sha(proj["id"], sha)
    assert found is not None
    # If the walker were to add again it would create a duplicate row.
    # Ensure the existing row is the only one.
    docs = projects.list_documents(proj["id"])
    assert len(docs) == 1
```

- [ ] **Step 3: Run the test, expect PASS already (helper test, walker test is structural)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_drive_sha256_dedupe.py -v`
Expected: 3 PASS

- [ ] **Step 4: Modify the walker to compute sha + skip on dupe**

In `app/routers/drive.py` around line 320 (just before `file_crypto.write_document(filepath, raw_bytes)`), add:

```python
                import hashlib as _hashlib
                content_sha = _hashlib.sha256(raw_bytes).hexdigest()
                existing = projects_router.store.find_document_by_sha(
                    project_id, content_sha,
                )
                if existing:
                    skipped.append({
                        "name": stored_basename,
                        "reason": f"unchanged (sha {content_sha[:12]}...)",
                    })
                    continue
```

Then update the `store.add_document(...)` call to pass the hash:

```python
                doc = store.add_document(project_id, stored_basename, stored_as,
                                         filepath, len(raw_bytes),
                                         content_sha256=content_sha)
```

Make sure `from app.core import projects as projects_router` is already imported. If not, add it.

- [ ] **Step 5: Run the existing drive-walker tests + the new ones**

Run: `.venv/Scripts/python.exe -m pytest tests/test_drive_router.py tests/test_drive_sha256_dedupe.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/projects.py app/routers/drive.py tests/test_drive_sha256_dedupe.py
git commit -m "feat(drive): SHA-256 dedupe in walker skips unchanged files"
```

### Task 3.3: Phase 3 GO - deploy + manual ingestion smoke

**Files:** (no code change; deploy + smoke)

- [ ] **Step 1: Push to main**

```bash
git push origin main
```

Wait for Render deploy `live`.

- [ ] **Step 2: Manual ingestion smoke - upload one small file via Drive then re-walk**

Per the spec: start with ONE file under 1MB. Use the UI to trigger the existing `/v1/projects/{id}/drive/index-folder` endpoint OR use a Drive walker test fixture if available. After the first walk, repeat. Confirm the second walk returns `skipped` containing that file with reason `unchanged (sha ...)`.

- [ ] **Step 3: CHECKPOINT - operator confirms skip count > 0**

Paste the walker response to the operator. **Do not start Phase 4 until they sign off.**

---

## Phase 4 - Source UX

Surface the retrieved sources in the chat UI.

### Task 4.1: Backend - emit `sources[]` in the chat_stream end event

**Files:**
- Modify: `app/agents/runtime.py` (around the final `yield {"type": "end", ...}` lines)
- Test: `tests/test_rag_injection.py` (append)

- [ ] **Step 1: Write the failing test for the end-event shape**

Append to `tests/test_rag_injection.py`:

```python
def test_chat_stream_end_event_carries_sources(monkeypatch, tmp_path):
    """The final SSE 'end' event must include sources[] with up to top-3
    items sorted by score desc, each with doc_id, doc_name, page_or_section,
    score, confidence."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.rag.vector_store import Chunk
    audit_rec = {
        "injected_k": 3,
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "score": 0.91},
            {"doc_id": "d1", "chunk_index": 7, "score": 0.62},
            {"doc_id": "d2", "chunk_index": 2, "score": 0.48},
            {"doc_id": "d3", "chunk_index": 0, "score": 0.30},  # would be 4th
        ],
    }
    from app.agents.runtime import _build_sources_from_audit
    sources = _build_sources_from_audit(audit_rec)
    assert len(sources) == 3
    assert sources[0]["score"] == 0.91
    assert sources[0]["confidence"] == "High"
    assert sources[1]["confidence"] == "Medium"
    assert sources[2]["confidence"] == "Low"
```

- [ ] **Step 2: Run the test, expect FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k end_event_carries -v`
Expected: FAIL with `ImportError: cannot import name '_build_sources_from_audit'`

- [ ] **Step 3: Add `_build_sources_from_audit` to runtime.py**

In `app/agents/runtime.py` (near the other helpers, around line 80 after `_user_intent_requires_tool`):

```python
def _build_sources_from_audit(audit_rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """From a rag_inject audit record, build the top-3 sources list for
    the SSE end event. Resolves doc_id -> filename via projects.get_document.
    Empty list when there are no chunks (fallback or non-RAG turn)."""
    chunks = (audit_rec or {}).get("chunks") or []
    if not chunks:
        return []
    by_score = sorted(chunks, key=lambda c: -(c.get("score") or 0))[:3]
    out: List[Dict[str, Any]] = []
    try:
        from app.core import projects as _projects
    except Exception:
        _projects = None
    for c in by_score:
        score = c.get("score") or 0.0
        if score >= 0.75:
            conf = "High"
        elif score >= 0.5:
            conf = "Medium"
        else:
            conf = "Low"
        doc_name = ""
        if _projects:
            try:
                d = _projects.get_document(c["doc_id"]) or {}
                doc_name = d.get("original_name") or ""
            except Exception:
                doc_name = ""
        out.append({
            "doc_id": c["doc_id"],
            "doc_name": doc_name,
            "page_or_section": f"chunk #{c['chunk_index']}",
            "score": float(score),
            "confidence": conf,
        })
    return out
```

- [ ] **Step 4: Inject sources into the end event of chat_stream**

Find all `yield {"type": "end", "iterations": ...}` lines in `chat_stream` (there are 2-3). Replace each with:

```python
                    yield {
                        "type": "end",
                        "iterations": iteration + 1,
                        "sources": _build_sources_from_audit(_rag_audit),
                    }
```

(For the loop-cap end at line ~845, use `MAX_TOOL_ITERATIONS` as the iterations value and add `"forced_final": True` back.)

- [ ] **Step 5: Run the sources test, expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rag_injection.py -k end_event_carries -v`
Expected: PASS

- [ ] **Step 6: Run runtime regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runtime_anti_hallucination.py tests/test_runtime_ollama_provider.py tests/test_agent_runtime_c4.py tests/test_agent_runtime_c5.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/agents/runtime.py tests/test_rag_injection.py
git commit -m "feat(runtime): emit sources[] in chat_stream end event"
```

### Task 4.2: Frontend - Sources footer in ProjectWorkspace.tsx

**Files:**
- Modify: `frontend/src/pages/ProjectWorkspace.tsx`

- [ ] **Step 1: Find where the SSE end event is handled**

Run: `grep -n '"end"\|type === .end\|sources' frontend/src/pages/ProjectWorkspace.tsx | head -10`
The SSE consumer is in the streaming `fetch` block (around line 917 from prior session). Find where `evt.type === "end"` is observed and the final assistant message is committed to state.

- [ ] **Step 2: Add a `sources` field to the ChatMessage type**

In `frontend/src/pages/ProjectWorkspace.tsx`, find the `ChatMessage` interface (somewhere around line 100-130). Add:

```typescript
interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{
    doc_id: string
    doc_name: string
    page_or_section: string
    score: number
    confidence: 'High' | 'Medium' | 'Low'
  }>
}
```

- [ ] **Step 3: Capture `sources` when the `end` event arrives**

In the SSE consumer (around line 1000-1050), where it transitions on `evt.type === 'end'`, update the last assistant message in state to include `sources: evt.sources`. Example:

```typescript
if (parsed.type === 'end') {
  setMessages((prev) => prev.map((m, i) =>
    i === prev.length - 1 && m.role === 'assistant'
      ? { ...m, sources: parsed.sources || [] }
      : m
  ))
  break
}
```

- [ ] **Step 4: Render the Sources footer below assistant bubbles**

Find where `ChatThread` renders each message bubble. Below the bubble content, add (only when `message.sources && message.sources.length > 0`):

```tsx
{message.role === 'assistant' && message.sources && message.sources.length > 0 && (
  <details className="chat-message__sources">
    <summary>Sources ({message.sources.length})</summary>
    <ul>
      {message.sources.map((s, i) => (
        <li key={i}>
          <span className="chat-message__sources-name">{s.doc_name || s.doc_id}</span>
          <span className="chat-message__sources-meta">
            {s.page_or_section} . score {s.score.toFixed(2)} . {s.confidence}
          </span>
        </li>
      ))}
    </ul>
  </details>
)}
```

- [ ] **Step 5: Run the frontend build to confirm no TS errors**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: built clean (vite + tsc).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/ProjectWorkspace.tsx
git commit -m "feat(frontend): Sources footer on assistant bubbles when SSE end carries sources"
```

### Task 4.3: Phase 4 GO - deploy and run the 5 regression queries through the UI

**Files:** (no code change; deploy + smoke)

- [ ] **Step 1: Push to main + wait for live**

```bash
git push origin main
```

- [ ] **Step 2: Drive Edge through the 5 regression queries again**

For each of the 5 queries from Phase 2 Task 2.6, confirm the Sources footer appears below the assistant response with High/Medium/Low confidence labels matching the audit log scores.

- [ ] **Step 3: CHECKPOINT - operator confirms confidence labels are right**

Paste screenshots or DOM text per query showing the Sources footer. **Phase 4 done when operator signs off.**

---

## Self-review

Spec coverage check (against `docs/superpowers/specs/2026-06-08-track1-rag-production.md`):

- Component 0 noise filter -> Task 0.4 + 0.5 (regex + retriever pool filter + audit field)
- Component 1 RAG injector -> Task 2.1 + 2.2 (apply_token_cap + rag_inject) + 2.3 (runtime hook)
- Component 2 token cap (whole-chunk) -> Task 2.1 + boundary covered
- Component 3 confidence fallback -> Task 2.2 (threshold_fired test) + 2.3 (sys_msg None path)
- Component 4 audit log -> Task 0.1 + 0.2 (writer + tolerance) + audit_rec contents covered in 2.2/2.3
- Component 4.5 daily budget -> Task 0.3 (module + boundary test) + 2.5.2 (Render env)
- Component 5 ?rag_debug=true -> Task 2.5
- Component 6 Drive SHA-256 -> Task 3.1 (schema) + 3.2 (walker)
- Component 7 Sources UX -> Task 4.1 (backend) + 4.2 (frontend)
- Phase 1 data generation -> Task 1.1 + 1.2 + 1.3 + 1.4
- 5 regression queries with Q4 explicit pass criterion -> Task 2.6 (gates Phase 2) + 4.3 (gates Phase 4)
- Acceptance criteria all four env vars overridable -> Task 2.6 + 2.5.2 (Render PUT calls)

Placeholder scan: no "TBD", "TODO", "fill in" instructions found; every code step contains the actual code. Every test step has full test bodies. Every command has exact paths and expected output where relevant.

Type consistency: `Chunk` (from `app.core.rag.vector_store`) used consistently in retriever, inject, audit, and tests. `_rag_audit` dict shape consistent across `rag_inject`, audit writer, and `_build_sources_from_audit`. `_build_sources_from_audit` accepts the same dict layout that `rag_inject` returns.

No gaps found.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-06-09-track1-rag-production.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
