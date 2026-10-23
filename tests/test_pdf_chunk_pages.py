"""Chunks carry the real page of the source PDF their text starts on.

Citations used to read "chunk #N": a chunk had no page. PDF extraction now
records where each page's text starts, each chunk is given the page its text
begins on (the chunk TEXT is unchanged -- no page markers), the page travels
through the vector store, and a citation reads "p. N" when it is known.

Synthetic PDFs and text only.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import doc_index


def _pdf(path, pages):
    """A text-layer PDF, one synthetic paragraph per page (None = blank page)."""
    import fitz

    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), body, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def _page_body(n: int, words: int = 160) -> str:
    return " ".join(f"pg{n}word{i}" for i in range(words))


# ── page assignment ──────────────────────────────────────────────────────────


def test_each_chunk_gets_the_page_its_text_starts_on():
    pages = [_page_body(n) for n in (1, 2, 3, 4)]
    text = "\n".join(pages)
    starts, off = [], 0
    for n, body in enumerate(pages, start=1):
        starts.append([n, off])
        off += len(body) + 1
    chunks = doc_index.chunk_text(text, words_per_chunk=100)  # 160-word pages

    got = doc_index.chunk_start_pages(text, starts, chunks)

    expected = []
    for chunk in chunks:
        first = chunk.split()[0]          # e.g. "pg3word40"
        expected.append(int(first[2:first.index("word")]))
    assert got == expected
    assert got[0] == 1 and got[-1] == 4 and len(set(got)) == 4


def test_overlapping_chunks_and_repeated_headers_resolve_in_reading_order():
    header = "SYNTHETIC HANDBOOK RUNNING HEADER SECTION GENERAL REQUIREMENTS"
    pages = [f"{header}\n{_page_body(n, 60)}" for n in (1, 2, 3)]
    text = "\n".join(pages)
    starts, off = [], 0
    for n, body in enumerate(pages, start=1):
        starts.append([n, off])
        off += len(body) + 1
    chunks = doc_index.chunk_text_with_overlap(text, target_chars=300, overlap=50)
    got = doc_index.chunk_start_pages(text, starts, chunks)
    assert None not in got
    assert got == sorted(got)               # never jumps back to an earlier page
    for chunk, page in zip(chunks, got):
        if chunk.startswith(header):
            first_word = chunk[len(header):].split()[0]
            assert first_word.startswith(f"pg{page}word")


def test_no_page_offsets_means_no_pages():
    assert doc_index.chunk_start_pages("some text", None, ["some text"]) == [None]
    assert doc_index.chunk_start_pages("", [[1, 0]], ["x"]) == [None]


# ── extraction → chunks ──────────────────────────────────────────────────────


def test_pdf_extraction_records_pages_and_chunks_keep_their_text(tmp_path):
    # 400 words a page, 500-word chunks: chunks start on pages 1, 2 and 4.
    path = _pdf(tmp_path / "synthetic_book.pdf",
                [_page_body(1, 400), _page_body(2, 400), None, _page_body(4, 400)])
    text, meta = doc_index._extract_pdf(path, "synthetic_book.pdf")
    assert [p for p, _ in meta["page_starts"]] == [1, 2, 4]  # blank page has no text
    for page, offset in meta["page_starts"]:
        assert text[offset:].startswith(f"pg{page}word0")

    chunks, out_meta = doc_index._produce_chunks(
        path, "synthetic_book.pdf", ".pdf", "proj-x", "default", False,
    )
    # Chunk text is exactly what chunking produced before pages existed.
    assert chunks == doc_index.chunk_extracted_document(text, filename="synthetic_book.pdf")
    assert not any("page_starts" in c for c in chunks)
    assert "page_starts" not in out_meta
    pages = out_meta["chunk_pages"]
    assert len(pages) == len(chunks)
    for chunk, page in zip(chunks, pages):
        first = chunk.split()[0]
        assert page == int(first[2:first.index("word")])
    assert pages == [1, 2, 4]


def test_batched_extraction_gives_the_same_text_and_pages(tmp_path, monkeypatch):
    path = _pdf(tmp_path / "synthetic_batches.pdf",
                [_page_body(n, 40) for n in range(1, 8)])
    whole_text, whole_meta = doc_index._extract_pdf(path, "synthetic_batches.pdf")
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", "3")
    text, meta = doc_index._extract_pdf_batched(path, "synthetic_batches.pdf")
    assert text == whole_text
    assert meta["page_starts"] == whole_meta["page_starts"]
    assert [p for p, _ in meta["page_starts"]] == list(range(1, 8))


def test_nul_bytes_removed_from_text_move_the_page_offsets():
    meta = {"page_starts": [[1, 0], [2, 6], [3, 13]]}
    text = "ab\x00cd\n" + "ef\x00\x00gh\n" + "ij"
    out = doc_index._strip_nul(text, meta)
    assert out == "abcd\nefgh\nij"
    for page, offset in meta["page_starts"]:
        assert out[offset:offset + 1] == {1: "a", 2: "e", 3: "i"}[page]


def test_non_pdf_sources_have_no_page(tmp_path):
    path = tmp_path / "synthetic_note.txt"
    path.write_text(_page_body(1, 600), encoding="utf-8")
    chunks, meta = doc_index._produce_chunks(
        str(path), "synthetic_note.txt", ".txt", "proj-x", "default", False,
    )
    assert chunks and meta["chunk_pages"] == [None] * len(chunks)
    assert doc_index._pages_for(chunks, meta) is None


# ── vector store ─────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core.rag import embeddings as _emb, vector_store as _vs

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    embedder = _emb.get_embedder()
    st = _vs.VectorStore(db_path=str(tmp_path / "pages.db"), dim=embedder.dim,
                         namespace="pagetest")
    yield st, embedder
    st.close()
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_page_is_stored_and_returned_by_every_read_path(store):
    st, embedder = store
    texts = ["alpha synthetic clause on PAGEPROBE anchors", "beta synthetic clause later"]
    st.upsert_chunks("p1", "d1", texts, embedder.encode(texts), pages=[12, 13])

    by_doc = st.chunks_for_docs("p1", ["d1"], all_rows=True)
    assert [c.page for c in by_doc] == [12, 13]
    sem = st._semantic_search("p1", embedder.encode([texts[0]])[0], k=2)
    assert {c.chunk_index: c.page for c in sem} == {0: 12, 1: 13}
    assert [c.page for c in st.bm25_search("p1", "PAGEPROBE", k=5)] == [12]
    assert [c.page for c in st.identifier_search("p1", ["PAGEPROBE"], k=5)] == [12]
    assert [c.page for c in st.chunks_containing_all("p1", ["pageprobe"])] == [12]
    assert [c.page for c in st.chunks_following("p1", [("d1", 0)])] == [13]
    assert by_doc[0].to_dict()["page"] == 12


def test_chunks_without_pages_store_null(store):
    st, embedder = store
    texts = ["gamma synthetic"]
    st.upsert_chunks("p1", "d2", texts, embedder.encode(texts))
    [chunk] = st.chunks_for_docs("p1", ["d2"])
    assert chunk.page is None
    assert "page" not in chunk.to_dict()


def test_a_table_created_before_pages_gains_the_column(tmp_path):
    import sqlite3

    from app.core.rag import vector_store as _vs

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE chunks_oldpages (chunk_id TEXT PRIMARY KEY, project_id TEXT, "
        "doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, "
        "embedding BLOB NOT NULL, created_at TEXT NOT NULL, knowledge_layer TEXT, "
        "authority TEXT, embedding_model TEXT, embedding_dim INTEGER, "
        "embedding_normalized BOOLEAN)"
    )
    con.commit()
    con.close()
    _vs.reset_store_cache()
    st = _vs.VectorStore(db_path=str(db), dim=8, namespace="oldpages", model_name="fake")
    texts = ["delta synthetic"]
    st.upsert_chunks("p1", "d3", texts, np.ones((1, 8), dtype=np.float32), pages=[5])
    assert [c.page for c in st.chunks_for_docs("p1", ["d3"])] == [5]
    st.close()


def test_migration_0021_adds_page_to_every_chunk_table_once(tmp_path):
    import importlib.util
    import pathlib

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "alembic" / "versions" / "0021_chunks_page.py")
    spec = importlib.util.spec_from_file_location("m0021", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "0021" and mod.down_revision == "0020"

    eng = sa.create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    with eng.begin() as conn:
        for tbl in ("chunks", "chunks_v2"):
            conn.execute(sa.text(
                f"CREATE TABLE {tbl} (chunk_id TEXT PRIMARY KEY, doc_id TEXT, "
                "chunk_index INTEGER, text TEXT)"))
        conn.execute(sa.text("CREATE TABLE chunks_notes (note TEXT)"))
    for _ in range(2):  # idempotent
        with eng.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                mod.upgrade()
    insp = sa.inspect(eng)
    for tbl in ("chunks", "chunks_v2"):
        assert "page" in {c["name"] for c in insp.get_columns(tbl)}
    assert "page" not in {c["name"] for c in insp.get_columns("chunks_notes")}


# ── citations ────────────────────────────────────────────────────────────────


def test_citation_shows_the_page_when_known_else_nothing():
    from app.agents.runtime import page_or_section_label

    assert page_or_section_label({"page": 123, "chunk_index": 7}) == "p. 123"
    assert page_or_section_label({"page": None, "chunk_index": 7}) == ""
    assert page_or_section_label({"chunk_index": 7}) == ""


def test_injected_chunk_audit_carries_the_page():
    from app.core.rag.inject import _audit_chunk
    from app.core.rag.vector_store import Chunk

    c = Chunk(chunk_id="p:d:0", project_id="p", doc_id="d", chunk_index=0,
              text="synthetic", page=41)
    assert _audit_chunk(c)["page"] == 41


def test_sources_panel_cites_the_page():
    from app.agents import runtime

    audit = {
        "project_id": "p",
        "chunks": [{"doc_id": "d", "chunk_index": 3, "chunk_id": "p:d:3",
                    "project_id": "p", "score": 0.9, "layer": "own", "page": 88}],
    }
    sources = runtime._build_sources_from_audit(audit, "The synthetic handbook says so.")
    assert sources and sources[0]["page_or_section"] == "p. 88"
    assert sources[0]["page"] == 88


def test_index_chunks_hands_the_pages_to_the_store(store, monkeypatch):
    st, embedder = store
    from app.core.rag import retriever

    monkeypatch.setattr(retriever, "available", lambda: True)
    monkeypatch.setattr(retriever, "get_store", lambda dim=None: st)
    texts = ["epsilon synthetic", "zeta synthetic"]
    assert retriever.index_chunks("p2", "d4", texts, pages=[7, None]) == 2
    assert [c.page for c in st.chunks_for_docs("p2", ["d4"])] == [7, None]
