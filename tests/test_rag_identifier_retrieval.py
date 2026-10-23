"""Tests for identifier-aware RAG retrieval boost.

The retriever must surface chunks that contain exact construction
reference identifiers (VO/RFI/NCR/PRC/drawing codes/etc.) above
semantically-similar boilerplate that lacks the requested identifier.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Fresh vector store + fake embedder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as _emb, vector_store as _vs
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store
    from sqlalchemy import delete as _sa_delete
    e = Embedder(model_name="fake")
    # Use the default project database path (honours DATA_DIR) so that
    # retriever.get_store() returns the same cached instance.
    store = get_store(dim=e.dim)
    # On PostgreSQL, DATA_DIR does not change the database URL, so rows
    # written by earlier tests survive into this test's store. Truncate the
    # chunk table up-front to guarantee isolation regardless of backend.
    with store._lock:
        with store._session_factory()() as session:
            session.execute(_sa_delete(store._rag_chunk_cls))
            session.commit()
    yield store, e
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_extract_identifiers_detects_common_reference_patterns():
    from app.core.rag.retriever import extract_query_identifiers

    ids = extract_query_identifiers('What is the status of VO Ref 31?')
    assert any("vo" in i and "31" in i for i in ids)

    ids = extract_query_identifiers('Is APPROVED valid per XYZ-501?')
    assert any("xyz-501" in i for i in ids)

    ids = extract_query_identifiers('Show drawing QZ-ENG-320-0000-EXC-DWG-LI-200-0001056-04')
    assert any("qz-eng-320" in i for i in ids)

    ids = extract_query_identifiers('What about BOQ item D999.46?')
    assert any("d999.46" in i for i in ids)

    ids = extract_query_identifiers('Find RFI 12-A and NCR-007')
    assert any("rfi" in i and "12-a" in i for i in ids)
    assert any("ncr-007" in i for i in ids)

    ids = extract_query_identifiers('Tell me about concrete')
    assert ids == []


def test_extract_identifiers_ignores_common_label_words_without_a_code():
    """Regression (2026-06-30 pilot): the labeled-reference regex treated
    common words in _REFERENCE_LABELS (Contract, Spec, Package, ...) as a
    reference label and grabbed the FOLLOWING plain English word as a 'code'
    — so "contract cover" -> ['contract cover', 'cover'] and "specification"
    -> ['spec ification', 'ification']. Those false identifiers then earned a
    +2.0 retrieval bonus, flooding the top-K with any boilerplate chunk that
    merely contained the word 'cover', and the model answered "I cannot find."

    A real reference code contains a digit (VO 99, RFI 42, Clause 13.1). A
    label followed by a digit-less word is NOT an identifier.
    """
    from app.core.rag.retriever import extract_query_identifiers

    # The two prod questions that broke. Neither contains a real reference.
    assert extract_query_identifiers(
        "What does the the client project demolition contract cover?"
    ) == []
    ids = extract_query_identifiers(
        "What does the specification say about reinforcement joints "
        "and crack control in concrete?"
    )
    assert ids == [], f"expected no identifiers, got {ids}"

    # A bare label word with no code must not self-extract.
    assert extract_query_identifiers("Tell me about the contract") == []
    assert extract_query_identifiers("Summarize the package scope") == []

    # ...but a label followed by a real (digit-bearing) code STILL extracts.
    assert any("99" in i for i in extract_query_identifiers("status of VO 99?"))
    assert any("42" in i for i in extract_query_identifiers("RFI 42 update"))
    assert any("13.1" in i for i in extract_query_identifiers("see Clause 13.1"))


def test_extract_identifiers_preserves_quoted_phrases():
    from app.core.rag.retriever import extract_query_identifiers

    ids = extract_query_identifiers('What does "Clause 13.1" require for "VO Ref 31"?')
    assert '"clause 13.1"' in ids or 'clause 13.1' in ids
    assert '"vo ref 31"' in ids or 'vo ref 31' in ids


def test_extract_identifiers_excludes_measurement_units():
    """A spec unit in a query is NOT a reference code. Before this guard,
    "concrete (250 kg/cm2)" extracted 'kg/cm2' as an identifier, which earned
    the +2.0 identifier bonus and matched every drawing dimension-table chunk
    containing the same numbers — burying the real rate chunk and inducing a
    fabricated figure lifted from the number-soup (2026-07-14 live incident)."""
    from app.core.rag.retriever import extract_query_identifiers, _looks_like_unit

    # The exact live-incident query must yield NO polluting identifier.
    assert extract_query_identifiers(
        "What is the unit rate for ready-mix concrete (250 kg/cm2) in Saudi Arabia?"
    ) == []
    # Unit-ratios and bare units never become identifiers.
    for unit_q in ("rebar n/mm2 grade", "density kn/m3", "pour 350 kg/cm2",
                   "area in m2", "volume m3"):
        assert extract_query_identifiers(unit_q) == [], unit_q
    for unit in ("kg/cm2", "n/mm2", "kn/m3", "m3", "cm2", "mm2"):
        assert _looks_like_unit(unit), unit
    for unit in ("aed/m2", "usd/ft2", "sar/m3"):
        assert _looks_like_unit(unit), unit
    assert extract_query_identifiers(
        "Stay on self-coding. Convert 1250 AED/m2 to USD/ft2 using 3.6725 AED "
        "= 1 USD and 1 m2 = 10.7639 ft2."
    ) == []

    # ...but real document reference codes are UNTOUCHED (identifier lane intact).
    for code in ("xyz-501", "qz-eng-320", "d999.46", "ncr-007", "12-a", "13.1"):
        assert not _looks_like_unit(code), code
    assert any("xyz-501" in i for i in extract_query_identifiers("valid per XYZ-501?"))
    assert any("d999.46" in i for i in extract_query_identifiers("BOQ item D999.46"))
    assert any("qz-eng-320" in i for i in
               extract_query_identifiers("Show drawing QZ-ENG-320-0000-EXC-DWG"))


def test_identifier_chunk_outranks_semantic_boilerplate(isolated_store, monkeypatch):
    """A generic chunk with high semantic similarity must not beat the
    chunk that actually contains the requested identifier."""
    from app.core.rag import retriever as ret
    from app.core.rag.embeddings import Embedder

    store, e = isolated_store
    # Generic boilerplate that would score highly on a status query.
    boilerplate = (
        "Status tracking is important for project controls. "
        "The contractor shall maintain a register of all variations, "
        "requests for information, and non-conformance reports."
    )
    # Exact chunk containing the identifier the user asked for.
    exact = (
        "VO Ref: 99 | Status: Closed | Closed date: 2024-02-12 | "
        "Description: additional drainage works"
    )
    store.upsert_chunks("proj_a", "doc_boilerplate", [boilerplate], e.encode([boilerplate]))
    store.upsert_chunks("proj_a", "doc_exact", [exact], e.encode([exact]))

    # Ensure the boilerplate doc is not treated as noise.
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "doc.pdf")

    chunks, _ = ret.retrieve_with_filter(
        "What was the status of VO Ref 99?", "proj_a", k=2
    )
    assert len(chunks) >= 1
    # The exact identifier chunk must be #1.
    assert "VO Ref: 99" in chunks[0].text
    assert chunks[0].score > chunks[1].score if len(chunks) > 1 else True


def test_retrieval_without_identifier_uses_semantic_ordering(isolated_store, monkeypatch):
    """Non-identifier queries should not be perturbed by an empty identifier leg."""
    from app.core.rag import retriever as ret

    store, e = isolated_store
    chunks = ["concrete pour schedule", "rebar inventory", "drawing revisions"]
    store.upsert_chunks("proj_a", "doc_x", chunks, e.encode(chunks))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "doc.pdf")

    results, _ = ret.retrieve_with_filter("concrete schedule", "proj_a", k=3)
    # Semantic ranking should return the three chunks; exact first-place
    # order depends on the deterministic fake embedder, so just assert
    # coverage without over-fitting to a particular hash-based ordering.
    assert len(results) == 3
    assert any("concrete pour schedule" in c.text for c in results)


def test_identifier_search_is_case_insensitive(isolated_store):
    from app.core.rag.embeddings import Embedder

    store, e = isolated_store
    text = "VO Ref: 99 was closed on 2024-02-12"
    store.upsert_chunks("proj_a", "doc_x", [text], e.encode([text]))

    results = store.identifier_search("proj_a", ["vo ref 99"], k=5)
    assert len(results) == 1
    assert results[0].score == 1.0


def test_identifier_search_escapes_like_wildcards(isolated_store):
    """Identifiers containing SQL LIKE wildcards (% or _) still match literally."""
    from app.core.rag.embeddings import Embedder

    store, e = isolated_store
    text = "Item 50% complete, code A_1"
    store.upsert_chunks("proj_a", "doc_x", [text], e.encode([text]))

    results = store.identifier_search("proj_a", ["50%", "A_1"], k=5)
    assert len(results) == 1
    # Both identifiers match.
    assert results[0].score == 1.0


def test_semantic_chunk_with_identifier_outranks_identifier_only_soup(
    isolated_store, monkeypatch
):
    """2026-07-26 live find (corpus project): identifier_search returns an
    ARBITRARY top-k of the many chunks containing a code, so the semantically
    best chunk that ALSO carries the identifier can miss that set — and then
    flat-bonused label-soup (drawing station tables) displaced it. The fused
    ranking must award the identifier bonus to semantic candidates whose TEXT
    contains the code, so cosine+bonus always beats bonus-alone."""
    from app.core.rag import retriever as ret

    store, e = isolated_store
    spec = (
        "Pump station schedule: WWPS-99 submersible pumps, 2 duty + 1 standby, "
        "total flow rate 366 l/s, total head 35.5 m."
    )
    soup = [
        "0+234.64 0+250.00 WWPS-99 DATUM 635.00 MHA-10 647.250 INVERT LEVEL",
        "SUBMISSION OF WWPS-99 STR. GA PLANS 26-May-25 17-Jun-25 APPROVALS",
    ]
    store.upsert_chunks("proj_id2", "doc_spec", [spec], e.encode([spec]))
    store.upsert_chunks("proj_id2", "doc_soup", soup, e.encode(soup))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "doc.pdf")

    # Simulate the arbitrary-k miss: identifier_search only surfaces the soup.
    real_id_search = store.identifier_search

    def missing_the_spec(project_id, identifiers, k=20):
        hits = real_id_search(project_id, identifiers, k=k)
        return [c for c in hits if "total flow rate" not in c.text]

    monkeypatch.setattr(store, "identifier_search", missing_the_spec)

    # Deterministic semantics (the fake embedder's hash cosine is arbitrary):
    # the spec chunk is the semantic winner, the soup is near-noise — the
    # real-corpus shape (0.75 spec vs ~0 drawing tables).
    real_search = store.search

    def scored_search(project_id, query_vec, k=20, query_text=None):
        out = real_search(project_id, query_vec, k=k, query_text=query_text)
        for c in out:
            c.score = 0.75 if "total flow rate" in c.text else 0.05
        return out

    monkeypatch.setattr(store, "search", scored_search)

    chunks, _ = ret.retrieve_with_filter(
        "What is the total flow rate of WWPS-99?", "proj_id2", k=3
    )
    assert chunks, "expected results"
    assert "total flow rate 366" in chunks[0].text, [c.text[:60] for c in chunks]


# ── WAVE 2 B4 / B5: CESMM OCR space + wrong-row ranking ──────────────────────


def test_normalize_cesmm_item_codes_collapses_ocr_space():
    from app.core.rag.vector_store import normalize_cesmm_item_codes

    assert normalize_cesmm_item_codes("D 549.2") == "D549.2"
    assert normalize_cesmm_item_codes("D549.2") == "D549.2"
    assert normalize_cesmm_item_codes("d 599.5") == "d599.5"
    assert normalize_cesmm_item_codes("I  112.3") == "I112.3"
    # Drawing / contract ids must not be rewritten.
    assert "QZ-ENG-320" in normalize_cesmm_item_codes(
        "drawing QZ-ENG-320-0000-EXC-DWG"
    )
    assert "AB-2023-101" in normalize_cesmm_item_codes("see AB-2023-101 Vol 1")


def test_extract_identifiers_collapses_ocr_spaced_cesmm():
    from app.core.rag.retriever import extract_query_identifiers

    assert "d549.2" in extract_query_identifiers("rate for D 549.2 please")
    assert "d549.2" in extract_query_identifiers("rate for D549.2 please")
    assert "d599.5" in extract_query_identifiers("D 599.5 carriageway breakout")
    # WAVE 2 B5 live prompt (parentheses around the compact code).
    assert "d549.2" in extract_query_identifiers(
        "What is the amount for removal of existing chain link fence (D549.2)?"
    )


def test_ocr_spaced_d5492_is_retrievable_as_compact(isolated_store, monkeypatch):
    """WAVE 2 B5: stored OCR text is ``D 549.2``; query is ``D549.2``.

    Live Neon: ILIKE D549.2 = 0, ILIKE 'D 549.2' = 2. Query-time match
    must hit the unreindexed spaced row so the RAG-miss short-circuit
    does not fire. Expected figures: 3,504 m @ SAR 80.00.
    """
    from app.core.rag import retriever as ret

    store, e = isolated_store
    spaced = (
        "D 549.2 Removal of existing chain link fence 3,504 m 80.00 280,320.00"
    )
    assert "D549.2" not in spaced
    store.upsert_chunks("proj_b5", "doc_fence", [spaced], e.encode([spaced]))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "priced-boq.pdf")

    hits = store.identifier_search("proj_b5", ["D549.2"], k=5)
    assert hits, "D549.2 must match stored 'D 549.2'"
    assert "80.00" in hits[0].text
    assert "3,504" in hits[0].text or "3504" in hits[0].text.replace(",", "")

    chunks, _ = ret.retrieve_with_filter(
        "What is the rate for D549.2 chain link fence?",
        "proj_b5",
        k=3,
    )
    assert chunks, "compact D549.2 must retrieve the OCR-spaced row"
    top = chunks[0].text
    assert "80.00" in top
    assert "3,504" in top or "3504" in top.replace(",", "")


def test_d5995_carriageway_outranks_excluded_culvert(isolated_store, monkeypatch):
    """WAVE 2 B4: two D599.5 rows — do not prefer the Excluded culvert.

    Live hit was chunk 329: ``J |Breakout and remove existing storm water
    culverts D599.5 | sum 1 Excluded``. Expected: 340,904 m2 @ SAR 31.00.
    The culvert line is not the carriageway quantity.
    """
    from app.core.rag import retriever as ret

    store, e = isolated_store
    culvert = (
        "J |Breakout and remove existing storm water culverts D599.5 "
        "| sum 1 Excluded"
    )
    # Plant the priced row with the OCR space so B4 and B5 share one path.
    carriageway = (
        "D 599.5 Breaking out existing carriageway including road markings "
        "340904 m2 31.00 10568024"
    )
    store.upsert_chunks("proj_b4", "doc_culvert", [culvert], e.encode([culvert]))
    store.upsert_chunks("proj_b4", "doc_cway", [carriageway], e.encode([carriageway]))
    monkeypatch.setattr(
        ret,
        "_doc_name_for_id",
        lambda _id: (
            "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"
        ),
    )

    real_search = store.search

    def scored_search(project_id, query_vec, k=20, query_text=None):
        out = real_search(project_id, query_vec, k=k, query_text=query_text)
        for c in out:
            # Live shape: the Excluded mention outranked on cosine.
            c.score = 0.85 if "Excluded" in c.text else 0.05
        return out

    monkeypatch.setattr(store, "search", scored_search)

    chunks, _ = ret.retrieve_with_filter(
        "BOQ item D599.5 breaking out existing carriageway 340904",
        "proj_b4",
        k=3,
    )
    assert chunks, "expected identifier hits"
    top = chunks[0].text
    assert "340904" in top.replace(",", "")
    assert "31.00" in top
    assert "Excluded" not in top
    assert "culvert" not in top.lower()


def test_unknown_cesmm_code_still_misses(isolated_store):
    """Do not weaken the identifier miss fence — a code not in the index
    must still return no identifier hits."""
    store, e = isolated_store
    text = "D549.2 Removal of existing chain link fence 3504 m 80.00"
    store.upsert_chunks("proj_miss", "doc_x", [text], e.encode([text]))
    assert store.identifier_search("proj_miss", ["D888.9"], k=5) == []


def test_identifier_present_in_text_collapses_ocr_spaced_cesmm():
    """Chat-path presence check (rag_inject identifier-miss gate).

    Live WAVE 2 B5 2026-08-30: retrieve_with_filter already returned the
    OCR-spaced row, but the inject gate split stored ``D 549.2`` into
    ``d``+``549`` and query ``D549.2`` into ``d549``+``2``, then AND-missed.
    Collapse must happen on both sides. An unknown code must still miss.
    """
    from app.core.rag.retriever import identifier_present_in_text

    priced = (
        "D 549.2 Removal of existing chain link fence 3,504 m 80.00 280,320.00"
    )
    assert identifier_present_in_text("D549.2", priced)
    assert identifier_present_in_text("d549.2", priced)
    assert identifier_present_in_text("D 549.2", priced)
    assert not identifier_present_in_text("D888.9", priced)
    # Existing VO Ref matching must keep working.
    assert identifier_present_in_text("VO 99", "VO Ref: 99 was closed")


def test_chat_path_retrieves_ocr_spaced_d5492_without_reindex(
    isolated_store, monkeypatch
):
    """WAVE 2 B5: stored OCR ``D 549.2`` for query ``D549.2`` on the same
    ``retrieve_with_filter`` path chat uses (``rag_inject``).

    #450 covered identifier_search + retrieve_with_filter in isolation.
    Live chat still short-circuited because rag_inject's identifier-miss
    gate did not collapse CESMM spaces. This test drives the real
    retrieve_with_filter (not a mock) through rag_inject, which is what
    ``/v1/chat/stream`` calls. No compact reindex of the stored row.
    """
    from app.core.rag import retriever as ret
    from app.core.rag.inject import rag_inject

    store, e = isolated_store
    spaced = (
        "D 549.2 Removal of existing chain link fence 3,504 m @ SAR 80.00 "
        "= SAR 280,320.00"
    )
    assert "D549.2" not in spaced
    store.upsert_chunks(
        "drive_archive", "d0c00001", [spaced], e.encode([spaced]),
    )
    monkeypatch.setattr(
        ret,
        "_doc_name_for_id",
        lambda _id: "AGII - Infra-1 - Demolition BOQ.pdf",
    )
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")

    b5 = "What is the amount for removal of existing chain link fence (D549.2)?"

    chunks, _ = ret.retrieve_with_filter(b5, "drive_archive", k=5)
    assert chunks, "retrieve_with_filter must surface the OCR-spaced D 549.2 row"
    blob = " ".join(c.text for c in chunks)
    assert "80.00" in blob
    assert "3,504" in blob or "3504" in blob.replace(",", "")

    msg, audit = rag_inject(
        user_message=b5,
        project_id="drive_archive",
        conversation_id="ws-master_corpus-1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert msg is not None, (
        f"chat path discarded the hit: identifier_miss="
        f"{audit.get('identifier_miss')} extracted={audit.get('extracted_identifiers')}"
    )
    assert audit.get("identifier_miss") is not True
    assert "80.00" in msg["content"]
    assert "280,320" in msg["content"] or "280320" in msg["content"].replace(",", "")


def test_chat_path_unknown_cesmm_still_identifier_misses(
    isolated_store, monkeypatch
):
    """Unknown CESMM codes must still trip the identifier-miss fence."""
    from app.core.rag import retriever as ret
    from app.core.rag.inject import rag_inject

    store, e = isolated_store
    text = "D 549.2 Removal of existing chain link fence 3,504 m 80.00 280,320.00"
    store.upsert_chunks("drive_archive", "doc_x", [text], e.encode([text]))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "priced-boq.pdf")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)

    msg, audit = rag_inject(
        user_message="What is the amount for D888.9?",
        project_id="drive_archive",
        conversation_id="ws-master_corpus-1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert msg is None
    assert audit.get("identifier_miss") is True or audit.get("threshold_fired") is True
    assert any("d888.9" in i for i in (audit.get("extracted_identifiers") or []))


def test_a_class_code_matches_compact_and_ocr_spaced_through_indexable_sql(isolated_store):
    """``K418.3`` matches both ``K418.3`` and the OCR-spaced ``K 418.3``, and the
    filter is plain ``LOWER(text) LIKE`` -- the expression the trigram index
    on ``lower(text)`` serves. A ``REPLACE(text, ...)`` filter has no index and
    reads every row of the project."""
    from sqlalchemy import event

    store, e = isolated_store
    texts = ["K418.3 Kerb laying 140 m", "K 418.3 Kerb laying 60 m", "K419.1 Gully 4 nr"]
    for i, t in enumerate(texts):
        store.upsert_chunks("proj_a", f"doc_{i}", [t], e.encode([t]))

    seen = []
    engine = store._session_factory()().get_bind()

    def _capture(conn, cursor, statement, params, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _capture)
    try:
        results = store.identifier_search("proj_a", ["K418.3"], k=5)
    finally:
        event.remove(engine, "before_cursor_execute", _capture)

    assert sorted(c.text for c in results) == sorted(texts[:2])
    reads = [s for s in seen if "LIKE" in s.upper()]
    assert reads and not any("REPLACE(" in s.upper() for s in reads)
