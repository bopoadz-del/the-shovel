"""WAVE 2 B4/B5: priced CESMM rows must not elect storm-water Rate Only.

Live Master Corpus on fa07b2f (tip #540) answered both asks with:

    D599.5 / D549.2 (removal of storm water culverts) is Rate Only.
    No amount exists for this item in the client BOQ — do not invent a total.

The Demolition BOQ OCR line holds D529.3 Rate Only beside the priced
D599.5 carriageway (340,904 @ 31.00 = 10,568,024) and D549.2 fence
(3,504 × 80.00 = 280,320). G4 must still elect Rate Only for D529.3.
A2/E1 compose and R2 credentials are not this path. Fixture wording only.
"""
from __future__ import annotations

from app.agents.runtime import _graft_rate_only_item, _postprocess_answer
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


LIVE_PREFIX = "Answer only from the client project documents. "
B4_ASK = (
    "What is the quantity and amount for breaking out existing "
    "carriageway including road markings (D599.5)?"
)
B5_ASK = (
    "What is the amount for removal of existing chain link fence (D549.2)?"
)
G4_ASK = (
    "What is the total amount for removal of storm water culverts (D529.3)?"
)
LIVE_B4 = LIVE_PREFIX + B4_ASK
LIVE_B5 = LIVE_PREFIX + B5_ASK
LIVE_G4 = LIVE_PREFIX + G4_ASK

# Live figures. Do not invent a different client total.
B4_QTY = "340904"
B4_RATE = "31.00"
B4_AMT = "10568024"
B5_QTY = "3504"
B5_RATE = "80.00"
B5_AMT = "280320"

SOUP = (
    "D529.3 Removal of storm water culverts — m 1,370.00 Rate Only "
    f"D549.2 Removal of existing chain link fence {B5_QTY} m {B5_RATE} "
    f"{B5_AMT}.00 "
    "D599.5 Breaking out existing carriageway including road markings "
    f"{B4_QTY} m2 {B4_RATE} {B4_AMT}"
)
OCR_SPACED_SOUP = (
    "D 529.3 Removal of storm water culverts — m 1,370.00 Rate Only "
    f"D 549.2 Removal of existing chain link fence 3,504 m {B5_RATE} "
    "280,320.00 "
    "D 599.5 Breaking out existing carriageway including road markings "
    "340,904 m2 31.00 10,568,024"
)
PIPE_SOUP = (
    "| D529.3 | Removal of storm water culverts | — | m | 1,370.00 "
    "| Rate Only | D549.2 | Removal of existing chain link fence | "
    f"{B5_QTY} | m | {B5_RATE} | {B5_AMT} | D599.5 | Breaking out "
    f"existing carriageway including road markings | {B4_QTY} | m2 | "
    f"{B4_RATE} | {B4_AMT} |"
)

PRICED_B4_ANSWER = (
    "D599.5 breaking out existing carriageway: quantity 340,904 m2 "
    "@ 31.00 = 10,568,024."
)
PRICED_B5_ANSWER = (
    "D549.2 removal of existing chain link fence amount is 280,320 "
    "(3,504 × 80.00)."
)
STORM_WATER_RATE_ONLY = "removal of storm water culverts"

ACTIVE = "p_master"
SOUP_DOC = "soup"


def _chunk(cid, doc_id, score, text):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_live_b4_b5_asks_are_frozen_item_amount_queries():
    from app.core.rag.retriever import (
        extract_asked_cesmm_codes,
        query_asks_for_boq_item_amount,
    )

    assert query_asks_for_boq_item_amount(B4_ASK)
    assert query_asks_for_boq_item_amount(LIVE_B4)
    assert query_asks_for_boq_item_amount(B5_ASK)
    assert query_asks_for_boq_item_amount(LIVE_B5)
    assert extract_asked_cesmm_codes(LIVE_B4) == ["d599.5"]
    assert extract_asked_cesmm_codes(LIVE_B5) == ["d549.2"]
    assert extract_asked_cesmm_codes(LIVE_G4) == ["d529.3"]


def test_same_line_ocr_soup_does_not_stain_priced_b4_b5_rows():
    from app.core.rag.retriever import (
        _cesmm_row_windows,
        chunk_states_rate_only_item,
    )

    for blob in (SOUP, OCR_SPACED_SOUP, PIPE_SOUP):
        assert chunk_states_rate_only_item(blob, ["d529.3"]), blob
        assert not chunk_states_rate_only_item(blob, ["d599.5"]), blob
        assert not chunk_states_rate_only_item(blob, ["d549.2"]), blob
        cway = " ".join(_cesmm_row_windows(blob, "d599.5"))
        fence = " ".join(_cesmm_row_windows(blob, "d549.2"))
        culvert = " ".join(_cesmm_row_windows(blob, "d529.3"))
        assert "Rate Only" in culvert
        assert STORM_WATER_RATE_ONLY in culvert.lower()
        assert B4_QTY in cway.replace(",", "")
        assert "Rate Only" not in cway
        assert STORM_WATER_RATE_ONLY not in cway.lower()
        assert B5_AMT in fence.replace(",", "")
        assert "Rate Only" not in fence
        assert STORM_WATER_RATE_ONLY not in fence.lower()


def test_format_rate_only_line_does_not_blame_carriageway_or_fence():
    from app.core.rag.retriever import format_rate_only_line

    b4 = format_rate_only_line(["d599.5"], SOUP)
    b5 = format_rate_only_line(["d549.2"], SOUP)
    g4 = format_rate_only_line(["d529.3"], SOUP)
    assert STORM_WATER_RATE_ONLY not in b4.lower()
    assert STORM_WATER_RATE_ONLY not in b5.lower()
    assert STORM_WATER_RATE_ONLY in g4.lower()
    assert "D529.3" in g4
    assert "Rate Only" in g4


def _soup_rag() -> dict:
    return {
        "role": "system",
        "content": (
            "[doc_id=soup chunk=0 score=0.90] " + SOUP
        ),
    }


def test_graft_does_not_replace_priced_b4_b5_when_soup_is_in_rag():
    rag = _soup_rag()
    b4 = _graft_rate_only_item(
        PRICED_B4_ANSWER, rag, [{"role": "user", "content": LIVE_B4}],
    )
    b5 = _graft_rate_only_item(
        PRICED_B5_ANSWER, rag, [{"role": "user", "content": LIVE_B5}],
    )
    assert b4 == PRICED_B4_ANSWER
    assert b5 == PRICED_B5_ANSWER
    assert STORM_WATER_RATE_ONLY not in b4.lower()
    assert STORM_WATER_RATE_ONLY not in b5.lower()


def test_graft_still_states_rate_only_for_g4_soup():
    rag = _soup_rag()
    out = _graft_rate_only_item(
        "I'm ready to help. Please let me know what you need.",
        rag,
        [{"role": "user", "content": LIVE_G4}],
    )
    assert "Rate Only" in out
    assert "D529.3" in out
    assert "no amount" in out.lower()
    assert B4_AMT not in out.replace(",", "")


def test_postprocess_b4_b5_keeps_priced_figures_on_soup():
    rag = _soup_rag()
    b4 = _postprocess_answer(
        PRICED_B4_ANSWER, rag, [{"role": "user", "content": LIVE_B4}],
    )
    b5 = _postprocess_answer(
        PRICED_B5_ANSWER, rag, [{"role": "user", "content": LIVE_B5}],
    )
    assert "10,568,024" in b4 or "10568024" in b4.replace(",", "")
    assert "280,320" in b5 or "280320" in b5.replace(",", "")
    assert STORM_WATER_RATE_ONLY not in b4.lower()
    assert STORM_WATER_RATE_ONLY not in b5.lower()


def test_inject_does_not_fire_rate_only_header_on_b4_or_b5_soup():
    chunks = [_chunk("soup", SOUP_DOC, 0.9, SOUP)]
    b4 = format_chunks_as_system_message(chunks, 4, query=LIVE_B4)["content"]
    b5 = format_chunks_as_system_message(chunks, 4, query=LIVE_B5)["content"]
    g4 = format_chunks_as_system_message(chunks, 4, query=LIVE_G4)["content"]
    assert "RATE ONLY" not in b4
    assert "RATE ONLY" not in b5
    assert "RATE ONLY" in g4
    assert "PRICED BOQ ROW" in b4
    assert "PRICED BOQ ROW" in b5
    assert "PRICED BOQ ROW" not in g4
    assert B4_QTY in b4
    assert B5_AMT in b5


def _install_soup_corpus(monkeypatch, *, extra_priced: bool):
    from app.core.rag import retriever as ret

    soup = _chunk("soup", SOUP_DOC, 0.92, SOUP)
    priced_cway = _chunk(
        "cway",
        "cway",
        0.11,
        "D599.5 Breaking out existing carriageway including road markings "
        f"{B4_QTY} m2 {B4_RATE} {B4_AMT}",
    )
    priced_fence = _chunk(
        "fence",
        "fence",
        0.10,
        "D549.2 Removal of existing chain link fence "
        f"{B5_QTY} m {B5_RATE} {B5_AMT}.00",
    )
    semantic = [soup]
    if extra_priced:
        semantic.extend([priced_cway, priced_fence])

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers).lower()
        hits = []
        if "599.5" in blob or "d599" in blob:
            hits.append(priced_cway if extra_priced else soup)
        if "549.2" in blob or "d549" in blob:
            hits.append(priced_fence if extra_priced else soup)
        if "529.3" in blob or "d529" in blob:
            hits.append(soup)
        return hits[:k]

    def fake_containing_all(self, project_id, needles, k=20):
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        hay = SOUP.lower().replace(" ", "")
        if cleaned and all(
            n.replace(" ", "") in hay or n in SOUP.lower() for n in cleaned
        ):
            return [soup]
        return []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, project_id, doc_ids, k_per_doc=12: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 3,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(
        ret, "_doc_name_for_id",
        lambda did: "AGII - Infra-1 - Demolition BOQ.pdf",
        raising=False,
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: [],
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_retrieve_b4_elects_carriageway_not_storm_water_rate_only(monkeypatch):
    ret = _install_soup_corpus(monkeypatch, extra_priced=True)
    chunks, _ = ret.retrieve_with_filter(LIVE_B4, ACTIVE, k=5)
    assert chunks
    top = chunks[0].text
    blob = " ".join(c.text for c in chunks)
    assert B4_QTY in top.replace(",", "")
    assert B4_RATE in top
    assert B4_AMT in top.replace(",", "")
    assert not ret.chunk_states_rate_only_item(top, ["d599.5"])
    # Do not drop the priced row just because D529.3 Rate Only is in-pool.
    assert any(B4_AMT in (c.text or "").replace(",", "") for c in chunks)
    assert not all(
        ret.chunk_states_rate_only_item(c.text, ["d599.5"]) for c in chunks
    )
    # Storm-water Rate Only must not be the elected answer class.
    assert "Rate Only" not in top or "carriageway" in top.lower()
    assert STORM_WATER_RATE_ONLY not in top.lower() or B4_QTY in top.replace(",", "")
    # Fence amount is a neighbor, not the B4 answer — top must be carriageway.
    assert "carriageway" in top.lower()
    _ = blob


def test_retrieve_b5_elects_fence_not_storm_water_rate_only(monkeypatch):
    ret = _install_soup_corpus(monkeypatch, extra_priced=True)
    chunks, _ = ret.retrieve_with_filter(LIVE_B5, ACTIVE, k=5)
    assert chunks
    top = chunks[0].text
    assert B5_AMT in top.replace(",", "")
    assert B5_RATE in top
    assert "chain link" in top.lower() or "fence" in top.lower()
    assert not ret.chunk_states_rate_only_item(top, ["d549.2"])
    assert not all(
        ret.chunk_states_rate_only_item(c.text, ["d549.2"]) for c in chunks
    )


def test_retrieve_b4_soup_only_keeps_priced_carriageway_figures(monkeypatch):
    """Live shape: one OCR chunk holds every D-row. Do not fence to Rate Only."""
    ret = _install_soup_corpus(monkeypatch, extra_priced=False)
    chunks, _ = ret.retrieve_with_filter(LIVE_B4, ACTIVE, k=5)
    assert chunks
    blob = " ".join(c.text for c in chunks)
    assert B4_QTY in blob.replace(",", "")
    assert B4_AMT in blob.replace(",", "")
    assert any(not ret.chunk_states_rate_only_item(c.text, ["d599.5"]) for c in chunks)


def test_retrieve_g4_still_elects_rate_only_from_soup(monkeypatch):
    ret = _install_soup_corpus(monkeypatch, extra_priced=True)
    chunks, _ = ret.retrieve_with_filter(LIVE_G4, ACTIVE, k=5)
    assert chunks
    assert any(ret.chunk_states_rate_only_item(c.text, ["d529.3"]) for c in chunks)
    assert "Rate Only" in chunks[0].text


PART_NR_3_PRICED = (
    "PART NR. 3 DEMOLITION\n"
    "D 549.2 Removal of existing chain link fence 3,504 m @ SAR 80.00 "
    f"= SAR 280,320.00"
)
RATE_ONLY_D549_SIBLING = (
    "PART NR. 3\n"
    "D549.2 Removal of existing chain link fence — m Rate Only"
)
EXCLUDED_D549_SIBLING = (
    "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B Bill of Quantities (Priced)\n"
    "D549.2 Removal of existing chain link fence | sum 1 Excluded"
)


def _install_b5_sibling_corpus(monkeypatch):
    """Live be93dee shape: priced Part Nr. 3 plus Rate Only / Excluded."""
    from app.core.rag import retriever as ret

    priced = _chunk("part3", "part3", 0.11, PART_NR_3_PRICED)
    rate_only = _chunk("ro", "ro", 0.93, RATE_ONLY_D549_SIBLING)
    excluded = _chunk("ex", "ex", 0.88, EXCLUDED_D549_SIBLING)
    semantic = [rate_only, excluded, priced]

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        blob = " ".join(identifiers).lower()
        if "549.2" in blob or "d549" in blob:
            return [rate_only, excluded, priced][:k]
        return []

    def fake_containing_all(self, project_id, needles, k=20):
        cleaned = [" ".join((n or "").lower().split()) for n in (needles or [])]
        hits = []
        for chunk in semantic:
            hay = (chunk.text or "").lower()
            if cleaned and all(n in hay for n in cleaned):
                hits.append(chunk)
        return hits[:k]

    def _name(did):
        if did == "ex":
            return (
                "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_"
                "Bill of Quantities (Priced).pdf"
            )
        return "Demolition BOQ Part Nr. 3.pdf"

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, project_id, doc_ids, k_per_doc=12: [],
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 3,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", _name, raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: [],
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_filename_terms",
        lambda *a, **k: [],
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("COMPOSE_PRICED_BOQ_ROW", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_retrieve_b5_prefers_priced_part_nr_3_over_rate_only_and_excluded(
    monkeypatch,
):
    """Rate Only + Excluded siblings must not fence out 280,320."""
    ret = _install_b5_sibling_corpus(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(LIVE_B5, ACTIVE, k=5)
    assert chunks
    blob = " ".join(c.text for c in chunks)
    assert B5_AMT in blob.replace(",", "")
    top = chunks[0].text
    assert B5_AMT in top.replace(",", "")
    assert ret.chunk_states_priced_item(top, ["d549.2"])
    assert not ret.chunk_states_rate_only_item(top, ["d549.2"])
    assert not ret.chunk_states_excluded_item(top, ["d549.2"])
    # Priced fence drops Rate Only / Excluded-only siblings.
    assert all(ret.chunk_states_priced_item(c.text, ["d549.2"]) for c in chunks)
    assert not any(
        ret.chunk_states_rate_only_item(c.text, ["d549.2"])
        and not ret.chunk_states_priced_item(c.text, ["d549.2"])
        for c in chunks
    )


def test_named_rate_only_survives_other_contract_priced_row():
    """Named-year Rate Only must not empty when another year is priced.

    Codex #558: AB-2022-202 D529.3 is Rate Only; AB-2023-101 has a
    priced lookalike. The priced fence used the whole pool, dropped
    the named Rate Only row, then the named-id check dropped the
    other year's priced row — retrieval went empty.
    """
    from app.core.rag.retriever import _ContractScope

    ask = (
        LIVE_PREFIX
        + "Under AB-2022-202, what is the total amount for removal of "
        "storm water culverts (D529.3)?"
    )
    named_ro = (
        "AB-2022-202 - Demolition BOQ.pdf",
        "D529.3 Removal of storm water culverts — m 1,370.00 Rate Only",
    )
    other_priced = (
        "AB-2023-101 - Demolition BOQ.pdf",
        "D529.3 Removal of storm water culverts 1,370 m 50.00 68,500.00",
    )
    scope = _ContractScope(ask, [named_ro, other_priced])
    assert scope.named
    assert scope._rate_only_in_pool
    assert not scope._priced_item_in_pool
    assert scope.allow(*named_ro)
    assert not scope.allow(*other_priced)


def test_named_priced_still_fences_same_contract_rate_only_sibling():
    """Naming the year must not weaken #558 same-contract priced election."""
    from app.core.rag.retriever import _ContractScope

    ask = LIVE_PREFIX + "Under AB-2023-101, " + B5_ASK
    named_priced = (
        "AB-2023-101 - Demolition BOQ Part Nr. 3.pdf",
        PART_NR_3_PRICED,
    )
    named_ro = (
        "AB-2023-101 - Demolition BOQ Rate Only.pdf",
        RATE_ONLY_D549_SIBLING,
    )
    other_ex = (
        "AB-2022-202 - Bill of Quantities (Priced).pdf",
        EXCLUDED_D549_SIBLING,
    )
    scope = _ContractScope(ask, [named_priced, named_ro, other_ex])
    assert scope.named
    assert scope._priced_item_in_pool
    assert scope.allow(*named_priced)
    assert not scope.allow(*named_ro)
    assert not scope.allow(*other_ex)


def test_e1_scope_keeps_cd_operands_when_priced_fence_rows_are_in_pool():
    """Leftover E1 after #559: priced D549.2 must not drop Contract Data.

    Unnamed calculate-in-SAR is rate × ACA, not a CESMM quote. A priced
    fence row in the same pool used to look like a unit-rate miss.
    """
    from app.core.rag.retriever import (
        _ContractScope,
        query_asks_delay_damages_daily_amount,
        query_asks_for_boq_item_amount,
    )

    e1 = (
        LIVE_PREFIX
        + "Calculate the delay damages per calendar day in SAR for the "
        "whole of the Works."
    )
    rate = (
        "AB-2023-101 - Contract Data.pdf",
        "CONTRACT DATA particulars.\n"
        "8.8 Delay Damages for the whole of the Works: "
        "0.1% of the Contract Price per calendar day",
    )
    aca = (
        "AB-2023-101 - Contract Data.pdf",
        "CONTRACT DATA particulars.\n"
        "1.1.1 Accepted Contract Amount excluding VAT | "
        "SAR 1,754,504,456.25",
    )
    priced = (
        "AB-2023-101 - Demolition BOQ Part Nr. 3.pdf",
        PART_NR_3_PRICED,
    )
    assert query_asks_delay_damages_daily_amount(e1)
    assert not query_asks_for_boq_item_amount(e1)
    scope = _ContractScope(e1, [priced, rate, aca])
    assert not scope.named
    assert not scope._priced_item_in_pool
    assert not scope._rate_only_in_pool
    assert scope.allow(*rate)
    assert scope.allow(*aca)
    # A CESMM row is not an E1 operand. The compose fence may drop it;
    # the priced fence must not have been the reason.
