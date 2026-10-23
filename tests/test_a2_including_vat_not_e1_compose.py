"""Wave-1 A2 on tip 9ad62cc: including-VAT particular, not leftover-E1 compose.

Live Master Corpus (cold New-chat ×3, not contamination):

* Ask: Accepted Contract Amount including VAT (Wave-1 A2).
* Expect: SAR 2,017,680,124.69.
* Got: Delay damages SAR 39,098.39/day (0.1% of Accepted Contract
  Amount SAR 39,098,392.98) citing Contract Data chunk #0.

Leftover E1 on the same SHA can still compose the excl-VAT ACA →
SAR 1,754,504.46/day when it passes. A2 must not enter that compose
path and must late-scan the filled including-VAT row past chunk #0.
Do not invent a figure.
Do not steal A3/A5/A6/A9/B2/E1/C1/F1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.agents.runtime import (
    _CG_REFUSAL,
    _apply_rag_context,
    _graft_asked_contract_particular,
    _graft_composed_delay_damages_daily,
    _latest_operator_ask,
    _latest_user_text,
    _postprocess_answer,
)
from app.core.rag.vector_store import Chunk
from app.lib.construction_formulas_commercial import (
    compose_delay_damages_daily_from_excerpts,
    query_asks_delay_damages_daily_amount,
)


CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json")
    .read_text(encoding="utf-8")
)
A2_ASK = CATALOG["cases"]["A2"]["ask"]
A3_ASK = CATALOG["cases"]["A3"]["ask"]
A5_ASK = CATALOG["cases"]["A5"]["ask"]
A6_ASK = CATALOG["cases"]["A6"]["ask"]
A9_ASK = CATALOG["cases"]["A9"]["ask"]
B2_ASK = CATALOG["cases"]["B2"]["ask"]
C1_ASK = CATALOG["cases"]["C1"]["ask"]
E1_ASK = CATALOG["cases"]["E1"]["ask"]
F1_ASK = CATALOG["cases"]["F1"]["ask"]
LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_A2 = LIVE_PREFIX + A2_ASK
LIVE_E1 = LIVE_PREFIX + (
    "Calculate the delay damages per calendar day in SAR for the "
    "whole of the Works."
)

NET_ACA = 1_754_504_456.25
GROSS_ACA = 2_017_680_124.69
DAILY = 1_754_504.46
# Live A2 FAIL on 9ad62cc: Contract Data chunk #0 partial / toy ACA.
PARTIAL_ACA = 39_098_392.98
PARTIAL_DAILY = 39_098.39
RATE = "0.1% of the Contract Price per calendar day"
ACA_INCL = f"SAR {GROSS_ACA:,.2f}"
ACA_EXCL = f"SAR {NET_ACA:,.2f}"
PARTIAL_ACA_TXT = f"SAR {PARTIAL_ACA:,.2f}"

CD_NAME = (
    "AB-2023-101_QP2 Infra P1_Vol 1.0_Cond of Contract "
    "(complete)_Contract Data.pdf"
)
CHUNK0_DELAY_PARTIAL = (
    "CONTRACT DATA\n"
    "this contract class=project_corpus\n"
    "8.8 Delay Damages for the whole of the Works\n"
    f"{RATE}\n"
    f"Accepted Contract Amount {PARTIAL_ACA_TXT}\n"
)
SCANNED_ACA_INCL = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted \nContract \nAmount (including VAT)\n"
    f"{ACA_INCL}\n"
)
SCANNED_ACA_EXCL = (
    "CONTRACT DATA\n"
    "1.1.1\nAccepted \nContract \nAmount (excluding VAT)\n"
    f"{ACA_EXCL}\n"
)
LIVE_A2_FAIL = (
    "Delay damages for the whole of the Works are "
    f"SAR {PARTIAL_DAILY:,.2f} per calendar day "
    f"(0.1% of Accepted Contract Amount {PARTIAL_ACA_TXT})."
)
RATE_ROW = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    f"8.8 Delay Damages for the whole of the Works: {RATE}"
)
NET_ACA_ROW = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
    f"1.1.1 Accepted Contract Amount excluding VAT | {ACA_EXCL}"
)

ACTIVE = "p_master"
CD_DOC = "cd118"
LATE_INCL_INDEX = 80


def _chunk(cid, doc_id, score, text, chunk_index=0):
    return Chunk(
        chunk_id=cid,
        project_id=ACTIVE,
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def _sys(*chunk_texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def test_a2_is_including_vat_particular_not_e1_compose():
    from app.core.rag.retriever import (
        query_asks_for_aca_including_vat,
        query_is_aca_including_vat_particular,
    )

    assert A2_ASK == "What is the Accepted Contract Amount including VAT?"
    assert query_asks_for_aca_including_vat(LIVE_A2)
    assert query_is_aca_including_vat_particular(LIVE_A2)
    assert query_is_aca_including_vat_particular(A2_ASK)
    assert not query_asks_delay_damages_daily_amount(LIVE_A2)
    assert not query_asks_delay_damages_daily_amount(A2_ASK)
    assert query_asks_delay_damages_daily_amount(LIVE_E1)
    assert not query_is_aca_including_vat_particular(LIVE_E1)
    # Combined wording stays leftover E1 (do not steal #523).
    combined = (
        "Calculate the delay damages per calendar day in SAR using the "
        "Accepted Contract Amount including VAT."
    )
    assert query_asks_delay_damages_daily_amount(combined)
    assert not query_is_aca_including_vat_particular(combined)


def test_compose_does_not_fire_on_a2_even_when_operands_are_present():
    excerpts = "\n\n".join((CHUNK0_DELAY_PARTIAL, SCANNED_ACA_EXCL, SCANNED_ACA_INCL))
    assert compose_delay_damages_daily_from_excerpts(LIVE_A2, excerpts) is None
    assert compose_delay_damages_daily_from_excerpts(A2_ASK, excerpts) is None
    e1 = compose_delay_damages_daily_from_excerpts(
        LIVE_E1, "\n\n".join((RATE_ROW, NET_ACA_ROW)),
    )
    assert e1 is not None
    assert e1["daily_amount"] == DAILY
    assert e1["contract_amount"] == NET_ACA


def test_graft_compose_is_noop_on_a2_live_fail_shape():
    rag = _sys(CHUNK0_DELAY_PARTIAL, RATE_ROW, NET_ACA_ROW, SCANNED_ACA_INCL)
    msgs = [{"role": "user", "content": LIVE_A2}]
    assert _graft_composed_delay_damages_daily(LIVE_A2_FAIL, rag, msgs) == (
        LIVE_A2_FAIL
    )
    e1_msgs = [{"role": "user", "content": LIVE_E1}]
    out = _graft_composed_delay_damages_daily(LIVE_A2_FAIL, rag, e1_msgs)
    assert "1,754,504.46" in out
    assert f"{PARTIAL_DAILY:,.2f}" not in out.split("\n", 1)[0]


def test_graft_a2_replaces_live_partial_daily_when_incl_is_in_excerpts():
    rag = _sys(CHUNK0_DELAY_PARTIAL, SCANNED_ACA_INCL)
    msgs = [{"role": "user", "content": LIVE_A2}]
    out = _graft_asked_contract_particular(LIVE_A2_FAIL, rag, msgs)
    assert "2,017,680,124.69" in out
    assert "including VAT" in out
    assert "39,098.39" not in out
    assert "delay damages" not in out.lower()


def test_postprocess_a2_live_fail_becomes_including_vat():
    rag = _sys(CHUNK0_DELAY_PARTIAL, SCANNED_ACA_INCL)
    msgs = [{"role": "user", "content": LIVE_A2}]
    out = _postprocess_answer(LIVE_A2_FAIL, rag, msgs)
    assert "2,017,680,124.69" in out
    assert out != LIVE_A2_FAIL
    assert out != _CG_REFUSAL
    first = out.split("\n", 1)[0]
    assert "2,017,680,124.69" in first
    assert "39,098.39" not in first
    assert "delay damages" not in first.lower()


def _chunk0_and_late_incl():
    delay = _chunk("cd0", CD_DOC, 0.94, CHUNK0_DELAY_PARTIAL, chunk_index=0)
    incl = _chunk(
        "cd80", CD_DOC, 0.18, SCANNED_ACA_INCL, chunk_index=LATE_INCL_INDEX,
    )
    names = {CD_DOC: CD_NAME}
    seeded = [{"id": CD_DOC, "original_name": CD_NAME, "file_path": CD_NAME}]
    return delay, incl, names, seeded


def _install_chunk0_misses_incl(monkeypatch, *, delay, late, names, seeded):
    """identifier_search + first-N stay on chunk #0; all_rows sees late incl."""
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in [delay] if c.project_id == project_id][:k]

    def fake_id_search(self, project_id, identifiers, k=20):
        return [delay][:k]

    def fake_chunks_for_docs(
        self, project_id, doc_ids, k_per_doc=12, from_end=False, all_rows=False,
    ):
        rows = [c for c in (delay, late) if c.doc_id in (doc_ids or [])]
        rows = sorted(rows, key=lambda c: int(c.chunk_index or 0))
        if all_rows:
            return rows
        n = max(1, int(k_per_doc or 12))
        # Live 9ad62cc: first-N is refuse-prone chunk #0; the filled
        # including-VAT row sits at appendix index 80.
        if from_end:
            return [late][:n]
        return [delay][:n]

    def fake_containing_all(self, project_id, needles, k=20, doc_ids=None):
        needles_l = [str(n).lower() for n in (needles or [])]
        out = []
        for chunk in (delay, late):
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            text_l = (chunk.text or "").lower()
            if needles_l and all(n in text_l for n in needles_l):
                out.append(chunk)
        return out[: max(1, int(k or 20))]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.identifier_search", fake_id_search,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        fake_chunks_for_docs,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_containing_all",
        fake_containing_all,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.count", lambda self, pid=None: 2,
    )
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""),
                        raising=False)
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda pid, phrase, limit=8: (
            list(seeded)
            if "contract data" in (phrase or "").lower()
            or "conditions of contract" in (phrase or "").lower()
            else []
        ),
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


def test_a2_late_scan_surfaces_incl_vat_past_chunk0_delay(monkeypatch):
    delay, incl, names, seeded = _chunk0_and_late_incl()
    ret = _install_chunk0_misses_incl(
        monkeypatch, delay=delay, late=incl, names=names, seeded=seeded,
    )
    chunks, _ = ret.retrieve_with_filter(LIVE_A2, ACTIVE, k=5)
    blob = " ".join(c.text for c in chunks)
    assert chunks
    assert ACA_INCL in blob
    assert ACA_INCL in chunks[0].text
    assert PARTIAL_ACA_TXT not in blob
    assert "39,098.39" not in blob
    assert compose_delay_damages_daily_from_excerpts(LIVE_A2, blob) is None


def test_a2_volume_helper_returns_incl(monkeypatch):
    from app.core.rag.retriever import including_vat_excerpts_from_loaded_cd_volume

    delay, incl, _names, _seeded = _chunk0_and_late_incl()

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12,
            from_end=False, all_rows=False,
        ):
            rows = [delay, incl]
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
            return []

    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda *a, **k: [],
    )
    extra = including_vat_excerpts_from_loaded_cd_volume(
        LIVE_A2, ACTIVE, _Store(),
        rag_context=_sys(CHUNK0_DELAY_PARTIAL)["content"],
        doc_ids=[CD_DOC],
    )
    assert ACA_INCL in extra
    assert PARTIAL_ACA_TXT not in extra
    assert including_vat_excerpts_from_loaded_cd_volume(
        LIVE_E1, ACTIVE, _Store(), doc_ids=[CD_DOC],
    ) == ""


def test_graft_a2_last_chance_from_loaded_cd_volume(monkeypatch):
    from app.core.rag.retriever import including_vat_excerpts_from_loaded_cd_volume

    delay, incl, _names, _seeded = _chunk0_and_late_incl()

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12,
            from_end=False, all_rows=False,
        ):
            return [delay, incl] if all_rows else [delay]

        def chunks_containing_all(self, *a, **k):
            return []

    monkeypatch.setattr(
        "app.core.rag.retriever.get_lexical_store",
        lambda: _Store(),
    )
    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda *a, **k: [{"id": CD_DOC, "original_name": CD_NAME}],
    )
    extra = including_vat_excerpts_from_loaded_cd_volume(
        LIVE_A2, ACTIVE, rag_context=_sys(CHUNK0_DELAY_PARTIAL)["content"],
    )
    assert ACA_INCL in extra
    rag = _sys(CHUNK0_DELAY_PARTIAL)
    msgs = [{"role": "user", "content": LIVE_A2}]
    out = _graft_asked_contract_particular(
        LIVE_A2_FAIL, rag, msgs, project_id=ACTIVE,
    )
    assert "2,017,680,124.69" in out
    assert "39,098.39" not in out
    assert "delay damages" not in out.lower()


def test_a2_volume_helper_does_not_steal_neighbor_asks(monkeypatch):
    from app.core.rag.retriever import including_vat_excerpts_from_loaded_cd_volume

    delay, incl, _names, _seeded = _chunk0_and_late_incl()

    class _Store:
        def chunks_for_docs(self, *a, **k):
            return [delay, incl]

        def chunks_containing_all(self, *a, **k):
            return []

    for ask in (A3_ASK, A5_ASK, A6_ASK, A9_ASK, B2_ASK, C1_ASK, E1_ASK, F1_ASK):
        assert including_vat_excerpts_from_loaded_cd_volume(
            ask, ACTIVE, _Store(), doc_ids=[CD_DOC],
        ) == ""


def test_ensure_a2_kept_replaces_chunk0_delay_with_incl():
    from app.core.rag.retriever import ensure_kept_has_including_vat

    delay, incl, _names, _seeded = _chunk0_and_late_incl()
    kept = [delay]
    ranked = [delay, incl]
    assert ensure_kept_has_including_vat(LIVE_A2, kept, ranked) is True
    assert ACA_INCL in kept[0].text
    assert PARTIAL_ACA_TXT not in kept[0].text
    assert ensure_kept_has_including_vat(LIVE_E1, [delay], ranked) is False


def test_chunk0_partial_aca_is_not_including_vat():
    from app.core.rag.retriever import (
        chunk_states_aca_including_vat,
        extract_aca_including_vat,
    )

    assert not chunk_states_aca_including_vat(CHUNK0_DELAY_PARTIAL)
    assert extract_aca_including_vat(CHUNK0_DELAY_PARTIAL) is None
    assert extract_aca_including_vat(
        CHUNK0_DELAY_PARTIAL + "\n\n" + SCANNED_ACA_INCL,
    ) == (GROSS_ACA, "SAR")


def _fold_live_a2_chunk0(rag=None):
    """Live 396cc7b path: RAG-fold chunk #0 into the last user bubble."""
    rag = rag or _sys(CHUNK0_DELAY_PARTIAL)
    msgs = [{"role": "user", "content": LIVE_A2}]
    assert _apply_rag_context(msgs, rag) is True
    return rag, msgs


def test_rag_folded_chunk0_looks_like_e1_until_unwrapped():
    """#539 live miss: folded chunk #0 + lookup 'compute' classifies as E1."""
    from app.core.rag.retriever import (
        query_asks_delay_damages_daily_amount,
        query_is_aca_including_vat_particular,
    )

    _rag, msgs = _fold_live_a2_chunk0()
    folded = _latest_user_text(msgs)
    assert "Delay Damages" in folded
    assert re.search(r"(?i)\bcompute\b", folded)
    assert "including VAT" in folded
    # Detectors that read the folded bubble steal A2 onto leftover E1.
    assert query_asks_delay_damages_daily_amount(folded)
    assert not query_is_aca_including_vat_particular(folded)
    ask = _latest_operator_ask(msgs)
    assert ask.strip() == LIVE_A2
    assert query_is_aca_including_vat_particular(ask)
    assert not query_asks_delay_damages_daily_amount(ask)


def test_graft_compose_is_noop_on_rag_folded_a2_live_fail():
    rag, msgs = _fold_live_a2_chunk0(
        _sys(CHUNK0_DELAY_PARTIAL, RATE_ROW, NET_ACA_ROW, SCANNED_ACA_INCL),
    )
    assert _graft_composed_delay_damages_daily(LIVE_A2_FAIL, rag, msgs) == (
        LIVE_A2_FAIL
    )
    e1_msgs = [{"role": "user", "content": LIVE_E1}]
    _apply_rag_context(e1_msgs, rag)
    out = _graft_composed_delay_damages_daily(LIVE_A2_FAIL, rag, e1_msgs)
    assert "1,754,504.46" in out
    assert f"{PARTIAL_DAILY:,.2f}" not in out.split("\n", 1)[0]


def test_postprocess_rag_folded_a2_live_fail_becomes_including_vat():
    """Live 396cc7b New-chat A2: folded chunk #0 compose must be replaced."""
    rag, msgs = _fold_live_a2_chunk0(_sys(CHUNK0_DELAY_PARTIAL, SCANNED_ACA_INCL))
    out = _postprocess_answer(LIVE_A2_FAIL, rag, msgs, project_id=ACTIVE)
    assert "2,017,680,124.69" in out
    assert out != LIVE_A2_FAIL
    assert out != _CG_REFUSAL
    first = out.split("\n", 1)[0]
    assert "2,017,680,124.69" in first
    assert "39,098.39" not in first
    assert "delay damages" not in first.lower()


def test_graft_a2_last_chance_scans_cited_chunk_owner_pid(monkeypatch):
    """Master Corpus UI id is empty; including-VAT lives on the source pid."""
    from app.core.rag.retriever import including_vat_excerpts_from_loaded_cd_volume

    source_pid = "p_dd118"
    delay = _chunk("cd0", CD_DOC, 0.94, CHUNK0_DELAY_PARTIAL, chunk_index=0)
    delay.project_id = source_pid
    incl = _chunk(
        "cd80", CD_DOC, 0.18, SCANNED_ACA_INCL, chunk_index=LATE_INCL_INDEX,
    )
    incl.project_id = source_pid

    class _Store:
        def chunks_for_docs(
            self, project_id, doc_ids, k_per_doc=12,
            from_end=False, all_rows=False,
        ):
            if project_id != source_pid:
                return []
            rows = [delay, incl]
            if all_rows:
                return rows
            n = max(1, int(k_per_doc or 12))
            return rows[-n:] if from_end else rows[:n]

        def chunks_containing_all(self, *a, **k):
            return []

    monkeypatch.setattr(
        "app.core.projects.documents_matching_title_phrase",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "app.core.rag.retriever.get_lexical_store",
        lambda: _Store(),
    )
    rag_ctx = (
        f"[doc_id={CD_DOC} chunk=0 score=0.80] {CHUNK0_DELAY_PARTIAL}"
    )
    extra = including_vat_excerpts_from_loaded_cd_volume(
        LIVE_A2, ACTIVE, _Store(),
        rag_context=rag_ctx,
        extra_pids=[source_pid],
    )
    assert ACA_INCL in extra
    assert PARTIAL_ACA_TXT not in extra
    rag = {"role": "system", "content": "Reference context:\n" + rag_ctx}
    msgs = [{"role": "user", "content": LIVE_A2}]
    _apply_rag_context(msgs, rag)
    out = _graft_asked_contract_particular(
        LIVE_A2_FAIL, rag, msgs, project_id=ACTIVE,
        extra_project_ids=[source_pid],
    )
    assert "2,017,680,124.69" in out
    assert "39,098.39" not in out
    assert "delay damages" not in out.lower()
