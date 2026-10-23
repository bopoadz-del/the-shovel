"""A PDF that fakes bold by printing every glyph twice is still readable text.

Live bcb5bbf, Set 1 A9 -- the last failing question of 34, 0/4, deterministic:

    "Who is the Engineer under this contract?"
    -> "I hit an internal search formatting issue before I could produce a
        grounded answer."

The Contract Data PDF draws its bold cells by over-printing, and the text
layer comes out as

    PPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss
    EEnnggiinneeeerr EEXXAAMMPPLLEECCOO((EEXX22MM SSaauuddii LLiimmiitteedd))

Every rule downstream then reads junk. The row parser produced
key="PPLLEECCOO((EE: XX22MM SSaauuddii ..." value="Clause (as", the Engineer
extractor returned **"Clause (as"** as the appointed firm, those chunks were
judged to "state the Engineer", and the fence that keeps only such chunks
threw away the one clean copy of the row. The model, told to "state ONLY that
firm's name" over garbage, produced something the leak guard replaced.

Repairing it once, where chunk text is read, fixes every rule at the same
time. The hard part is what must NOT be touched: 1100, 2233, AABB-0011 are
real numbers and codes.
"""
from __future__ import annotations

import pytest

from app.core.rag.text_repair import repair_fake_bold


def test_the_live_lines_read_as_text():
    assert repair_fake_bold("PPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss") == (
        "Party and Engineer details")
    assert repair_fake_bold("EEnnggiinneeeerr EEXXAAMMPPLLEECCOO ((EEXX22MM SSaauuddii LLiimmiitteedd))") == (
        "Engineer EXAMPLECO (EX2M Saudi Limited)")


def test_doubled_punctuation_and_digits_inside_a_doubled_run_are_repaired():
    line = "NN aammee SSuurrnnaammee ((ppeerrssoonn@@eexxaammppllee..ssaa)) 88550000 EExxaammpplleevviillllee"
    assert repair_fake_bold(line) == "N ame Surname (person@example.sa) 8500 Exampleville"


def test_a_tripled_run_is_repaired_too():
    assert repair_fake_bold("PPaarrttyy aaannnddd EEnnggiinneeeerr") == "Party and Engineer"


@pytest.mark.parametrize(
    "text",
    [
        # Real numbers and codes, in ordinary text: never touched.
        "Quantity 1100 m at SAR 2233.00 per unit, drawing AABB-0011, item 7788.",
        "The total is 1,100,000.00 and the page is 44 of 66.",
        # Ordinary words with double letters.
        "The committee will assess the address book and the bookkeeper's success.",
        "See Annex II, Appendix AA and zone CC for access.",
        "Aconex",
        "",
    ],
)
def test_ordinary_text_is_returned_byte_for_byte(text):
    assert repair_fake_bold(text) == text


def test_a_lone_doubled_looking_token_in_clean_text_is_left_alone():
    """One token is a coincidence ("AABB" is a drawing zone). The artefact
    comes in runs: a cell, a line."""
    assert repair_fake_bold("Refer to zone AABBCC on the drawing.") == (
        "Refer to zone AABBCC on the drawing.")


def test_a_clean_word_inside_a_doubled_line_keeps_its_own_double_letters():
    line = "CCoommppaannyy address LLiimmiitteedd committee EExxaammpplleevviillllee"
    assert repair_fake_bold(line) == "Company address Limited committee Exampleville"


def test_two_doubled_looking_tokens_are_still_not_enough_evidence():
    """Three is the bar for a chunk. Two can be a pair of zone codes."""
    text = "Zones AABBCC and DDEEFF are shown on drawing 1100."
    assert repair_fake_bold(text) == text


def test_a_doubled_cell_holding_a_single_word_is_repaired():
    """Found on the real index. A table breaks lines at cells, so the word that
    matters most sits ALONE on its line -- ``|: | EEnnggiinneeeerr`` -- and a
    rule that wanted two doubled words per line left exactly that one alone.
    The evidence is the chunk's; the repair is the line's."""
    text = ("PPaarrttyy aanndd ddeettaaiillss\n"
            "|: | EEnnggiinneeeerr\n"
            "EXAMPLECO(EX2M Saudi Limited): | |\n")
    assert repair_fake_bold(text) == (
        "Party and details\n|: | Engineer\nEXAMPLECO(EX2M Saudi Limited): | |\n")


def test_only_the_affected_lines_change():
    text = "4.3.3(a): | Value of Performance Bond: 10 %\nPPaarrttyy aanndd EEnnggiinneeeerr\n1100 units"
    assert repair_fake_bold(text) == (
        "4.3.3(a): | Value of Performance Bond: 10 %\nParty and Engineer\n1100 units")


def test_it_is_idempotent():
    once = repair_fake_bold("EEnnggiinneeeerr EEXXAAMMPPLLEECCOO ((EEXX22MM))")
    assert repair_fake_bold(once) == once


# ── and the point of it: the Engineer is found ────────────────────────────

LIVE_GARBLED_CHUNK = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
    "11..33..11((bb)): | PPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss: | |\n"
    "|: | EEmmppllooyyeerr | EExxaammppllee CClliieenntt LLiimmiitteedd |\n"
    # Live shape: the role and the firm share ONE cell, and no space
    # survives between the firm and its bracket.
    "|: | EEnnggiinneeeerr EEXXAAMMPPLLEECCOO((EExxaammppllee SSaauuddii LLiimmiitteedd)) |\n"
    "|: | CCoonnttrraaccttoorr | EExxaammppllee CCoonnttrraaccttiinngg CCoommppaannyy |\n"
    "Clause (as amended): | Description | Data |\n"
)


def test_the_engineer_is_read_from_a_repaired_chunk_not_from_the_table_header():
    from app.core.rag.retriever import extract_engineer_identity

    assert extract_engineer_identity(LIVE_GARBLED_CHUNK) in (None, "Clause (as"), (
        "premise: unrepaired, the extractor returns nothing useful")
    fixed = repair_fake_bold(LIVE_GARBLED_CHUNK)
    assert "Engineer EXAMPLECO(Example Saudi Limited)" in fixed
    assert extract_engineer_identity(fixed) == "EXAMPLECO(Example Saudi Limited)"


def test_every_chunk_is_repaired_the_moment_it_is_built():
    """The choke point: not eight read paths that each have to remember."""
    from app.core.rag.vector_store import Chunk

    c = Chunk(chunk_id="x", project_id="p", doc_id="d", chunk_index=0,
              text=LIVE_GARBLED_CHUNK)
    assert "Engineer EXAMPLECO(Example Saudi Limited)" in c.text
    clean = Chunk(chunk_id="y", project_id="p", doc_id="d", chunk_index=1,
                  text="Quantity 1100 m, drawing AABB-0011.")
    assert clean.text == "Quantity 1100 m, drawing AABB-0011."


def test_a_page_sized_key_is_not_a_row_and_a_table_header_is_not_a_firm():
    """Live: a flattened page came through as one 400-character "key" that
    merely contained the word Engineer, value "Clause (as"."""
    from app.core.rag.retriever import extract_engineer_identity

    page = (
        "CONTRACT DATA particulars — filled-in amount [XX_Contract Data.pdf].\n"
        "CONTRACT DATA\n"
        + "Target Local Content Score 42% the Engineer shall review each submission "
        + "and respond within the period stated " * 6
        + "Part No. 1B (Provisional Sums): Clause (as\n"
    )
    assert extract_engineer_identity(page) is None
    assert extract_engineer_identity(
        "CONTRACT DATA\nEngineer: Clause (as amended) | Description | Data\n") is None


def test_a9_the_engineer_chunk_survives_and_leads(monkeypatch):
    """The live pool: garbled Contract Data windows that do NOT name the firm
    rank first, and used to be the only chunks the Engineer fence kept."""
    from app.core.rag import retriever as ret
    from app.core.rag.vector_store import Chunk

    def chunk(cid, text, score, index):
        return Chunk(chunk_id=cid, project_id="p_master", doc_id="cd", chunk_index=index,
                     text=text, score=score)

    header_only = (
        "CONTRACT DATA particulars — filled-in amount [XX-2099-001_Contract Data.pdf].\n"
        "CONTRACT DATA\nPPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss: | | | |\n"
        "aaannnddd: | aaa | ffoorr CCoommmmuunniiccaattiioonnss\n"
        "Clause (as amended): | Description | Data |\n"
    )
    pool = [chunk("hdr", header_only, 0.80, 6), chunk("eng", LIVE_GARBLED_CHUNK, 0.40, 4)]
    vs = "app.core.rag.vector_store.VectorStore."
    monkeypatch.setattr(vs + "search", lambda self, pid, qvec, k, query_text=None: pool[:k])
    monkeypatch.setattr(vs + "identifier_search", lambda self, pid, identifiers, k=20: [])
    monkeypatch.setattr(vs + "chunks_for_docs", lambda self, pid, doc_ids, k_per_doc=12, **_kw: [])
    monkeypatch.setattr(vs + "chunks_containing_all", lambda self, pid, needles, k=20, **_kw: [])
    monkeypatch.setattr(vs + "count", lambda self, pid=None: 11)
    monkeypatch.setattr(vs + "_verify_embedding_identity", lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id",
                        lambda did: "XX-2099-001_Contract Data.pdf", raising=False)
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter("Who is the Engineer under this contract?", "p_master", k=5)

    assert chunks and chunks[0].chunk_id == "eng"
    assert "Engineer EXAMPLECO(Example Saudi Limited)" in chunks[0].text
    assert ret.extract_engineer_identity(chunks[0].text) == "EXAMPLECO(Example Saudi Limited)"


# ── what must NOT be touched, inside a chunk that IS fake-bold ─────────────

DOUBLED_HEAD = "PPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss\n"


def test_a_clean_number_beside_clean_words_survives_in_a_doubled_line():
    """1100 is pair-shaped. On a doubled line it is collapsed only when BOTH
    neighbours were doubled; here its neighbours are ordinary words."""
    line = "CCoommppaannyy LLiimmiitteedd EExxaammpplleevviillllee quantity 1100 metres"
    assert repair_fake_bold(line) == "Company Limited Exampleville quantity 1100 metres"


def test_a_clean_line_in_a_doubled_chunk_is_left_exactly_as_it_was():
    """"Annex II", "zone AA" are pair-shaped and short. The chunk is fake-bold;
    THIS line is not, so nothing on it changes."""
    text = DOUBLED_HEAD + "See Annex II and zone AA, item 1100.\n"
    assert repair_fake_bold(text) == "Party and Engineer details\nSee Annex II and zone AA, item 1100.\n"


def test_a_label_keeps_its_single_colon():
    text = DOUBLED_HEAD + "CCoommmmuunniiccaattiioonnss: 11..33..11((bb)):\n"
    assert repair_fake_bold(text) == "Party and Engineer details\nCommunications: 1.3.1(b):\n"


def test_a_page_sized_key_is_rejected_even_when_its_value_looks_like_a_firm():
    from app.core.rag.retriever import extract_engineer_identity

    page = (
        "CONTRACT DATA particulars — filled-in amount [XX_Contract Data.pdf].\n"
        "CONTRACT DATA\n"
        + "Target Local Content Score 42% the Engineer shall review each submission "
        + "and respond within the period stated " * 6
        + "Part No. 1B (Provisional Sums): Acme Builders Limited\n"
    )
    assert extract_engineer_identity(page) is None
