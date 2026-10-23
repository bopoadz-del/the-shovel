"""A page total belongs to the page printed under it -- and to no other.

Live on b64bbd2, three runs, 0/3:

    "What is the Part Summary total for page d/3/1 of the Demolition and
     Site Clearance bill?"

The figure is in the index, in one chunk, next to its page number. It was not
retrieved, and the reason is worse than a ranking miss.

A scanned bill prints each page like this, and the chunker cuts it like this:

    ...last item  To Part Summary  <stamp noise>  SAR 1,234,567.00
    Date: ...  RFP No. ...  Classification - Public  Page d/3/1  124 of 675
    PROJECT: ...  BILL SECTION d  PART Nr. 3  PAGE Nr. d/3/2  Item Description

Three things follow from that shape, and the code had all three wrong:

1. The page number sits ~170 characters AFTER the total. The composer looked
   80 characters ahead, never saw it, and treated the total as unlabelled.
2. An unlabelled total was accepted as the asked page. With 675 pages, dozens
   of footers have an unreadable page number; every one of them tied at the
   same score as the right chunk, and a lone one would have been STATED to
   the user as "the d/3/1 total".
3. ``PAGE Nr. d/3/2`` on the same chunk is the NEXT page's header. It made
   the d/3/1 total count as evidence for d/3/2 as well.

And OCR glues the footer: ``Paged/3/12`` has no word boundary before the
``d``, so it was read as "no page number" -- and then accepted as d/3/1.
"""
from __future__ import annotations

import pytest

from app.core.rag.retriever import (
    chunk_states_part_summary_total,
    compose_part_summary_total,
)
from app.core.rag.vector_store import Chunk

ASK = ("What is the Part Summary total for page d/3/1 of the Demolition and "
       "Site Clearance bill?")
ASK_NEXT_PAGE = ASK.replace("d/3/1", "d/3/2")
ASK_PAGE_12 = ASK.replace("d/3/1", "d/3/12")

FOOTER = (
    "Date: July 10, 2099 Procurement of Contractor for Example Package 1 "
    "RFP No. XX-2099-001 Classification - Public "
)
HEADER = (
    "PROJECT: EXAMPLE INFRASTRUCTURE PACKAGE 1 BILL OF QUANTITIES DEMOLITION "
    "AND SITE CLEARANCE BILL SECTION d PART Nr. 3 "
)

# The real cut: total, then the footer naming ITS page, then the next header.
PAGE_1 = (
    "scape areas D290.2 Nr 1,239 1,185.00 1,468,215.00 liy jlu 4 J !l r "
    "To Part Summary U,. .Io. 1 K'l.A SAR 1,234,567.00 "
    + FOOTER + "Page d/3/1 124 of 675 " + HEADER + "PAGE Nr. d/3/2 Item Description"
)
PAGE_2 = (
    "D599.4 m 36.00 Rate only 1,852,848.00 To Part Summary SAR 7,654,321.00 "
    + FOOTER + "Page d/3/2 125 of 675 " + HEADER + "PAGE Nr. d/3/3 Item Description"
)
# OCR glued the footer. Still page 12, and still not page 1.
PAGE_12_GLUED = (
    "D999.48 m 825 364.00 300,300.00 To Part Summary SAR Closed "
    "9,999,999.00 134 of 675 RFP No. XX-2099-001 Classification - Public "
    "Paged/3/12 Date: July 10, 2099"
)
# The page number did not survive the scan at all.
UNLABELLED = (
    "A 250.26 J Coating Stripping Test; ASTM D 1664. "
    "To Part Summary SAR 2,717.00 C.R. stamp noise"
)


# ── which page does a total belong to ─────────────────────────────────────

def test_the_real_chunk_is_evidence_for_its_own_page():
    assert chunk_states_part_summary_total(PAGE_1, ["d/3/1"])


def test_the_next_pages_header_is_not_this_pages_total():
    """``PAGE Nr. d/3/2`` is printed above the items that FOLLOW."""
    assert not chunk_states_part_summary_total(PAGE_1, ["d/3/2"])
    assert chunk_states_part_summary_total(PAGE_2, ["d/3/2"])


def test_a_glued_footer_is_still_a_page_number():
    assert chunk_states_part_summary_total(PAGE_12_GLUED, ["d/3/12"])
    assert not chunk_states_part_summary_total(PAGE_12_GLUED, ["d/3/1"])


def test_a_total_with_no_page_number_is_evidence_for_no_page():
    assert not chunk_states_part_summary_total(UNLABELLED, ["d/3/1"])
    # Asked without a page, it is still a Part Summary total.
    assert chunk_states_part_summary_total(UNLABELLED)


# ── what gets STATED to the user ──────────────────────────────────────────

def test_compose_reads_the_page_number_printed_after_the_total():
    parsed = compose_part_summary_total(ASK, PAGE_1)
    assert parsed and parsed["page"] == "d/3/1"
    assert parsed["amount"] == 1234567.0
    assert parsed["currency"] == "SAR"


@pytest.mark.parametrize("wrong", [UNLABELLED, PAGE_12_GLUED, PAGE_2])
def test_compose_never_states_another_pages_total(wrong):
    """The dangerous one. Alone in the excerpt, each of these used to come
    back as {'page': 'd/3/1', 'amount': <its own figure>}."""
    assert compose_part_summary_total(ASK, wrong) is None


def test_compose_picks_the_asked_page_out_of_a_crowded_excerpt():
    """Live shape: five footers retrieved, one of them the right one. The old
    composer returned None here (ambiguous) and the answer was "not found"."""
    excerpt = "\n\n".join((UNLABELLED, PAGE_12_GLUED, PAGE_2, PAGE_1))
    parsed = compose_part_summary_total(ASK, excerpt)
    assert parsed and parsed["amount"] == 1234567.0


def test_an_unlabelled_total_does_not_borrow_the_next_totals_footer():
    """Two totals in one excerpt, the first with no page number of its own.
    The footer after the SECOND total names the second total's page."""
    excerpt = (
        "A 250.26 Coating Stripping Test To Part Summary SAR 2,717.00 "
        "D290.2 Nr 1,239 1,185.00 To Part Summary SAR 1,234,567.00 "
        + FOOTER + "Page d/3/1 124 of 675"
    )
    assert compose_part_summary_total(ASK, excerpt)["amount"] == 1234567.0


# ── the currency is the one printed beside the amount ─────────────────────
#
# Live d8d9573, unseen Set 3: the same page total came back as
# "SAR 34,645,529.00" for one question and "INR 34,645,529.00" for another.
# The currency was the FIRST currency-shaped token anywhere near the label,
# matched case-insensitively -- so a scrap of stamp OCR ("Inr", "Sr", "Usd")
# ahead of the label outranked the "SAR" printed against the figure.

@pytest.mark.parametrize("noise", ["Inr", "sr", "Usd", "K'l.A inr J"])
def test_stamp_noise_before_the_label_is_not_the_currency(noise):
    page = (
        f"D290.2 Nr 1,239 1,185.00 1,468,215.00 {noise} To Part Summary U,. "
        "SAR 1,234,567.00 " + FOOTER + "Page d/3/1 124 of 675"
    )
    assert compose_part_summary_total(ASK, page)["currency"] == "SAR"


def test_a_total_printed_without_a_currency_does_not_borrow_one_from_noise():
    page = ("1,468,215.00 Inr stamp To Part Summary 1,234,567.00 "
            + FOOTER + "Page d/3/1 124 of 675")
    assert compose_part_summary_total(ASK, page)["currency"] == ""


def test_a_real_currency_elsewhere_on_the_page_is_still_a_fallback():
    """Genuine upper-case code in a column header, none beside the figure."""
    page = ("Unit Rate SAR Amount SAR D290.2 Nr 1,239 To Part Summary "
            "1,234,567.00 " + FOOTER + "Page d/3/1 124 of 675")
    assert compose_part_summary_total(ASK, page)["currency"] == "SAR"


# ── the one-page shortcut answers one-page questions only ─────────────────

@pytest.mark.parametrize(
    "question",
    [
        "What is the combined Part Summary total of pages d/3/1, d/3/2 and d/3/3?",
        "Verify: do the three priced items on page d/3/1 add up to its Part Summary total?",
        "What is the sum of the Part Summary totals for d/3/1 and d/3/2?",
        "Is the Part Summary total for page d/3/1 larger than the one for d/3/3?",
        # No "combined", no "sum": two pages is enough on its own.
        "What are the Part Summary totals for pages d/3/1 and d/3/2?",
    ],
)
def test_a_question_over_several_pages_or_a_check_is_not_the_shortcut(question):
    """Both live questions were answered with page d/3/1's total alone --
    a correct number, to a different question."""
    from app.core.rag.retriever import query_asks_for_part_summary_total

    assert not query_asks_for_part_summary_total(question), question


# ── a label the chunker cut in half ───────────────────────────────────────
#
# Unseen Set 3 B3, page d/3/3, 0/3 on every build. On the live index one
# chunk ENDS "...storm water culverts 1,370.00 To Par" and the next BEGINS
# "t Summary ... SAR 17,496,857.00 ... Page d/3/3". Neither half says "Part
# Summary", so the detector saw no total and the lookup (LIKE '%part
# summary%') could not find the chunk at all.

SPLIT_HEAD = ("G |Breakout and remove existing chain link fence D549.2 m 3,504 80.00 "
              "280,320.00 H_ |storm water culverts D529.3 m 1,370.00 Rate Only To Par")
SPLIT_TAIL = ("t Summary U,. stamp SAR 5,000,000.00 " + FOOTER + "Page d/3/3 126 of 675 "
              + HEADER + "PAGE Nr. d/3/4 Item Description")
OCR_NOISE_TAIL = ("r^. Summary 28/7/99 * SAR 5,000,000.00 " + FOOTER
                  + "Page d/3/3 126 of 675")
ASK_D33 = ASK.replace("d/3/1", "d/3/3")


@pytest.mark.parametrize("tail", [SPLIT_TAIL, OCR_NOISE_TAIL])
def test_the_second_half_of_a_split_label_is_still_the_pages_total(tail):
    assert chunk_states_part_summary_total(tail, ["d/3/3"])
    parsed = compose_part_summary_total(ASK_D33, tail)
    assert parsed and parsed["amount"] == 5000000.0 and parsed["currency"] == "SAR"


def test_the_first_half_states_no_total():
    assert not chunk_states_part_summary_total(SPLIT_HEAD, ["d/3/3"])


def test_a_split_label_is_only_a_label_at_the_very_start_of_a_chunk():
    """"Summary" opening a CHUNK, before a money figure, under a bill page
    footer. Not the word wherever it appears."""
    prose = ("Executive overview of the demolition works. Summary of costs to date is "
             "SAR 5,000,000.00 as reported. " + FOOTER + "Page d/3/3 126 of 675")
    assert not chunk_states_part_summary_total(prose, ["d/3/3"])
    assert compose_part_summary_total(ASK_D33, prose) is None


def test_in_a_joined_excerpt_each_chunk_start_counts():
    """The model's context is several chunks joined; the split tail is rarely
    the first of them."""
    excerpt = "\n\n".join((PAGE_1, "[source: PART NR. 3.pdf, chunk 12]\n" + SPLIT_TAIL))
    assert compose_part_summary_total(ASK_D33, excerpt)["amount"] == 5000000.0


def test_a_several_page_question_still_gets_every_named_pages_total(monkeypatch):
    """Taking E6 off the shortcut must not take it off the RESCUE. It needs
    three footers; cosine offers none of them, and only the first page used
    to be looked for."""
    from app.core.rag import retriever as ret

    # Page d/3/3 is the chunk whose label the chunker cut in half: it does
    # not contain the words "part summary" at all.
    page_3 = SPLIT_TAIL
    footers = {"d/3/1": _chunk("p1", 0.0, PAGE_1, 3), "d/3/2": _chunk("p2", 0.0, PAGE_2, 5),
               "d/3/3": _chunk("p3", 0.0, page_3, 7)}
    noise = [_chunk(f"n{i}", 0.80 - i / 100, f"General specification clause {i} on demolition.", 300 + i)
             for i in range(6)]

    def containing_all(self, pid, needles, k=20, **_kw):
        want = [n.lower() for n in needles]
        if len(want) == 1:
            # A bare "part summary" LIKE on a 675-page bill is cut at LIMIT 20
            # long before it reaches these pages. Only a page-scoped needle
            # finds them -- which is why every named page needs its own.
            return []
        return [c for c in footers.values()
                if all(w in c.text.lower() for w in want)]

    vs = "app.core.rag.vector_store.VectorStore."
    monkeypatch.setattr(vs + "search", lambda self, pid, qvec, k, query_text=None: noise[:k])
    monkeypatch.setattr(vs + "identifier_search", lambda self, pid, identifiers, k=20: [])
    monkeypatch.setattr(vs + "chunks_for_docs", lambda self, pid, doc_ids, k_per_doc=12, **_kw: [])
    monkeypatch.setattr(vs + "chunks_containing_all", containing_all)
    monkeypatch.setattr(vs + "count", lambda self, pid=None: 11)
    monkeypatch.setattr(vs + "_verify_embedding_identity", lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id",
                        lambda did: "Example - Demolition BOQ.pdf", raising=False)
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        lambda *a, **k: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("COMPOSE_PART_SUMMARY", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    ask = "What is the combined Part Summary total of pages d/3/1, d/3/2 and d/3/3?"
    chunks, _ = ret.retrieve_with_filter(ask, "p_master", k=5)
    got = {c.chunk_id for c in chunks}
    assert {"p1", "p2", "p3"} <= got, got


def test_the_plain_one_page_question_still_takes_the_shortcut():
    from app.core.rag.retriever import query_asks_for_part_summary_total

    assert query_asks_for_part_summary_total(ASK)
    assert query_asks_for_part_summary_total(
        "what does page d/3/1 of the demolition bill total?")


def test_compose_follows_the_question_to_the_other_page():
    excerpt = "\n\n".join((PAGE_1, PAGE_2, PAGE_12_GLUED))
    assert compose_part_summary_total(ASK_NEXT_PAGE, excerpt)["amount"] == 7654321.0
    assert compose_part_summary_total(ASK_PAGE_12, excerpt)["amount"] == 9999999.0


# ── retrieval: the named page beats the crowd ─────────────────────────────

def _chunk(cid, score, text, index):
    return Chunk(chunk_id=cid, project_id="p_master", doc_id="boq",
                 chunk_index=index, text=text, score=score)


def test_retrieval_returns_the_asked_page_and_not_the_other_footers(monkeypatch):
    from app.core.rag import retriever as ret

    crowd = [
        _chunk(f"u{i}", 0.80 - i / 100, UNLABELLED + f" sheet {i}", 200 + i)
        for i in range(8)
    ]
    others = [_chunk("p12", 0.79, PAGE_12_GLUED, 40), _chunk("p2", 0.78, PAGE_2, 5)]
    right = _chunk("p1", 0.40, PAGE_1, 3)
    semantic = crowd + others + [right]

    vs = "app.core.rag.vector_store.VectorStore."
    monkeypatch.setattr(
        vs + "search",
        lambda self, pid, qvec, k, query_text=None: semantic[:k],
    )
    monkeypatch.setattr(vs + "identifier_search",
                        lambda self, pid, identifiers, k=20: [])
    monkeypatch.setattr(vs + "chunks_for_docs",
                        lambda self, pid, doc_ids, k_per_doc=12, **_kw: [])
    monkeypatch.setattr(vs + "chunks_containing_all",
                        lambda self, pid, needles, k=20, **_kw: [])
    monkeypatch.setattr(vs + "count", lambda self, pid=None: 11)
    monkeypatch.setattr(vs + "_verify_embedding_identity", lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id",
                        lambda did: "Example - Demolition BOQ.pdf", raising=False)
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        lambda *a, **k: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("COMPOSE_PART_SUMMARY", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)

    chunks, _ = ret.retrieve_with_filter(ASK, "p_master", k=5)

    assert [c.chunk_id for c in chunks] == ["p1"]


# ── the collection page: every page's total, as a list ────────────────────
#
# Unseen Set 3 B3 (page d/3/3), 0/3 on every build. The bill ends each Part
# with a collection page, and on the live index it reads:
#
#   PART SUMMARY From Page Nr. d/3/1 34,645,529.00 From Page Nr. d/3/2
#   1,852,848.00 From Page Nr. d/3/3 17,496,857.00 From Page Nr. d/3/4 ...
#
# The page reference comes BEFORE its amount, in a list -- the opposite of a
# page footer, which is all the ownership logic understood. And the old
# label-window rule read that list as one total belonging to every page
# reference within 80 characters: 34,645,529.00 "for" d/3/1 AND d/3/2.

COLLECTION = (
    "BILL OF QUANTITIES DEMOLITION AND SITE CLEARANCE Description Qty PART SUMMARY "
    "From Page Nr. d/3/1 1,234,567.00 From Page Nr. d/3/2 7,654,321.00 "
    "From Page Nr. d/3/3 5,000,000.00 From Page Nr. d/3/4 8,240,875.00) "
    "From Page Nr. d/3/12 9,999,999.00 Carried to Grand Summary 99,000,000.00"
)


@pytest.mark.parametrize(
    "page, amount",
    [("d/3/1", 1234567.0), ("d/3/2", 7654321.0), ("d/3/3", 5000000.0),
     ("d/3/4", 8240875.0), ("d/3/12", 9999999.0)],
)
def test_each_page_gets_its_own_row_of_the_collection(page, amount):
    ask = ASK.replace("d/3/1", page)
    assert chunk_states_part_summary_total(COLLECTION, [page])
    assert compose_part_summary_total(ask, COLLECTION)["amount"] == amount


def test_a_page_the_collection_does_not_list_gets_nothing():
    assert not chunk_states_part_summary_total(COLLECTION, ["d/3/9"])
    assert compose_part_summary_total(ASK.replace("d/3/1", "d/3/9"), COLLECTION) is None


def test_page_1_is_not_page_12_in_a_list_either():
    assert compose_part_summary_total(ASK, COLLECTION)["amount"] == 1234567.0


def test_the_collection_and_the_pages_own_footer_agree_and_either_answers():
    excerpt = COLLECTION + "\n\n[doc_id=x chunk=3 score=1.000] " + PAGE_1
    assert compose_part_summary_total(ASK, excerpt)["amount"] == 1234567.0


def test_no_total_ever_belongs_to_two_pages():
    """Structural, so it does not depend on which candidate is listed first:
    the label that HEADS a collection list must not claim the list."""
    from app.core.rag.retriever import _part_summary_totals

    totals = _part_summary_totals(COLLECTION)
    assert totals and all(len(t["pages"]) == 1 for t in totals), totals
    assert sorted(t["pages"][0] for t in totals) == ["d/3/1", "d/3/12", "d/3/2", "d/3/3", "d/3/4"]
