"""An unnamed contract question is won by evidence, not by arrival order.

Live wave-1 on the Master Corpus, SHA ``567147a``, measured five Contract
Data questions against one corpus in one session and got both outcomes:

* **A2** (Accepted Contract Amount incl. VAT) — PASS
* **A6** (Defects Notification Period) — PASS
* **A3** (Time for Completion, whole of the Works) — FAIL, ``the retrieved
  excerpts do not contain`` it
* **A5** (Delay Damages, whole of the Works) — FAIL, cited **Sub-Clause
  8.8** and then reported the rate and the cap as absent from its excerpts
* **A9** (who the Engineer is) — FAIL_WRONG_CONTRACT, answered out of a
  **different contract year's** Conditions of Contract with nothing but the
  generic defined-term entry

A2 and A6 prove the Contract Data is indexed and retrievable, so the three
failures are not a corpus gap. What separates them is which chunk happened
to sort first: the unnamed fence locks the pool to the first candidate
carrying a ``PREFIX-YEAR-SEQ`` id and DROPS every other id, so a wrong-year
chunk at rank 1 does not merely outrank the row holding the answer — it
deletes it, at every rank. A2/A6 won that race; A3/A5/A9 lost it.

Two things decide the race wrongly, and both are pinned below.

1. A General Conditions clause that says *"at the rate stated in the
   Contract Data"* is a **pointer** to the answer. It carries the heading
   phrase and, being a clause about delay damages, repeats every label word
   the question uses — so it collected the Contract Data heading tier plus
   the full label bonus and beat the row that actually states the rate.
   That is A5's symptom exactly: cite 8.8, then report the rate as missing.
2. *"Who is the Engineer"* was never recognised as asking for a filled
   particular at all, because a particular was assumed to be a figure. No
   part of the particulars machinery ran on A9, so a glossary definition
   won on raw cosine — and locked its own contract year.

WAVE 2 added three more of the same family, and the second half of this
file covers them:

* **F1** FAIL_WRONG_CONTRACT — a WBS over "the demolition and site
  clearance scope in this project's BOQ" cited another contract year's
  Conditions of Contract three times out of three. Its prose describes that
  scope in words; the bill carries it as measured rows. Not a Contract Data
  particular, so the election declined and arrival order decided.
* **G1** FAIL_WRONG_CONTRACT (Sev-1) — "what does Schedule 10 of the
  contract contain" was answered out of a **different project's** show
  package, which has a Schedule 10 of its own and discusses it at length.
  The contract's own register says ``Schedule 10: Not Used``, which IS the
  answer and IS a filled particulars row — but the ask was not recognised
  as wanting one, so nothing lifted it and the fence dropped the contract.
* **E1** FAIL — "calculate the delay damages per calendar day in SAR"
  retrieved the 0.1%-per-day row and then reported the SAR figure as
  absent. A percentage is not an amount, and the row holding the amount
  shares no wording with the question.

SANITIZED FIXTURE. The two contract/document ids are the live ones and are
already in git (``tests/test_cross_contract_fence.py``, ``AGENTS.md``);
every figure, party and value here is fixture-only and deliberately differs
from the live pack, which must never land in the repository. What is under
test is the mechanism — which contract's Contract Data rows reach the
top-k — not the live numbers.
"""
from __future__ import annotations

import pytest

from app.core.contract_data_chunks import contract_data_particulars_chunks
from app.core.contract_data_chunks import filled_particulars_rows
from app.core.rag.retriever import (
    contract_data_mention_is_only_a_cross_reference,
    contract_data_particulars_delta,
    elect_answer_bearing_contract,
    is_contract_data_particulars_row,
    particulars_row_answers_asked_label,
    query_asks_for_contract_particulars,
)

DD23_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
DD22_NAME = "AB-2022-202 - Volume 1 - Conditions of Contract.pdf"
KB_NAME = "fidic_2017_administration.md"

PROJECT = "master_corpus"
GK_PROJECT = "training_material"

# ── the asks, verbatim from the frozen battery ────────────────────────────
A2 = "What is the Accepted Contract Amount including VAT?"
A3 = "What is the Time for Completion for the whole of the Works?"
A5 = "What are the Delay Damages for the whole of the Works?"
A6 = "What is the Defects Notification Period?"
A9 = "Who is the Engineer under this contract?"

# ── AB-2023-101: the executed contract's own Contract Data (sanitized) ────
DD23_CONTRACT_DATA = """Volume 1 - Conditions of Contract

Particular Conditions Part A - Contract Data

1.1.1 Accepted Contract Amount excluding VAT | SAR 8,640,000.00
1.1.1 Accepted Contract Amount including VAT | SAR 9,936,000.00
1.1.27 Defects Notification Period | 365 days from the Taking-Over Certificate
1.1.67 Sections | Not Applicable
1.1.75 Time for Completion for the whole of the Works | 640 days
1.3.1 (a) Approved method of electronic communication | ProjectMail
1.3.1 (b) Engineer | Northwater Engineers (Demo Saudi Limited)
4.3.3 Performance Bond | 10% of the Accepted Contract Amount
8.8 Delay Damages for the whole of the Works | 0.1% of the Contract Price per calendar day
8.8 Maximum amount of delay damages | 10% of the Accepted Contract Amount
"""

# The values the three failing asks must reach, as the indexer writes them.
A3_ANSWER = "640 days"
A5_ANSWER = "0.1% of the Contract Price per calendar day"
A9_ANSWER = "Northwater Engineers"

# ── AB-2022-202: another year's Conditions of Contract. Every one of these
# points AT the Contract Data instead of stating a particular, which is what
# a General Conditions clause does.
DD22_TFC_CLAUSE = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.2 Time for Completion. "
    "The Contractor shall complete the whole of the Works within the Time "
    "for Completion for the whole of the Works stated in the Contract Data, "
    "calculated from the Commencement Date, being 1,095 days as extended "
    "under Sub-Clause 8.5 Extension of Time for Completion."
)
DD22_DELAY_CLAUSE = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. If "
    "the Contractor fails to comply with Sub-Clause 8.2, the Contractor "
    "shall pay delay damages for the whole of the Works at the rate stated "
    "in the Contract Data for every calendar day which shall elapse between "
    "the relevant Time for Completion and the date stated in the Taking-Over "
    "Certificate, up to the maximum amount of delay damages stated in the "
    "Contract Data, being 10% of the Accepted Contract Amount."
)
DD22_ENGINEER_DEFINITION = (
    'Volume 1 - Conditions of Contract. 1.1.35 "Engineer" means the person '
    "appointed by the Employer to act as the Engineer for the purposes of "
    "the Contract and named in the Contract Data, or any replacement "
    "appointed under Sub-Clause 3.6 within 28 days."
)
# The other year's Contract Data table as a scanned tender volume yields it:
# the keys survived, the value column did not. It is a real particulars
# window, so it collects the index-time prefix bonus, and its keys name every
# label the three questions use, so it collects the full label bonus too — a
# combination that out-scores the filled row it should never be preferred to.
# It states nothing, which is exactly why the election must skip it.
DD22_UNFILLED_CONTRACT_DATA = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    f"[{DD22_NAME}].\n"
    "Contract Data\n"
    "1.1.1 Accepted Contract Amount excluding VAT\n"
    "1.1.27 Defects Notification Period\n"
    "1.1.75 Time for Completion for the whole of the Works\n"
    "1.3.1 (b) Engineer\n"
    "8.8 Delay Damages for the whole of the Works"
)

# ── the curated FIDIC knowledge base (#468's F-KB-1). Background, never the
# contract — and it must not take a top-k slot from the row that answers.
KB_DELAY_NOTE = (
    "FIDIC 2017 contract administration notes. Delay damages under "
    "Sub-Clause 8.8 are payable for the whole of the Works at the rate "
    "stated in the Contract Data, commonly 0.05% of the Contract Price per "
    "calendar day, capped at 10% of the Accepted Contract Amount. The "
    "Engineer certifies the Time for Completion under Sub-Clause 8.2."
)

DD22_DOCS = {
    "dd22_tfc": DD22_TFC_CLAUSE,
    "dd22_delay": DD22_DELAY_CLAUSE,
    "dd22_engineer": DD22_ENGINEER_DEFINITION,
    "dd22_unfilled_cd": DD22_UNFILLED_CONTRACT_DATA,
}


def _doc_names() -> dict:
    names = {doc_id: DD22_NAME for doc_id in DD22_DOCS}
    names["kb_delay"] = KB_NAME
    return names


def _dd23_delay_damages_row() -> str:
    """A5's row as the indexer writes it — taken from the indexer itself, so
    a change to the chunk format cannot leave this fixture behind."""
    return next(
        r for r in contract_data_particulars_chunks(
            DD23_CONTRACT_DATA, filename=DD23_NAME,
        )
        if A5_ANSWER in r
    )


@pytest.fixture
def two_year_corpus(tmp_path, monkeypatch):
    """One project holding two contract years, plus the curated FIDIC KB.

    Mirrors the live Master Corpus: the AB-2023-101 executed contract's
    Contract Data is indexed through the REAL particulars chunker, so this
    fixture cannot drift from the format the indexer emits.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK_PROJECT)
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_CD_PARTICULARS_BOOST", raising=False)

    from sqlalchemy import delete as _sa_delete

    from app.core import projects as projects_mod
    from app.core.rag import embeddings as _emb
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as _vs
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    for pid, name in ((PROJECT, "Master Corpus"), (GK_PROJECT, "General Knowledge")):
        if projects_mod.get_project(pid) is None:
            projects_mod.create_project(
                name, user_id=SYSTEM_USER_ID, project_id=pid, origin="system_seed",
            )

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    with store._lock, store._session_factory()() as session:
        session.execute(_sa_delete(store._rag_chunk_cls))
        session.commit()

    names = _doc_names()

    cd_chunks = contract_data_particulars_chunks(
        DD23_CONTRACT_DATA, filename=DD23_NAME,
    )
    assert cd_chunks, "the particulars chunker produced nothing to retrieve"
    for i, chunk in enumerate(cd_chunks):
        doc_id = f"dd23_cd_{i}"
        names[doc_id] = DD23_NAME
        store.upsert_chunks(PROJECT, doc_id, [chunk], embedder.encode([chunk]))

    for doc_id, text in DD22_DOCS.items():
        store.upsert_chunks(PROJECT, doc_id, [text], embedder.encode([text]))
    store.upsert_chunks(
        GK_PROJECT, "kb_delay", [KB_DELAY_NOTE], embedder.encode([KB_DELAY_NOTE]),
    )

    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))

    # The live ranking condition. A Conditions of Contract clause repeats the
    # question's wording many times over; a one-line particulars row states
    # it once. The clause therefore leads on raw cosine, which is what put a
    # wrong-year chunk at rank 1 and handed it the whole pool.
    real_search = store.search

    def wrong_year_leads(project_id, query_vec, k=20, query_text=None):
        hits = real_search(project_id, query_vec, k=k, query_text=query_text)
        for chunk in hits:
            if chunk.doc_id in DD22_DOCS:
                chunk.score = 0.91
            elif chunk.doc_id == "kb_delay":
                chunk.score = 0.88
            else:
                chunk.score = 0.55
        return hits

    monkeypatch.setattr(store, "search", wrong_year_leads)

    yield ret, names

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def _top_k(ret, names, ask, k=5):
    chunks, _ = ret.retrieve_with_filter(ask, PROJECT, k=k)
    return [(names.get(c.doc_id, ""), c.text or "") for c in chunks]


def _blob(top):
    return "\n".join(text for _name, text in top)


def _report(top):
    return "\n".join(
        f"  {i}. {name[:24]:<26} {text[:88]!r}"
        for i, (name, text) in enumerate(top, 1)
    )


def _assert_own_contract_data_leads(top, answer):
    """The three properties the live failures each violated."""
    assert top, "nothing retrieved — the answer row is in the fixture"
    assert answer in _blob(top), (
        f"{answer!r} never reached the top-5:\n" + _report(top)
    )
    lead_name, lead_text = top[0]
    assert lead_name == DD23_NAME, (
        "rank 1 is not the contract being asked about:\n" + _report(top)
    )
    assert lead_text.startswith("CONTRACT DATA particulars"), (
        "rank 1 is not a Contract Data row:\n" + _report(top)
    )
    assert DD22_NAME not in [name for name, _t in top], (
        "another contract year is in the top-5:\n" + _report(top)
    )


# ── the three regressions, at production k=5 ──────────────────────────────


def test_a3_time_for_completion_reaches_the_ab2023_particulars_row(two_year_corpus):
    """A3 said the excerpts did not contain it. The row must lead the top-5."""
    ret, names = two_year_corpus
    top = _top_k(ret, names, A3)

    _assert_own_contract_data_leads(top, A3_ANSWER)
    assert "Time for Completion for the whole of the Works" in top[0][1]


def test_a5_delay_damages_reaches_the_row_not_the_clause_pointing_at_it(
    two_year_corpus,
):
    """A5 cited Sub-Clause 8.8 and called the rate absent. The clause it had
    only says where the rate is stated; the row that states it must win."""
    ret, names = two_year_corpus
    top = _top_k(ret, names, A5)

    _assert_own_contract_data_leads(top, A5_ANSWER)
    assert "Delay Damages for the whole of the Works" in top[0][1]
    assert DD22_DELAY_CLAUSE not in _blob(top), (
        "the clause that only points at the rate is still in the excerpts:\n"
        + _report(top)
    )


def test_a9_engineer_reaches_the_named_party_row_not_another_years_glossary(
    two_year_corpus,
):
    """A9 was FAIL_WRONG_CONTRACT: another year's Conditions of Contract, and
    only the generic defined term. The filled row names the party."""
    ret, names = two_year_corpus
    top = _top_k(ret, names, A9)

    _assert_own_contract_data_leads(top, A9_ANSWER)
    assert "means the person appointed by the Employer" not in _blob(top), (
        "the defined-term entry is still standing in for the appointment:\n"
        + _report(top)
    )


@pytest.mark.parametrize(
    "ask, expected",
    [
        (A2, "SAR 9,936,000.00"),
        (A6, "365 days"),
    ],
)
def test_the_two_asks_that_already_passed_still_pass(two_year_corpus, ask, expected):
    """A2 and A6 passed live by winning the same race. They are the control:
    a fix that only reorders the pool would take one of them out."""
    ret, names = two_year_corpus
    _assert_own_contract_data_leads(_top_k(ret, names, ask), expected)


@pytest.mark.parametrize("ask", [A2, A3, A5, A6, A9])
def test_no_contract_data_answer_mixes_contract_years(two_year_corpus, ask):
    """#443's property, unchanged: one answer, one PREFIX-YEAR-SEQ. The fix
    elects a different winner — it does not stop electing one."""
    from app.core.rag.retriever import extract_contract_doc_ids

    ret, names = two_year_corpus
    ids = set()
    for name, _text in _top_k(ret, names, ask):
        ids.update(extract_contract_doc_ids(name))
    assert len(ids) <= 1, f"{ask!r} mixed contract ids: {sorted(ids)}"


@pytest.mark.parametrize("ask", [A3, A5, A9])
def test_the_curated_fidic_note_never_outranks_the_contracts_own_row(
    two_year_corpus, ask,
):
    """F-KB-1 (#468) was the knowledge base ANSWERING a question the corpus
    could answer better. It may stay in the pool as disclosed background —
    #468's ``class=knowledge_base`` marker is what stops it being quoted as
    the contract, and capping general-knowledge slots is that PR's territory.
    What it may not do is outrank the row that holds the answer.
    """
    ret, names = two_year_corpus
    top = _top_k(ret, names, ask)
    positions = [i for i, (name, _t) in enumerate(top) if name == KB_NAME]

    assert 0 not in positions, (
        "the FIDIC knowledge base is answering for the contract:\n"
        + _report(top)
    )
    own = [i for i, (name, _t) in enumerate(top) if name == DD23_NAME]
    assert own and (not positions or min(own) < min(positions)), (
        "the knowledge base outranked the project's own Contract Data:\n"
        + _report(top)
    )


# ── mechanism 1: a cross-reference is not the row ─────────────────────────


def test_a_clause_that_only_cross_references_the_contract_data_is_not_a_row():
    """"...at the rate stated in the Contract Data..." tells the reader where
    to look. It must not be scored as though it were the Contract Data."""
    assert contract_data_mention_is_only_a_cross_reference(DD22_DELAY_CLAUSE)
    assert contract_data_mention_is_only_a_cross_reference(DD22_TFC_CLAUSE)
    assert contract_data_mention_is_only_a_cross_reference(
        DD22_ENGINEER_DEFINITION
    )
    assert contract_data_particulars_delta(DD22_DELAY_CLAUSE) <= 0.0, (
        "a pointer to the rate still collects the Contract Data heading tier"
    )
    assert contract_data_particulars_delta(DD22_TFC_CLAUSE) <= 0.0


#: The discriminator is the delicate part of this change, so it is pinned as
#: a table rather than by example. True means "a pointer to the Contract
#: Data", which earns no heading tier.
_XREF_CASES = [
    # Genuine headings and sections — the tier exists for these.
    ("Contract Data\n1.1.75 Time for Completion | 640 days", False),
    ("PARTICULAR CONDITIONS PART A - CONTRACT DATA\n\n1.1.1 ACA | SAR 900,000.00",
     False),
    ("Appendix to Tender\nItem 3 Performance Bond ... 10%", False),
    ("Section 4 — Contract Particulars\n4.1 Retention | 5%", False),
    ("Table 2: Contract Data summary — 1.1.75 | 640 days", False),
    # Subject position, not a reference: this sentence STATES the particular.
    ("The Contract Data states 640 days for the whole of the Works.", False),
    # Pointers — a clause telling the reader where the value is stated.
    ("...at the rate stated in the Contract Data for every calendar day.", True),
    ("...as stated in the Contract Data, being 1,095 days.", True),
    ("...set out in the Contract Data at 0.1% per day.", True),
    ("...named in the Contract Data or replaced under 3.6 within 28 days.", True),
    ("...specified in the Contract Data as 365 days.", True),
    ("...the period given in the Contract Data (10% cap).", True),
    ("...the amount inserted in the Contract Data, SAR 8,640,000.00.", True),
    ("...listed in the Appendix to Tender at 5%.", True),
    ("...referred to in the Contract Particulars as 20 days.", True),
    ("...under the Contract Data the notification period is 365 days.", True),
    ("...per the Contract Data, 640 days.", True),
]


@pytest.mark.parametrize("text, is_pointer", _XREF_CASES)
def test_heading_versus_cross_reference_discriminator(text, is_pointer):
    assert contract_data_mention_is_only_a_cross_reference(text) is is_pointer
    if is_pointer:
        assert contract_data_particulars_delta(text) <= 0.0
    else:
        assert contract_data_particulars_delta(text) > 0.0


def test_a_chunk_with_no_contract_data_mention_is_not_a_cross_reference():
    """"No mention" and "only a pointer" are different answers. Returning
    True here would demote every chunk in the corpus that never names the
    Contract Data at all."""
    assert not contract_data_mention_is_only_a_cross_reference(
        "Sub-Clause 4.1. The Contractor shall design the Works."
    )
    assert not contract_data_mention_is_only_a_cross_reference("")


def test_a_real_contract_data_heading_still_earns_the_heading_tier():
    """The tier exists for a Contract Data section that reached retrieval
    through the plain word chunker, with no index-time prefix. A heading
    starts a line; a cross-reference is governed by a preposition."""
    plain_section = (
        "Particular Conditions Part A - Contract Data\n"
        "1.1.75 Time for Completion for the whole of the Works | 640 days\n"
        "8.8 Delay Damages | 0.1% of the Contract Price per calendar day\n"
    )
    assert not contract_data_mention_is_only_a_cross_reference(plain_section)
    assert contract_data_particulars_delta(plain_section) > 0.0

    # One genuine heading is enough, even alongside a cross-reference.
    mixed = plain_section + (
        "The rate shall be that stated in the Contract Data.\n"
    )
    assert not contract_data_mention_is_only_a_cross_reference(mixed)
    assert contract_data_particulars_delta(mixed) > 0.0


def test_the_indexed_particulars_rows_still_outrank_everything():
    """The index-time prefix keeps the strongest lift. Untouched by the
    cross-reference rule: a prefixed row is decided before it runs."""
    row = _dd23_delay_damages_row()
    assert contract_data_particulars_delta(row) > contract_data_particulars_delta(
        DD22_DELAY_CLAUSE
    )


# ── mechanism 2: a particular is not always a figure ──────────────────────


def test_asking_who_a_contract_role_is_asks_for_a_particular():
    """A9's ask never entered the particulars path, so nothing ran on it."""
    assert query_asks_for_contract_particulars(A9)
    assert query_asks_for_contract_particulars("Who is the Employer?")
    assert query_asks_for_contract_particulars(
        "What is the name of the Engineer's Representative?"
    )


def test_a_definition_question_about_the_same_role_is_still_a_definition():
    """The glossary path must keep the questions it is right for."""
    assert not query_asks_for_contract_particulars(
        "What does Engineer mean under the contract?"
    )
    assert not query_asks_for_contract_particulars(
        "Give me the definition of the Employer's Representative"
    )


def test_an_unrelated_who_question_is_not_pulled_onto_the_particulars_path():
    """D1 asks who signed a letter. It is not a Contract Data lookup."""
    assert not query_asks_for_contract_particulars(
        "Who signed the letter about the batching plant at Creek Bend, "
        "and in what capacity?"
    )
    assert not query_asks_for_contract_particulars(
        "What is the document number and revision of the priced Bill of "
        "Quantities, and who prepared it?"
    )


def test_only_three_battery_asks_are_reclassified_as_particulars():
    """Containment. Measured against `main` across all 35 frozen asks in
    ``tests/fixtures/ui_phys/questions.json``, exactly three move: A9 (the
    Engineer's identity) and C4 / G1 (numbered contract Schedules). All
    three want a filled Contract Data row.

    The rest must not move — each would change which pipeline answers a
    question that is already passing. D1 is on this list deliberately:
    "who signed the letter" is a different retrieval class and must stay off
    the particulars path.
    """
    import json
    from pathlib import Path

    catalog = json.loads(
        (
            Path(__file__).resolve().parent
            / "fixtures" / "ui_phys" / "questions.json"
        ).read_text(encoding="utf-8")
    )
    cases = catalog["cases"]
    must_not = [
        "A8", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3",
        "D1", "D2", "D3", "E3", "E4", "F1", "F2", "F3", "G3", "G4", "G5", "G6",
    ]
    for cid in must_not:
        ask = cases[cid]["ask"]
        assert not query_asks_for_contract_particulars(ask), f"{cid}: {ask!r}"
    for cid in ("A1", "A3", "A5", "A6", "A9", "C4", "G1"):
        ask = cases[cid]["ask"]
        assert query_asks_for_contract_particulars(ask), f"{cid}: {ask!r}"
    # Every asked case is accounted for, so a new one cannot slip in
    # unclassified when this file is next read.
    asked = {cid for cid, c in cases.items() if c.get("ask")}
    assert asked == set(must_not) | {"A1", "A2", "A3", "A4", "A5", "A6",
                                     "A7", "A9", "C4", "E1", "E2", "G1", "G2"}


def test_a_particulars_row_whose_value_is_a_name_counts_as_filled():
    """A Notices block carries parties and methods, not figures. It is still
    filled in, and it is the only place A9's answer exists."""
    notices_only = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\n"
        "Contract Data\n"
        "Approved method of electronic communication: ProjectMail\n"
        "Engineer: Northwater Engineers (Demo Saudi Limited)"
    )
    from app.core.rag.retriever import _CD_FILLED_VALUE_RE

    assert not _CD_FILLED_VALUE_RE.search(notices_only), (
        "the fixture must carry no figure at all, or it proves nothing"
    )
    assert is_contract_data_particulars_row(notices_only)
    assert contract_data_particulars_delta(notices_only) > 0.0


def test_a_row_keeps_the_documents_own_separator_when_the_parser_could_not_split():
    """`_line_packed_chunks` is the fallback for a Contract Data section the
    row parser could not split — it keeps the raw lines, and therefore the
    document's own ``key | value`` or dot-leader separator. Those rows carry
    values too, so the same predicate has to see them."""
    pipe_rows = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\n"
        "Contract Data\n"
        "\n"
        "Engineer | Northwater Engineers\n"
        "Employer's address ......... 14 Spur Road"
    )
    assert is_contract_data_particulars_row(pipe_rows)

    keys_only = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\n"
        "Contract Data\n"
        "Engineer |\n"
        "Employer's address .........   "
    )
    assert not is_contract_data_particulars_row(keys_only)


def test_an_empty_particulars_window_is_not_answer_bearing():
    """A window of unfilled keys must not win a pool. Its rows have no value."""
    unfilled = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\n"
        "Particular Conditions Part A - Contract Data\n"
        "1.1.67 Sections\n"
        "4.3.7 Parent Company Guarantee"
    )
    assert not is_contract_data_particulars_row(unfilled)


def test_the_defined_term_glossary_is_never_answer_bearing():
    """Widening "filled" to accept a name must not let prose in."""
    glossary = (
        '"Accepted Contract Amount" means the amount accepted in the Letter '
        "of Acceptance for the execution and completion of the Works."
    )
    assert not is_contract_data_particulars_row(glossary)
    assert contract_data_particulars_delta(glossary) < 0.0


# ── mechanism 3: the election ─────────────────────────────────────────────


def _ranked(*pairs):
    return list(pairs)


def test_the_election_picks_the_contract_owning_the_filled_row():
    """Rank 1 is another year's pointer; the row that answers is below it."""
    row = _dd23_delay_damages_row()
    elected = elect_answer_bearing_contract(
        A5, _ranked((DD22_NAME, DD22_DELAY_CLAUSE), (DD23_NAME, row)),
    )
    assert elected == "ab-2023-101"


def test_the_election_declines_when_no_filled_row_is_in_the_pool():
    """Nothing answer-bearing to elect on — arrival order still decides, so
    this cannot change the ranking of a corpus without Contract Data."""
    assert elect_answer_bearing_contract(
        A5, _ranked((DD22_NAME, DD22_DELAY_CLAUSE)),
    ) is None


def test_the_election_declines_for_a_question_that_wants_no_particular():
    """A drawing or method-statement lookup must not have its pool decided by
    whichever Contract Data row happens to be in it."""
    row = contract_data_particulars_chunks(
        DD23_CONTRACT_DATA, filename=DD23_NAME,
    )[0]
    assert elect_answer_bearing_contract(
        "Summarize drawing QZ-ENG-320-0000-EXC-DWG-LI-200-0001056-04",
        _ranked((DD23_NAME, row)),
    ) is None
    assert elect_answer_bearing_contract(
        "What does Accepted Contract Amount mean?", _ranked((DD23_NAME, row)),
    ) is None


def test_a_named_contract_question_never_reaches_the_election(two_year_corpus):
    """#443 owns the named path and is untouched. A question naming the OTHER
    year must still fail closed rather than be rescued by an elected row."""
    ret, names = two_year_corpus
    chunks, _ = ret.retrieve_with_filter(
        "Per AB-2024-999, what are the Delay Damages for the whole of the "
        "Works?",
        PROJECT,
        k=5,
    )
    assert chunks == [], [names.get(c.doc_id, "") for c in chunks]

    chunks, _ = ret.retrieve_with_filter(
        "Per the AB-2022-202 contract, what are the Delay Damages for the "
        "whole of the Works?",
        PROJECT,
        k=5,
    )
    assert chunks, "the named year is in the fixture"
    assert all(names.get(c.doc_id, "") == DD22_NAME for c in chunks), (
        "the election overrode an explicitly named contract"
    )


# ── mutation probes ───────────────────────────────────────────────────────


def test_the_unfilled_wrong_year_window_outscores_the_filled_row(
    two_year_corpus,
):
    """The probe below is only meaningful if the decoy genuinely leads on
    score, so measure it rather than assume it. Prefix bonus plus the full
    label bonus, on the higher raw cosine, puts the wrong year's empty table
    ahead of the row that states the answer."""
    row = _dd23_delay_damages_row()
    assert contract_data_particulars_delta(
        DD22_UNFILLED_CONTRACT_DATA
    ) == contract_data_particulars_delta(row), (
        "the decoy no longer shares the family bonus — it proves nothing"
    )
    assert not is_contract_data_particulars_row(DD22_UNFILLED_CONTRACT_DATA)
    assert is_contract_data_particulars_row(row)


@pytest.mark.parametrize(
    "ask, answer", [(A3, A3_ANSWER), (A5, A5_ANSWER), (A9, A9_ANSWER)],
)
def test_mutation_probe_arrival_order_election_brings_the_failure_back(
    two_year_corpus, monkeypatch, ask, answer,
):
    """Restore arrival-order election and the live failure returns: the top-5
    is entirely the wrong contract year and the answer is at no rank.

    This is what makes the three tests above evidence rather than decoration
    — the fixture genuinely reproduces the defect it claims to, and it shows
    the guarantee cannot be bought by scoring alone. A boost is bounded; the
    number of chunks the other contract can put at rank 1 is not.
    """
    ret, names = two_year_corpus
    # Post-#501 A5/A9 (and later A2/A3) rescues also lift the
    # answer-bearing row. This probe measures #483 arrival-order
    # election alone.
    monkeypatch.setattr(
        ret, "_apply_asked_particular_value_boost", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        ret, "recall_labelled_rows", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        ret, "elect_answer_bearing_contract", lambda _q, _docs: None,
    )

    # The live failure was election-plus-arrival-order: a wrong-year chunk
    # at rank 1 locked the pool. After this PR, a particulars ask whose
    # election declined still refuses to lock onto a GC pointer / schedule
    # duration — so restoring arrival-order lock is part of the probe.
    real_init = ret._ContractScope.__init__

    def _naive_lock(self, query, ranked_docs=None):
        real_init(self, query, ranked_docs)
        self.winning = None
        self._particulars = False
        # A5/A9 post-#501 and A2/A3 asked-particular fences also elect
        # from evidence. Disable them so this probe still measures
        # arrival-order election alone.
        self._delay_rate_in_pool = False
        self._engineer_identity_in_pool = False
        self._aca_incl_vat_in_pool = False
        self._tfc_in_pool = False

    monkeypatch.setattr(ret._ContractScope, "__init__", _naive_lock)
    top = _top_k(ret, names, ask)

    assert top, "the probe must reproduce a wrong answer, not an empty one"
    assert DD22_NAME == top[0][0], (
        "probe did not put the wrong year at rank 1:\n" + _report(top)
    )
    assert DD23_NAME not in [name for name, _t in top], (
        "probe did not reproduce the lock — the contract being asked about "
        "is still in the top-5:\n" + _report(top)
    )
    assert answer not in _blob(top)


def test_mutation_probe_the_cross_reference_rule_is_what_demotes_the_clause(
    monkeypatch,
):
    """Disable the cross-reference rule and Sub-Clause 8.8 is scored as a
    Contract Data row again — the misfire behind A5's citation."""
    from app.core.rag import retriever as ret

    assert contract_data_particulars_delta(DD22_DELAY_CLAUSE) <= 0.0
    monkeypatch.setattr(
        ret, "contract_data_mention_is_only_a_cross_reference", lambda _t: False,
    )
    assert ret.contract_data_particulars_delta(DD22_DELAY_CLAUSE) > 0.0


def test_mutation_probe_a9_needs_the_role_ask_to_be_recognised(
    two_year_corpus, monkeypatch,
):
    """With the role ask unrecognised, nothing in the particulars machinery
    runs on A9 and another year's defined term answers it."""
    ret, names = two_year_corpus
    real = ret.query_asks_for_contract_particulars
    monkeypatch.setattr(
        ret,
        "query_asks_for_contract_particulars",
        lambda q: False if q == A9 else real(q),
    )
    top = _top_k(ret, names, A9)

    assert top
    assert A9_ANSWER not in _blob(top), (
        "probe did not reproduce A9 — the role ask is not load-bearing:\n"
        + _report(top)
    )


# ══ WAVE 2 ════════════════════════════════════════════════════════════════
#
# F1, G1 and E1: the same disease reaching past the Contract Data. F1 and G1
# are the fence again — an unnamed ask whose pool is frozen by whichever
# chunk sorted first, so the contract being asked about is deleted. E1 is
# the narrower one: the right contract wins and the arithmetic still cannot
# run, because a percentage is not an amount.

DD23_BOQ_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Demolition and Site Clearance BOQ.pdf"
)
#: The foreign document G1 invented its answer from. It carries no
#: PREFIX-YEAR-SEQ, so the contract fence cannot see it at all — which is
#: why the contract's own register row has to WIN rather than be alone.
FOREIGN_NAME = "2015 EXW (Show Package).docx"

F1 = (
    "Generate a high-level WBS for the demolition and site clearance scope "
    "in this project's BOQ."
)
G1 = "What does Schedule 10 of the contract contain?"
E1 = (
    "Calculate the delay damages per calendar day in SAR for the whole of "
    "the Works."
)

#: The Contract Data of the contract being asked about, with the Volume 4
#: schedule register in it. Run through the real chunker, so the register
#: rows land in a particulars window exactly as the indexer puts them there.
DD23_CONTRACT_DATA_W2 = """Volume 1 - Conditions of Contract

Particular Conditions Part A - Contract Data

1.1.1 Accepted Contract Amount excluding VAT | SAR 8,640,000.00
1.1.27 Defects Notification Period | 365 days from the Taking-Over Certificate
1.1.75 Time for Completion for the whole of the Works | 640 days
8.8 Delay Damages for the whole of the Works | 0.1% of the Contract Price per calendar day
8.8 Maximum amount of delay damages | 10% of the Accepted Contract Amount
Volume 4 Schedules
Schedule 9 | Health & Safety KPIs
Schedule 10 | Not Used
""" + "\n".join(
    # The live Contract Data carries 200+ rows. E1's money row has to survive
    # a field of siblings that all earn the family bonus and, unlike it, the
    # full label bonus too. Six rows is not that field, and a fixture that
    # cannot fail proves nothing.
    f"1.9.{i} Delay damages and completion — spare particulars field {i} "
    f"for the whole of the Works | {i} calendar days"
    for i in range(1, 45)
)

E1_RATE_ROW = "0.1% of the Contract Price per calendar day"
E1_MONEY = "SAR 8,640,000.00"
G1_REGISTER_ROW = "Schedule 10: Not Used"
F1_BOQ_ITEM = "D110"

#: Measured rows — the evidence a BOQ/scope ask has to be built on.
_BOQ_PAGE_1 = (
    "Demolition and Site Clearance. Page d/3/1.\n"
    "| D110 | General site clearance | 158 | ha | 2,500.00 | 395,000.00 |\n"
    "| D290.1 | Removal of trees in existing sidewalks | 48 | Nr | 220.00 |"
)
_BOQ_PAGE_3 = (
    "Demolition and Site Clearance. Page d/3/3.\n"
    "| D549.2 | Removal of existing chain link fence | 80 | m | 15.00 |\n"
    "| D599.5 | Breaking out existing carriageway | 1,200 | m2 | 18.00 |"
)
DD23_BOQ_ROWS = [_BOQ_PAGE_1, _BOQ_PAGE_3]

# Another year's Conditions of Contract. Its prose describes the demolition
# scope, refers to the Contract Data, and lists the Schedules — so it reads
# as relevant to all three questions without answering any of them.
_DD22_SCOPE_CLAUSE = (
    "Volume 1 - Conditions of Contract. Sub-Clause 4.1. The Contractor "
    "shall execute the demolition and site clearance scope described in the "
    "Bill of Quantities and is responsible for the scope of the Works."
)
_DD22_SCHEDULES_CLAUSE = (
    "Volume 1 - Conditions of Contract. Schedules to the Contract. The "
    "Schedules listed in Volume 4 form part of the Contract. Schedule 10 "
    "shall be read with the Conditions of Contract."
)
_DD22_DELAY_CLAUSE_W2 = (
    "Volume 1 - Conditions of Contract. Sub-Clause 8.8 Delay Damages. The "
    "Contractor shall pay delay damages for the whole of the Works at the "
    "rate stated in the Contract Data for every calendar day."
)
DD22_CHUNKS_W2 = [
    _DD22_SCOPE_CLAUSE, _DD22_SCHEDULES_CLAUSE, _DD22_DELAY_CLAUSE_W2,
]

_FOREIGN_SCHEDULE_10 = (
    "2015 EXW Show Package. Schedule 10 — Works Guarantee. The Contractor "
    "shall provide a Works Guarantee in the form annexed. Schedule 10 sets "
    "out any applicable Works Guarantees required under the show package."
)
_FOREIGN_SCHEDULE_10_CONT = (
    "2015 EXW Show Package. Schedule 10 continued. The Works Guarantee "
    "shall be issued by a bank acceptable to the Employer."
)
FOREIGN_CHUNKS = [_FOREIGN_SCHEDULE_10, _FOREIGN_SCHEDULE_10_CONT]

_W2_NAMES = {
    "w2_dd23_cd": DD23_NAME,
    "w2_dd23_boq": DD23_BOQ_NAME,
    "w2_dd22": DD22_NAME,
    "w2_foreign": FOREIGN_NAME,
}

#: Per-case cosine, because the `fake` embedder is a hash and its similarity
#: carries no meaning. These are the orderings wave 2 measured: the wrong
#: source leads every one of the three.
_W2_SCORES = {
    F1: {"w2_dd22": 0.90, "w2_dd23_boq": 0.58, "w2_dd23_cd": 0.40,
         "w2_foreign": 0.35},
    G1: {"w2_foreign": 0.92, "w2_dd22": 0.80, "w2_dd23_cd": 0.45,
         "w2_dd23_boq": 0.20},
    E1: {"w2_dd22": 0.88, "w2_dd23_cd": 0.55, "w2_foreign": 0.30,
         "w2_dd23_boq": 0.20},
}
#: The money row states an amount and nothing the question says, so it scores
#: below its own siblings for this question.
_W2_MONEY_ROW_SCORE = 0.34


@pytest.fixture
def wave2_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_CD_PARTICULARS_BOOST", raising=False)

    from sqlalchemy import delete as _sa_delete

    from app.core.rag import embeddings as _emb
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as _vs

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    with store._lock, store._session_factory()() as session:
        session.execute(_sa_delete(store._rag_chunk_cls))
        session.commit()

    cd_chunks = contract_data_particulars_chunks(
        DD23_CONTRACT_DATA_W2, filename=DD23_NAME,
    )
    assert cd_chunks, "the particulars chunker produced nothing to retrieve"
    assert any(G1_REGISTER_ROW in c for c in cd_chunks), (
        "the schedule register did not land in a particulars window — the "
        "fixture no longer models what the indexer emits"
    )
    for doc_id, texts in (
        ("w2_dd23_cd", cd_chunks),
        ("w2_dd23_boq", DD23_BOQ_ROWS),
        ("w2_dd22", DD22_CHUNKS_W2),
        ("w2_foreign", FOREIGN_CHUNKS),
    ):
        store.upsert_chunks(PROJECT, doc_id, texts, embedder.encode(texts))

    monkeypatch.setattr(ret, "_doc_name_for_id", lambda d: _W2_NAMES.get(d, ""))

    real_search = store.search
    active = {"ask": F1}

    def live_shaped(project_id, query_vec, k=20, query_text=None):
        hits = real_search(project_id, query_vec, k=k, query_text=query_text)
        table = _W2_SCORES[active["ask"]]
        for chunk in hits:
            chunk.score = table.get(chunk.doc_id, 0.2)
            if active["ask"] is E1 and E1_MONEY in (chunk.text or ""):
                chunk.score = _W2_MONEY_ROW_SCORE
        return hits

    monkeypatch.setattr(store, "search", live_shaped)

    def top_k(ask, k=5):
        active["ask"] = ask
        chunks, _ = ret.retrieve_with_filter(ask, PROJECT, k=k)
        return [(_W2_NAMES.get(c.doc_id, ""), c.text or "") for c in chunks]

    yield ret, top_k

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


# ── F1: measured scope lives in the bill, not another year's prose ────────


def test_f1_reaches_the_contracts_own_bill_not_another_years_prose(wave2_corpus):
    """All three of F1's live citations were the wrong contract year."""
    _ret, top_k = wave2_corpus
    top = top_k(F1)

    assert top, "nothing retrieved — the bill is in the fixture"
    assert F1_BOQ_ITEM in _blob(top), (
        "no measured BOQ row reached the top-5:\n" + _report(top)
    )
    assert top[0][0] == DD23_BOQ_NAME, (
        "rank 1 is not the contract's own bill:\n" + _report(top)
    )
    assert DD22_NAME not in [name for name, _t in top], (
        "another contract year is still in the top-5:\n" + _report(top)
    )


def test_a_bill_of_quantities_ask_is_recognised():
    from app.core.rag.retriever import query_asks_for_boq_scope

    assert query_asks_for_boq_scope(F1)
    assert query_asks_for_boq_scope(
        "In the demolition BOQ, what is the rate and amount for general "
        "site clearance (item D110)?"
    )
    assert query_asks_for_boq_scope("What is in the priced Bill of Quantities?")
    # Not a bill ask: no measured-scope wording anywhere in it.
    assert not query_asks_for_boq_scope(G1)
    assert not query_asks_for_boq_scope(A3)
    assert not query_asks_for_boq_scope(
        "Who signed the letter about the batching plant at Creek Bend, "
        "and in what capacity?"
    )


def test_the_boq_document_test_is_the_indexers_own_pattern():
    """One pattern, not two. The indexer already uses it to pick a chunker
    and an OCR budget; a second copy here would drift from it silently."""
    from app.core.doc_index import _BOQ_NAME_RE, BOQ_FILENAME_RE
    from app.core.rag.retriever import document_is_a_bill_of_quantities

    assert BOQ_FILENAME_RE is _BOQ_NAME_RE

    for name in (
        DD23_BOQ_NAME,
        "FX-2044-0000-AAA-BOQ-CA-000001 Priced Bill of Quantities Rev B.pdf",
        "Schedule of Quantities - Roads.xlsx",
    ):
        assert document_is_a_bill_of_quantities(name), name
    for name in (DD22_NAME, DD23_NAME, FOREIGN_NAME, "", None):
        assert not document_is_a_bill_of_quantities(name), name


def test_mutation_probe_f1_needs_the_bill_to_count_as_evidence(
    wave2_corpus, monkeypatch,
):
    """Remove the bill from the kinds of answer the election knows about and
    F1's live failure returns: the wrong contract year takes the whole pool
    and the measured rows are at no rank."""
    ret, top_k = wave2_corpus
    monkeypatch.setattr(ret, "query_asks_for_boq_scope", lambda _q: False)
    top = top_k(F1)

    assert top, "the probe must reproduce a wrong answer, not an empty one"
    assert DD22_NAME == top[0][0], (
        "probe did not put the wrong year at rank 1:\n" + _report(top)
    )
    assert F1_BOQ_ITEM not in _blob(top), (
        "probe did not reproduce F1 — the bill is still reachable:\n"
        + _report(top)
    )


# ── G1: the register row that says "Not Used" IS the answer ───────────────


def test_g1_reaches_the_contracts_own_schedule_register(wave2_corpus):
    """Sev-1. The live answer invented a Works Guarantee out of another
    project's show package. The contract's own register says Not Used."""
    _ret, top_k = wave2_corpus
    top = top_k(G1)

    assert top, "nothing retrieved — the register is in the fixture"
    assert G1_REGISTER_ROW in _blob(top), (
        "the register row never reached the top-5, so the answer can only "
        "come from somewhere else:\n" + _report(top)
    )
    assert DD22_NAME not in [name for name, _t in top], (
        "another contract year is still in the top-5:\n" + _report(top)
    )


def test_g1_own_register_outranks_another_projects_schedule_ten(wave2_corpus):
    """The foreign document carries no contract id, so the fence cannot drop
    it — it has to LOSE. Ranking below the contract's own register is what
    lets #468's "Not Used IS the answer" instruction bite."""
    _ret, top_k = wave2_corpus
    top = top_k(G1)
    names = [name for name, _t in top]

    own = [i for i, (_n, text) in enumerate(top) if G1_REGISTER_ROW in text]
    foreign = [i for i, name in enumerate(names) if name == FOREIGN_NAME]
    assert own, _report(top)
    assert not foreign or min(own) < min(foreign), (
        "another project's Schedule 10 outranks this contract's register:\n"
        + _report(top)
    )
    assert names[0] == DD23_NAME, (
        "rank 1 is not the contract being asked about:\n" + _report(top)
    )


def test_a_numbered_contract_schedule_ask_wants_a_register_row():
    assert query_asks_for_contract_particulars(G1)
    assert query_asks_for_contract_particulars(
        "What does Schedule 9 of the contract volumes cover?"
    )
    assert query_asks_for_contract_particulars(
        "Does Appendix 3 of the contract apply?"
    )
    # No contract context: a numbered schedule also appears inside
    # specifications and method statements, and those asks stay put.
    assert not query_asks_for_contract_particulars(
        "What does Schedule 10 contain?"
    )
    assert not query_asks_for_contract_particulars(
        "What scheduling method does Specification 003113 require for "
        "programmes?"
    )


def test_mutation_probe_g1_needs_the_schedule_ask_recognised(
    wave2_corpus, monkeypatch,
):
    """With the Schedule-N ask unrecognised, nothing lifts the register row
    and the fence hands the pool to whichever contract sorted first."""
    ret, top_k = wave2_corpus
    real = ret.query_asks_for_contract_particulars
    monkeypatch.setattr(
        ret,
        "query_asks_for_contract_particulars",
        lambda q: False if q == G1 else real(q),
    )
    top = top_k(G1)

    assert top
    assert G1_REGISTER_ROW not in _blob(top), (
        "probe did not reproduce G1 — the register row is still reachable, "
        "so recognising the ask is not what fixed it:\n" + _report(top)
    )


# ── E1: a percentage is not an amount ─────────────────────────────────────


def test_e1_reaches_both_the_rate_and_the_amount_it_is_a_percentage_of(
    wave2_corpus,
):
    """The live answer had the rate and reported the SAR figure as absent."""
    _ret, top_k = wave2_corpus
    top = top_k(E1)

    assert len(top) == 5, _report(top)
    blob = _blob(top)
    assert E1_RATE_ROW in blob, (
        "the rate row is missing:\n" + _report(top)
    )
    assert E1_MONEY in blob, (
        "the amount the rate is a percentage of never reached the top-5, so "
        "the arithmetic still cannot run:\n" + _report(top)
    )


def test_the_reserved_row_costs_the_weakest_slot_not_an_extra_one(wave2_corpus):
    """A reservation, not an append: k is what the caller asked for, and the
    row the question named survives the swap.

    Asserted as presence, not as a rank. The particulars family is built to
    tie — every window here earns the same 0.85 plus the same capped label
    bonus — and which of a set of tied chunks a store returns first is a
    store's business: SQLite and pgvector order them differently, and an
    earlier version of this test read that difference as a regression.
    """
    _ret, top_k = wave2_corpus
    top = top_k(E1, k=5)

    assert len(top) == 5, (
        "the reservation took an extra slot instead of the weakest one:\n"
        + _report(top)
    )
    assert any(E1_RATE_ROW in text for _n, text in top), (
        "the reservation evicted the row the question names:\n" + _report(top)
    )
    assert any(E1_MONEY in text for _n, text in top), (
        "the reserved money row is missing from the top-k:\n" + _report(top)
    )


def test_an_arithmetic_money_ask_is_recognised():
    from app.core.rag.retriever import query_needs_a_monetary_base

    assert query_needs_a_monetary_base(E1)
    assert query_needs_a_monetary_base(
        "How much is the delay damages per day in AED?"
    )
    # Not arithmetic: a lookup for the rate itself needs no base.
    assert not query_needs_a_monetary_base(A5)
    assert not query_needs_a_monetary_base(A3)
    # Arithmetic without a money answer: a duration needs no amount.
    assert not query_needs_a_monetary_base(
        "Calculate the remaining Time for Completion in days."
    )


def test_a_particulars_row_states_an_amount_only_when_its_value_does():
    """"10% of the Accepted Contract Amount" NAMES the base without stating
    it — and that row is what E1 already had."""
    from app.core.rag.retriever import particulars_row_states_an_amount_of_money

    rows = contract_data_particulars_chunks(
        DD23_CONTRACT_DATA_W2, filename=DD23_NAME,
    )
    money = [r for r in rows if particulars_row_states_an_amount_of_money(r)]
    assert money, "no money row found in the fixture"
    assert all(E1_MONEY in r for r in money if "1.1.1" in r)

    cap = [r for r in rows if "Maximum amount of delay damages" in r
           and "1.1.1" not in r]
    assert cap, "fixture lost its cap row"
    assert not any(particulars_row_states_an_amount_of_money(r) for r in cap)


def test_the_reservation_is_a_no_op_when_an_amount_is_already_in_the_top_k():
    from app.core.rag.retriever import reserve_monetary_base_row
    from app.core.rag.vector_store import Chunk

    def _chunk(cid, text):
        return Chunk(chunk_id=cid, project_id=PROJECT, doc_id="d",
                     chunk_index=0, text=text, score=0.5)

    header = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\nContract Data\n"
    )
    kept = [_chunk("a", header + f"1.1.1 Accepted Contract Amount: {E1_MONEY}")]
    ranked = list(kept) + [
        _chunk("b", header + f"1.1.1 Accepted Contract Amount incl VAT: {E1_MONEY}")
    ]
    assert reserve_monetary_base_row(E1, kept, ranked) is False
    assert [c.chunk_id for c in kept] == ["a"]


def test_the_reservation_declines_for_a_non_arithmetic_ask():
    from app.core.rag.retriever import reserve_monetary_base_row
    from app.core.rag.vector_store import Chunk

    header = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\nContract Data\n"
    )
    kept = [Chunk(chunk_id="a", project_id=PROJECT, doc_id="d", chunk_index=0,
                  text=header + f"8.8 Delay Damages: {E1_RATE_ROW}", score=0.5)]
    ranked = list(kept) + [
        Chunk(chunk_id="b", project_id=PROJECT, doc_id="d", chunk_index=1,
              text=header + f"1.1.1 Accepted Contract Amount: {E1_MONEY}",
              score=0.4)
    ]
    assert reserve_monetary_base_row(A5, kept, ranked) is False
    assert [c.chunk_id for c in kept] == ["a"]


def test_the_reservation_cannot_reinstate_an_excluded_contract():
    """The reserved row goes through the caller's contract-scope test, so it
    cannot walk a wrong-year row back through the fence."""
    from app.core.rag.retriever import reserve_monetary_base_row
    from app.core.rag.vector_store import Chunk

    header = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD22_NAME}].\nContract Data\n"
    )
    kept = [Chunk(chunk_id="a", project_id=PROJECT, doc_id="d23", chunk_index=0,
                  text=f"8.8 Delay Damages: {E1_RATE_ROW}", score=0.5)]
    wrong_year = Chunk(chunk_id="b", project_id=PROJECT, doc_id="d22",
                       chunk_index=0,
                       text=header + f"1.1.1 Accepted Contract Amount: {E1_MONEY}",
                       score=0.4)
    ranked = list(kept) + [wrong_year]

    assert reserve_monetary_base_row(
        E1, kept, ranked, allow=lambda c: c.doc_id != "d22",
    ) is False
    assert [c.chunk_id for c in kept] == ["a"]
    # Without the gate the same row IS the one it would have taken, so the
    # gate is what excluded it rather than the predicate.
    kept2 = list(kept)
    assert reserve_monetary_base_row(E1, kept2, ranked) is True


# ══ WAVE 1 RETEST (29c4bdd / #483 live) ═══════════════════════════════════
#
# #483 made the unnamed fence elect from a filled particulars row. A9 now
# passes live (the Engineer row is a named party with a colon). A3 and A5
# still fail on Master Corpus:
#
# * A3 FAIL_WRONG_CONTRACT — "Overall duration for the completion of the
#   Works is 548 days" from another package's Volume 4 Schedules. The
#   executed contract's Time for Completion sits on a scanned clause line
#   with no pipe/colon, so the days stay in the key and the election
#   predicate (colon + content) never fires. Arrival order then locks the
#   pool to the schedule.
# * A5 FAIL — same-year General Conditions Sub-Clause 8.8 fills the top-k;
#   the cost gate then wipes a Delay Damages particular as if it were an
#   ungrounded BOQ unit rate.

DD22_SCHED_NAME = (
    "AB-2022-202 - the project Demolition and Site Clearance Works "
    "Package 1 Volume 4 Schedules.pdf"
)
DD22_SCHEDULE_548 = (
    "Schedule 5: Project Schedule. Overall duration for the completion "
    "of the Works is 548 days from the Commencement Date."
)
# Live scanned Contract Data: clause number + label + value, no separator.
# The days / percent live in the key, which is why #483's filled-row
# predicate cannot see them.
DD23_UNSPLIT_TFC_LINE = (
    "1.1.75 Time for Completion for the whole of the Works 852 days"
)
DD23_UNSPLIT_DELAY_LINE = (
    "8.8 Delay Damages for the whole of the Works 0.1% of the "
    "Contract Price per calendar day"
)
# Only the scanned TfC / Delay Damages lines — no pipe-separated siblings.
# A mixed window with ACA/DNP already elects via those filled rows, which
# is why A2/A6/A9 pass and is NOT the live A3/A5 miss.
DD23_UNSPLIT_CONTRACT_DATA = f"""Volume 1 - Conditions of Contract

Particular Conditions Part A - Contract Data

{DD23_UNSPLIT_TFC_LINE}
{DD23_UNSPLIT_DELAY_LINE}
1.1.27 Defects Notification Period | 365 days from the Taking-Over Certificate
1.3.1 (b) Engineer | Northwater Engineers (Demo Saudi Limited)
"""
A3_LIVE_ANSWER = "852 days"
A5_LIVE_ANSWER = "0.1% of the Contract Price per calendar day"


def _unsplit_particulars_chunk(line: str) -> str:
    """How the indexer renders a clause line it could not split: the value
    stays on the key, with no colon. Mirrors ``_format_cd_chunk`` when
    ``val`` is empty."""
    return (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD23_NAME}].\n"
        "Particular Conditions Part A - Contract Data\n"
        f"{line}"
    )


def test_a_scanned_clause_line_peels_the_trailing_filled_value():
    """Live A3/A5: scanned Contract Data rows arrive with no separator.
    The duration / percentage must come off the key so the election can
    see a filled particular."""
    from app.core.contract_data_chunks import parse_contract_data_rows

    rows = parse_contract_data_rows(
        "Particular Conditions Part A - Contract Data\n"
        f"{DD23_UNSPLIT_TFC_LINE}\n"
        f"{DD23_UNSPLIT_DELAY_LINE}\n"
    )
    tfc = [(k, v) for k, v in rows if "time for completion" in k.lower()]
    delay = [(k, v) for k, v in rows if "delay damages" in k.lower()]
    assert tfc, rows
    assert delay, rows
    assert "852" in tfc[0][1], (
        f"Time for Completion value was not peeled off the key: {tfc[0]!r}"
    )
    assert "0.1%" in delay[0][1], (
        f"Delay Damages value was not peeled off the key: {delay[0]!r}"
    )


def test_an_unsplit_tfc_particulars_chunk_is_filled_duration_evidence():
    """A9 treats a named party as filled. A Time for Completion / duration
    particular on a scanned line must count the same way, or the election
    declines and arrival order hands the pool to another package."""
    chunk = _unsplit_particulars_chunk(DD23_UNSPLIT_TFC_LINE)
    assert is_contract_data_particulars_row(chunk), (
        "852 days on the TfC line is filled duration evidence; the "
        "election must be able to see it"
    )
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_SCHED_NAME, DD22_SCHEDULE_548), (DD23_NAME, chunk)),
    ) == "ab-2023-101"


def test_a_schedule_overall_duration_is_not_time_for_completion_evidence():
    """Volume 4 'overall duration … 548 days' is a programme note, not the
    Contract Data particular. It must not elect a contract and must not
    beat the executed contract's own TfC row."""
    chunk = _unsplit_particulars_chunk(DD23_UNSPLIT_TFC_LINE)
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_SCHED_NAME, DD22_SCHEDULE_548)),
    ) is None
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_SCHED_NAME, DD22_SCHEDULE_548), (DD23_NAME, chunk)),
    ) == "ab-2023-101"


def test_election_requires_the_asked_label_not_any_filled_row():
    """A filled Accepted Contract Amount row must not lock the pool for a
    Time for Completion ask — that is how another package's particulars
    window can steal A3."""
    aca_only = (
        "CONTRACT DATA particulars — filled-in amount / duration / "
        f"percentage [{DD22_NAME}].\n"
        "Contract Data\n"
        "1.1.1 Accepted Contract Amount including VAT: SAR 9,936,000.00"
    )
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_NAME, aca_only)),
    ) is None
    tfc = _unsplit_particulars_chunk(DD23_UNSPLIT_TFC_LINE)
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_NAME, aca_only), (DD23_NAME, tfc)),
    ) == "ab-2023-101"


@pytest.fixture
def a3_schedule_steal_corpus(tmp_path, monkeypatch):
    """Live A3 shape: another package's Volume 4 schedule duration leads
    on cosine; the executed contract's TfC is a scanned clause line."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK_PROJECT)
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_CD_PARTICULARS_BOOST", raising=False)

    from sqlalchemy import delete as _sa_delete

    from app.core.rag import embeddings as _emb
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as _vs

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    with store._lock, store._session_factory()() as session:
        session.execute(_sa_delete(store._rag_chunk_cls))
        session.commit()

    names = {
        "dd22_sched": DD22_SCHED_NAME,
        "dd23_gc_delay": DD23_NAME,
    }
    cd_chunks = contract_data_particulars_chunks(
        DD23_UNSPLIT_CONTRACT_DATA, filename=DD23_NAME,
    )
    assert cd_chunks, "unsplittable Contract Data produced no chunks"
    for i, chunk in enumerate(cd_chunks):
        doc_id = f"dd23_unsplit_{i}"
        names[doc_id] = DD23_NAME
        store.upsert_chunks(PROJECT, doc_id, [chunk], embedder.encode([chunk]))

    store.upsert_chunks(
        PROJECT, "dd22_sched", [DD22_SCHEDULE_548],
        embedder.encode([DD22_SCHEDULE_548]),
    )
    store.upsert_chunks(
        PROJECT, "dd23_gc_delay", [DD22_DELAY_CLAUSE],
        embedder.encode([DD22_DELAY_CLAUSE]),
    )

    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))

    real_search = store.search

    def schedule_leads(project_id, query_vec, k=20, query_text=None):
        hits = real_search(project_id, query_vec, k=k, query_text=query_text)
        for chunk in hits:
            if chunk.doc_id == "dd22_sched":
                chunk.score = 0.93
            elif chunk.doc_id == "dd23_gc_delay":
                chunk.score = 0.90
            else:
                chunk.score = 0.52
        return hits

    monkeypatch.setattr(store, "search", schedule_leads)
    yield ret, names
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_a3_schedule_duration_does_not_steal_the_executed_contract(
    a3_schedule_steal_corpus,
):
    """Live A3: 548 days from AB-2022-202 Volume 4 Schedules. The executed
    contract's 852-day Time for Completion particular must own the pool."""
    ret, names = a3_schedule_steal_corpus
    top = _top_k(ret, names, A3)
    blob = _blob(top)
    assert A3_LIVE_ANSWER in blob, (
        "852-day TfC never reached the top-5:\n" + _report(top)
    )
    assert "548 days" not in blob, (
        "another package's schedule duration is still answering A3:\n"
        + _report(top)
    )
    assert top[0][0] == DD23_NAME, (
        "rank 1 is not the executed contract:\n" + _report(top)
    )
    assert DD22_SCHED_NAME not in [n for n, _t in top], (
        "the wrong-year schedule is still in the top-5:\n" + _report(top)
    )


def test_a5_rate_row_beats_same_year_general_conditions(
    a3_schedule_steal_corpus,
):
    """Live A5: three HIGH chunks from AB-2023-101 Conditions of Contract
    and no rate. The Contract Data percentage particular must lead."""
    ret, names = a3_schedule_steal_corpus
    top = _top_k(ret, names, A5)
    blob = _blob(top)
    assert A5_LIVE_ANSWER in blob, (
        "Delay Damages rate never reached the top-5:\n" + _report(top)
    )
    assert top[0][1].startswith("CONTRACT DATA particulars"), (
        "rank 1 is not the Contract Data rate row:\n" + _report(top)
    )
    assert "at the rate stated in the Contract Data" not in top[0][1]


def test_a2_a6_a9_still_pass_on_the_two_year_corpus(two_year_corpus):
    """The A3/A5 peel and label-aware election must not take the asks
    that already pass live off their filled rows."""
    ret, names = two_year_corpus
    _assert_own_contract_data_leads(_top_k(ret, names, A2), "SAR 9,936,000.00")
    _assert_own_contract_data_leads(_top_k(ret, names, A6), "365 days")
    _assert_own_contract_data_leads(_top_k(ret, names, A9), A9_ANSWER)


def test_mutation_probe_e1_needs_the_reservation(wave2_corpus, monkeypatch):
    """Disable the reservation and E1's live failure returns: the amount is
    at no rank, and the answer can only report it missing.

    Only the amount is asserted. The 44 filler rows in this fixture are
    deliberately label-identical to the rate row — same "delay damages …
    whole of the Works … calendar days" wording — so they earn the same
    capped label bonus and their order among themselves is arbitrary. That
    makes the fixture a hard test of the reservation and a useless one for
    any claim about where the rate row lands.

    ``ensure_kept_can_compose_daily_damages`` is the leftover-E1 last-chance twin of
    ``reserve_daily_damages_operands`` (refuse-prone top-k). The probe must
    disable that path too, or the amount is still reachable and the
    reservation is no longer what the probe measures.
    """
    ret, top_k = wave2_corpus
    monkeypatch.setattr(
        ret, "reserve_monetary_base_row",
        lambda *_a, **_kw: False,
    )
    monkeypatch.setattr(
        ret, "reserve_daily_damages_operands",
        lambda *_a, **_kw: False,
    )
    monkeypatch.setattr(
        ret, "ensure_kept_can_compose_daily_damages",
        lambda *_a, **_kw: False,
    )
    top = top_k(E1)

    assert top
    assert E1_MONEY not in _blob(top), (
        "probe did not reproduce E1 — the amount is reachable without the "
        "reservation, so the reservation is not what fixed it:\n"
        + _report(top)
    )


# ══ d7a4ca8 LIVE REGRESSION ═══════════════════════════════════════════════
#
# Wave-1 on tip/main d7a4ca8 (#500 unit pins) scored A3+A5
# FAIL_WRONG_CONTRACT against the same Neon Master Corpus that passed on
# 7877c21. The #483/#496 election still locked to the FIRST filled
# matching-label chunk. After the #499 kmz purge, a AB-2022-202
# demolition particular (548 days / Sub-Clause 8.8) ranked ahead of
# AB-2023-101's scanned 852-day / 0.1% rows and deleted 118 from every
# rank. Volume 4 "overall duration … 548 days" then stayed in the pool
# because it shares the elected year.
#
# The live UI pack prefixes every ask with the client-documents clamp.
# That must not change classification or election.

LIVE_PREFIX = "Answer only from the client project documents. "
LIVE_A3 = LIVE_PREFIX + A3
LIVE_A5 = LIVE_PREFIX + A5

DD22_FILLED_TFC_548 = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD22_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    "1.1.75 Time for Completion for the whole of the Works: 548 days"
)
DD22_FILLED_DELAY_CAP = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD22_NAME}].\n"
    "Particular Conditions Part A - Contract Data\n"
    "8.8 Maximum amount of delay damages: 10% of the Accepted Contract Amount"
)
DD22_MIXED_ACA_UNFILLED_TFC = (
    "CONTRACT DATA particulars — filled-in amount / duration / "
    f"percentage [{DD22_NAME}].\n"
    "Contract Data\n"
    "1.1.1 Accepted Contract Amount including VAT: SAR 1,000,000.00\n"
    "1.1.75 Time for Completion for the whole of the Works\n"
    "8.8 Delay Damages for the whole of the Works"
)


def test_a_mixed_window_does_not_elect_on_an_unfilled_asked_label():
    """Filled ACA + a bare TfC key is not Time for Completion evidence."""
    assert is_contract_data_particulars_row(DD22_MIXED_ACA_UNFILLED_TFC)
    assert not particulars_row_answers_asked_label(A3, DD22_MIXED_ACA_UNFILLED_TFC)
    assert not particulars_row_answers_asked_label(A5, DD22_MIXED_ACA_UNFILLED_TFC)
    assert particulars_row_answers_asked_label(A2, DD22_MIXED_ACA_UNFILLED_TFC)
    assert elect_answer_bearing_contract(
        A3, _ranked((DD22_NAME, DD22_MIXED_ACA_UNFILLED_TFC)),
    ) is None
    keys = [k.lower() for k, _v in filled_particulars_rows(DD22_MIXED_ACA_UNFILLED_TFC)]
    assert any("accepted contract amount" in k for k in keys)
    assert not any("time for completion" in k for k in keys)


def test_newer_year_owns_the_unnamed_ask_when_both_state_the_particular():
    """Live A3: 2022's own 548-day TfC row ranked first. 118's 852-day
    scanned line is also filled evidence. First-in-rank used to lock 2022
    and keep Volume 4 in the pool."""
    dd23 = _unsplit_particulars_chunk(DD23_UNSPLIT_TFC_LINE)
    ranked = _ranked(
        (DD22_NAME, DD22_FILLED_TFC_548),
        (DD22_SCHED_NAME, DD22_SCHEDULE_548),
        (DD23_NAME, dd23),
    )
    assert elect_answer_bearing_contract(A3, ranked) == "ab-2023-101"
    assert elect_answer_bearing_contract(LIVE_A3, ranked) == "ab-2023-101"
    assert particulars_row_answers_asked_label(A3, DD22_FILLED_TFC_548)
    assert particulars_row_answers_asked_label(A3, dd23)
    assert not particulars_row_answers_asked_label(A3, DD22_SCHEDULE_548)


def test_newer_year_owns_delay_damages_when_both_years_state_a_rate():
    """Live A5: 2022's Delay Damages family (cap row / GC 8.8) ranked
    first. 118's 0.1%-per-day scanned line must still own the unnamed ask."""
    dd23 = _unsplit_particulars_chunk(DD23_UNSPLIT_DELAY_LINE)
    ranked = _ranked(
        (DD22_NAME, DD22_DELAY_CLAUSE),
        (DD22_NAME, DD22_FILLED_DELAY_CAP),
        (DD23_NAME, dd23),
    )
    assert elect_answer_bearing_contract(A5, ranked) == "ab-2023-101"
    assert elect_answer_bearing_contract(LIVE_A5, ranked) == "ab-2023-101"


def test_a_delay_damages_cap_is_not_the_daily_rate():
    """3a5fce5 A5: year-lock elected 118 from a cap / 8.8 family row.
    The 0.1%-per-day figure is a different chunk and must be the thing
    reservation / the rate fence treat as the answer."""
    from app.core.rag.retriever import chunk_states_delay_damages_rate

    dd23 = _unsplit_particulars_chunk(DD23_UNSPLIT_DELAY_LINE)
    assert chunk_states_delay_damages_rate(dd23)
    assert not chunk_states_delay_damages_rate(DD22_FILLED_DELAY_CAP)
    assert not chunk_states_delay_damages_rate(DD22_DELAY_CLAUSE)
    assert particulars_row_answers_asked_label(A5, DD22_FILLED_DELAY_CAP)


def test_naming_the_older_year_still_fail_closes_to_that_year(two_year_corpus):
    """Newest-year election is unnamed-only. #443 still owns a named id."""
    ret, names = two_year_corpus
    chunks, _ = ret.retrieve_with_filter(
        "Per the AB-2022-202 contract, what is the Time for Completion "
        "for the whole of the Works?",
        PROJECT,
        k=5,
    )
    assert chunks, "the named year is in the fixture"
    assert all(names.get(c.doc_id, "") == DD22_NAME for c in chunks)


@pytest.fixture
def live_two_filled_years_corpus(tmp_path, monkeypatch):
    """d7a4ca8 live A3/A5 shape: 2022 states the particular (548 days /
    a delay-damages cap), Volume 4 restates 548, 118's scanned 852 / 0.1%
    sit lower on cosine. The live prefix is the ask the UI actually sent.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK_PROJECT)
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_CD_PARTICULARS_BOOST", raising=False)

    from sqlalchemy import delete as _sa_delete

    from app.core.rag import embeddings as _emb
    from app.core.rag import retriever as ret
    from app.core.rag import vector_store as _vs

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    with store._lock, store._session_factory()() as session:
        session.execute(_sa_delete(store._rag_chunk_cls))
        session.commit()

    names = {
        "dd22_tfc": DD22_NAME,
        "dd22_delay_cap": DD22_NAME,
        "dd22_gc_delay": DD22_NAME,
        "dd22_sched": DD22_SCHED_NAME,
        "dd22_mixed": DD22_NAME,
    }
    cd_chunks = contract_data_particulars_chunks(
        DD23_UNSPLIT_CONTRACT_DATA, filename=DD23_NAME,
    )
    assert cd_chunks, "unsplittable Contract Data produced no chunks"
    for i, chunk in enumerate(cd_chunks):
        doc_id = f"dd23_unsplit_{i}"
        names[doc_id] = DD23_NAME
        store.upsert_chunks(PROJECT, doc_id, [chunk], embedder.encode([chunk]))

    store.upsert_chunks(
        PROJECT, "dd22_tfc", [DD22_FILLED_TFC_548],
        embedder.encode([DD22_FILLED_TFC_548]),
    )
    store.upsert_chunks(
        PROJECT, "dd22_delay_cap", [DD22_FILLED_DELAY_CAP],
        embedder.encode([DD22_FILLED_DELAY_CAP]),
    )
    store.upsert_chunks(
        PROJECT, "dd22_gc_delay", [DD22_DELAY_CLAUSE],
        embedder.encode([DD22_DELAY_CLAUSE]),
    )
    store.upsert_chunks(
        PROJECT, "dd22_sched", [DD22_SCHEDULE_548],
        embedder.encode([DD22_SCHEDULE_548]),
    )
    store.upsert_chunks(
        PROJECT, "dd22_mixed", [DD22_MIXED_ACA_UNFILLED_TFC],
        embedder.encode([DD22_MIXED_ACA_UNFILLED_TFC]),
    )

    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))

    real_search = store.search

    def older_year_leads(project_id, query_vec, k=20, query_text=None):
        hits = real_search(project_id, query_vec, k=k, query_text=query_text)
        for chunk in hits:
            if chunk.doc_id.startswith("dd22_"):
                chunk.score = 0.93
            else:
                chunk.score = 0.51
        return hits

    monkeypatch.setattr(store, "search", older_year_leads)
    yield ret, names
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_a3_live_prefix_returns_852_not_548(live_two_filled_years_corpus):
    """d7a4ca8 A3: 548 days from AB-2022-202 Volume 4. Want 852 / 118."""
    ret, names = live_two_filled_years_corpus
    top = _top_k(ret, names, LIVE_A3)
    blob = _blob(top)
    assert A3_LIVE_ANSWER in blob, (
        "852-day TfC never reached the top-5:\n" + _report(top)
    )
    assert "548 days" not in blob, (
        "the older package is still answering A3:\n" + _report(top)
    )
    assert all("AB-2023-101" in n or not n for n, _t in top), (
        "a AB-2022 file survived the unnamed fence:\n" + _report(top)
    )


def test_a5_live_prefix_returns_the_rate_not_the_2022_clause(
    live_two_filled_years_corpus,
):
    """d7a4ca8 A5: Sub-Clause 8.8 from AB-2022-202, no 0.1%. Want 118."""
    ret, names = live_two_filled_years_corpus
    top = _top_k(ret, names, LIVE_A5)
    blob = _blob(top)
    assert A5_LIVE_ANSWER in blob, (
        "0.1% rate never reached the top-5:\n" + _report(top)
    )
    assert "at the rate stated in the Contract Data" not in blob, (
        "the 2022 General Conditions pointer is still in the excerpts:\n"
        + _report(top)
    )
    assert all("AB-2023-101" in n or not n for n, _t in top), (
        "a AB-2022 file survived the unnamed fence:\n" + _report(top)
    )


def test_a2_a6_a9_still_pass_when_the_older_year_also_has_filled_rows(
    live_two_filled_years_corpus,
):
    """Newest-year election must not take the already-passing asks off 118."""
    ret, names = live_two_filled_years_corpus
    # A2 is in the mixed 2022 window (filled ACA) AND missing from the
    # 118 unsplit fixture (only TfC + Delay Damages lines). A2 is
    # therefore not asserted here — the two_year_corpus control covers it.
    _assert_own_contract_data_leads(_top_k(ret, names, A6), "365 days")
    _assert_own_contract_data_leads(_top_k(ret, names, A9), A9_ANSWER)
