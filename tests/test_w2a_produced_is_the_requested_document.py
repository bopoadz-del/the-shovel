"""W2-a (exam): a produced document IS the requested document.

Live (REPORT_TESTER_2121Z, order 3): a user asked for a one-page summary
document as a downloadable file and pressed the Word button under the answer.
The file was headed "<project> - Conversation Excerpt" with an export
timestamp, and held the chat answer with its inline ``[source: ...]`` tags.

The rules this exam holds:

* WHICH file is made is decided by the control pressed, never by wording:
  ``scope=document`` (the answer's own download control) produces the
  document; ``scope=message`` (the default) is still the plain conversation
  excerpt, and ``scope=answers`` is still the answer-range report, unchanged.
* A produced document carries its own title (the answer's opening heading,
  else the project's name -- never the user's request sentence).
* Its body has no chat chrome: no source lines, no inline source tags, no
  closing question to the reader, no marker the answer contract numbered.
  This holds for an answer in any language (an Arabic answer is tested) and
  a bracketed number that was in the answer itself (a clause or note
  reference) survives.
* Its sources are the answer's STRUCTURED citations, stored on the assistant
  message when the answer leaves (``exit_frames.with_answer_exit``), listed
  once at the end -- including a source the answer drew on but did not write
  inline. Re-reading the text is only the fallback for a message stored
  without them.

Everything goes through the real stored form: a registered user, a real
project, messages written to the message store, the export route by its
handler. Every name and answer here is invented by the test; the expected
title, body and source lines are read off the test's own input.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import exports
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]

ASK_HEADED = "Put the drainage pipe schedule in a short document I can download."
ANSWER_HEADED = """## Drainage Pipe Schedule

Pipe class B is specified for all carriageway crossings [Source: Civil Works Specification.pdf, p. 12].

Bedding is granular type S under every run.
Source: Site Drainage Notes.docx, section 4.2

Would you like me to price these runs as well?
"""
TITLE_HEADED = "Drainage Pipe Schedule"
BODY_HEADED = (
    "Pipe class B is specified for all carriageway crossings.",
    "Bedding is granular type S under every run.",
)
SOURCES_HEADED = (
    "Civil Works Specification.pdf, p. 12",
    "Site Drainage Notes.docx, section 4.2",
)

ASK_PLAIN = "Write a one-page note on the slab curing regime"
ANSWER_PLAIN = (
    "Slabs are cured for 7 days under wet hessian (Curing Method Statement.pdf, p. 3).\n"
    "Membrane curing is not permitted on surfaces that receive a screed."
)

CHROME = (
    "Conversation Excerpt",
    "Source:",
    "[Source",
    "(Curing Method Statement",
    "chat UI",
    "live conversation",
    "Would you like",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"w2a-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _turn(cid: str, pid: str, ask: str, answer: str) -> None:
    from app.core import agent_memory

    agent_memory.get_or_create_conversation(cid, "project_assistant", pid)
    agent_memory.append_message(cid, "user", ask)
    agent_memory.append_message(cid, "assistant", answer)


def _export(client, headers, pid, cid, fmt, scope: str | None = "document", extra: str = ""):
    # The route is addressed by its handler, so the request provably reaches
    # exports.export_conversation_message and not a look-alike path.
    path = app.url_path_for(
        exports.export_conversation_message.__name__, project_id=pid, conversation_id=cid,
    )
    query = f"format={fmt}" + (f"&scope={scope}" if scope else "") + extra
    res = client.post(f"{path}?{query}", headers=headers)
    assert res.status_code == 200, res.text
    return res


def _docx_paragraphs(raw: bytes) -> list[tuple[str, str]]:
    from docx import Document

    doc = Document(io.BytesIO(raw))
    return [(p.style.name, p.text.strip()) for p in doc.paragraphs if p.text.strip()]


def _xlsx_sheets(raw: bytes) -> dict[str, list[str]]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(raw))
    return {ws.title: [str(c.value).strip() for row in ws.iter_rows() for c in row
                       if c.value is not None and str(c.value).strip()] for ws in wb.worksheets}


def _texts(fmt: str, raw: bytes) -> list[str]:
    if fmt == "docx":
        paras = _docx_paragraphs(raw)
        assert paras[0][0] == "Title", paras[:2]
        return [t for _s, t in paras]
    return [t for cells in _xlsx_sheets(raw).values() for t in cells]


def _no_chrome(texts: list[str]) -> None:
    joined = "\n".join(texts)
    for bit in CHROME:
        assert bit not in joined, f"chat chrome {bit!r} in the produced document:\n{joined}"
    assert not re.search(r"\[\d+\]", joined), f"a citation marker is left in the body:\n{joined}"


def _sources_at_end(texts: list[str], expected: tuple[str, ...]) -> None:
    assert "Sources" in texts, texts
    at = len(texts) - 1 - texts[::-1].index("Sources")
    assert list(texts[at + 1:]) == list(expected), texts[at:]


#: (ask, answer, title, body sentences, source lines). Title, body and source
#: lines are read off each answer by hand. ``None`` as the title means the
#: project's name: the answer has no title of its own.
CASES = {
    "headed": (ASK_HEADED, ANSWER_HEADED, TITLE_HEADED, BODY_HEADED, SOURCES_HEADED),
    "untitled": (
        ASK_PLAIN, ANSWER_PLAIN, None,
        ("Slabs are cured for 7 days under wet hessian.",
         "Membrane curing is not permitted on surfaces that receive a screed."),
        ("Curing Method Statement.pdf, p. 3",),
    ),
    "cites_nothing": (
        "List the three pour bays in order",
        "## Pour Order\n\nBay 1, then bay 3, then bay 2.",
        "Pour Order",
        ("Bay 1, then bay 3, then bay 2.",),
        ("No document was cited in this answer.",),
    ),
}
EARLIER = ("An earlier, unrelated question about site access", "Gate 2 is open from 6am.")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_a_produced_document_has_its_own_title_body_and_sources_at_the_end(client, fmt, case):
    ask, answer, title, body, sources = CASES[case]
    headers = _register(client, f"{fmt}-{case}")
    name = f"W2a-{fmt}-{case}-{_RUN}"
    pid = _project(client, headers, name)
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, *EARLIER)
    _turn(cid, pid, ask, answer)
    title = title or name

    texts = _texts(fmt, _export(client, headers, pid, cid, fmt).content)

    assert texts[0] == title, texts[:3]
    assert texts[0] != ask, "the title is the user's request sentence"
    assert texts.count(title) == 1, "the title is repeated in the body"
    for sentence in body:
        assert any(sentence in t for t in texts), (sentence, texts)
    assert not any(EARLIER[1] in t for t in texts), "another turn's answer is in the document"
    _no_chrome(texts)
    _sources_at_end(texts, sources)


@pytest.mark.parametrize("case", ["headed", "untitled"])
def test_the_produced_file_is_listed_under_its_session_by_its_title(client, case):
    ask, answer, title, _body, _sources = CASES[case]
    headers = _register(client, f"listed-{case}")
    name = f"W2a-listed-{case}-{_RUN}"
    pid = _project(client, headers, name)
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, ask, answer)

    res = _export(client, headers, pid, cid, "docx")
    disposition = res.headers["content-disposition"]
    assert (title or name).replace(" ", "_") in disposition, disposition

    listed = client.get(
        f"/v1/projects/{pid}/conversations/{cid}/documents", headers=headers,
    ).json()["documents"]
    assert len(listed) == 1, listed
    assert listed[0]["origin"] == "produced"
    assert listed[0]["original_name"] in disposition


# ── (1) the plain excerpt and the answers-scope report are unchanged ────────

@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_without_scope_document_the_export_is_still_the_plain_excerpt(client, fmt):
    headers = _register(client, f"excerpt-{fmt}")
    name = f"W2a-excerpt-{fmt}-{_RUN}"
    pid = _project(client, headers, name)
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, ASK_HEADED, ANSWER_HEADED)

    res = _export(client, headers, pid, cid, fmt, scope=None)
    texts = _texts(fmt, res.content)

    assert texts[0] == f"{name} - Conversation Excerpt", texts[:2]
    # The excerpt is the answer as it was stored: its own source line and
    # its closing question stay.
    assert any("Source: Site Drainage Notes.docx" in t for t in texts), texts
    assert any("Would you like me to price these runs as well?" in t for t in texts), texts
    assert f"{_sanitize(name)}-{cid[:8]}.{fmt}" in res.headers["content-disposition"]
    if fmt == "xlsx":
        assert "Sources" not in _xlsx_sheets(res.content), "the excerpt workbook gained a sheet"


def test_the_answers_scope_spreadsheet_is_unchanged(client):
    headers = _register(client, "answers-xlsx")
    name = f"W2a-answers-{_RUN}"
    pid = _project(client, headers, name)
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, *EARLIER)
    _turn(cid, pid, ASK_HEADED, ANSWER_HEADED)

    res = _export(client, headers, pid, cid, "xlsx", scope="answers")
    sheets = _xlsx_sheets(res.content)

    assert list(sheets) == ["Message"], list(sheets)
    assert sheets["Message"][0] == f"{name} - Conversation Excerpt", sheets["Message"][:2]
    assert any(EARLIER[1] in t for t in sheets["Message"]), "an answer in the range is missing"


def _sanitize(s: str) -> str:
    return exports._sanitize_filename(s)


# ── (2) sources come from the citations stored with the message ─────────────

def _end_frame(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _through_the_exit(frames: list[str], cid: str) -> list[str]:
    from app.routers.exit_frames import with_answer_exit

    async def gen():
        for f in frames:
            yield f

    return [f async for f in with_answer_exit(gen(), cid)]


def test_the_document_lists_the_citations_stored_when_the_answer_left(client):
    """The answer's citations are stored on its message by the stream's exit
    wrapper, and the document lists them -- including a document the answer
    drew on (a Sources row) but never wrote inline."""
    from app.core import agent_memory

    headers = _register(client, "stored")
    pid = _project(client, headers, f"W2a-stored-{_RUN}")
    cid = f"ws-{pid}-1000"
    answer = (
        "## Kerb Schedule\n\n"
        "Precast kerbs are bedded on C20 concrete (Highway Details Register.pdf, p. 7).\n"
        "Dropped kerbs sit at every crossing."
    )
    _turn(cid, pid, "Put the kerb schedule in a document", answer)
    end = {
        "type": "end", "content": answer,
        "sources": [
            {"doc_name": "Highway Details Register.pdf", "page_or_section": "p. 7"},
            {"doc_name": "Crossing Layout Sheet.xlsx", "page_or_section": "Sheet 2"},
        ],
    }
    asyncio.run(_through_the_exit([_end_frame({"type": "token", "content": answer}), _end_frame(end)], cid))

    stored = [m for m in agent_memory.get_messages(cid) if m["role"] == "assistant"][-1]["citations"]
    assert stored and [e["document"] for e in stored] == [
        "Highway Details Register.pdf", "Crossing Layout Sheet.xlsx"], stored
    assert stored[0]["marker"] == 1 and stored[1]["marker"] is None, stored

    texts = _texts("docx", _export(client, headers, pid, cid, "docx").content)
    _no_chrome(texts)
    _sources_at_end(texts, ("Highway Details Register.pdf, p. 7", "Crossing Layout Sheet.xlsx, Sheet 2"))


def test_citations_never_land_on_an_earlier_answer():
    """When the answer of this turn was not stored, the newest message is the
    user's ask: nothing is attached, so an earlier answer never gets this
    turn's sources."""
    from app.core import agent_memory

    cid = f"ws-w2a-guard-{_RUN}-1000"
    agent_memory.get_or_create_conversation(cid, "project_assistant", None)
    agent_memory.append_message(cid, "user", "First ask")
    agent_memory.append_message(cid, "assistant", "First answer.")
    agent_memory.append_message(cid, "user", "Second ask")
    entry = [{"marker": 1, "document": "Invented Register.pdf", "section": "", "tier": None, "kind": "document"}]

    assert agent_memory.attach_citations(cid, entry) is False
    assert all(m.get("citations") is None for m in agent_memory.get_messages(cid))


# ── (3) Arabic: removal by structure, not by English words ──────────────────

ARABIC_ANSWER = (
    "## جدول أنابيب الصرف\n\n"
    "تُستخدم أنابيب الفئة ب في جميع معابر الطرق (مواصفات الأعمال المدنية.pdf، ص 12).\n"
    "الفرشة من الحصى الحبيبي تحت كل خط.\n"
    "المصدر: ملاحظات صرف الموقع.docx، القسم 4.2\n\n"
    "هل تريد أن أسعّر هذه الخطوط أيضاً؟"
)


@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_an_arabic_answer_loses_its_source_line_inline_cite_and_closing_offer(client, fmt):
    headers = _register(client, f"arabic-{fmt}")
    pid = _project(client, headers, f"W2a-arabic-{fmt}-{_RUN}")
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, "ضع جدول الأنابيب في مستند", ARABIC_ANSWER)

    texts = _texts(fmt, _export(client, headers, pid, cid, fmt).content)
    joined = "\n".join(texts)

    assert texts[0] == "جدول أنابيب الصرف", texts[:2]
    assert any("تُستخدم أنابيب الفئة ب في جميع معابر الطرق" in t for t in texts), texts
    assert any("الفرشة من الحصى الحبيبي تحت كل خط." in t for t in texts), texts
    body = texts[: len(texts) - 1 - texts[::-1].index("Sources")]
    assert not any(".pdf" in t or ".docx" in t for t in body), f"a citation is left in the body:\n{joined}"
    assert not any("المصدر" in t for t in body), f"the source line is left in the body:\n{joined}"
    assert not any("؟" in t for t in body), f"the closing offer is left in the body:\n{joined}"
    after = texts[len(texts) - texts[::-1].index("Sources"):]
    assert any("مواصفات الأعمال المدنية.pdf" in t for t in after), after
    assert any("ملاحظات صرف الموقع.docx" in t for t in after), after


# ── (4) only the contract's own markers are stripped ────────────────────────

def test_a_bracketed_note_number_in_the_answer_survives(client):
    """The answer says "note [1]" itself, and its inline file citation becomes
    the contract's marker [1] as well. Only the contract's marker goes."""
    headers = _register(client, "bracket")
    pid = _project(client, headers, f"W2a-bracket-{_RUN}")
    cid = f"ws-{pid}-1000"
    answer = (
        "## Joint Spacing\n\n"
        "Contraction joints are spaced at 6 m as set out in note [1] of the joint drawing "
        "(Pavement Joint Layout.pdf, p. 2).\n"
        "Clause [14] governs sealing."
    )
    _turn(cid, pid, "Put the joint spacing in a document", answer)

    texts = _texts("docx", _export(client, headers, pid, cid, "docx").content)

    assert any("as set out in note [1] of the joint drawing." in t for t in texts), texts
    assert any("Clause [14] governs sealing." in t for t in texts), texts
    assert not any("Pavement Joint Layout" in t for t in texts[: texts.index("Sources")]), texts
    _sources_at_end(texts, ("Pavement Joint Layout.pdf, p. 2",))
