"""W2-a (exam): a produced document IS the requested document.

Live (REPORT_TESTER_2121Z, order 3): a user asked for a one-page summary
document as a downloadable file. The Word file was headed "<project> -
Conversation Excerpt" with an export timestamp, and held the chat answer with
its inline ``[source: ...]`` tags. That is an export of the conversation,
not the document that was asked for.

The rule this exam holds, for ANY produced-document export of an answer:

* The document carries its own title: the answer's own opening heading, or
  else the user's ask the answer replied to. Never "Conversation Excerpt".
* The body is the answer's content with no chat chrome: no inline source
  tags, no ``Source:`` lines, no numbered markers, no offer of further help,
  no pointer to "the chat UI" or "the live conversation".
* Sources are listed once, at the end, one line per cited document with its
  page or section, in the order the answer cited them.
* The same file is listed under its session, named after its title.

Everything goes through the real stored form: a registered user, a real
project, messages written to the message store, the export route by name and
the session's documents route. Every name and answer here is invented by the
test; the expected title, body and source lines are read off the test's own
input text, never off the product's output.
"""
from __future__ import annotations

import io
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
#: Read off ANSWER_HEADED by hand: its heading, its two content sentences,
#: its two citations in the order they appear.
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


def _export(client, headers, pid, cid, fmt):
    # The route is addressed by its handler, so the request provably reaches
    # exports.export_conversation_message and not a look-alike path.
    path = app.url_path_for(
        exports.export_conversation_message.__name__, project_id=pid, conversation_id=cid,
    )
    res = client.post(f"{path}?format={fmt}", headers=headers)
    assert res.status_code == 200, res.text
    return res


def _docx_paragraphs(raw: bytes) -> list[tuple[str, str]]:
    from docx import Document

    doc = Document(io.BytesIO(raw))
    return [(p.style.name, p.text.strip()) for p in doc.paragraphs if p.text.strip()]


def _xlsx_cells(raw: bytes) -> list[str]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(raw))
    return [str(c.value).strip() for ws in wb.worksheets for row in ws.iter_rows()
            for c in row if c.value is not None and str(c.value).strip()]


def _no_chrome(texts: list[str]) -> None:
    joined = "\n".join(texts)
    for bit in CHROME:
        assert bit not in joined, f"chat chrome {bit!r} in the produced document:\n{joined}"
    import re
    assert not re.search(r"\[\d+\]", joined), f"a citation marker is left in the body:\n{joined}"


def _sources_at_end(texts: list[str], expected: tuple[str, ...]) -> None:
    assert "Sources" in texts, texts
    at = len(texts) - 1 - texts[::-1].index("Sources")
    assert list(texts[at + 1:]) == list(expected), texts[at:]


#: (ask, answer, title, body sentences, source lines). Title, body and source
#: lines are read off each answer by hand: its opening heading or else the ask,
#: its content sentences, its citations in the order they appear.
CASES = {
    "headed": (ASK_HEADED, ANSWER_HEADED, TITLE_HEADED, BODY_HEADED, SOURCES_HEADED),
    "titled_by_ask": (
        ASK_PLAIN, ANSWER_PLAIN, ASK_PLAIN,
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


def _texts(fmt: str, raw: bytes) -> list[str]:
    if fmt == "docx":
        paras = _docx_paragraphs(raw)
        assert paras[0][0] == "Title", paras[:2]
        return [t for _s, t in paras]
    return _xlsx_cells(raw)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("fmt", ["docx", "xlsx"])
def test_a_produced_document_has_its_own_title_body_and_sources_at_the_end(client, fmt, case):
    ask, answer, title, body, sources = CASES[case]
    headers = _register(client, f"{fmt}-{case}")
    pid = _project(client, headers, f"W2a-{fmt}-{case}-{_RUN}")
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, *EARLIER)
    _turn(cid, pid, ask, answer)

    texts = _texts(fmt, _export(client, headers, pid, cid, fmt).content)

    assert texts[0] == title, texts[:3]
    assert texts.count(title) == 1, "the title is repeated in the body"
    for sentence in body:
        assert any(sentence in t for t in texts), (sentence, texts)
    assert not any(EARLIER[1] in t for t in texts), "another turn's answer is in the document"
    _no_chrome(texts)
    _sources_at_end(texts, sources)


@pytest.mark.parametrize("case", ["headed", "titled_by_ask"])
def test_the_produced_file_is_listed_under_its_session_by_its_title(client, case):
    ask, answer, title, _body, _sources = CASES[case]
    headers = _register(client, f"listed-{case}")
    pid = _project(client, headers, f"W2a-listed-{case}-{_RUN}")
    cid = f"ws-{pid}-1000"
    _turn(cid, pid, ask, answer)

    res = _export(client, headers, pid, cid, "docx")
    disposition = res.headers["content-disposition"]
    assert title.replace(" ", "_") in disposition, disposition

    listed = client.get(
        f"/v1/projects/{pid}/conversations/{cid}/documents", headers=headers,
    ).json()["documents"]
    assert len(listed) == 1, listed
    assert listed[0]["origin"] == "produced"
    assert listed[0]["original_name"] in disposition
