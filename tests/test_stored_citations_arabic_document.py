"""An answer about an Arabic document is stored with its citations, and the
document produced from it lists them (#395 follow-up 2, advisor 17:32Z,
17 Nov).

Found by this test, three places where the answer contract read a citation
by Latin letters only (``app.agents.answer_contract``):

* the key a cited name is matched to its Sources row by kept ``[a-z0-9]``,
  so every Arabic name keyed to the same empty string and never took its
  row (no tier, no page from the row);
* a locator after the name was found only behind a Latin comma and an
  English word (``p.``, ``section``), so ``X.docx، صفحة 3`` was stored as
  one document name with no section;
* a label before the name was dropped only when it was the English
  ``source:``, so ``[المصدر: X.pdf]`` was stored as a document called
  ``المصدر: X.pdf``.

The rule now, by structure and no word of any language: a file name ends at
its extension and holds no colon; a label ends at the last colon before the
name, and what follows the extension past its punctuation is the locator.
A name is keyed by its letters and digits in any script. Conflict lines
name a document whose name holds letters of any script.

Every user, project, document name, question and answer is invented here;
the expected lines are derived from the test's own documents and answer.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents import answer_contract
from app.agents.runtime import Agent
from app.core.rag.vector_store import Chunk
from app.main import app
from app.routers import exports
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]

# Separators between a name and its locator: Latin comma, Arabic comma,
# Arabic semicolon, a dash.
SEPARATORS = [", ", "، ", "؛ ", " — "]
NAMES = ["مواصفات الأسوار.docx", "مخطط حدود الموقع.pdf", "Pump Schedule.xlsx"]
LOCATORS = ["صفحة 3", "البند 4.2", "p. 7"]


# ── the parser, by structure ────────────────────────────────────────────────

@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("locator", LOCATORS)
def test_the_locator_after_the_name_is_its_section_in_any_script(sep, name, locator):
    done = answer_contract.apply(f"تُثبّت الأعمدة كل 2.4 م [source: {name}{sep}{locator}].", sources=[])
    assert [(c["document"], c["section"]) for c in done.citations] == [(name, locator)], done.citations


@pytest.mark.parametrize("label", ["المصدر", "Source", "Référence", "المراجع"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bracket", ["[]", "()"])
def test_a_label_before_the_name_is_not_part_of_the_name(label, name, bracket):
    text = f"الحد بطول 640 م {bracket[0]}{label}: {name}{bracket[1]}"
    done = answer_contract.apply(text, sources=[])
    assert [c["document"] for c in done.citations] == [name], done.citations
    assert label not in done.text and name not in done.text, done.text


@pytest.mark.parametrize("names", [
    ("مواصفات الأسوار.docx", "مخطط حدود الموقع.pdf"),
    ("سجل المنشآت.pdf", "جدول البوابات.xlsx"),
])
def test_each_arabic_name_takes_its_own_sources_row(names):
    rows = [{"doc_name": n, "page_or_section": f"p. {i + 2}", "layer_label": f"tier-{i}"}
            for i, n in enumerate(names)]
    # Cited with no locator: the page and the tier come from the row.
    text = "\n".join(f"جملة {i} [source: {n}]" for i, n in enumerate(reversed(names)))
    done = answer_contract.apply(text, sources=rows)
    got = [(c["document"], c["section"], c["tier"]) for c in done.citations]
    want = [(r["doc_name"], r["page_or_section"], r["layer_label"]) for r in reversed(rows)]
    assert got == want, got


@pytest.mark.parametrize("name,words", [
    ("مخطط حدود الموقع.pdf", True), ("Fence Spec.docx", True),
    ("4417-22.pdf", False), ("٤٤١٧.pdf", False),
])
def test_a_document_is_nameable_by_letters_of_any_script(name, words):
    assert bool(answer_contract._doc_words(name)) is words


@pytest.mark.parametrize("note", ["البند [7] من العقد يبقى.", "See note [3] of the drawing."])
def test_a_bracketed_number_that_cites_no_file_survives(note):
    done = answer_contract.apply(f"{note}\nتُثبّت الأعمدة كل 2.4 م [المصدر: مواصفات الأسوار.docx، صفحة 3]",
                                 sources=[])
    assert note in done.text and [c["marker"] for c in done.citations] == [1], done.text


# ── through the real route: ask in Arabic, store, export ────────────────────

TURNS = [
    {
        "ask": "ما متطلبات السياج الحدودي في مستندات المشروع؟",
        "title": "متطلبات السياج الحدودي",
        "docs": [("مواصفات الأسوار.docx", "تُثبّت الأعمدة كل 2.4 م."),
                 ("مخطط حدود الموقع.pdf", "يمتد الحد مسافة 640 م."),
                 ("جدول البوابات.xlsx", "تُفتح البوابات إلى الداخل.")],
        # (document index, how the answer writes the citation around the name)
        "cites": [(1, "[المصدر: {name}، صفحة 2]"), (0, "[source: {name}؛ البند 4.2]")],
        "sections": ["صفحة 2", "البند 4.2"],
    },
    {
        "ask": "ماذا تقول مستندات العبّارة عن جدران الرأس؟",
        "title": "جدران الرأس",
        "docs": [("سجل المنشآت.pdf", "تُصبّ جدران الرأس بخرسانة C32/40."),
                 ("مقاطع جدار الرأس.dwg", "تنحرف الجدران الجانبية بزاوية 45 درجة."),
                 ("ملاحظات المئزر.txt", "سماكة المئزر 300 مم.")],
        "cites": [(0, "({name}، ص 12)"), (1, "[المرجع: {name}]")],
        "sections": ["ص 12", ""],
    },
]


def _answer(turn: dict) -> str:
    lines = [f"{turn['docs'][i][1]} {form.format(name=turn['docs'][i][0])}" for i, form in turn["cites"]]
    return f"## {turn['title']}\n\n" + "\n".join(lines)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def real_agent(monkeypatch):
    """The real agent; only the model and the retrieval store are stubbed,
    each answering by the turn's own question."""
    def _turn_of(text: str) -> dict | None:
        return next((t for t in TURNS if t["ask"] in text), None)

    def fake_retrieve(query, project_id, k=5, **_kw):
        turn = _turn_of(query)
        docs = turn["docs"] if turn else []
        return [Chunk(chunk_id=f"{project_id}:d{i}:0", project_id=project_id, doc_id=f"d{i}",
                      chunk_index=0, text=text, score=0.9 - i * 0.1, source_name=name)
                for i, (name, text) in enumerate(docs)][:k], 0

    async def fake_llm(self, messages, api_key, project_id=None, **_kw):
        asks = " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        turn = _turn_of(asks)
        await asyncio.sleep(0)
        return {"status": "success",
                "choice": {"message": {"content": _answer(turn) if turn else "لا إجابة.", "tool_calls": []}},
                "raw": {}}

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setenv("CHAT_SSE_WATCHDOG", "1")
    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    monkeypatch.setattr("app.routers.chat._predefined_enabled", lambda: False)


def _user_project(client, tag: str) -> tuple[dict, str]:
    email = f"arcite-{tag}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    res = client.post("/v1/projects", json={"name": f"ArCite-{tag}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return headers, res.json()["id"]


def _ask(client, headers: dict, pid: str, ask: str) -> tuple[str, dict]:
    cid = f"ws-{pid}-{int(time.time() * 1000)}"
    body = {"message": ask, "project_id": pid, "conversation_id": cid, "agent": "project-assistant"}
    with client.stream("POST", "/v1/chat/stream", json=body, headers=headers) as stream:
        assert stream.status_code == 200
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    end = [e for e in events if e.get("type") == "end"]
    assert end, events
    return cid, end[-1]


@pytest.mark.parametrize("case", range(len(TURNS)))
def test_an_arabic_answer_is_stored_with_its_citations_and_exported_from_them(client, real_agent, case):
    from docx import Document

    from app.core import agent_memory

    turn = TURNS[case]
    headers, pid = _user_project(client, str(case))
    cid, end = _ask(client, headers, pid, turn["ask"])
    panel = [row["doc_name"] for row in end.get("sources") or []]
    assert panel, end

    # Stored on the assistant message: the cited documents first, in the
    # order the answer cited them, each with the locator it was written
    # with; then the turn's uncited sources, unnumbered.
    stored = [m for m in agent_memory.get_messages(cid) if m.get("role") == "assistant"][-1]
    cites = stored.get("citations") or []
    cited = [turn["docs"][i][0] for i, _form in turn["cites"]]
    numbered = [c for c in cites if c.get("marker") is not None]
    assert [(c["document"], c["section"]) for c in numbered] == list(zip(cited, turn["sections"])), cites
    assert [c["marker"] for c in numbered] == list(range(1, len(cited) + 1)), cites
    rest = [c["document"] for c in cites if c.get("marker") is None]
    assert rest == [n for n in panel if n not in cited], (rest, panel)

    # The produced document: its Sources are those entries; its body keeps
    # every sentence and no citation text, label or marker.
    path = app.url_path_for(exports.export_conversation_message.__name__,
                            project_id=pid, conversation_id=cid)
    got = client.post(f"{path}?format=docx&scope=document", headers=headers)
    assert got.status_code == 200, got.text
    texts = [p.text.strip() for p in Document(io.BytesIO(got.content)).paragraphs if p.text.strip()]
    tail = texts[len(texts) - texts[::-1].index("Sources"):]
    assert tail == [f"{c['document']}, {c['section']}" if c["section"] else c["document"] for c in cites], tail
    body = "\n".join(texts[: len(texts) - len(tail) - 1])
    for i, _form in turn["cites"]:
        assert turn["docs"][i][1] in body, body
    for residue in ("[1]", "[2]", "source", "المصدر", "المرجع", *cited):
        assert residue not in body, (residue, body)
