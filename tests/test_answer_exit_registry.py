"""Nothing internal leaves in an answer.

One exit check (``app.agents.answer_exit``) runs on every answer a user
reads: streamed, returned, stored and reopened. The rules are tested over
generated inputs, not a list of known strings, so a document, formula, tool
or block added later is covered without editing this file:

* the retrieval marker the injector actually writes carries only keys the
  check knows;
* every citation form (the marker, a bracket closed or not, a ``Source:``
  line, an inline chunk or document id) over generated documents, ids and
  pages resolves to the document's name and page, or is removed; no chunk
  number, document id or marker key is left;
* a citation nothing backs is removed, never guessed;
* a Sources row shows the page or nothing, and has a document name;
* every registered formula and tool id reads as its display name; a block
  id never shows; an unregistered code name reads as words;
* a ``key=literal`` dump goes;
* a tool's error text copied into the answer goes; what is left is an ask
  for the input it named, never a sentence about the step failing;
* refusal wording that names the platform's machinery reads in the user's
  terms;
* text with nothing internal in it is left exactly as written;
* the check is idempotent.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import random
import re
import uuid

import pytest

from app.lib import formula_registry
from tests._fixture_values import STUB_TOKEN_02

try:
    answer_exit = importlib.import_module("app.agents.answer_exit")
except ImportError:  # the check does not exist yet: every case fails, not collection
    answer_exit = None

_RNG_SEED = 20261009
_WORDS = ("Contract", "Data", "Particular", "Conditions", "Specification", "Section", "HSE",
          "Plan", "Method", "Statement", "Volume", "Drainage", "Pile", "Load", "Test", "Report",
          "Tender", "Addendum", "Minutes", "Meeting", "Site", "Survey", "NOC", "Lighting")
_EXTS = (".pdf", ".docx", ".pptx", ".xlsx", ".txt")
_SEPS = (" ", "_", "-")
_FACT = "The stated value is 42 days."


def _ax():
    if answer_exit is None:
        pytest.fail("no exit check: app.agents.answer_exit is missing")
    return answer_exit


@pytest.fixture(autouse=True)
def _no_document_store(monkeypatch):
    """Ids the turn did not read resolve through the document store; here it
    knows nothing, so an unbacked id must be removed, not looked up."""
    from app.core import projects

    monkeypatch.setattr(projects, "get_document", lambda _doc_id: None)


# ── generated corpora ───────────────────────────────────────────────────────

def _documents(seed: int, n: int = 5) -> list[dict]:
    rng = random.Random(seed)
    docs, chunks = [], rng.sample(range(900), n)
    for i in range(n):
        sep = rng.choice(_SEPS)
        words = rng.sample(_WORDS, rng.randint(2, 4))
        if rng.random() < 0.4:
            words.insert(0, f"{rng.choice('ABCDEFGHJK')}{rng.choice('LMNPQRSTUV')}-{rng.randint(100, 999)}")
        name = sep.join(words) + rng.choice(_EXTS)
        doc_id = uuid.UUID(int=rng.getrandbits(128)).hex
        if rng.random() < 0.3:
            doc_id = str(uuid.UUID(doc_id))
        page = rng.choice([None, rng.randint(1, 300)])
        docs.append({"name": name, "doc_id": doc_id, "chunk": chunks[i], "page": page})
    return docs


def _chunk_objects(docs: list[dict]):
    from app.core.rag.vector_store import Chunk

    out = []
    for d in docs:
        c = Chunk(chunk_id=f"p:{d['doc_id']}:{d['chunk']}", project_id="p", doc_id=d["doc_id"],
                  chunk_index=d["chunk"], text=_FACT, score=0.8)
        c.page = d["page"]
        c.source_name = d["name"]
        c.revision = "B"
        c.superseded = False
        out.append(c)
    return out


def _injected(docs: list[dict]) -> dict:
    from app.core.rag.inject import format_chunks_as_system_message

    return format_chunks_as_system_message(_chunk_objects(docs), total_candidates=len(docs), query="q")


def _turn(docs: list[dict]):
    ax = _ax()
    turn = ax.Turn()
    ax.note_evidence(_injected(docs), [], turn)
    return turn


def _marker_of(doc: dict) -> str:
    content = _injected([doc])["content"]
    m = re.search(r"\[doc_id=[^\]]*\]", content)
    assert m, "the injector wrote no marker"
    return m.group(0)


def _place(doc: dict) -> str:
    return _ax().render_place(doc["name"], [doc["page"]] if doc["page"] else [])


_INTERNAL_RE = re.compile(
    r"\bchunks?\b[\s#=:]*\d|\bdoc(?:ument)?[ _-]?id\b|\b(?:score|src|class|layer|rev)\s*=|\x00|\x01|\x02",
    re.IGNORECASE,
)


def _assert_nothing_internal(out: str, docs: list[dict]) -> None:
    assert not _INTERNAL_RE.search(out), out
    for d in docs:
        assert d["doc_id"].lower() not in out.lower(), out
        assert d["doc_id"][:8].lower() not in out.lower(), out


def _assert_idempotent(out: str, **kwargs) -> None:
    assert _ax().check_text(out, **kwargs) == out


# ── the marker the injector writes ──────────────────────────────────────────

def test_marker_keys_are_all_known_to_the_exit_check():
    keys = set(getattr(answer_exit, "MARKER_KEYS", ()) or ())
    doc = {"name": "Any Document.pdf", "doc_id": uuid.uuid4().hex, "chunk": 3, "page": 4}
    chunk = _chunk_objects([doc])[0]
    chunk.superseded = True
    from app.core.rag.inject import format_chunks_as_system_message

    content = format_chunks_as_system_message([chunk], 1, "q")["content"]
    marker = re.search(r"\[doc_id=[^\]]*\]", content).group(0)
    written = set(re.findall(r"\b([a-z_]+)=", marker))
    assert written and written <= keys, f"marker keys the exit check does not know: {written - keys}"


# ── every citation form resolves to name and page, or goes ──────────────────

def _forms():
    """(form id, renderer(doc) -> citation text, resolvable)."""
    return [
        ("marker", lambda d: _marker_of(d), True),
        ("square-source", lambda d: f"[source: {d['name']}, chunk {d['chunk']}]", True),
        ("round-source", lambda d: f"(source: {d['name']}, chunk {d['chunk']})", True),
        ("lenticular", lambda d: f"【{d['name']}, chunk {d['chunk']}】", True),
        ("square-docid", lambda d: f"[doc_id={d['doc_id']} chunk={d['chunk']}]", True),
        ("round-docid-prefix", lambda d: f"(doc_id={d['doc_id'][:8]}, chunk {d['chunk']})", True),
        ("unclosed-docid", lambda d: f"(doc_id={d['doc_id'][:8]}", True),
        ("unclosed-chunk", lambda d: f"[chunk {d['chunk']}", True),
        ("hash-chunk", lambda d: f"(chunk #{d['chunk']})", True),
        ("inline-cue-chunk", lambda d: f"per chunk {d['chunk']}", True),
        ("inline-name-chunk", lambda d: f"in {d['name']}, chunk {d['chunk']}", True),
        ("inline-docid", lambda d: f"see doc_id {d['doc_id']}", True),
    ]


_SEEDS = range(_RNG_SEED, _RNG_SEED + 4)
_CASES = [(seed, fid) for seed in _SEEDS for fid, _r, _ok in _forms()]


@pytest.mark.parametrize("seed,form", _CASES, ids=[f"{s}-{f}" for s, f in _CASES])
def test_inline_citation_resolves_to_name_and_page(seed, form):
    ax = _ax()
    docs = _documents(seed)
    turn = _turn(docs)
    render = {f: r for f, r, _ok in _forms()}[form]
    for doc in docs:
        text = f"{_FACT[:-1]} {render(doc)}. Nothing else."
        out = ax.check_text(text, turn=turn)
        _assert_nothing_internal(out, docs)
        assert doc["name"] in out, (text, out)
        if doc["page"]:
            assert _place(doc) in out, (text, out)
        assert out.startswith("The stated value is 42 days") and out.endswith("Nothing else."), out
        _assert_idempotent(out, turn=turn)


_LINE_FORMS = [
    ("name-chunk", lambda d: f"Source: {d['name']}, chunk {d['chunk']}."),
    ("name-chunks-plural", lambda d: f"Sources: {d['name']}, chunks {d['chunk']}."),
    ("name-docid-unclosed", lambda d: f"Source: {d['name']} (doc_id={d['doc_id'][:8]}."),
    ("bold-name-chunk", lambda d: f"**Source:** {d['name']}, chunk {d['chunk']}"),
    ("bullet-marker", lambda d: f"- Source: {_marker_of(d)}"),
]
_LINE_CASES = [(seed, fid) for seed in _SEEDS for fid, _r in _LINE_FORMS]


@pytest.mark.parametrize("seed,form", _LINE_CASES, ids=[f"{s}-{f}" for s, f in _LINE_CASES])
def test_source_line_resolves_to_name_and_page(seed, form):
    ax = _ax()
    docs = _documents(seed)
    turn = _turn(docs)
    render = dict(_LINE_FORMS)[form]
    for doc in docs:
        text = f"{_FACT}\n\n{render(doc)}"
        out = ax.check_text(text, turn=turn)
        _assert_nothing_internal(out, docs)
        assert _place(doc) in out, (text, out)
        assert re.search(r"(?m)^\W*Sources?\W*:", out), out
        _assert_idempotent(out, turn=turn)


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_citation_nothing_backs_is_removed(seed):
    """A chunk number or id the turn never read is removed, never guessed;
    the claim it was attached to stays."""
    ax = _ax()
    docs = _documents(seed)
    turn = _turn(docs)
    unread = max(d["chunk"] for d in docs) + 1
    stranger = uuid.UUID(int=random.Random(seed + 99).getrandbits(128)).hex
    for cite in (f"(see chunk {unread})", f"[chunk {unread}]", f"[doc_id={stranger} chunk={unread}]",
                 f"(doc_id={stranger[:8]}", f"per chunk {unread}", stranger):
        text = f"{_FACT[:-1]} {cite}."
        out = ax.check_text(text, turn=turn)
        _assert_nothing_internal(out, docs)
        assert stranger[:8] not in out and str(unread) not in out, (text, out)
        assert out.startswith("The stated value is 42 days"), out
        _assert_idempotent(out, turn=turn)


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_chunk_number_shared_by_two_documents_is_not_guessed(seed):
    ax = _ax()
    docs = _documents(seed, n=2)
    docs[1]["chunk"] = docs[0]["chunk"]
    turn = _turn(docs)
    out = ax.check_text(f"{_FACT[:-1]} (chunk {docs[0]['chunk']}).", turn=turn)
    _assert_nothing_internal(out, docs)
    assert docs[0]["name"] not in out and docs[1]["name"] not in out, out


# ── Sources rows ────────────────────────────────────────────────────────────

_SECTION_FORMS = ("chunk #{c}", "chunk {c}", "#{c}", "", "{p}")


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_sources_rows_show_name_and_page_or_nothing(seed):
    ax = _ax()
    docs = _documents(seed)
    turn = _turn(docs)
    rows = []
    for d in docs:
        for form in _SECTION_FORMS:
            rows.append({"doc_id": d["doc_id"], "doc_name": d["name"], "chunk_index": d["chunk"],
                         "page": d["page"], "page_or_section": form.format(c=d["chunk"], p=d["page"] or "")})
        rows.append({"doc_id": d["doc_id"], "doc_name": "", "chunk_index": d["chunk"], "page": d["page"],
                     "page_or_section": f"chunk #{d['chunk']}"})
        rows.append({"doc_id": d["doc_id"], "doc_name": d["doc_id"], "chunk_index": d["chunk"],
                     "page": d["page"], "page_or_section": ""})
    rows.append({"doc_id": uuid.uuid4().hex, "doc_name": "", "chunk_index": 1, "page_or_section": "chunk #1"})
    out = ax.check_sources(rows, turn=turn)
    assert out, "every row was dropped"
    names = {d["name"] for d in docs}
    for row in out:
        assert row["doc_name"] in names, row
        section = row["page_or_section"]
        assert not re.search(r"chunk|#", section, re.IGNORECASE), row
        page = next(d["page"] for d in docs if d["name"] == row["doc_name"])
        assert section in ("", f"p. {page}"), row
    assert len(out) == len(rows) - 1


@pytest.mark.parametrize("chunk_index", [0, 1, 7, 65, 941])
def test_built_sources_label_is_never_a_chunk_number(chunk_index):
    from app.agents.runtime import page_or_section_label

    assert page_or_section_label({"page": None, "chunk_index": chunk_index}) == ""
    assert page_or_section_label({"page": chunk_index + 1, "chunk_index": chunk_index}) == f"p. {chunk_index + 1}"


# ── registered ids, block ids and code names ────────────────────────────────

def _formula_ids():
    return [(s.name, s.display_name) for s in formula_registry.all_specs() if s.display_name and "_" in s.name]


def _tool_ids():
    from app.agents.core import tool_registry

    tool_registry.load()
    return [(n, s.display_name) for n, s in sorted(tool_registry._TOOLS.items()) if s.display_name and "_" in n]


def _block_ids():
    from app.blocks import BLOCK_REGISTRY

    return sorted(n for n in BLOCK_REGISTRY if "_" in n)


def _param_ids():
    seen = {}
    for spec in formula_registry.all_specs():
        for name in formula_registry.parameters(spec):
            if "_" in name and re.search(r"[a-z]{3}", name):
                seen.setdefault(name, spec.name)
    return sorted(seen)


@pytest.mark.parametrize("ident,display", _formula_ids() + _tool_ids())
def test_registered_id_reads_as_its_display_name(ident, display):
    ax = _ax()
    out = ax.check_text(f"I used {ident} for this. The result follows.")
    assert ident not in out and display in out, out
    _assert_idempotent(out)


@pytest.mark.parametrize("ident", _block_ids())
def test_block_id_never_shows(ident):
    ax = _ax()
    for text in (f"I called {ident} first.", f"The `{ident}` step ran.", f"Routed via {ident}.summary next."):
        out = ax.check_text(text)
        assert ident not in out and "_" not in out, (text, out)
        _assert_idempotent(out)


@pytest.mark.parametrize("key", _param_ids())
def test_parameter_assignment_reads_as_words(key):
    ax = _ax()
    out = ax.check_text(f"It ran with {key}=12.5 as given.")
    assert key not in out and re.search(r"\bwith \D*\d", out), out
    _assert_idempotent(out)


_LITERALS = ("{}", "[]", "true", "false", "null", '{"a": 1}', '["x", "y"]')
_DUMP_KEYS = _block_ids()[:6] + _param_ids()[:6] + [f"{b}.summary" for b in _block_ids()[:3]]
_DUMP_CASES = [(k, lit, sep) for k in _DUMP_KEYS for lit in _LITERALS for sep in ("=", ": ", " ")
               if not (sep == " " and lit[0].isalpha())]


@pytest.mark.parametrize("key,literal,sep", _DUMP_CASES)
def test_key_value_dump_is_removed(key, literal, sep):
    ax = _ax()
    out = ax.check_text(f"The total is 40 m3. {key}{sep}{literal}. Check the levels.")
    assert key not in out and literal not in out, out
    assert "The total is 40 m3." in out and "Check the levels." in out, out
    _assert_idempotent(out)


# ── raw tool errors ─────────────────────────────────────────────────────────

def _error_texts(param: str):
    words = param.replace("_", " ")
    return (
        f"Missing required parameter: {param}",
        f"{param} must be greater than 0, got -1.",
        f"No {words} supplied. Pass {param}=[{{...}}] or project_type inferred.",
        f"KeyError: '{param}'",
    )


_ERROR_CASES = [(tool, display, param, i) for tool, display in _tool_ids()[:4]
                for param in _param_ids()[:3] for i in range(4)]


@pytest.mark.parametrize("tool,display,param,variant", _ERROR_CASES)
def test_copied_tool_error_becomes_one_plain_sentence(tool, display, param, variant):
    ax = _ax()
    err = _error_texts(param)[variant]
    turn = ax.Turn()
    ax.note_tool_result(tool, {"status": "error", "error": err}, turn)
    out = ax.check_text(f"I tried the step. {err} Let me know the value.", turn=turn)
    assert param not in out and "{" not in out and "Error:" not in out, out
    assert "did not return" not in out and out.endswith("Let me know the value."), out
    assert f"{display} needs the " in out, out
    _assert_idempotent(out, turn=turn)


@pytest.mark.parametrize("head", ["Error: ", "Exception: ", "Traceback (most recent call last): ",
                                  "ValueError: ", "status: error "])
def test_structural_error_line_is_replaced(head):
    ax = _ax()
    out = ax.check_text(f"Here is what happened.\n{head}something broke at line 3")
    assert head.strip() not in out and "broke" not in out, out
    assert out == "Here is what happened.", out
    _assert_idempotent(out)


def test_block_error_is_recorded_wherever_the_block_runs():
    """A block returning an error records it for the turn, so a copy in the
    answer is recognised whichever path ran the block."""
    ax = _ax()
    from app.core.universal_base import UniversalBlock

    err = f"No inputs supplied. Pass {_param_ids()[0]}=[{{...}}]"

    class _Failing(UniversalBlock):
        name = "failing_probe_block"

        async def process(self, input_data, params=None):
            return {"status": "error", "error": err}

    turn, token = ax.begin_turn()
    try:
        asyncio.run(_Failing().execute({}, {"action": "any_action"}))
    finally:
        ax.reset_turn(token)
    assert any(e == err for _label, e in turn.tool_errors), turn.tool_errors
    out = ax.check_text(f"It said: {err}", turn=turn)
    assert "I need the " in out and "{" not in out and "did not return" not in out, out


# ── machinery wording ───────────────────────────────────────────────────────

_DETS = ("The", "No", "These", "Any")
_ADJS = ("retrieved", "provided", "supplied", "injected", "indexed")
_NOUNS = ("excerpts", "chunks", "snippets", "passages", "context")
_MACHINE_CASES = [(d, a, n) for d in _DETS for a in _ADJS for n in _NOUNS]


@pytest.mark.parametrize("det,adj,noun", _MACHINE_CASES)
def test_machinery_refusal_reads_in_user_terms(det, adj, noun):
    ax = _ax()
    out = ax.check_text(f"{det} {adj} {noun} do not state the bond amount.")
    assert not re.search(rf"\b(?:{adj}|{noun})\b", out, re.IGNORECASE), out
    assert out.endswith("do not state the bond amount.") and out[:1].isupper(), out
    _assert_idempotent(out)


@pytest.mark.parametrize("phrase", ["tool call", "tool calls", "tool results", "function output",
                                    "search results", "retrieval results", "supplied parameters"])
def test_machinery_terms_read_in_user_terms(phrase):
    ax = _ax()
    out = ax.check_text(f"The {phrase} did not include a rate.")
    assert phrase not in out.lower(), out
    _assert_idempotent(out)


# ── text with nothing internal is untouched ─────────────────────────────────

def _preserved_texts():
    texts = [f"Source: {s.display_name} (platform formula)." for s in formula_registry.all_specs()
             if s.display_name]
    texts += [
        "Volume = 30 × 20 × 1.5 = 900 m³; with 5% waste, 945 m³.",
        "Design strength f_ck = 30 MPa and f_yk = 500 MPa.",
        "See https://example.org/a_b/c_d?x_y=1 or mail site_team@example.org.",
        "The concrete chunks were removed; the chunks of slab went to tip.",
        "Class A finish, Rev 3 drawings, layer of blinding 50 mm.",
        "Open Drainage_Layout_Rev_B.pdf and Method Statement.docx.",
        "```\nrun(length_m=10)\n```",
        "The context of the claim is the late access.",
        "Section 3.2.1 and clause 14.7 apply; see theshovel.ai for help.",
        "",
        "   ",
    ]
    return texts


@pytest.mark.parametrize("text", _preserved_texts())
def test_text_with_nothing_internal_is_unchanged(text):
    assert _ax().check_text(text) == text


@pytest.mark.parametrize("seed", list(_SEEDS))
def test_document_names_survive_in_prose(seed):
    ax = _ax()
    docs = _documents(seed)
    turn = _turn(docs)
    for d in docs:
        text = f"Refer to {d['name']} for the method."
        assert ax.check_text(text, turn=turn) == text


# ── the exits are wired ─────────────────────────────────────────────────────

def _llm_agent(monkeypatch, reply: str, docs: list[dict]):
    from app.agents.runtime import Agent

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    agent = Agent(name="exit-check-agent", description="t", system_prompt="t", allowed_blocks=[])

    async def fake_call_llm(messages, api_key, **kwargs):
        return {"status": "success",
                "choice": {"message": {"content": reply, "tool_calls": []}, "finish_reason": "stop"}}

    def fake_rag_inject(**kwargs):
        return _injected(docs), {"project_id": "proj_x", "chunks": [
            {"doc_id": d["doc_id"], "chunk_index": d["chunk"], "chunk_id": f"x:{d['doc_id']}:{d['chunk']}",
             "score": 0.8, "page": d["page"]} for d in docs]}

    by_id = {d["doc_id"]: {"original_name": d["name"]} for d in docs}
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.core.projects.get_document", lambda did: by_id.get(did))
    return agent


def _raw_reply(doc: dict) -> str:
    # A whole echoed marker (doc_id, chunk and score) trips the runtime's
    # context-leak guard, which replaces the answer outright; these are the
    # citation shapes that reach the exit.
    return (f"The stated value is 42 days [doc_id={doc['doc_id']} chunk={doc['chunk']}]. "
            f"I used smart_orchestrator; delta_ok=false.\n\nSource: {doc['name']}, chunk {doc['chunk']}.")


@pytest.mark.asyncio
async def test_agent_chat_answer_and_sources_pass_the_exit(monkeypatch):
    _ax()
    docs = _documents(_RNG_SEED, n=1)
    agent = _llm_agent(monkeypatch, _raw_reply(docs[0]), docs)
    result = await agent.chat("What is the value?", project_id="proj_x")
    _assert_nothing_internal(result["answer"], docs)
    assert "smart_orchestrator" not in result["answer"]
    assert not re.search(r"delta[ _]ok", result["answer"]), "a dump read as words is still a dump"
    assert docs[0]["name"] in result["answer"]
    for row in result.get("sources") or []:
        assert row.get("doc_name") and not re.search(r"chunk|#", row.get("page_or_section") or "")


@pytest.mark.asyncio
async def test_agent_stream_end_passes_the_exit(monkeypatch):
    _ax()
    docs = _documents(_RNG_SEED + 1, n=1)
    agent = _llm_agent(monkeypatch, _raw_reply(docs[0]), docs)
    events = [e async for e in agent.chat_stream("What is the value?", project_id="proj_x")]
    end = [e for e in events if e.get("type") == "end"][-1]
    content = end.get("content") or ""
    assert content.strip(), "the end event carries no checked answer for the client to show"
    _assert_nothing_internal(content, docs)
    assert "smart_orchestrator" not in content and docs[0]["name"] in content
    assert not re.search(r"delta[ _]ok", content), "a dump read as words is still a dump"
    for row in end.get("sources") or []:
        assert row.get("doc_name") and not re.search(r"chunk|#", row.get("page_or_section") or "")


def test_router_stream_end_passes_the_exit():
    """Every router stream is wrapped, including paths that build their own
    ``end`` with no content."""
    _ax()
    exit_frames = importlib.import_module("app.routers.exit_frames")
    docs = _documents(_RNG_SEED + 2, n=1)
    d = docs[0]

    async def frames():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        for word in _raw_reply(d).split(" "):
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'sources': [{'doc_id': d['doc_id'], 'doc_name': d['name'], 'chunk_index': d['chunk'], 'page': d['page'], 'page_or_section': 'chunk #' + str(d['chunk'])}]})}\n\n"

    async def run():
        return [f async for f in exit_frames.with_answer_exit(frames())]

    out = asyncio.run(run())
    end = json.loads(out[-1][len("data:"):])
    _assert_nothing_internal(end["content"], docs)
    assert d["name"] in end["content"]
    assert end["sources"][0]["page_or_section"] in ("", f"p. {d['page']}")


def test_stored_and_reopened_answers_pass_the_exit(monkeypatch):
    _ax()
    from app.core import agent_memory

    docs = _documents(_RNG_SEED + 3, n=1)
    cid = f"exit-check-{uuid.uuid4().hex[:8]}"
    agent_memory.get_or_create_conversation(cid, "project-assistant", "proj_x")
    agent_memory.append_message(cid, "user", "What is the value?")
    stored = agent_memory.append_message(cid, "assistant", _raw_reply(docs[0]))
    _assert_nothing_internal(stored["content"], docs)
    assert "smart_orchestrator" not in stored["content"]

    from app.routers import agents as agents_router

    raw = [{"role": "assistant", "content": _raw_reply(docs[0])}, {"role": "user", "content": "delta_ok=false"}]
    monkeypatch.setattr(agent_memory, "get_messages", lambda _cid, limit=40: [dict(m) for m in raw])
    reopened = agents_router._reopened_messages(cid)
    _assert_nothing_internal(reopened[0]["content"], docs)
    assert reopened[1]["content"] == "delta_ok=false", "a user's own words are shown as typed"


def test_answer_of_nothing_but_internals_is_never_an_empty_bubble():
    ax = _ax()
    out = ax.check_end_event({"type": "end"}, streamed="delta_ok=false (see chunk 4)", turn=ax.Turn())
    assert out["content"].strip() and "delta_ok" not in out["content"] and "chunk" not in out["content"]


# ── the queued evidence, as instances of the rules above ────────────────────

def test_queued_evidence():
    ax = _ax()
    doc = {"name": "fui_upload_note.pptx", "doc_id": "5c1ecd75" + uuid.uuid4().hex[8:], "chunk": 0, "page": None}
    turn = _turn([doc])
    risk_error = "No risks supplied. Pass risks=[{...}] or project_type inferred."
    ax.note_tool_result("construction", {"action": "risk_register", "status": "error", "error": risk_error}, turn)
    cases = {
        "Source: fui_upload_note.pptx, chunk 0.": "Source: fui_upload_note.pptx.",
        "Source: fui_upload_note.pptx (doc_id=5c1ecd75.": "Source: fui_upload_note.pptx.",
    }
    for text, expected in cases.items():
        assert ax.check_text(text, turn=turn) == expected
    prose = ("Checked delta_ok=false and learning_engine.summary {} first. "
             "I routed it through smart_orchestrator and boq_processor. " + risk_error)
    out = ax.check_text(prose, turn=turn)
    for raw in ("delta_ok", "learning_engine", "{}", "smart_orchestrator", "boq_processor",
                "project_type inferred", "No risks supplied"):
        assert raw not in out, (raw, out)
    assert "did not return" not in out and "platform step" not in out, out
    rows = ax.check_sources([{"doc_id": doc["doc_id"], "doc_name": doc["name"], "chunk_index": 0,
                              "page_or_section": "chunk #0"}], turn=turn)
    assert rows == [{"doc_id": doc["doc_id"], "doc_name": doc["name"], "chunk_index": 0, "page_or_section": ""}]


# ── Source labels: display units, one line per credit ───────────────────────

def _exponent_units():
    from app.agents.base.formulas.construction_formulas_planning import _FAMILIES

    known = {u for table in _FAMILIES.values() for u in table}
    declared = {p.unit for s in formula_registry.all_specs() for p in formula_registry.parameters(s).values()}
    return sorted(u for u in known | declared if u and re.search(r"[A-Za-z][234](?![0-9A-Za-z])", u)
                  and not re.search(r"co2", u, re.IGNORECASE))


_RAW_EXPONENT_RE = re.compile(r"(?<=[A-Za-z])(?<![Cc][Oo])[234](?![0-9A-Za-z])")


@pytest.mark.parametrize("unit", _exponent_units())
def test_unit_codes_in_a_calculator_label_show_as_display_units(unit):
    from app.lib.source_labels import calculator_label

    label = calculator_label("pe_unit_convert", {"value": 250, "from_unit": unit, "to_unit": unit})
    assert not _RAW_EXPONENT_RE.search(label), label
    assert re.search("[²³⁴]", label), label


def _formulas_with_exponent_inputs():
    out = []
    for spec in formula_registry.all_specs():
        params = formula_registry.parameters(spec)
        hit = next((n for n, p in params.items() if p.unit and _RAW_EXPONENT_RE.search(p.unit)
                    and p.kind in formula_registry.NUMERIC_KINDS and p.kind in ("number", "integer")), None)
        if hit and spec.display_name:
            out.append((spec.name, hit))
    return out


@pytest.mark.parametrize("formula,param", _formulas_with_exponent_inputs())
def test_declared_units_in_a_calculator_label_show_as_display_units(formula, param):
    from app.lib.source_labels import calculator_label

    label = calculator_label(formula, {param: 12})
    assert not _RAW_EXPONENT_RE.search(label.split("(", 1)[-1]), label


_JOINS = (" → ", " -> ", " > ", " / ", ", ", " via ")
_CREDIT_CASES = [(s.name, j) for s in formula_registry.all_specs() if s.display_name for j in _JOINS]


@pytest.mark.parametrize("formula,join", _CREDIT_CASES[::7] + _CREDIT_CASES[-len(_JOINS):])
def test_one_source_line_per_calculator_credit(formula, join):
    ax = _ax()
    from app.lib.source_labels import (
        calculator_label,
        formula_display_name,
        tool_display_name,
    )

    credit = f"Source: {calculator_label(formula, {})}."
    platform = tool_display_name("construction_calc")
    name = formula_display_name(formula)
    for extra in (f"Source: {platform}{join}{name}", f"Source: {name}{join}{platform}",
                  f"Source: {platform}", credit):
        for lines in ((extra, credit), (credit, extra)):
            text = "The result is 42.\n\n" + "\n".join(lines)
            out = ax.check_text(text)
            assert out.count("Source:") == 1 and credit.rstrip(".") in out, (text, out)
            _assert_idempotent(out)


def test_a_document_source_beside_a_calculator_credit_is_kept():
    ax = _ax()
    from app.lib.source_labels import calculator_label

    credit = f"Source: {calculator_label('concrete_volume', {})}."
    text = f"The slab is 900 m³.\n\nSource: Structural Notes.pdf, p. 4.\n{credit}"
    assert ax.check_text(text) == text
