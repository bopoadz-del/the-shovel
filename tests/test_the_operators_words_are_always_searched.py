"""Document search runs the model's query once.

One query string, one ``retrieve_with_filter`` call, no second result set.
"""
from __future__ import annotations

import inspect

import pytest

from app.core import doc_index


class _Chunk:
    def __init__(self, chunk_id, doc_id, text, score):
        self.chunk_id = chunk_id
        self.project_id = "p1"
        self.doc_id = doc_id
        self.chunk_index = 0
        self.text = text
        self.score = score
        self.layer = "own"


CLAUSE = ("compaction of the backfill to minimum 98 percent of maximum dry "
          "density of the modified proctor test at near optimum moisture")

# What the model's own phrasing finds: confidently wrong, never empty.
MODEL_HITS = [
    _Chunk("c1", "spec2", "cable duct trench soft ground sand cover after tamping", 0.88),
    _Chunk("c2", "spec3", "embankment compaction testing to the highway standard", 0.81),
]
# What a second, operator-worded query used to find. It must not be merged in.
VERBATIM_HITS = [
    _Chunk("c9", "spec4", CLAUSE, 0.77),
    _Chunk("c1", "spec2", "cable duct trench soft ground sand cover after tamping", 0.70),
]

OPERATOR_ASK = ("Per the project specification, to what degree must structural "
                "backfill under foundations be compacted, and by which test?")
MODEL_QUERY = "backfill compaction requirements"


@pytest.fixture
def two_sided_corpus(monkeypatch):
    """retrieve_with_filter answers differently for each query string."""
    seen = []

    def fake_retrieve(query, project_id, k=5, **_kw):
        seen.append(query)
        if query == OPERATOR_ASK:
            return list(VERBATIM_HITS), 0
        return list(MODEL_HITS), 0

    import app.core.rag.retriever as retr
    monkeypatch.setattr(retr, "retrieve_with_filter", fake_retrieve)
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    return seen


def _text(results):
    return " ".join(r["snippet"] for r in results)


async def test_verbatim_co_search_flag_on_issues_exactly_one_query(two_sided_corpus, monkeypatch):
    """With the verbatim co-search flag on, the tool still issues one query.

    ``SEARCH_ALSO_VERBATIM=1`` must not run a second retrieval and must not
    merge a second result set into the model's hits.
    """
    monkeypatch.setenv("SEARCH_ALSO_VERBATIM", "1")
    results = await doc_index.search_project_documents(
        "p1", MODEL_QUERY, top_k=5)
    assert two_sided_corpus == [MODEL_QUERY]
    assert "modified proctor" not in _text(results).lower()


async def test_one_model_query_is_the_only_search(two_sided_corpus):
    await doc_index.search_project_documents("p1", MODEL_QUERY, top_k=5)
    assert two_sided_corpus == [MODEL_QUERY]


async def test_both_retrievals_stay_inside_the_one_thread_hop(two_sided_corpus, monkeypatch):
    """The one retrieval runs inside the existing to_thread call.

    A second hop would put retrieval back on the event loop. One hop, one
    query, and that query is not on the loop thread.
    """
    import asyncio
    import threading

    hops = []
    retrieve_threads = []
    loop_thread = threading.get_ident()
    real_to_thread = asyncio.to_thread

    async def counting(fn, /, *args, **kwargs):
        hops.append(threading.get_ident())
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(doc_index.asyncio, "to_thread", counting)

    import app.core.rag.retriever as retr
    current = retr.retrieve_with_filter

    def marking(query, project_id, k=5, **kw):
        retrieve_threads.append(threading.get_ident())
        return current(query, project_id, k=k, **kw)

    monkeypatch.setattr(retr, "retrieve_with_filter", marking)

    await doc_index.search_project_documents("p1", MODEL_QUERY, top_k=5)
    assert hops == [loop_thread]
    assert retrieve_threads == [retrieve_threads[0]]
    assert len(retrieve_threads) == 1
    assert retrieve_threads[0] != loop_thread


def _split_name(*parts: str) -> str:
    """Build a forbidden identifier without embedding it in this source."""
    return "".join(parts)


def test_search_signature_schema_and_sources_omit_co_search():
    """The tool signature, the schema shown to the model, and non-test sources
    do not carry the removed parameter or the removed flag."""
    import json
    import subprocess
    from pathlib import Path

    param = _split_name("also_", "query")
    flag = _split_name("SEARCH_", "ALSO_", "VERBATIM")
    helper = _split_name("_same_", "search")
    reader = _split_name("_also_", "verbatim_", "enabled")

    assert param not in inspect.signature(doc_index.search_project_documents).parameters
    assert param not in inspect.signature(
        doc_index._search_project_documents_sync).parameters

    from app.agents.runtime import Agent
    agent = Agent(
        name="co-search-excision",
        description="schema check",
        system_prompt="schema check",
        allowed_blocks=[],
        can_delegate=False,
    )
    tools = agent.tool_definitions(project_id="p")
    search = next(
        t for t in tools if t["function"]["name"] == "search_project_documents")
    props = search["function"]["parameters"]["properties"]
    assert param not in props
    assert flag not in json.dumps(search)

    root = Path(__file__).resolve().parents[1]
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    hits = []
    for raw in listed.split(b"\0"):
        if not raw or raw.startswith(b"tests/"):
            continue
        path = root / raw.decode()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in (param, flag, helper, reader):
            if needle in text:
                hits.append(f"{raw.decode()}: {needle}")
    assert hits == []


def test_the_single_query_regression_does_not_pass_a_second_argument():
    param = _split_name("also_", "query")
    src = inspect.getsource(test_verbatim_co_search_flag_on_issues_exactly_one_query)
    assert f"{param}=" not in src
    assert f'["{param}"]' not in src
    assert f"['{param}']" not in src


def test_the_flag_is_named_only_inside_that_regression():
    """The flag and the removed parameter appear only inside the one regression."""
    from pathlib import Path

    flag = _split_name("SEARCH_", "ALSO_", "VERBATIM")
    param = _split_name("also_", "query")
    text = Path(__file__).read_text(encoding="utf-8")
    regression = inspect.getsource(test_verbatim_co_search_flag_on_issues_exactly_one_query)
    outside = text.replace(regression, "", 1)
    assert flag not in outside
    assert param not in outside
