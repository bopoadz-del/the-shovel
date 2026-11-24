"""G2 exams of the exams: a must-pass and a must-fail for each scorer.

A scorer that can only say PASS (or only FAIL) proves nothing about the
answers it grades. Each test below builds its own answer, hands it to the
scorer's real decision function and checks that a good answer is passed and
a bad one is failed, one defect per must-fail case, so every rule in the
scorer is shown to bite on its own.

Scorers covered here:
- scripts/golden_set_gate.py ``judge`` (the pilot golden-set gate).
- scripts/consistency_oracle.py ``run_oracle`` and ``preflight_corpus_health``
  (the standing prod-health oracle). Their network calls are replaced by
  in-test stubs; the verdict logic is the script's own.

The entries and answers are invented for the test; they are not the golden
set's queries or any live ask.
"""
from __future__ import annotations

import pytest

from scripts import consistency_oracle as oracle
from scripts.golden_set_gate import judge

# ── golden_set_gate.judge ────────────────────────────────────────────────────


def _entry(expect=None, reject=None, retrieval=None) -> dict:
    return {
        "answer_expect": list(expect or []),
        "answer_reject": list(reject or []),
        "retrieval_expect": retrieval,
    }


def _res(answer="", sources=None, errors=None, http_error=None) -> dict:
    out = {"answer": answer, "sources": list(sources or []),
           "errors": list(errors or [])}
    if http_error is not None:
        out["http_error"] = http_error
    return out


# Two independent entries, each with a passing and a failing answer, so the
# verdict is shown to follow the answer and not the entry.
JUDGE_CASES = [
    (
        _entry(expect=[r"\b412\s*m3\b"], reject=[r"\bnot found\b"],
               retrieval="pour_register"),
        _res("Total poured is 412 m3 [1].",
             sources=[{"doc_name": "Pour_Register_Q3.xlsx", "doc_id": "d1"}]),
        _res("The quantity was not found [1].",
             sources=[{"doc_name": "Pour_Register_Q3.xlsx", "doc_id": "d1"}]),
    ),
    (
        _entry(expect=[r"Engineer", r"28\s*days"], reject=[r"\bguess\b"],
               retrieval="doc-77"),
        _res("Notice goes to the Engineer within 28 days.",
             sources=[{"doc_name": "", "doc_id": "doc-77"}]),
        _res("Notice goes to the Engineer within 14 days.",
             sources=[{"doc_name": "", "doc_id": "doc-77"}]),
    ),
]


@pytest.mark.parametrize("entry,good,bad", JUDGE_CASES)
def test_judge_must_pass_and_must_fail(entry, good, bad):
    ok = judge(entry, good)
    assert ok["verdict"] == "PASS", ok["reasons"]
    assert ok["reasons"] == []
    no = judge(entry, bad)
    assert no["verdict"] == "FAIL"
    assert no["reasons"]


@pytest.mark.parametrize("answer", ["", "   \n\t"])
def test_judge_fails_an_empty_answer(answer):
    v = judge(_entry(), _res(answer))
    assert v["verdict"] == "FAIL"
    assert "execution: empty answer" in v["reasons"]


@pytest.mark.parametrize("http_error", ["HTTP 502", "timeout after 240s"])
def test_judge_fails_a_transport_error_even_with_a_good_answer(http_error):
    entry = _entry(expect=[r"\b9\b"])
    assert judge(entry, _res("9 bores"))["verdict"] == "PASS"
    v = judge(entry, _res("9 bores", http_error=http_error))
    assert v["verdict"] == "FAIL"
    assert v["reasons"] == [f"execution: {http_error}"]


@pytest.mark.parametrize("err", ["tool crashed", "stream aborted"])
def test_judge_fails_an_error_event_even_with_a_good_answer(err):
    entry = _entry(expect=[r"\bC40\b"])
    v = judge(entry, _res("Grade is C40.", errors=[err]))
    assert v["verdict"] == "FAIL"
    assert v["reasons"] == [f"execution: error event: {err}"]


@pytest.mark.parametrize("pattern,good,bad", [
    (r"\b1,?250\b", "Length 1,250 m.", "Length 1,205 m."),
    (r"^Yes\b", "Yes, it is required.", "It is required. Yes."),
])
def test_judge_expect_rule_alone(pattern, good, bad):
    entry = _entry(expect=[pattern])
    assert judge(entry, _res(good))["verdict"] == "PASS"
    v = judge(entry, _res(bad))
    assert v["verdict"] == "FAIL"
    assert v["expect_misses"] == [pattern]
    assert v["reject_hits"] == []


@pytest.mark.parametrize("pattern,good,bad", [
    (r"I (?:cannot|can't) find", "The bond is 10%.", "I cannot find the bond."),
    (r"\bassum", "Rate is 45 USD/m2.", "Assuming 45 USD/m2."),
])
def test_judge_reject_rule_alone(pattern, good, bad):
    entry = _entry(reject=[pattern])
    assert judge(entry, _res(good))["verdict"] == "PASS"
    v = judge(entry, _res(bad))
    assert v["verdict"] == "FAIL"
    assert v["reject_hits"] == [pattern]
    assert v["expect_misses"] == []


@pytest.mark.parametrize("needle,hit,miss", [
    ("spec_section_03", {"doc_name": "Spec_Section_03_Concrete.pdf"},
     {"doc_name": "Spec_Section_05_Steel.pdf"}),
    ("abc-123", {"doc_name": "", "doc_id": "ABC-123"},
     {"doc_name": "abc.pdf", "doc_id": "abc-124"}),
])
def test_judge_retrieval_rule_alone(needle, hit, miss):
    entry = _entry(retrieval=needle)
    good = judge(entry, _res("An answer.", sources=[hit]))
    assert good["verdict"] == "PASS"
    assert good["retrieval_ok"] is True
    bad = judge(entry, _res("An answer.", sources=[miss]))
    assert bad["verdict"] == "FAIL"
    assert bad["retrieval_ok"] is False
    none = judge(entry, _res("An answer.", sources=[]))
    assert none["verdict"] == "FAIL"
    assert "(no sources cited)" in none["reasons"][0]


@pytest.mark.parametrize("sources", [[], [{"doc_name": "x.pdf"}]])
def test_judge_without_a_retrieval_rule_does_not_grade_sources(sources):
    v = judge(_entry(expect=[r"ok"]), _res("ok", sources=sources))
    assert v["verdict"] == "PASS"
    assert v["retrieval_ok"] is None


# ── consistency_oracle ───────────────────────────────────────────────────────


def _phrase(value: str, unit: str) -> str:
    return f"The station is rated at {value} {unit} total."


def _stub_stream(monkeypatch, replies):
    """Replace the network call with replies built in the test, one per
    question, in order."""
    queue = list(replies)
    asked = []

    def fake(base, tok, project_id, message, timeout=240):
        asked.append((project_id, message))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(oracle, "_stream_chat", fake)
    return asked


SRC = [{"doc_name": "design_basis.pdf"}]


@pytest.mark.parametrize("unit", ["l/s", "litres per second"])
def test_oracle_must_pass_every_phrasing_cited_with_the_fact(monkeypatch, unit):
    n = len(oracle.FACT_QUESTIONS)
    asked = _stub_stream(
        monkeypatch, [(_phrase(oracle.FACT_EXPECTED, unit), SRC)] * n)
    assert oracle.run_oracle("http://x", "t") is True
    assert [m for _, m in asked] == oracle.FACT_QUESTIONS
    assert {p for p, _ in asked} == {oracle.FACT_PROJECT}


def test_oracle_must_pass_keeps_derived_figures(monkeypatch):
    # The oracle's own doc: extra derived numbers are elaboration, not
    # divergence. The fact plus a second figure still passes.
    half = str(int(oracle.FACT_EXPECTED) // 2)
    text = (f"{_phrase(oracle.FACT_EXPECTED, 'l/s')} "
            f"Each duty pump gives {half} l/s.")
    _stub_stream(monkeypatch, [(text, SRC)] * len(oracle.FACT_QUESTIONS))
    assert oracle.run_oracle("http://x", "t") is True


def _wrong_value() -> str:
    return str(int(oracle.FACT_EXPECTED) + 1)


@pytest.mark.parametrize("bad_index", [0, 2])
def test_oracle_must_fail_one_uncited_answer(monkeypatch, bad_index):
    n = len(oracle.FACT_QUESTIONS)
    replies = [(_phrase(oracle.FACT_EXPECTED, "l/s"), SRC)] * n
    replies[bad_index] = (_phrase(oracle.FACT_EXPECTED, "l/s"), [])
    _stub_stream(monkeypatch, replies)
    assert oracle.run_oracle("http://x", "t") is False


@pytest.mark.parametrize("bad_text", [
    _phrase(_wrong_value(), "l/s"),
    # The right digits but not as a flow figure: not the fact.
    f"Drawing {oracle.FACT_EXPECTED} shows the station.",
])
def test_oracle_must_fail_one_answer_without_the_fact(monkeypatch, bad_text):
    n = len(oracle.FACT_QUESTIONS)
    replies = [(_phrase(oracle.FACT_EXPECTED, "l/s"), SRC)] * n
    replies[1] = (bad_text, SRC)
    _stub_stream(monkeypatch, replies)
    assert oracle.run_oracle("http://x", "t") is False


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"),
                                 ConnectionError("reset")])
def test_oracle_must_fail_a_stream_error_and_keep_asking(monkeypatch, exc):
    n = len(oracle.FACT_QUESTIONS)
    replies = [exc] + [(_phrase(oracle.FACT_EXPECTED, "l/s"), SRC)] * (n - 1)
    asked = _stub_stream(monkeypatch, replies)
    assert oracle.run_oracle("http://x", "t") is False
    assert len(asked) == n


@pytest.mark.parametrize("payload", [
    {"status": "active"},
    {"status": "active", "name": "corpus", "documents": []},
])
def test_preflight_must_pass_an_active_project(monkeypatch, payload):
    monkeypatch.setattr(oracle, "_get",
                        lambda base, path, tok, timeout=90: payload)
    assert oracle.preflight_corpus_health("http://x", "t") is True


@pytest.mark.parametrize("payload", [{"status": "archived"}, {}])
def test_preflight_must_fail_a_project_that_is_not_active(monkeypatch, payload):
    monkeypatch.setattr(oracle, "_get",
                        lambda base, path, tok, timeout=90: payload)
    assert oracle.preflight_corpus_health("http://x", "t") is False


@pytest.mark.parametrize("exc", [OSError("dns"), ValueError("bad json")])
def test_preflight_must_fail_when_the_project_cannot_load(monkeypatch, exc):
    def boom(base, path, tok, timeout=90):
        raise exc

    monkeypatch.setattr(oracle, "_get", boom)
    assert oracle.preflight_corpus_health("http://x", "t") is False


@pytest.mark.parametrize("healthy,consistent,code", [
    (True, True, 0), (True, False, 1), (False, True, 1), (False, False, 1),
])
def test_oracle_main_exit_code_needs_both_stages(monkeypatch, healthy,
                                                 consistent, code):
    monkeypatch.setenv("FORK_TOKEN", "t")
    monkeypatch.setattr("sys.argv", ["consistency_oracle.py", "--base", "http://x"])
    monkeypatch.setattr(oracle, "preflight_corpus_health", lambda b, t: healthy)
    monkeypatch.setattr(oracle, "run_oracle", lambda b, t: consistent)
    assert oracle.main() == code
