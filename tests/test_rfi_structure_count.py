"""Q-3: the read-only RFI structure count. Every input is built here; no
document id, name or corpus text from production appears."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rfi_structure_count.py"
_SRC = _SCRIPT.read_text(encoding="utf-8")


def _load():
    spec = importlib.util.spec_from_file_location("rfi_structure_count", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()
PATS = m.compile_headings(m.DEFAULT_HEADINGS)

Q = "Question:\nWhich grade applies to the slab?"
A = "Response:\nUse the grade on the general notes."
R = "References:\nSheet 7, clause 4."


@pytest.mark.parametrize("chunks,expected", [
    ([(0, f"{Q}\n{A}\n{R}")], True),          # all three blocks
    ([(0, f"{Q}\n{R}")], False),               # no answer block
    ([(0, f"{Q}\n{A}")], False),               # no references block
    ([(0, f"{A}\n{R}\n{Q}")], False),          # answer only BEFORE the question
])
def test_rfi_like_needs_question_then_answer_and_references(chunks, expected):
    assert m.classify(chunks, PATS)["rfi_like"] is expected


@pytest.mark.parametrize("body,hit", [
    ("Response: see sheet", True),           # heading at line start
    ("the contractor's response: see sheet", False),  # word inside prose
    ("2.1 Response\nsee sheet", True),       # numbered heading ending the line
    ("Responsibility: the contractor", False),  # longer word, not the heading
])
def test_a_heading_is_a_label_at_line_start(body, hit):
    assert bool(m.heading_hits(body, PATS)["answer"]) is hit


@pytest.mark.parametrize("chunks,split,qa", [
    ([(0, Q), (1, A), (2, R)], True, (0, 1)),
    ([(0, f"{Q}\n{A}"), (1, R)], False, (0, 0)),
    ([(5, f"{A}\n{R}"), (3, Q)], True, (3, 5)),   # read in chunk_index order
])
def test_split_is_question_and_answer_in_different_chunks(chunks, split, qa):
    r = m.classify(chunks, PATS)
    assert r["rfi_like"] is True
    assert r["split"] is split
    assert (r["question_chunk"], r["answer_chunk"]) == qa


@pytest.mark.parametrize("raw,words", [
    ("", m.DEFAULT_HEADINGS["answer"]),
    (json.dumps({"question": ["Ask"], "answer": ["Reply Given"], "references": ["See"]}),
     ("reply given",)),
])
def test_heading_table_reads_the_operator_table(raw, words):
    assert m.headings(raw)["answer"] == words


@pytest.mark.parametrize("raw", [
    json.dumps({"question": ["Ask"], "answer": [], "references": ["See"]}),
    json.dumps({"question": ["Ask"], "answer": ["Reply"]}),
])
def test_a_malformed_table_is_refused_not_counted_as_zero(raw):
    with pytest.raises(ValueError):
        m.headings(raw)


@pytest.mark.parametrize("table,text_,expected", [
    ({"question": ["ask"], "answer": ["reply given"], "references": ["see"]},
     "Ask:\nx\nReply   Given:\ny\nSee:\nz", True),
    ({"question": ["ask"], "answer": ["reply given"], "references": ["see"]},
     f"{Q}\n{A}\n{R}", False),
])
def test_an_operator_table_replaces_the_default(table, text_, expected):
    pats = m.compile_headings(m.headings(json.dumps(table)))
    assert m.classify([(0, text_)], pats)["rfi_like"] is expected


@pytest.mark.parametrize("rows,expected", [
    ([("d1", 0, "a"), ("d1", 1, "b"), ("d2", 0, "c")], [("d1", 2), ("d2", 1)]),
    ([("d9", 3, "x")], [("d9", 1)]),
])
def test_group_by_doc(rows, expected):
    assert [(d, len(c)) for d, c in m.group_by_doc(rows)] == expected


@pytest.mark.parametrize("docs,counts", [
    ({"a": [(0, Q), (1, A), (2, R)], "b": [(0, f"{Q}\n{A}\n{R}")], "c": [(0, "plain text")]},
     (3, 2, 1, 1)),
    ({"c": [(0, "plain text")]}, (1, 0, 0, 0)),
])
def test_summary_counts(docs, counts):
    s = m.summarise((d, m.classify(c, PATS)) for d, c in docs.items())
    assert (s["documents"], s["rfi_like"], s["split"], s["same_chunk"]) == counts


@pytest.mark.parametrize("docs,roles,q_then_a", [
    # one complete form, one with question and answer but no references,
    # one with only a references heading, one with an answer BEFORE its question
    ({"a": [(0, Q), (1, A), (2, R)], "b": [(0, f"{Q}\n{A}")], "c": [(0, R)],
      "d": [(0, f"{A}\n{Q}")]},
     {"question": 3, "answer": 3, "references": 2}, 2),
    ({"c": [(0, "plain text")], "e": [(0, Q)]},
     {"question": 1, "answer": 0, "references": 0}, 0),
])
def test_summary_counts_each_role_heading_even_when_no_document_is_rfi_like(docs, roles, q_then_a):
    s = m.summarise((d, m.classify(c, PATS)) for d, c in docs.items())
    assert s["with_role"] == roles
    assert s["q_then_a"] == q_then_a
    line = next(x for x in m.render("anc", s) if "documents with a heading of" in x)
    assert line.endswith(" ".join(f"{r}={roles[r]}" for r in m.ROLES)
                         + f" question_then_answer={q_then_a}")


@pytest.mark.parametrize("body,roles", [
    (f"{Q}\n{R}", ("question", "references")),
    ("plain text", ()),
])
def test_classify_names_the_roles_found(body, roles):
    assert m.classify([(0, body)], PATS)["roles"] == roles


@pytest.mark.parametrize("secret", ["Which grade applies to the slab", "general notes"])
def test_render_prints_ids_and_counts_never_text(secret):
    docs = {"doc1": [(0, Q), (1, A), (2, R)]}
    s = m.summarise((d, m.classify(c, PATS)) for d, c in docs.items())
    out = "\n".join(m.render("anc", s))
    assert secret not in out
    assert "doc1 | 3 | 0 | 1 | yes" in out


@pytest.mark.parametrize("cap,shown", [(1, 1), (5, 2)])
def test_render_caps_the_id_list(cap, shown):
    docs = {"a": [(0, f"{Q}\n{A}\n{R}")], "b": [(0, f"{Q}\n{A}\n{R}")]}
    s = m.summarise(((d, m.classify(c, PATS)) for d, c in docs.items()), cap=cap)
    assert len(s["rows"]) == shown


class _FakeSession:
    """Records the one chunk read and returns invented rows."""

    def __init__(self, rows):
        self.rows, self.sql, self.params = rows, None, None

    def execute(self, stmt, params):
        self.sql, self.params = str(stmt), params
        return iter(self.rows)


_ROWS = [("doc-a", 0, "x"), ("doc-a", 1, "y"), ("doc-s", 0, "z"), ("doc-b", 0, "w")]


@pytest.mark.parametrize("hidden, kept", [
    (set(), ["doc-a", "doc-a", "doc-s", "doc-b"]),
    ({"doc-s"}, ["doc-a", "doc-a", "doc-b"]),
])
def test_project_chunk_rows_drop_the_session_hidden_documents(monkeypatch, hidden, kept):
    import app.core.projects as projects
    seen = {}

    def fake_hidden(project_id, db_session=None):
        seen["args"] = (project_id, db_session)
        return set(hidden)

    monkeypatch.setattr(projects, "session_hidden_document_ids", fake_hidden)
    sess = _FakeSession(_ROWS)
    out = list(m.project_chunk_rows(sess, "chunks_v2", "proj-1"))
    assert [r[0] for r in out] == kept
    assert seen["args"] == ("proj-1", sess) and sess.params == {"pr": "proj-1"}
    test_sql_is_a_select_with_bound_parameters(sess.sql)


@pytest.mark.parametrize("rows, hidden, kept", [
    (_ROWS, {"doc-a", "doc-b"}, [("doc-s", 0, "z")]),
    (_ROWS, {"absent"}, _ROWS),
])
def test_drop_hidden_keeps_order_and_only_visible_rows(rows, hidden, kept):
    assert list(m.drop_hidden(rows, hidden)) == kept


@pytest.mark.parametrize("sql", [m._DOC_SQL])
def test_sql_is_a_select_with_bound_parameters(sql):
    assert sql.lstrip().upper().startswith("SELECT ")
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\b", sql, re.I)
    assert ":p" in sql or ":pr" in sql


@pytest.mark.parametrize("env,code", [({"RFI_ANCHOR_ID": ""}, 2), ({"RFI_ANCHOR_ID": "a%b"}, 2)])
def test_main_refuses_a_missing_or_unsafe_anchor(monkeypatch, env, code):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert m.main() == code


@pytest.mark.parametrize("call", ["commit(", ".add("])
def test_script_never_writes(call):
    assert call not in _SRC


# --- letter script: what the English heading rule cannot see -------------

_AR = "".join(chr(c) for c in range(0x0628, 0x0638))  # sixteen Arabic letters
_GR = "".join(chr(c) for c in range(0x03B1, 0x03BA))  # nine Greek letters


@pytest.mark.parametrize("ch, script", [
    ("a", "LATIN"), (_AR[0], "ARABIC"), (_GR[0], "GREEK"), ("7", None), (":", None),
])
def test_letter_script_is_the_unicode_script_of_a_letter_only(ch, script):
    assert m.letter_script(ch) == script


@pytest.mark.parametrize("table, scripts", [
    (m.DEFAULT_HEADINGS, {"LATIN"}),
    ({"question": [_AR[:4]], "answer": [_AR[4:8]], "references": [_AR[8:12]]}, {"ARABIC"}),
    ({"question": ["query"], "answer": [_AR[4:8]], "references": ["ref"]}, {"LATIN", "ARABIC"}),
])
def test_the_rule_sees_the_scripts_its_own_table_is_written_in(table, scripts):
    assert m.table_scripts(table) == frozenset(scripts)


@pytest.mark.parametrize("texts, script, unseen, mixed", [
    (["plain words only here"], "LATIN", False, False),
    ([_AR * 3, "ab"], "ARABIC", True, False),
    (["a" * 90, _AR[:10]], "LATIN", False, True),     # 10 of 100 letters outside
    (["a" * 99, _AR[:1]], "LATIN", False, False),     # 1 of 100: under the share
    (["12 34 -- ::"], None, False, False),
])
def test_script_profile_mainly_or_in_part_outside_the_table(texts, script, unseen, mixed):
    p = m.script_profile(m.script_counts(texts), frozenset({"LATIN"}))
    assert (p["script"], p["unseen"], p["mixed"]) == (script, unseen, mixed)


@pytest.mark.parametrize("seen, unseen", [
    (frozenset({"LATIN"}), 1),
    (frozenset({"LATIN", "ARABIC"}), 0),
])
def test_summary_counts_documents_by_script_over_every_document(seen, unseen):
    docs = {"d1": [(0, "Question: x\nAnswer: y\nReference: z")],
            "d2": [(0, _AR * 2)],
            "d3": [(0, "123")]}
    s = m.summarise((d, m.classify(c, PATS, seen)) for d, c in docs.items())
    assert dict(s["by_script"]) == {"LATIN": 1, "ARABIC": 1, "(no letters)": 1}
    assert s["unseen_script"] == unseen
    assert s["rfi_like"] == 1


@pytest.mark.parametrize("secret", ["zebra-" + _GR, _AR + " hidden"])
def test_render_prints_script_names_never_text(secret):
    docs = {"d1": [(0, secret * 3)], "d2": [(0, "Question: a\nAnswer: b\nRefs: c")]}
    s = m.summarise((d, m.classify(c, PATS, frozenset({"LATIN"}))) for d, c in docs.items())
    out = "\n".join(m.render("anc", s))
    assert secret not in out and secret.split()[0] not in out
    assert "outside the heading table's scripts: mainly=" in out
