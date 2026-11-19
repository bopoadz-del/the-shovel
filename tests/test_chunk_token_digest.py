"""chunk_token_digest: digests carry no text, and the stand-in text built from
them scores exactly like the real text. Every input here is invented."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SALT = b"invented-salt-for-tests-only"

_CHUNKS = {
    "c-1": "The slab is poured at C40 grade with 1,250 m3 of concrete over 14 days.",
    "c-2": "Curing lasts 7 days; formwork is struck when cube strength reaches 0.00005 margin.",
}


def _row(answer):
    return {
        "id": "r1",
        "answer": answer,
        "handed": [{"chunk_id": k, "doc_id": "d" + k, "doc_name": n}
                   for k, n in (("c-1", "Pour Plan.pdf"), ("c-2", "Curing Note.pdf"))],
        "tool_results": [],
        "sources_list": [],
    }


@pytest.mark.parametrize("answer", [
    # every figure and word present in the cited chunk
    "The slab takes 1,250 m3 of concrete over 14 days [source: Pour Plan.pdf].",
    # a figure absent from the chunk, low overlap, one chunk cited each way
    "Curing lasts 9 days under wet hessian [source: Curing Note.pdf]. "
    "Concrete volume is 1,250 m3 [source: Pour Plan.pdf]. Small margin 0.00005 [source: Curing Note.pdf].",
])
def test_stand_in_text_scores_exactly_like_the_real_text(answer):
    dig, sc = _load("chunk_token_digest"), _load("answer_source_score")
    real = _row(answer)
    for c in real["handed"]:
        c["text"] = _CHUNKS[c["chunk_id"]]
    records = dig.digest_records(_CHUNKS, list(_CHUNKS), _SALT)
    stand_in = _row(answer)
    sc.attach_chunk_text([stand_in], dig.surrogate_texts(records, [stand_in], _SALT))
    assert sc.score_row(stand_in).as_dict() == sc.score_row(real).as_dict()


@pytest.mark.parametrize("text", list(_CHUNKS.values()))
def test_a_record_carries_no_word_and_no_number_of_the_text(text):
    rec = _load("chunk_token_digest").chunk_digest("c-x", text, _SALT)
    printed = json.dumps(rec).casefold()
    for tok in ("slab", "concrete", "curing", "formwork", "1250", "1,250", "days"):
        assert tok not in printed
    assert rec["w"] == sorted(rec["w"]) and rec["n"] == sorted(rec["n"]) and rec["chunk"] == "c-x"


@pytest.mark.parametrize("salt_a, salt_b", [(_SALT, b"another-invented-salt-xx"), (b"x" * 16, b"y" * 16)])
def test_a_different_salt_gives_different_digests(salt_a, salt_b):
    dig = _load("chunk_token_digest")
    assert dig.chunk_digest("c", _CHUNKS["c-1"], salt_a)["w"] != dig.chunk_digest("c", _CHUNKS["c-1"], salt_b)["w"]


@pytest.mark.parametrize("raw, ok", [(None, False), ("short", False), ("s" * 15, False), ("s" * 16, True)])
def test_salt_is_required_and_long_enough(raw, ok):
    dig = _load("chunk_token_digest")
    if ok:
        assert dig.read_salt(raw) == raw.encode()
    else:
        with pytest.raises(ValueError):
            dig.read_salt(raw)


@pytest.mark.parametrize("n, ok", [(200, True), (201, False)])
def test_ids_are_capped_per_run(n, ok):
    dig = _load("chunk_token_digest")
    raw = ",".join(f"id-{i}" for i in range(n))
    if ok:
        assert len(dig.parse_ids(raw)) == n
    else:
        with pytest.raises(ValueError):
            dig.parse_ids(raw)


@pytest.mark.parametrize("raw, ids", [(" a, b ,a,, ", ["a", "b"]), ("", [])])
def test_ids_are_distinct_and_in_order(raw, ids):
    assert _load("chunk_token_digest").parse_ids(raw) == ids


@pytest.mark.parametrize("asked, missing", [(["c-1", "gone"], ["gone"]), (["c-1", "c-2"], [])])
def test_an_id_the_index_lacks_prints_missing_and_gets_no_stand_in(asked, missing):
    dig = _load("chunk_token_digest")
    recs = dig.digest_records(_CHUNKS, asked, _SALT)
    assert [r["chunk"] for r in recs if r.get("missing")] == missing
    texts = dig.surrogate_texts(recs, [_row("concrete slab 14")], _SALT)
    assert set(texts) == set(asked) - set(missing)


@pytest.mark.parametrize("table", ["chunks_v2", "chunks"])
def test_chunk_sql_is_a_select_by_bound_ids(table):
    sql = _load("chunk_token_digest")._chunk_text_sql(table)
    assert sql.startswith("SELECT ") and f"FROM {table} " in sql and ":ids" in sql
    assert not any(w in sql.upper() for w in ("INSERT", "UPDATE", "DELETE", "DROP"))


@pytest.mark.parametrize("line_prefix", ["", "2026-11-18T09:00:00Z "])
def test_saved_log_lines_are_read_back_with_or_without_a_prefix(tmp_path, line_prefix):
    dig = _load("chunk_token_digest")
    rec = dig.chunk_digest("c-1", _CHUNKS["c-1"], _SALT)
    log = tmp_path / "log.txt"
    log.write_text(line_prefix + json.dumps(rec) + "\nchunk_digest: asked=1 found=1 missing=0\nnoise {not json\n",
                   encoding="utf-8")
    assert dig._read_records(str(log)) == [rec]


@pytest.mark.parametrize("env, code", [({}, 2), ({"CHUNK_DIGEST_SALT": "s" * 16}, 2)])
def test_main_refuses_without_salt_or_ids(monkeypatch, env, code):
    monkeypatch.delenv("CHUNK_DIGEST_SALT", raising=False)
    monkeypatch.delenv("CHUNK_DIGEST_IDS", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _load("chunk_token_digest").main([]) == code
