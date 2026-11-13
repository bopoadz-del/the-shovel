"""The restore drill compares a restored database with live by counts only.
Every database here is a throwaway SQLite file built by the test."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from scripts import restore_drill as rd


def _db(tmp_path, name: str, rows: dict, statuses=()):
    eng = create_engine(f"sqlite:///{tmp_path / name}.db")
    with eng.begin() as c:
        for table, n in rows.items():
            extra = ", ingest_status TEXT" if table == "documents" else ""
            c.execute(text(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY{extra})'))
            for i in range(n):
                if table == "documents":
                    st = statuses[i] if i < len(statuses) else None
                    c.execute(text('INSERT INTO documents (id, ingest_status) VALUES (:i, :s)'), {"i": i, "s": st})
                else:
                    c.execute(text(f'INSERT INTO "{table}" (id) VALUES (:i)'), {"i": i})
    return sessionmaker(bind=eng)


TABLES = ("projects", "documents")


def test_counts_reads_each_table_and_status(tmp_path):
    f = _db(tmp_path, "a", {"projects": 2, "documents": 3, "chunks_x": 7}, ("INDEXED", "INDEXED", "OCR_REQUIRED"))
    c = rd.counts(f, TABLES, "chunks_x")
    assert c["projects"] == 2 and c["documents"] == 3 and c["chunks_x"] == 7
    assert c["documents_by_status"] == {"INDEXED": 2, "OCR_REQUIRED": 1}


def test_a_missing_table_is_unreadable_not_skipped(tmp_path):
    f = _db(tmp_path, "b", {"projects": 1, "documents": 1})
    c = rd.counts(f, TABLES, "chunks_absent")
    assert c["chunks_absent"] is None and c["projects"] == 1


@pytest.mark.parametrize("restore,live,ok,unreadable,empty", [
    ({"documents": 5, "chunks": 40}, {"documents": 6, "chunks": 44}, True, [], []),
    ({"documents": 0, "chunks": 40}, {"documents": 6, "chunks": 44}, False, [], ["documents"]),
    ({"documents": 0, "chunks": 0}, {"documents": 0, "chunks": 0}, True, [], []),
    ({"documents": None, "chunks": 40}, {"documents": 6, "chunks": 44}, False, ["documents"], []),
])
def test_compare_verdict(restore, live, ok, unreadable, empty):
    r = rd.compare(restore, live)
    assert r["ok"] is ok and r["unreadable"] == unreadable and r["empty_on_restore"] == empty


def test_compare_reports_the_delta():
    r = rd.compare({"documents": 5}, {"documents": 9})
    assert r["tables"]["documents"] == {"restore": 5, "live": 9, "delta": 4}


def test_run_end_to_end_ok_and_fail(tmp_path, capsys):
    live = _db(tmp_path, "live", {**{t: 2 for t in rd.TABLES}, "chunks_y": 9}, ("INDEXED", "INDEXED"))
    good = _db(tmp_path, "good", {**{t: 1 for t in rd.TABLES}, "chunks_y": 4}, ("INDEXED",))
    lost = _db(tmp_path, "lost", {**{t: 1 for t in rd.TABLES}, "chunks_y": 0}, ("INDEXED",))
    assert rd.run("ep-x", restore_factory=good, live_factory=live, chunk_table="chunks_y")["ok"] is True
    bad = rd.run("ep-x", restore_factory=lost, live_factory=live, chunk_table="chunks_y")
    assert bad["ok"] is False and bad["empty_on_restore"] == ["chunks_y"]
    out = capsys.readouterr().out
    assert "RESTORE_DRILL verdict=ok" in out and "RESTORE_DRILL verdict=FAIL" in out


@pytest.mark.parametrize("endpoint,code", [("not-an-endpoint", "bad_endpoint_id"), ("", "bad_endpoint_id")])
def test_main_refuses_a_malformed_endpoint(monkeypatch, capsys, endpoint, code):
    from scripts import scrub_verify as sv
    monkeypatch.setattr(sv, "_live_url", lambda: "postgresql://u:p@ep-live-1.example.neon.tech/db")
    assert rd.main(["--endpoint", endpoint]) == 2
    out = capsys.readouterr().out
    assert f"refused={code}" in out and "u:p" not in out


def test_main_refuses_the_live_endpoint(monkeypatch, capsys):
    from scripts import scrub_verify as sv
    monkeypatch.setattr(sv, "_live_url", lambda: "postgresql://u:p@ep-live-1.example.neon.tech/db")
    assert rd.main(["--endpoint", "ep-live-1"]) == 2
    assert "refused=restore_endpoint_is_live" in capsys.readouterr().out
