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


@pytest.mark.parametrize("n,statuses,by_status", [
    ((2, 3, 7), ("INDEXED", "INDEXED", "OCR_REQUIRED"), {"INDEXED": 2, "OCR_REQUIRED": 1}),
    ((1, 2, 0), ("FAILED", None), {"FAILED": 1, "unset": 1}),
])
def test_counts_reads_each_table_and_status(tmp_path, n, statuses, by_status):
    f = _db(tmp_path, "a", {"projects": n[0], "documents": n[1], "chunks_x": n[2]}, statuses)
    c = rd.counts(f, TABLES, "chunks_x")
    assert (c["projects"], c["documents"], c["chunks_x"]) == n
    assert c["documents_by_status"] == by_status


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


@pytest.mark.parametrize("restore,live,delta", [(5, 9, 4), (7, 7, 0), (None, 3, None)])
def test_compare_reports_the_delta(restore, live, delta):
    r = rd.compare({"documents": restore}, {"documents": live})
    assert r["tables"]["documents"] == {"restore": restore, "live": live, "delta": delta}


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


@pytest.mark.parametrize("live", ["ep-live-1", "ep-other-7"])
def test_main_refuses_the_live_endpoint(monkeypatch, capsys, live):
    from scripts import scrub_verify as sv
    monkeypatch.setattr(sv, "_live_url", lambda: f"postgresql://u:p@{live}.example.neon.tech/db")
    assert rd.main(["--endpoint", live]) == 2
    assert "refused=restore_endpoint_is_live" in capsys.readouterr().out


# ── scratch branch lifecycle: a fake Neon API built per test ─────────────────
import datetime as _dt  # noqa: E402

LIVE = "postgresql://u:p@ep-live-1.c-2.us-west-2.aws.neon.tech/db"


class FakeNeon:
    def __init__(self, statuses=("running", "finished"), made_branch="br-new",
                 made_endpoint="ep-new-9", projects=None):
        self.calls = []
        self.statuses = list(statuses)
        self.made_branch, self.made_endpoint = made_branch, made_endpoint
        self.projects = projects or {"p-other": [{"id": "ep-x", "branch_id": "br-x"}],
                                     "p-live": [{"id": "ep-live-1", "branch_id": "br-main"}]}

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.startswith("/projects?"):
            return {"projects": [{"id": p} for p in self.projects]}
        if path.endswith("/endpoints"):
            return {"endpoints": self.projects[path.split("/")[2]]}
        if method == "POST":
            return {"branch": {"id": self.made_branch},
                    "endpoints": [{"id": self.made_endpoint}],
                    "operations": [{"id": "op-1"}]}
        if "/operations/" in path:
            return {"operation": {"status": self.statuses.pop(0) if self.statuses else "finished"}}
        return {}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def NOW():
    return _dt.datetime(2026, 11, 15, 10, 0, tzinfo=_dt.timezone.utc)


def _drill(neon, minutes=60, compare=None, **kw):
    clk = Clock()
    seen = []

    def cmp(ep):
        seen.append(ep)
        return compare(ep) if compare else {"ok": True}
    r = rd.scratch_drill(neon, LIVE, minutes, cmp, clock=clk, sleep=clk.sleep, now_utc=NOW, **kw)
    return r, seen


@pytest.mark.parametrize("url,ep", [(LIVE, "ep-live-1"),
                                    ("postgresql://u:p@ep-live-1-pooler.x.neon.tech/db", "ep-live-1"),
                                    ("postgresql://u:p@ep-other-2.x.neon.tech:5432/db", "ep-other-2")])
def test_live_endpoint_id(url, ep):
    assert rd.live_endpoint_id(url) == ep


@pytest.mark.parametrize("url", ["postgresql://u:p@db.internal/db", "sqlite:///x.db"])
def test_live_endpoint_id_refuses_a_non_neon_host(url):
    with pytest.raises(rd.Refused) as e:
        rd.live_endpoint_id(url)
    assert e.value.code == "live_host_not_neon"


@pytest.mark.parametrize("minutes,point", [(60, "2026-11-15T09:00:00Z"), (5, "2026-11-15T09:55:00Z")])
def test_branch_made_on_the_live_branch_at_the_point_compared_then_deleted(minutes, point):
    neon = FakeNeon()
    r, seen = _drill(neon, minutes)
    post = [c for c in neon.calls if c[0] == "POST"][0]
    assert post[1] == "/projects/p-live/branches"
    assert post[2]["branch"]["parent_id"] == "br-main" and post[2]["branch"]["parent_timestamp"] == point
    assert post[2]["endpoints"] == [{"type": "read_only"}]
    assert seen == ["ep-new-9"]
    assert neon.calls[-1][:2] == ("DELETE", "/projects/p-live/branches/br-new")
    assert [c for c in neon.calls if c[0] == "DELETE"] == [neon.calls[-1]]
    assert r["times"]["restore_point"] == point and r["times"]["ready_s"] == 5.0
    assert r["ok"] is True


@pytest.mark.parametrize("statuses,ready", [(("finished",), 0.0), (("running", "running", "finished"), 10.0)])
def test_ready_time_counts_the_polls(statuses, ready):
    r, _ = _drill(FakeNeon(statuses=statuses))
    assert r["times"]["ready_s"] == ready


@pytest.mark.parametrize("err", [RuntimeError, ConnectionError])
def test_branch_is_deleted_even_when_the_comparison_raises(err):
    neon = FakeNeon()

    def boom(ep):
        raise err("db down")
    with pytest.raises(err):
        _drill(neon, compare=boom)
    assert neon.calls[-1][:2] == ("DELETE", "/projects/p-live/branches/br-new")


@pytest.mark.parametrize("statuses,code", [(("running", "failed"), "branch_operation_failed"),
                                           (("running",) * 1000, "branch_not_ready")])
def test_a_branch_that_never_serves_is_refused_and_deleted(statuses, code):
    neon = FakeNeon(statuses=statuses)
    with pytest.raises(rd.Refused) as e:
        _drill(neon, ready_timeout_s=60)
    assert e.value.code == code
    assert neon.calls[-1][:2] == ("DELETE", "/projects/p-live/branches/br-new")


@pytest.mark.parametrize("endpoint", ["", "ep-live-1"])
def test_an_unusable_restore_endpoint_is_refused_never_compared(endpoint):
    neon = FakeNeon(made_endpoint=endpoint)
    with pytest.raises(rd.Refused) as e:
        _drill(neon)
    assert e.value.code == "restore_endpoint_unusable"
    assert neon.calls[-1][:2] == ("DELETE", "/projects/p-live/branches/br-new")


@pytest.mark.parametrize("branch", ["", "br-main"])
def test_no_delete_when_no_new_branch_was_made(branch):
    neon = FakeNeon(made_branch=branch)
    with pytest.raises(rd.Refused) as e:
        _drill(neon)
    assert e.value.code == "branch_not_created"
    assert not [c for c in neon.calls if c[0] == "DELETE"]


@pytest.mark.parametrize("projects", [{"p-other": [{"id": "ep-x", "branch_id": "br-x"}]},
                                      {"p-a": [], "p-b": [{"id": "ep-live-1-old", "branch_id": "br-b"}]}])
def test_live_endpoint_not_in_any_project_is_refused_before_any_write(projects):
    neon = FakeNeon(projects=projects)
    with pytest.raises(rd.Refused) as e:
        _drill(neon)
    assert e.value.code == "live_endpoint_not_found"
    assert all(c[0] == "GET" for c in neon.calls)


@pytest.mark.parametrize("minutes", [0, -5])
def test_minutes_back_below_one_is_refused(minutes):
    neon = FakeNeon()
    with pytest.raises(rd.Refused) as e:
        _drill(neon, minutes)
    assert e.value.code == "minutes_back_below_1" and neon.calls == []


@pytest.mark.parametrize("set_empty", [True, False])
def test_create_without_a_key_refuses(monkeypatch, capsys, set_empty):
    if set_empty:
        monkeypatch.setenv("NEON_API_KEY", "")
    else:
        monkeypatch.delenv("NEON_API_KEY", raising=False)
    assert rd.main(["--create"]) == 2
    assert "refused=neon_api_key_missing" in capsys.readouterr().out


@pytest.mark.parametrize("compare_ok,code", [(True, 0), (False, 1)])
def test_create_with_a_key_runs_the_scratch_drill(monkeypatch, capsys, compare_ok, code):
    from scripts import scrub_verify as sv
    monkeypatch.setenv("NEON_API_KEY", "k-test")
    monkeypatch.setattr(sv, "_live_url", lambda: LIVE)
    got = {}

    def fake_http(key):
        got["key"] = key
        return FakeNeon()
    monkeypatch.setattr(rd, "neon_http", fake_http)
    monkeypatch.setattr(rd, "run", lambda ep: {"ok": compare_ok and ep == "ep-new-9"})
    assert rd.main(["--create", "--minutes-back", "30"]) == code
    out = capsys.readouterr().out
    assert got["key"] == "k-test" and "k-test" not in out
    assert "restore_point=2026-" in out and "deleted_s=" in out


# ── a refused Neon call names the call that failed (exam rule 5: the stage) ──
def _failing_urlopen(status):
    import io
    import urllib.error

    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, status, "x", {}, io.BytesIO(b'{"message":"m"}'))
    return urlopen


@pytest.mark.parametrize("method, path, body, code", [
    ("GET", "/projects?limit=400", None, "neon_http_400:GET:/projects"),
    ("POST", "/projects/proj-abc-123/branches", {"branch": {}},
     "neon_http_400:POST:/projects/*/branches"),
    ("GET", "/projects/proj-abc-123/operations/op-9", None,
     "neon_http_404:GET:/projects/*/operations/*"),
    ("DELETE", "/projects/proj-abc-123/branches/br-x-1", None,
     "neon_http_409:DELETE:/projects/*/branches/*"),
])
def test_a_refused_neon_call_names_its_route_without_ids_or_body(monkeypatch, method, path, body, code):
    import urllib.request

    status = int(code.split(":")[0].rsplit("_", 1)[1])
    monkeypatch.setattr(urllib.request, "urlopen", _failing_urlopen(status))
    with pytest.raises(rd.Refused) as e:
        rd.neon_http("k-secret")(method, path, body)
    assert e.value.code == code
    for leak in ("proj-abc-123", "op-9", "br-x-1", "k-secret", '"m"', "message"):
        assert leak not in e.value.code


@pytest.mark.parametrize("path, raw, parsed", [
    ("/projects?limit=400", b'{"projects": []}', {"projects": []}),
    ("/projects/p-1/branches", b'{"branch": {"id": "br-1"}, "operations": [{"id": "op-1"}]}',
     {"branch": {"id": "br-1"}, "operations": [{"id": "op-1"}]}),
])
def test_a_successful_neon_call_returns_its_json(monkeypatch, path, raw, parsed):
    import io
    import urllib.request

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(raw))
    assert rd.neon_http("k")("GET", path) == parsed


@pytest.mark.parametrize("path, route", [
    ("/projects", "/projects"),
    ("/projects?limit=400&org_id=o-1", "/projects"),
    ("/projects/p-1/endpoints", "/projects/*/endpoints"),
    ("/projects/p-1/branches/b-2", "/projects/*/branches/*"),
])
def test_neon_route_blanks_ids_and_drops_the_query(path, route):
    assert rd.neon_route(path) == route


# ── the code's own models must load from the restore branch ──────────────────
from sqlalchemy import Integer, String  # noqa: E402
from sqlalchemy.orm import DeclarativeBase, mapped_column  # noqa: E402


class _TBase(DeclarativeBase):
    pass


class _Project(_TBase):
    __tablename__ = "projects"
    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)


def _schema(tmp_path, name: str, project_cols: str, n_rows: int = 1):
    eng = create_engine(f"sqlite:///{tmp_path / name}.db")
    with eng.begin() as c:
        c.execute(text(f'CREATE TABLE "projects" ({project_cols})'))
        for i in range(n_rows):
            c.execute(text('INSERT INTO "projects" (id) VALUES (:i)'), {"i": i})
    return sessionmaker(bind=eng)


@pytest.mark.parametrize("cols,loads", [
    ("id INTEGER PRIMARY KEY, name TEXT", True),
    ("id INTEGER PRIMARY KEY", "OperationalError"),
])
def test_a_model_loads_only_when_every_mapped_column_is_there(tmp_path, cols, loads):
    f = _schema(tmp_path, "s", cols)
    assert rd.model_reads(f, [_Project]) == {"projects": loads}


@pytest.mark.parametrize("n_rows", [0, 3])
def test_a_model_read_loads_an_empty_or_a_full_table(tmp_path, n_rows):
    f = _schema(tmp_path, "e", "id INTEGER PRIMARY KEY, name TEXT", n_rows)
    assert rd.model_reads(f, [_Project]) == {"projects": True}


@pytest.mark.parametrize("restore,live,want", [
    ({"projects": "OperationalError"}, {"projects": True}, ["projects"]),
    ({"projects": True}, {"projects": True}, []),
    ({"projects": "OperationalError"}, {"projects": "OperationalError"}, []),
    ({}, {"projects": True}, ["projects"]),
])
def test_unservable_is_a_model_live_loads_and_restore_does_not(restore, live, want):
    assert rd.unservable(restore, live) == want


def test_app_models_maps_the_drill_tables_to_the_code_models():
    got = rd.app_models(("projects", "no_such_table", "documents"))
    assert [m.__table__.name for m in got] == ["projects", "documents"]
    assert [m.__table__.name for m in rd.app_models(rd.TABLES)] == list(rd.TABLES)


@pytest.mark.parametrize("restore_cols,ok,unservable", [
    ("id INTEGER PRIMARY KEY, name TEXT", True, []),
    ("id INTEGER PRIMARY KEY", False, ["projects"]),
])
def test_run_fails_a_restore_the_code_cannot_serve(tmp_path, capsys, monkeypatch, restore_cols, ok, unservable):
    monkeypatch.setattr(rd, "TABLES", ("projects",))
    live = _schema(tmp_path, "live", "id INTEGER PRIMARY KEY, name TEXT")
    restore = _schema(tmp_path, "restore", restore_cols)
    r = rd.run("ep-x", restore_factory=restore, live_factory=live,
               chunk_table="projects", models=[_Project])
    assert r["ok"] is ok and r["unservable"] == unservable
    out = capsys.readouterr().out
    assert f"unservable={','.join(unservable) or '-'}" in out
    assert "model=projects" in out and "name" not in out.split("model=projects")[1].split("\n")[0]
