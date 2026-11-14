"""Control flow of the bounded BM25 leg, without a database.

``test_bm25_bounded_is_exact`` proves on PostgreSQL that the bounded leg
returns what ranking every match returns. These pin, on any backend, WHEN it
answers from the bounded candidates and when it must fall back to the full
scan -- the decisions that keep it exact.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.rag import vector_store as vs


class _Result:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar


class _Session:
    """Answers the statements ``_bm25_postgres_rows`` sends, and records them."""

    def __init__(self, tsquery, rounds=(), full=(), stats=None, stats_fail=False):
        self.tsquery = tsquery
        self.rounds = list(rounds)  # one row list per bounded round, in order
        self.full = list(full)
        self.stats = stats
        self.stats_fail = stats_fail
        self.sent = []
        self.rolled_back = False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "::text" in sql and "websearch_to_tsquery" in sql and "FROM" not in sql:
            self.sent.append("tsquery")
            return _Result(scalar=self.tsquery)
        if "pg_stats" in sql:
            self.sent.append("stats")
            if self.stats_fail:
                raise SQLAlchemyError("no stats")
            return _Result(rows=[self.stats] if self.stats else [])
        if "CAST(:cand AS tsquery)" in sql:
            self.sent.append(("bounded", params["cand"]))
            return _Result(rows=self.rounds.pop(0) if self.rounds else [])
        self.sent.append("full")
        return _Result(rows=self.full)

    def rollback(self):
        self.rolled_back = True


def _store():
    st = object.__new__(vs.VectorStore)
    st._table_name = "chunks_test"
    return st


def _rows(ranks):
    return [SimpleNamespace(chunk_id=f"c{i}", rank=r) for i, r in enumerate(ranks)]


def _tsq(lexemes):
    return " | ".join(f"'{x}'" for x in lexemes)


def _first_bound(lexemes, stats=None):
    """The bound the first bounded round must prove, as the store computes it."""
    freq = dict(zip(*stats)) if stats else {}
    w = vs.lexeme_weights(lexemes, freq)
    n = len(lexemes)
    t = max(2, min(vs._BM25_START_THRESHOLD, (n + 1) // 2))
    return vs.bm25_excluded_bound(lexemes, w, t, 1) * (1 + 2 * vs._BM25_BOUND_MARGIN)


def test_stopwords_only_reads_nothing():
    s = _Session(tsquery="")
    assert _store()._bm25_postgres_rows(s, "p", "the or of", 50, "") == []
    assert s.sent == ["tsquery"]


@pytest.mark.parametrize("tsq", ["'concret'", "'a' <-> 'b' | 'c'"])
def test_one_lexeme_or_a_phrase_takes_the_full_scan(tsq):
    s = _Session(tsquery=tsq, full=_rows([0.1]))
    got = _store()._bm25_postgres_rows(s, "p", "q", 5, "")
    assert [r.chunk_id for r in got] == ["c0"]
    assert [x for x in s.sent if x != "stats"] == ["tsquery", "full"]


@pytest.mark.parametrize("stats", [None, (["l3", "l0"], [0.5, 0.01])])
def test_a_proving_first_round_is_the_answer(stats):
    lex = [f"l{i}" for i in range(7)]
    kth = _first_bound(lex, stats)  # just over the bound: nothing outside can beat it
    s = _Session(tsquery=_tsq(lex), rounds=[_rows([0.09, 0.08, kth])], stats=stats)
    got = _store()._bm25_postgres_rows(s, "p", "q", 3, "")
    assert len(got) == 3
    assert s.sent[0] == "tsquery" and "stats" in s.sent
    assert [x for x in s.sent if x == "full"] == []
    # The first round admits the rarest lexeme on its own.
    rarest = vs.rarest_first(lex, vs.lexeme_weights(lex, dict(zip(*stats)) if stats else {}))[0]
    assert f"('{rarest}')" in s.sent[-1][1]


def test_an_unproven_round_widens_then_proves():
    lex = [f"l{i}" for i in range(7)]
    weak = _first_bound(lex) * (1 - 1e-6)  # fails the first round's bound
    s = _Session(
        tsquery=_tsq(lex),
        rounds=[_rows([0.09, weak]), _rows([0.09, weak])],
    )
    got = _store()._bm25_postgres_rows(s, "p", "q", 2, "")
    assert len(got) == 2
    bounded = [x for x in s.sent if isinstance(x, tuple)]
    assert len(bounded) == 2 and "full" not in s.sent
    # The second round admits more rare lexemes on their own than the first.
    assert bounded[1][1].count(") | (") >= 1
    assert sum(1 for g in bounded[1][1].split(" | ") if "&" not in g) > sum(
        1 for g in bounded[0][1].split(" | ") if "&" not in g
    )


def test_short_rounds_fall_back_to_the_full_scan():
    lex = [f"l{i}" for i in range(9)]
    s = _Session(
        tsquery=_tsq(lex),
        rounds=[_rows([0.05]), _rows([0.05])],  # fewer than k each time
        full=_rows([0.05, 0.01]),
    )
    got = _store()._bm25_postgres_rows(s, "p", "q", 2, "")
    assert [r.chunk_id for r in got] == ["c0", "c1"]
    bounded = [x[1] for x in s.sent if isinstance(x, tuple)]
    assert len(bounded) == vs._BM25_MAX_SHORT_ROUNDS
    # A short round proves nothing: the next admits ONE more rare lexeme alone.
    singles = [sum(1 for g in b.split(" | ") if "&" not in g) for b in bounded]
    assert singles == [1, 2], bounded
    assert s.sent[-1] == "full"


def test_a_bound_that_never_holds_ends_in_the_full_scan():
    lex = [f"l{i}" for i in range(9)]
    s = _Session(
        tsquery=_tsq(lex),
        rounds=[_rows([1e-9, 1e-9])] * 10,
        full=_rows([0.2]),
    )
    got = _store()._bm25_postgres_rows(s, "p", "q", 2, "")
    assert [r.chunk_id for r in got] == ["c0"]
    assert s.sent[-1] == "full"


@pytest.mark.parametrize("stats", [(["l1"], [0.3]), (["l0", "l2"], [0.9, 0.05])])
def test_the_score_is_sent_the_query_weights(stats):
    lex = ["l0", "l1", "l2"]
    s = _Session(tsquery=_tsq(lex), rounds=[], full=_rows([0.05]), stats=stats)
    params = {}
    orig = s.execute

    def _capture(stmt, p=None):
        if p and "lw" in p:
            params.update(p)
        return orig(stmt, p)

    s.execute = _capture
    _store()._bm25_postgres_rows(s, "p", "q", 5, "")
    w = vs.lexeme_weights(lex, dict(zip(*stats)))
    assert params["lx"] == lex
    assert params["lw"] == [w[x] for x in lex]
    assert params["scale"] == pytest.approx(vs._LEXICAL_SCORE_SCALE / sum(w.values()))


def test_lexeme_statistics_are_cached_and_never_required():
    st = _store()
    s = _Session(tsquery="", stats=(["concret", "cover"], [0.06, 0.05]))
    assert st._lexeme_frequencies(s) == {"concret": 0.06, "cover": 0.05}
    assert st._lexeme_frequencies(s) == {"concret": 0.06, "cover": 0.05}
    assert s.sent.count("stats") == 1

    broken = _Session(tsquery="", stats_fail=True)
    assert _store()._lexeme_frequencies(broken) == {}
    assert broken.rolled_back


def test_rarest_lexemes_carry_the_candidate_query():
    # The statistics list runs below "mid" ("tail"), so the unlisted "rare"
    # takes that floor and is the rarest.
    w = vs.lexeme_weights(
        ["common", "rare", "mid"], {"common": 0.9, "mid": 0.1, "tail": 0.01},
    )
    tsq = vs.bm25_candidate_tsquery(["common", "rare", "mid"], 3, w)
    # t = n: only the full conjunction can hold all three.
    assert tsq == "('rare' & 'mid' & 'common')"
    # One rare lexeme admits alone; the other two need t of them.
    assert vs.bm25_candidate_tsquery(["common", "rare", "mid"], 2, w, 1) == (
        "('rare') | ('mid' & 'common')"
    )
    assert vs.bm25_candidate_tsquery([], 1) is None
    assert vs.bm25_candidate_tsquery(["a"], 2) is None


class _RecEngine:
    def __init__(self, has_ext):
        self.executed = []
        self._has_ext = has_ext

    def begin(self):
        eng = self

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, stmt):
                eng.executed.append(str(stmt))
                return _Result(rows=[1] if eng._has_ext else [])

        return _Ctx()


@pytest.mark.parametrize("has_ext", [True, False])
def test_trigram_index_is_ensured(has_ext):
    eng = _RecEngine(has_ext)
    vs._ensure_trigram_index(eng, "chunks_v2")
    assert any(
        "chunks_v2_text_trgm" in s and "gin_trgm_ops" in s and "IF NOT EXISTS" in s
        for s in eng.executed
    ), eng.executed
    assert any("CREATE EXTENSION" in s for s in eng.executed) is (not has_ext)


def test_trigram_index_never_raises():
    class _Bad:
        def begin(self):
            raise RuntimeError("connection dropped mid-DDL")

    vs._ensure_trigram_index(_Bad(), "chunks_v2")
