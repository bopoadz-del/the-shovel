"""scrub_verify: synthetic rules and rows only -- never a real identifier."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scrub_verify as sv  # noqa: E402


def test_a_short_identifier_inside_a_longer_word_is_never_replaced():
    rule = sv.whole_word_python("abel")
    assert rule.sub("the project", "labelled cable ABEL, abel.") == "labelled cable the project, the project."
    assert rule.sub("the project", "Abelton") == "Abelton"


def test_whole_word_wraps_the_postgres_pattern_on_both_sides():
    w = sv.whole_word(sv.pg_pattern(r"\bZQX\b"))
    assert w.startswith("(?<![[:alnum:]_])(?:") and w.endswith(")(?![[:alnum:]_])")
    assert r"\yZQX\y" in w


def test_rules_parse_in_the_order_the_scrubber_applied_them():
    rules = sv.parse_rules("ZQ => a\\nZQXLONG => b\n# note\nPLAIN")
    assert rules == [("ZQXLONG", "b"), ("PLAIN", "the project"), ("ZQ", "a")]


def test_python_only_rules_and_backreferences_are_refused():
    with pytest.raises(sv.Refused) as r:
        sv.translated([("(?P<x>ZQ)", "a"), ("ZQ", r"\1")])
    assert r.value.fields["rules_unsupported"] == 2


def test_restore_url_swaps_only_the_endpoint():
    live = "postgresql+psycopg://u:p@ep-live-123-pooler.us-west-2.aws.neon.tech/db?sslmode=require"
    out = sv.restore_url(live, "ep-soft-test-9")
    assert out == "postgresql+psycopg://u:p@ep-soft-test-9.us-west-2.aws.neon.tech/db?sslmode=require"
    with pytest.raises(sv.Refused):
        sv.restore_url(live, "evil.example.com")


_PG = os.getenv("DATABASE_URL", "").startswith("postgresql")


@pytest.mark.skipif(not _PG, reason="needs DATABASE_URL on PostgreSQL")
def test_postgres_whole_word_never_cuts_into_a_word():
    from sqlalchemy import text as sql

    from app.core.db import _session_factory_for_url

    factory = _session_factory_for_url(os.environ["DATABASE_URL"])
    p = sv.pg_pattern("abel")
    with factory() as s:
        as_run, whole = s.execute(sql(
            "SELECT regexp_replace(:t, :p, 'X', 'gi'), regexp_replace(:t, :w, 'X', 'gi')"),
            {"t": "labelled ABEL cable", "p": p, "w": sv.whole_word(p)}).one()
        counts = s.execute(sql("SELECT regexp_count(:t, :p, 1, 'i'), regexp_count(:t, :w, 1, 'i')"),
                           {"t": "labelled ABEL cable", "p": p, "w": sv.whole_word(p)}).one()
    assert as_run == "lXled X cable"
    assert whole == "labelled X cable"
    assert tuple(counts) == (2, 1)
