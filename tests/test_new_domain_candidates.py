"""S-7: the new-domain-candidate log is tenant-scoped, append-only, input-derived."""
from __future__ import annotations

from app.lib import new_domain_candidates as ndc


def test_rows_are_input_derived_and_appended(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ndc.log_new_domain_candidate("acme", "compute the thermal bridging psi-value",
                                 agent="construction-pm", sources=["ISO 14683"])
    ndc.log_new_domain_candidate("acme", "size the attenuation tank for a 1-in-100 storm")
    rows = ndc.read_new_domain_candidates("acme")
    assert len(rows) == 2
    # the row carries the actual question it received, not a fixed string
    qs = [r["question"] for r in rows]
    assert "thermal bridging" in qs[0] and "attenuation tank" in qs[1]
    assert qs[0] != qs[1]
    assert rows[0]["agent"] == "construction-pm" and rows[0]["sources"] == ["ISO 14683"]
    assert all(r["tenant"] == "acme" for r in rows)


def test_tenant_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ndc.log_new_domain_candidate("tenant-a", "a-only ask")
    ndc.log_new_domain_candidate("tenant-b", "b-only ask")
    a = ndc.read_new_domain_candidates("tenant-a")
    b = ndc.read_new_domain_candidates("tenant-b")
    assert [r["question"] for r in a] == ["a-only ask"]
    assert [r["question"] for r in b] == ["b-only ask"]


def test_read_before_any_write_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert ndc.read_new_domain_candidates("never-seen") == []


def test_tenant_slug_is_filesystem_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # a client-named / path-y tenant id must not escape the log dir
    row = ndc.log_new_domain_candidate("../../etc/passwd", "x")
    assert "/" not in row["tenant"] and ".." not in row["tenant"]
    assert ndc.read_new_domain_candidates("../../etc/passwd")[0]["question"] == "x"
