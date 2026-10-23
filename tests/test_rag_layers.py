"""Stage 1 / Task 1 — layered-RAG vocabulary + RAG_LAYERED flag.

Locks the persisted layer names, the authority precedence order, and the
default-OFF flag so the cloud dev/demo path is byte-unchanged until an
operator sets RAG_LAYERED. See docs/rag-deployment-plan.md.
"""
from app.core.rag import layers as L


def test_layer_and_authority_vocab():
    assert L.LAYERS == frozenset(
        {"shared_domain", "company_rules", "project_record", "user_session"})
    assert L.AUTHORITIES == (
        "contractual", "design", "commercial", "operational",
        "policy", "historical", "personal")


def test_authority_rank_precedence():
    # contract beats meeting note (policy) beats a personal note
    assert L.authority_rank("contractual") < L.authority_rank("policy")
    assert L.authority_rank("policy") < L.authority_rank("personal")
    assert L.authority_rank("does-not-exist") == len(L.AUTHORITIES)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    assert L.layered_enabled() is False
    monkeypatch.setenv("RAG_LAYERED", "1")
    assert L.layered_enabled() is True
    monkeypatch.setenv("RAG_LAYERED", "off")
    assert L.layered_enabled() is False
