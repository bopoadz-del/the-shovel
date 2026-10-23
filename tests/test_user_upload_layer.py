"""Stage 4 — user uploads land in the user_session layer, never polluting Main.

The operator's rule: an uploaded file goes to the user's own RAG layer, not the
authoritative main corpus. So a user upload is classified ``user_session`` and
carries the LOWEST layer precedence — it surfaces when it's the best semantic
match, but never outranks authoritative Main content (shared_domain / project_
record) at equal cosine.
"""
from app.core.rag import layers as L


def test_user_upload_classifies_user_session_regardless_of_project(monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")
    # even uploaded into a GK-designated project, a user upload is user_session
    layer, _ = L.classify("curated_kb", "my_boq.xlsx", is_user_upload=True)
    assert layer == "user_session"
    layer2, _ = L.classify("example_infra_pack", "notes.pdf", is_user_upload=True)
    assert layer2 == "user_session"


def test_user_session_is_lowest_layer_weight():
    # user layer must not override authoritative Main content
    assert L.layer_weight("user_session") < L.layer_weight("shared_domain")
    assert L.layer_weight("user_session") < L.layer_weight("project_record")
    assert L.layer_weight("user_session") < L.layer_weight("company_rules")


def test_user_upload_does_not_outrank_main_at_equal_cosine():
    # a user's personal upload vs an authoritative shared-domain policy note,
    # equal cosine -> the Main note must win on the precedence bonus
    user_bonus = L.precedence_bonus("user_session", "personal")
    main_bonus = L.precedence_bonus("shared_domain", "policy")
    assert main_bonus > user_bonus


def test_user_upload_still_wins_on_strong_semantic_match():
    # the precedence term is small (tie-breaker), so a user upload that is the
    # far better semantic match still ranks first: cosine dominates.
    user_score = 0.90 + L.precedence_bonus("user_session", "personal")
    main_score = 0.70 + L.precedence_bonus("shared_domain", "policy")
    assert user_score > main_score
