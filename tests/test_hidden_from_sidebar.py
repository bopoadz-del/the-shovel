"""Stage 5 — hidden_from_sidebar drops a project from the sidebar, keeps it live.

A hidden project (RAG corpus / GK / eval) does not appear in list_projects but
its row and chunks are untouched (still retrievable). Reversible; include_hidden
enumerates it for internal tooling.
"""
def test_hidden_project_excluded_from_sidebar_but_enumerable():
    from app.core import projects as store
    store.create_project("Visible s5", user_id="system", project_id="p_vis_s5")
    store.create_project("Corpus s5", user_id="system", project_id="p_corpus_s5")

    # both visible initially
    ids = {p["id"] for p in store.list_projects()}
    assert {"p_vis_s5", "p_corpus_s5"} <= ids

    assert store.set_hidden_from_sidebar("p_corpus_s5", True) is True

    sidebar = {p["id"] for p in store.list_projects()}
    assert "p_vis_s5" in sidebar
    assert "p_corpus_s5" not in sidebar          # hidden from sidebar

    # still enumerable + row intact (retrievable)
    all_ids = {p["id"] for p in store.list_projects(include_hidden=True)}
    assert "p_corpus_s5" in all_ids
    assert store.get_project("p_corpus_s5") is not None

    # reversible
    assert store.set_hidden_from_sidebar("p_corpus_s5", False) is True
    assert "p_corpus_s5" in {p["id"] for p in store.list_projects()}


def test_set_hidden_unknown_project_returns_false():
    from app.core import projects as store
    assert store.set_hidden_from_sidebar("nope_s5", True) is False
