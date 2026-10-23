"""Stage 4 disclosure — a user_session chunk reads 'Your upload' in Sources.

knowledge_layer flows inject audit -> runtime sources panel; a user-upload
chunk is labelled distinctly from Master/KB/project sources. None when unlayered
(flag off), so today's labels are unchanged.
"""
from app.agents.runtime import _build_sources_from_audit


def _audit(knowledge_layer):
    return {
        "project_id": "p1",
        "chunks": [{
            "doc_id": "doc-x", "chunk_index": 0, "chunk_id": "cx",
            "project_id": "p1", "score": 0.8, "layer": "own",
            "knowledge_layer": knowledge_layer,
        }],
    }


def test_user_upload_labelled_your_upload():
    src = _build_sources_from_audit(_audit("user_session"), final_text="")
    assert src and src[0]["layer_label"] == "Your upload"


def test_unlayered_keeps_project_document_label():
    src = _build_sources_from_audit(_audit(None), final_text="")
    assert src and src[0]["layer_label"] == "Project document"
