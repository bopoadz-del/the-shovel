"""Stage 1 / Task 2 — chunks.knowledge_layer + chunks.authority columns.

Persisted on both the static ``RagChunk`` (legacy ``chunks``) and every
dynamic ``chunks_<ns>`` class (prod writes to ``chunks_v2``). Nullable so
pre-migration rows and the flag-off path read as unlayered. ``knowledge_layer``
is deliberately NOT named ``layer`` — the Chunk dataclass already uses ``layer``
for the per-query STEP 0 isolation tag.
"""
from app.core.models import RagChunk, make_rag_chunk_class


def test_ragchunk_has_knowledge_layer_and_authority():
    cols = {c.name for c in RagChunk.__table__.columns}
    assert "knowledge_layer" in cols
    assert "authority" in cols
    # must NOT clash with the STEP 0 retrieval isolation tag name
    assert "layer" not in cols


def test_layer_columns_are_nullable():
    by = {c.name: c for c in RagChunk.__table__.columns}
    assert by["knowledge_layer"].nullable is True
    assert by["authority"].nullable is True


def test_namespaced_chunk_class_also_carries_columns():
    """Dynamic chunks_<ns> classes carry the layer columns, like RagChunk does.

    Uses a namespace of its own rather than "v2". The claim under test is about
    NAMESPACED classes in general, not about v2 specifically -- and pinning
    "v2" at dim 256 encoded a contradiction, because production maps v2 at 384
    (bge-small). In a full-suite run the live embedder registers ("v2", 384)
    first, so this then asked for a second width on the same namespace. That
    passed in isolation and failed only in a full run.
    """
    cls = make_rag_chunk_class("layercols_probe", 256, "fake-model")
    cols = {c.name for c in cls.__table__.columns}
    assert "knowledge_layer" in cols
    assert "authority" in cols


def test_one_namespace_cannot_be_remapped_to_a_second_dim():
    """A namespace owns one table, so it owns one embedding width.

    Keying the class cache by (namespace, dim) allowed two classes per
    namespace; the second declaration then hit
    "Table 'chunks_v2' is already defined for this MetaData instance".
    Refuse the remap with a readable error instead of a SQLAlchemy crash --
    and never hand back a class whose vector width differs from the request.
    """
    import pytest

    ns = "dimconflict_probe"
    first = make_rag_chunk_class(ns, 256, "fake-model")
    assert make_rag_chunk_class(ns, 256, "other-model") is first, (
        "same namespace + same dim must reuse the cached class; model_name is "
        "metadata only"
    )

    with pytest.raises(ValueError, match="already mapped at dim"):
        make_rag_chunk_class(ns, 384, "fake-model")
