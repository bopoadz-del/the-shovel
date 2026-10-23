"""Guard tests for rag_chunk_table_name — the single chokepoint for the RAG
chunk table identifier that is interpolated into raw SQL (``FROM {table}``).

Namespaces are internal today, but the identifier cannot be parameterised, so
the chokepoint must reject anything outside a safe identifier charset. This
locks in the defence-in-depth fix for the dynamic-``FROM`` sites in the audit.
"""

from __future__ import annotations

import pytest

from app.core.models import rag_chunk_table_name


def test_empty_namespace_is_legacy_chunks_table():
    assert rag_chunk_table_name("") == "chunks"


@pytest.mark.parametrize(
    "namespace",
    [
        "curated_kb",
        "drive_archive",
        "example_infra_pack",
        "projects_folder",
        "ABC123_x",
    ],
)
def test_valid_internal_namespaces_pass(namespace):
    assert rag_chunk_table_name(namespace) == f"chunks_{namespace}"


@pytest.mark.parametrize(
    "namespace",
    [
        "chunks; DROP TABLE users",
        "a b",
        "a-b",
        "a.b",
        "a'b",
        'a"b',
        "a)b",
        "a\nb",
        "évil",
        # Trailing-newline bypass: re `$` matches before a final newline, so a
        # `match()`-based guard accepted "foo\n" and leaked \n into the table
        # identifier. fullmatch closes it. Regression guard.
        "foo\n",
        "foo\r",
        "foo\r\n",
        "foo\n; DROP TABLE users",
        "\n",
    ],
)
def test_unsafe_namespace_raises(namespace):
    with pytest.raises(ValueError, match="unsafe RAG namespace"):
        rag_chunk_table_name(namespace)
