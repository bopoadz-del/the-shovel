"""Guard for scripts/rag_backfill_platform_index.py.

The Render-era R2 backfill pipeline (rag_backfill_loader / _batch_loader /
_local / _batch_local / _local_platform) was removed with the R2 archive:
the platform keeps no copy of originals and has no object store. Its loader
tests went with it; this guard covers the script that remains.
"""
from __future__ import annotations


def test_platform_index_imports_resolve_and_never_app_db():
    """Guard against regressing to the non-existent app.db package: every
    ``app.*`` module the script imports must exist."""
    import ast
    import importlib.util
    from pathlib import Path

    src = Path("scripts/rag_backfill_platform_index.py").read_text(encoding="utf-8")
    assert "app.db" not in src
    modules = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app"):
            modules.add(node.module)
            for alias in node.names:
                if importlib.util.find_spec(f"{node.module}.{alias.name}") is not None:
                    modules.add(f"{node.module}.{alias.name}")
    assert "app.core" in modules
    assert [m for m in modules if importlib.util.find_spec(m) is None] == []


def test_platform_index_writes_the_file_path_through_projects():
    """The row's file_path update goes through the shared projects helper,
    not a private session that could drift from the model."""
    from pathlib import Path

    src = Path("scripts/rag_backfill_platform_index.py").read_text(encoding="utf-8")
    assert "projects_mod.set_document_file_path(" in src
    assert "SessionLocal" not in src
