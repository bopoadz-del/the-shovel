"""System namespaces are declared once; every reader resolves them from there."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from app.core import knowledge_seed, projects, system_projects

ROOT = Path(__file__).resolve().parent.parent


def test_general_knowledge_default_comes_from_the_registry(monkeypatch):
    monkeypatch.delenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", raising=False)
    gk = system_projects.GENERAL_KNOWLEDGE_PROJECT_DEFAULT
    assert gk in system_projects.SYSTEM_PROJECTS
    assert projects.general_knowledge_project_ids() == frozenset({gk})
    assert knowledge_seed._gk_project_id() == gk


def test_env_overrides_every_reader(monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", " zz_kb_one , zz_kb_two ,")
    assert system_projects.primary_general_knowledge_project() == "zz_kb_one"
    assert projects.general_knowledge_project_ids() == frozenset({"zz_kb_one", "zz_kb_two"})
    assert knowledge_seed._gk_project_id() == "zz_kb_one"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", " , ")
    assert system_projects.primary_general_knowledge_project() == ""


def test_scanner_reads_the_same_registry():
    spec = importlib.util.spec_from_file_location("scan_hw_reg", ROOT / "scripts" / "scan_hardwiring.py")
    sh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sh)
    assert sh.system_project_ids() == set(system_projects.SYSTEM_PROJECTS)
