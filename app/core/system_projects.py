"""System namespaces: the projects the PRODUCT itself creates or defaults to.

A system project is not a customer's project. It is a namespace the platform
owns -- the general-knowledge layer it seeds from ``docs/knowledge/`` at boot.
Each is declared here ONCE and every other
module refers to it by the constant, never by the literal.

Customer project ids never belong here. They live in the database and arrive
at run time; product code, prompts and config name none of them.
``scripts/scan_hardwiring.py`` reads ``SYSTEM_PROJECTS`` from this file at run
time: a declared id is exempt from the PROJECT-ID leakage check only inside
this file, and any other live project id fails wherever it appears.

Deployments may point a role at a different id through its env var (for
example ``RAG_GENERAL_KNOWLEDGE_PROJECTS``); the values below are the product
defaults those env vars fall back to.
"""
from __future__ import annotations

import os

#: Default general-knowledge project: ``docs/knowledge/*.md`` is seeded into it
#: at boot (``app.core.knowledge_seed``) and it is merged into every project's
#: retrieval. Overridden by ``RAG_GENERAL_KNOWLEDGE_PROJECTS``.
GENERAL_KNOWLEDGE_PROJECT_DEFAULT = "training_material"

SYSTEM_PROJECTS: dict[str, str] = {
    GENERAL_KNOWLEDGE_PROJECT_DEFAULT: "general-knowledge layer seeded from docs/knowledge at boot",
}


def general_knowledge_env() -> str:
    """Raw ``RAG_GENERAL_KNOWLEDGE_PROJECTS``, defaulting to the product's GK project."""
    return os.getenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GENERAL_KNOWLEDGE_PROJECT_DEFAULT)


def primary_general_knowledge_project() -> str:
    """First configured general-knowledge project id ('' when none is configured)."""
    return next((p.strip() for p in general_knowledge_env().split(",") if p.strip()), "")
