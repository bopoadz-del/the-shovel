"""Layered-RAG vocabulary + flag (docs/rag-deployment-plan.md).

Layers are the WHERE of knowledge; authority is the HOW-MUCH-IT-WINS.
Both are persisted per chunk (Alembic 0011) and default-inert until
RAG_LAYERED is set. See docs/superpowers/plans/2026-07-23-layered-rag.md.
"""
from __future__ import annotations

import os
import re

from app.core.rag import revision as _revision

# L1 shared domain, L2A company/client rules, L2B live project record,
# L3 user/session. Names are the persisted ``chunks.layer`` values.
LAYERS = frozenset({"shared_domain", "company_rules", "project_record", "user_session"})
DEFAULT_LAYER = "shared_domain"

# Cross-cutting authority, highest precedence first. Contract beats meeting
# note; approved drawing beats draft; project BOQ beats generic benchmark.
AUTHORITIES = ("contractual", "design", "commercial", "operational",
               "policy", "historical", "personal")
DEFAULT_AUTHORITY = "operational"

# Legacy retrieval-time tags (STEP 0) map onto persisted layers so old
# rows and mixed corpora keep working during the migration.
LEGACY_LAYER_MAP = {
    "own": "project_record",
    "general_knowledge": "shared_domain",
    "master_corpus": "project_record",
    "user": "user_session",
}


def layered_enabled() -> bool:
    """Read RAG_LAYERED live so operators/tests can flip it without a
    re-import. Truthy: 1/true/yes/on (case-insensitive). Default: OFF."""
    return str(os.getenv("RAG_LAYERED", "")).strip().lower() in {"1", "true", "yes", "on"}


def authority_rank(name: str) -> int:
    """0 = strongest precedence; unknown authority sorts weakest."""
    try:
        return AUTHORITIES.index(name)
    except ValueError:
        return len(AUTHORITIES)


# ── ingest-time classification ─────────────────────────────────────────────

def _gk_project_ids() -> set:
    """Projects treated as the shared general-knowledge layer (env-configured)."""
    raw = os.getenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    return {p.strip() for p in raw.split(",") if p.strip()}

# A controlled company document (procedure, template, form) is named with a
# leading document code followed by its title: "XYZ-101_Change Control.pdf",
# "qrs_330 HSE Audit.docx". The shape is the signal -- no organisation's code
# prefix is listed. A contract id ("AB-2001-101_Vol 1") is not this shape: its
# digits run into another hyphenated group, not a title, and a year
# in a name ("<standard>_2017_<topic>") is not a document number.
_CONTROLLED_DOC_CODE = r"^\s*[a-z]{2,6}[\s_-](?!(?:19|20)\d\d(?!\d))\d{3,4}[a-z]?[\s_.-]+[a-z(]"

# Doc-name -> authority, checked in this order (first hit wins). The label is
# what the doc IS, not where it lives: a priced BOQ is commercial whether it's a
# project doc or a reference. Patterns are deliberately conservative; anything
# unmatched falls back to a layer-appropriate default (see classify()).
_AUTHORITY_PATTERNS = (
    ("contractual", r"contract|agreement|conditions of contract|fidic|variation|"
                    r"\bvo\b|\bclaim\b|particular conditions|general conditions"),
    ("design", r"drawing|\bdwg\b|\bdxf\b|\bdgn\b|\bifc\b|\bga\b|layout|"
               r"\bplan\b|detail|\bsection\b|elevation|\bdesign\b"),
    ("commercial", r"\bboq\b|bill of quant|\brate\b|priced|tender|\bcost\b|"
                   r"estimat|valuation|\bipc\b|\bipa\b|payment|budget|cash[\s_-]?flow"),
    ("policy", _CONTROLLED_DOC_CODE + r"|procedure|policy|method statement|"
               r"\bmethod\b|specification|\bspec\b|standard|\bcode\b|manual|"
               r"guideline|template|workflow|checklist"),
    ("operational", r"report|\blog\b|minutes|daily|weekly|\brfi\b|\bncr\b|"
                    r"inspection|submittal|schedule|programme|progress|transmittal"),
    # 'superseded' keyword set is owned by app.core.rag.revision (the always-on
    # revision-currency signal) so the flag-gated layered down-weight here and
    # the always-on penalty there can never disagree on what superseded means.
    ("historical", _revision.SUPERSEDED_PATTERN),
    ("personal", r"\bdraft\b|\bmemo\b|\bnote\b|working|scratch|personal"),
)

# When no doc-name keyword matches, the layer implies the authority: reference /
# rules content is policy-grade; project and user content is operational.
_LAYER_DEFAULT_AUTHORITY = {
    "shared_domain": "policy",
    "company_rules": "policy",
    "project_record": "operational",
    "user_session": "operational",
}

_COMPANY_RULE_NAME_RE = re.compile(
    _CONTROLLED_DOC_CODE + r"|procedure|template|policy|workflow|guideline",
    re.IGNORECASE,
)


def _classify_authority(name: str) -> str:
    for authority, pattern in _AUTHORITY_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return authority
    return ""


def classify(project_id, doc_name, *, is_user_upload=False):
    """Return ``(knowledge_layer, authority)`` for a doc at ingest time.

    Layer: user uploads -> ``user_session``; docs in a general-knowledge project
    -> ``shared_domain`` (or ``company_rules`` when the name reads as a
    procedure/template/policy); everything else -> ``project_record``. Authority
    comes from doc-name keywords, else the layer's default. Pure and side-effect
    free so it is trivially testable and safe to call on every ingest."""
    name = doc_name or ""
    if is_user_upload:
        layer = "user_session"
    elif project_id in _gk_project_ids():
        layer = "company_rules" if _COMPANY_RULE_NAME_RE.search(name) else "shared_domain"
    else:
        layer = "project_record"
    authority = _classify_authority(name) or _LAYER_DEFAULT_AUTHORITY[layer]
    return layer, authority


# ── retrieval-time authority precedence (Stage 3) ──────────────────────────
# How much each layer counts in the precedence bonus. The live project record
# dominates ("what does this project say"); company rules and shared general
# knowledge are the authoritative Main corpus. The user's own session (uploads)
# is LAST: it surfaces on semantic match but never outranks authoritative Main
# content at equal cosine — the operator's "uploads land in the user's RAG, not
# the main RAG" anti-pollution rule (docs/rag-deployment-plan.md L3: "never
# override project documents").
_LAYER_WEIGHT = {
    "project_record": 1.0,
    "company_rules": 0.7,
    "shared_domain": 0.5,
    "user_session": 0.3,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def layer_weight(layer) -> float:
    return _LAYER_WEIGHT.get(layer or "", 0.0)


def authority_weight(authority) -> float:
    """1.0 for the strongest authority (contractual), descending to ~1/N for the
    weakest; 0.0 for unknown/None."""
    r = authority_rank(authority or "")
    n = len(AUTHORITIES)
    return (n - r) / n if r < n else 0.0


def precedence_bonus(knowledge_layer, authority) -> float:
    """Small additive re-rank term so a higher-authority / higher-layer chunk
    outranks a comparably-relevant low-authority one among the fetched pool.
    Weights are env-tunable (RAG_AUTHORITY_WEIGHT / RAG_LAYER_WEIGHT) and default
    small: the term breaks ties and nudges, it does not override cosine."""
    w_auth = _env_float("RAG_AUTHORITY_WEIGHT", 0.05)
    w_layer = _env_float("RAG_LAYER_WEIGHT", 0.05)
    return w_auth * authority_weight(authority) + w_layer * layer_weight(knowledge_layer)
