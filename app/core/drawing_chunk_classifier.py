"""
Drawing Number-Table Chunk Classifier (W3 class fix)

Tags drawing dimension-table chunks at ingestion so they cannot ground
cost/rate claims. Drawing tables (e.g., "Item | Description | Quantity | Unit")
are low-semantic content that pollutes numeric queries -- a "450" from a
dimension table can be mistaken for a unit rate.

This is the CLASS fix behind the cost-fabrication incident:
  query: "concrete (250 kg/cm2)" -> drawing dimension tables pulled above
  real rate chunks -> LLM fabricates "450 SAR/m3"

The instance fix (_looks_like_unit + COST_GROUNDING_GATE) is already deployed.
This class fix tags the chunks at source so they are excluded from cost/rate
answering regardless of query construction.

Flag-gated: DRAWING_TABLE_CHUNK_TAGGING env var. OFF = no-op.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set

_ENV_FLAG = os.environ.get("DRAWING_TABLE_CHUNK_TAGGING", "").strip()
TAGGING_ENABLED = _ENV_FLAG in ("1", "true", "True", "TRUE", "yes")

# -- Drawing table detection patterns -------------------------------------

# Headers that strongly indicate a quantity/dimension table (not a rate table)
_DIMENSION_HEADERS = [
    r"\bitem\b.*\bdescription\b.*\bqty\b|\bquantity\b",
    r"\bitem\b.*\bdescription\b.*\bunit\b",
    r"\bno\.?\b.*\bdescription\b.*\bqty\b|\bquantity\b",
    r"\bmark\b.*\bdescription\b.*\bqty\b|\bquantity\b",
    r"\bdrawing\s+no\b.*\bdescription\b",
    r"\bitem\s+no\b.*\bsize\b.*\bunit\b",
    r"\bdescription\b.*\bnumber\b.*\blength\b",
    r"\bmaterial\b.*\bsize\b.*\bqty\b|\bquantity\b",
]

# Keywords that indicate a COST/PRICE table (these should NOT be tagged)
_COST_INDICATORS = [
    r"\brate\b", r"\bprice\b", r"\bcost\b", r"\bamount\b",
    r"\btotal\b.*\bprice\b|\bprice\b.*\btotal\b",
    r"\bunit\s+rate\b", r"\bsum\b", r"\bvalue\b",
    r"\bBOQ\b", r"\bbill\s+of\s+quantities\b",
    r"\bpayment\b", r"\binterim\b.*\bcertificate\b",
]

# Number patterns common in dimension tables (measurements, not prices)
_DIMENSION_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|m2|m3|kg|t|ton|nos?|ea|lm)\b",
    re.IGNORECASE,
)

# Currency patterns that indicate a cost table
_CURRENCY_RE = re.compile(
    r"(?:SAR|USD|EUR|AED|QAR|BHD|OMR|KWD|\$|\xe2\x82\x91|\xd9\x81\xd9\x90)\s*\d+",
    re.IGNORECASE,
)

_HEADER_RE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DIMENSION_HEADERS]
_COST_RE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _COST_INDICATORS]


def _looks_like_dimension_table(text: str) -> bool:
    """
    Heuristic: does this chunk text look like a drawing dimension/quantity table
    (NOT a cost/rate table)?

    Scores on:
    1. Header pattern match (item + description + qty/ unit)
    2. High density of dimension numbers (mm, m, m2, m3, kg) without currency
    3. Absence of cost/price indicators
    """
    text_lower = text.lower()

    # Check for cost indicators first -- if present, do NOT tag as dimension table
    for cost_re in _COST_RE_PATTERNS:
        if cost_re.search(text):
            return False

    # Check for currency symbols -- if present, likely a cost table
    if _CURRENCY_RE.search(text):
        return False

    score = 0

    # Header pattern match (strong signal)
    for header_re in _HEADER_RE_PATTERNS:
        if header_re.search(text):
            score += 3
            break  # Only count once

    # Dimension number density
    dim_matches = len(_DIMENSION_NUMBER_RE.findall(text))
    words = len(text.split())
    if words > 0:
        dim_density = dim_matches / words
        if dim_density > 0.15:  # >15% dimension numbers
            score += 2
        elif dim_matches >= 3:
            score += 1

    # Tabular structure (rows with | or consistent delimiters)
    pipe_rows = [r for r in text.split("\n") if "|" in r]
    if len(pipe_rows) >= 2:
        score += 1

    # Require score >= 3 to classify as dimension table
    return score >= 3


def classify_chunk(chunk_text: str, chunk_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Classify a chunk and return tags.

    Returns dict with:
        - is_drawing_dimension_table: bool
        - exclude_from_cost_grounding: bool
        - classifier_score: int
        - classifier_version: str
    """
    if not TAGGING_ENABLED:
        return {
            "is_drawing_dimension_table": False,
            "exclude_from_cost_grounding": False,
            "classifier_score": 0,
            "classifier_version": "1.0",
            "tagging_enabled": False,
        }

    is_dim_table = _looks_like_dimension_table(chunk_text)

    return {
        "is_drawing_dimension_table": is_dim_table,
        "exclude_from_cost_grounding": is_dim_table,
        "classifier_score": _score_dimension_table(chunk_text),
        "classifier_version": "1.0",
        "tagging_enabled": True,
    }


def _score_dimension_table(text: str) -> int:
    """Return the raw classification score (for debugging/auditing)."""
    score = 0
    for header_re in _HEADER_RE_PATTERNS:
        if header_re.search(text):
            score += 3
            break
    dim_matches = len(_DIMENSION_NUMBER_RE.findall(text))
    words = len(text.split())
    if words > 0 and dim_matches / words > 0.15:
        score += 2
    elif dim_matches >= 3:
        score += 1
    pipe_rows = [r for r in text.split("\n") if "|" in r]
    if len(pipe_rows) >= 2:
        score += 1
    return score


def tag_chunks_for_ingestion(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tag a list of chunks before ingestion. Mutates each chunk dict in place
    to add classification tags.

    Usage in ingestion pipeline:
        chunks = tag_chunks_for_ingestion(raw_chunks)
        # Chunks with exclude_from_cost_grounding=True are filtered from
        # cost/rate retrieval but remain available for other queries.
    """
    if not TAGGING_ENABLED:
        return chunks

    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("content", "")
        result = classify_chunk(text, chunk.get("meta"))
        chunk["_classifier"] = result

    return chunks


def filter_cost_grounding_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out drawing dimension-table chunks from a retrieval result set
    when the query is cost/rate related.

    Usage in cost-grounding gate:
        retrieved = rag_search(query, ...)
        safe_chunks = filter_cost_grounding_chunks(retrieved)
    """
    if not TAGGING_ENABLED:
        return chunks

    filtered = []
    for chunk in chunks:
        classifier = chunk.get("_classifier", {})
        if classifier.get("exclude_from_cost_grounding"):
            continue
        filtered.append(chunk)

    return filtered
