"""Construction Advisor — natural-language access to the construction KB.

Maps a free-text construction question to the relevant curated knowledge-base
rule(s) (app/knowledge/construction_kb.json) and returns each with its
provenance, credibility tier, and the loader's "verify against your spec"
warnings. If the caller supplies formula values, the top formula match is
evaluated. Every answer is a CITED rule — never an unsourced claim.
"""
from typing import Any, Dict

from app.core.universal_base import UniversalBlock
from app.blocks import _knowledge as kb


class ConstructionAdvisorBlock(UniversalBlock):
    name = "construction_advisor"
    description = (
        "Answer construction-engineering questions from the curated knowledge "
        "base (buildings/concrete/roads-earthworks-geotech/procurement), "
        "returning cited rules with provenance + credibility tier + warnings"
    )

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        query = self._query(input_data, params)
        if not query.strip():
            return {"status": "error", "error": "no query provided"}
        domain = params.get("domain")
        top_k = int(params.get("top_k", 5))

        matches = kb.search_knowledge(query, top_k=top_k, domain=domain)
        results = []
        for e in matches:
            results.append({
                "id": e.get("id"),
                "type": e.get("type"),
                "title": e.get("title"),
                "statement": e.get("statement"),
                "domain": e.get("domain"),
                "expression": e.get("expression"),
                "thresholds": e.get("thresholds"),
                "provenance": e.get("provenance", {}),
                "credibility_tier": e.get("credibility_tier"),
                "warnings": kb._build_warnings(e),
            })

        out: Dict[str, Any] = {
            "status": "success",
            "query": query,
            "matches": results,
            "count": len(results),
        }
        if not results:
            out["note"] = "no matching construction-KB rule for that query"

        # Optional: if the caller supplied variable values and the top match is
        # a formula, evaluate it (cited + tier + warnings included).
        values = params.get("values")
        if isinstance(input_data, dict) and not values:
            values = input_data.get("values")
        if values and matches and matches[0].get("type") == "formula":
            try:
                out["evaluation"] = kb.evaluate(matches[0]["id"], **values)
            except Exception as exc:  # noqa: BLE001
                out["evaluation_error"] = str(exc)[:120]
        return out

    @staticmethod
    def _query(input_data: Any, params: Dict) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            return (
                input_data.get("query")
                or input_data.get("text")
                or input_data.get("input")
                or params.get("query")
                or ""
            )
        return params.get("query", "") or ""
