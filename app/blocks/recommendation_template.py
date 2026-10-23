"""Recommendation Template Block - Rule-based recommendation generation from variance data"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)


# ── Built-in rule + template database ─────────────────────────────────────────
_RULE_DB: Dict[str, Dict] = {
    # Cost variance rules
    "cost_over_critical": {
        "condition": {"field": "variance_pct", "op": "gt", "value": 20},
        "template": "CRITICAL OVERRUN: {item} exceeds benchmark by {variance_pct:.1f}%. Estimated excess cost: {cost_impact_usd:,.0f} USD. Immediate action required.",
        "severity": "critical",
        "action_items": [
            "Suspend procurement pending cost review",
            "Issue Re-Tender to minimum 3 pre-qualified suppliers",
            "Escalate to Project Director within 24 hours",
            "Prepare Variation Order if scope changed",
        ],
        "category": "cost",
    },
    "cost_over_high": {
        "condition": {"field": "variance_pct", "op": "between", "value": [10, 20]},
        "template": "HIGH VARIANCE: {item} is {variance_pct:.1f}% above benchmark. Review pricing and negotiate.",
        "severity": "high",
        "action_items": [
            "Request detailed cost breakdown from supplier",
            "Check current market index (ENR, MEED)",
            "Negotiate volume discount or phased delivery",
        ],
        "category": "cost",
    },
    "cost_over_medium": {
        "condition": {"field": "variance_pct", "op": "between", "value": [5, 10]},
        "template": "MEDIUM VARIANCE: {item} is {variance_pct:.1f}% above benchmark. Monitor closely.",
        "severity": "medium",
        "action_items": [
            "Request at least one alternative quotation",
            "Compare with RS Means or regional index",
        ],
        "category": "cost",
    },
    "cost_under_warning": {
        "condition": {"field": "variance_pct", "op": "lt", "value": -15},
        "template": "QUALITY RISK: {item} is {variance_pct:.1f}% below benchmark. Verify specification compliance.",
        "severity": "warning",
        "action_items": [
            "Confirm material grade matches specification",
            "Check if scope items are missing",
            "Request material approval submission",
            "Audit contractor's method statement",
        ],
        "category": "cost",
    },
    # Schedule rules
    "schedule_delay_critical": {
        "condition": {"field": "delay_days", "op": "gt", "value": 30},
        "template": "CRITICAL DELAY: {item} is {delay_days} days behind schedule. Recovery plan required.",
        "severity": "critical",
        "action_items": [
            "Submit Recovery Programme within 7 days",
            "Add resources / shift to 24-hour operations",
            "Identify Critical Path impact",
            "Assess EOT entitlement",
        ],
        "category": "schedule",
    },
    "schedule_delay_high": {
        "condition": {"field": "delay_days", "op": "between", "value": [14, 30]},
        "template": "DELAY ALERT: {item} is {delay_days} days behind. Corrective action required.",
        "severity": "high",
        "action_items": [
            "Update programme with recovery logic",
            "Increase resource allocation on critical activities",
        ],
        "category": "schedule",
    },
    # Quality rules
    "qc_failure": {
        "condition": {"field": "test_result", "op": "eq", "value": "fail"},
        "template": "QC FAILURE: {item} failed inspection. Non-Conformance Report to be raised.",
        "severity": "critical",
        "action_items": [
            "Issue NCR within 24 hours",
            "Stop work on affected area",
            "Implement corrective action before reinspection",
            "Update QA/QC log",
        ],
        "category": "quality",
    },
    "qc_marginal": {
        "condition": {"field": "test_result", "op": "eq", "value": "marginal"},
        "template": "QC MARGINAL: {item} is at the edge of acceptance criteria. Monitor and retest.",
        "severity": "medium",
        "action_items": [
            "Perform repeat test within 48 hours",
            "Check curing conditions",
            "Review mix design",
        ],
        "category": "quality",
    },
    # Safety rules
    "safety_critical": {
        "condition": {"field": "risk_level", "op": "eq", "value": "critical"},
        "template": "SAFETY STOP: {item} presents critical safety risk. Work must stop immediately.",
        "severity": "critical",
        "action_items": [
            "Issue STOP WORK order immediately",
            "Report to HSE Manager",
            "Conduct incident investigation",
            "Retrain personnel before resuming",
        ],
        "category": "safety",
    },
    "safety_high": {
        "condition": {"field": "risk_level", "op": "eq", "value": "high"},
        "template": "SAFETY RISK: {item} requires immediate safety controls.",
        "severity": "high",
        "action_items": [
            "Update Risk Assessment and Method Statement (RAMS)",
            "Conduct toolbox talk",
            "Verify PPE compliance",
        ],
        "category": "safety",
    },
    # Carbon / ESG rules
    "carbon_high": {
        "condition": {"field": "carbon_kgco2e", "op": "gt", "value": 10000},
        "template": "HIGH CARBON: {item} contributes {carbon_kgco2e:,.0f} kgCO₂e. Review low-carbon alternatives.",
        "severity": "medium",
        "action_items": [
            "Consider supplementary cementitious materials (SCM)",
            "Evaluate recycled steel content",
            "Report in ESG dashboard",
        ],
        "category": "sustainability",
    },
}

_CATEGORY_ICONS: Dict[str, str] = {
    "cost": "💰",
    "schedule": "📅",
    "quality": "✅",
    "safety": "⚠️",
    "sustainability": "🌱",
}

_SEVERITY_ORDER = ["critical", "high", "warning", "medium", "low", "info"]


class RecommendationTemplateBlock(UniversalBlock):
    name = "recommendation_template"
    version = "1.0.0"
    description = "Rule-based construction recommendation engine: variance data → severity-ranked recommendation text + action items"
    layer = 3
    tags = ["domain", "construction", "recommendations", "rules", "templates", "reporting"]
    requires = []
    skip_input_validation_actions = ["list_rules"]
    required_input_one_of = ["variance_data", "rule_key"]

    default_config = {
        "max_recommendations": 20,
        "include_action_items": True,
        "custom_rules_env": "RECOMMENDATION_RULES_PATH",
    }

    ui_schema = {
        "input": {
            "type": "json",
            "placeholder": '{"variance_data": [{"item": "concrete_c30", "variance_pct": 18.5, "cost_impact_usd": 42000}], "rule_key": "cost_over_high"}',
            "multiline": True,
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "recommendation_text", "type": "text", "label": "Recommendation"},
                {"name": "severity", "type": "text", "label": "Severity"},
                {"name": "action_items", "type": "list", "label": "Actions"},
            ],
        },
        "quick_actions": [
            {"icon": "📋", "label": "Analyze Variances", "prompt": "Generate recommendations from variance data"},
            {"icon": "⚠️", "label": "Critical Only", "prompt": "Show only critical recommendations"},
            {"icon": "📚", "label": "Rule Library", "prompt": "Show all available recommendation rules"},
        ],
    }

    def __init__(self, hal_block=None, config: Dict = None):
        super().__init__(hal_block, config)
        self._rules = dict(_RULE_DB)
        self._load_custom_rules()

    def _load_custom_rules(self):
        env_path = os.environ.get(
            self.config.get("custom_rules_env", "RECOMMENDATION_RULES_PATH") if hasattr(self, "config") else "RECOMMENDATION_RULES_PATH",
            ""
        )
        if env_path and os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    custom = json.load(f)
                self._rules.update(custom)
            except Exception:
                logger.debug(
                    "swallowed %s in _load_custom_rules() — continuing",
                    "Exception", exc_info=True,
                )

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        operation = data.get("operation") or params.get("operation", "recommend")
        rule_key = data.get("rule_key") or params.get("rule_key")

        if operation == "list_rules":
            return self._list_rules()

        variance_data = data.get("variance_data", [])
        if isinstance(variance_data, dict):
            variance_data = [variance_data]

        # Single rule application
        if rule_key and rule_key in self._rules and not variance_data:
            item = data.get("item_data", data)
            return self._apply_single_rule(rule_key, item)

        # Auto-match rules to variance data
        recommendations = []
        max_recs = int(params.get("max_recommendations", self.config.get("max_recommendations", 20)))

        for item in variance_data:
            for rkey, rule in self._rules.items():
                if self._matches(item, rule["condition"]):
                    rec = self._render(rkey, rule, item, params)
                    if rec:
                        recommendations.append(rec)

        # If rule_key specified, also run that rule on each item
        if rule_key and rule_key in self._rules:
            rule = self._rules[rule_key]
            for item in variance_data:
                rec = self._render(rule_key, rule, item, params)
                if rec and rec not in recommendations:
                    recommendations.append(rec)

        # Sort by severity
        recommendations.sort(key=lambda r: _SEVERITY_ORDER.index(r["severity"]) if r["severity"] in _SEVERITY_ORDER else 99)
        recommendations = recommendations[:max_recs]

        if not recommendations and variance_data:
            recommendations = [self._default_recommendation(variance_data)]

        primary = recommendations[0] if recommendations else {}
        return {
            "status": "success",
            "recommendation_text": primary.get("recommendation_text", "No critical issues found."),
            "severity": primary.get("severity", "info"),
            "action_items": primary.get("action_items", []) if params.get("include_action_items", True) else [],
            "all_recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "rules_applied": list({r.get("rule_key") for r in recommendations}),
        }

    def _matches(self, item: Dict, condition: Dict) -> bool:
        field = condition.get("field", "")
        op = condition.get("op", "eq")
        val = condition.get("value")

        item_val = item.get(field)
        if item_val is None:
            return False

        try:
            if op == "eq":
                return str(item_val).lower() == str(val).lower()
            if op == "gt":
                return float(item_val) > float(val)
            if op == "lt":
                return float(item_val) < float(val)
            if op == "gte":
                return float(item_val) >= float(val)
            if op == "lte":
                return float(item_val) <= float(val)
            if op == "between":
                lo, hi = float(val[0]), float(val[1])
                return lo <= float(item_val) <= hi
            if op == "contains":
                return str(val).lower() in str(item_val).lower()
        except (TypeError, ValueError, IndexError):
            logger.debug(
                "swallowed %s in _matches() — continuing",
                "(TypeError, ValueError, IndexError)", exc_info=True,
            )
        return False

    def _render(self, rule_key: str, rule: Dict, item: Dict, params: Dict) -> Optional[Dict]:
        try:
            text = rule["template"].format_map(_DefaultDict(item))
        except Exception:
            text = rule["template"]

        category = rule.get("category", "general")
        icon = _CATEGORY_ICONS.get(category, "📌")
        include_actions = params.get("include_action_items", self.config.get("include_action_items", True))

        return {
            "rule_key": rule_key,
            "recommendation_text": f"{icon} {text}",
            "severity": rule.get("severity", "info"),
            "action_items": rule.get("action_items", []) if include_actions else [],
            "category": category,
            "item": item.get("item") or item.get("item_key") or item.get("description", ""),
        }

    def _apply_single_rule(self, rule_key: str, item: Dict) -> Dict:
        rule = self._rules[rule_key]
        rec = self._render(rule_key, rule, item, {})
        return {
            "status": "success",
            "recommendation_text": rec["recommendation_text"] if rec else "",
            "severity": rec["severity"] if rec else "info",
            "action_items": rec["action_items"] if rec else [],
            "all_recommendations": [rec] if rec else [],
            "recommendation_count": 1 if rec else 0,
            "rules_applied": [rule_key],
        }

    def _default_recommendation(self, variance_data: List[Dict]) -> Dict:
        return {
            "rule_key": "default",
            "recommendation_text": "📌 No critical issues detected. Continue monitoring per QA/QC plan.",
            "severity": "info",
            "action_items": ["Continue regular monitoring", "Update status in next progress report"],
            "category": "general",
            "item": "",
        }

    def _list_rules(self) -> Dict:
        summary = {
            k: {
                "condition": v["condition"],
                "severity": v["severity"],
                "category": v.get("category", "general"),
                "template_preview": v["template"][:80] + "..." if len(v["template"]) > 80 else v["template"],
            }
            for k, v in self._rules.items()
        }
        return {
            "status": "success",
            "recommendation_text": "",
            "severity": "info",
            "action_items": [],
            "rule_library": summary,
            "total_rules": len(self._rules),
            "categories": list({v.get("category", "general") for v in self._rules.values()}),
        }


class _DefaultDict(dict):
    """Format-map fallback: returns '' for missing keys."""
    def __missing__(self, key):
        return ""
