"""Construction container — boq submodule."""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.construction_types import Measurement, SpecItem, RiskItem

from .helpers import _parse_money_str, _safe_float, _safe_iso_date

logger = logging.getLogger(__name__)

# A money/quantity token: digits with optional thousands/decimal separators.
_FIG_NUM = r"([0-9][0-9,.]*[0-9]|[0-9])"
# Bounded label→number gap (currency code, "of", "total", spaces) — keeps a
# label from grabbing a figure that belongs to a later clause.
_FIG_GAP = r"[^0-9%]{0,30}?"
_FIG_PCT = r"([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)"


_CCY = r"(?:sar|aed|usd|qar|omr|bhd|kwd|\$)"
_MONEY_SUFFIX = {"m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}

# Live: "Draft a variation order" succeeded with VO-001 / Total: 0.0 and
# the ask as the description. A draft-ask is not a variation.
_DRAFT_VO_ASK_RE = re.compile(
    r"(?:draft|issue|create|generate|write|prepare|make)\s+"
    r"(?:a(?:n)?\s+)?(?:variation(?:\s+order)?|vo\b|change\s+order)",
    re.I,
)


def _vo_scope_after_ask(text: str) -> str:
    """Strip a draft-VO ask so leftover is actual works scope, if any."""
    cleaned = _DRAFT_VO_ASK_RE.sub(" ", text or "")
    cleaned = re.sub(
        r"\b(please|fidic|clause|under|a|an|the|for)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" .:-")


_VO_LINE_UNIT = (
    r"m[23]?|lm|sqm|cum|nr|nos?|ea|each|items?|fittings?|t|tonnes?|kg"
)
_VO_LINE_NUM = r"(?:[0-9][0-9,.]*[0-9]|[0-9])"
# Live tip d9d5971 ask2: "ADD: additional drainage 45 m @ AED 180/m" —
# optional colon/dash after the kind; "12 fittings @ AED 2500 each".
_VO_LINE_RE = re.compile(
    r"\b(?P<kind>ADD|OMIT|OMISSION|DELETE|DEDUCT)\b(?:\s*[:.\-])?\s+"
    r"(?:"
    r"(?P<qty1>" + _VO_LINE_NUM + r")\s*(?P<unit1>" + _VO_LINE_UNIT + r")\s+"
    r"(?P<desc1>[^\n@]+?)"
    r"|"
    r"(?P<desc2>[^\n@]+?)\s+(?P<qty2>" + _VO_LINE_NUM + r")\s*(?P<unit2>"
    + _VO_LINE_UNIT + r")"
    r")"
    r"\s+(?:@|at)\s*(?:(?:sar|aed|usd|qar|omr|bhd|kwd|\$)\s*)?"
    r"(?P<rate>" + _VO_LINE_NUM + r")"
    r"(?:\s*/\s*(?:" + _VO_LINE_UNIT + r"))?",
    re.I,
)


def _parse_vo_priced_lines(text: str) -> List[Dict[str, Any]]:
    """Parse ADD/OMIT qty-unit-desc @ rate lines from an operator ask."""
    lines: List[Dict[str, Any]] = []
    if not text:
        return lines
    seen: set[tuple] = set()
    for m in _VO_LINE_RE.finditer(text):
        kind = m.group("kind").upper()
        if kind in {"OMISSION", "DELETE", "DEDUCT"}:
            kind = "OMIT"
        qty = _parse_money_str(m.group("qty1") or m.group("qty2") or "0")
        unit = (m.group("unit1") or m.group("unit2") or "").strip()
        desc = (m.group("desc1") or m.group("desc2") or "").strip(" :-")
        rate = _parse_money_str(m.group("rate") or "0")
        if qty is None or rate is None or qty <= 0 or rate <= 0:
            continue
        amount = round(float(qty) * float(rate), 2)
        signed = -amount if kind == "OMIT" else amount
        key = (kind, float(qty), unit.lower(), desc.lower(), float(rate))
        if key in seen:
            continue
        seen.add(key)
        lines.append({
            "kind": kind,
            "description": desc,
            "quantity": float(qty),
            "unit": unit,
            "rate": float(rate),
            "amount": signed,
        })
    return lines


def _variation_has_draft_facts(vo_data: Any, text: str) -> bool:
    """True when there is a works description and a positive cost.

    Priced ADD/OMIT lines in the operator ask are draft facts — live Phase 2
    ask1 used to refuse because only ``vo_data.direct_cost`` counted.
    """
    data = vo_data if isinstance(vo_data, dict) else {}
    desc = str(data.get("description") or "").strip()
    joined = " ".join(x for x in (desc, text) if x)
    if _parse_vo_priced_lines(joined):
        return True
    try:
        cost = float(data.get("direct_cost") or 0)
    except (TypeError, ValueError):
        cost = 0.0
    scope = _vo_scope_after_ask(joined)
    return bool(scope) and cost > 0


def _follow_on_rfi_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Draft one follow-on RFI from the operator ask (live M10)."""
    t = text or ""
    if not re.search(
        r"follow-on rfi|\bdraft a(?:n)? (?:follow[- ]on )?rfi\b|"
        r"request for information",
        t,
        re.I,
    ):
        return None
    prior = re.search(r"\b(RFI\s*0*\d+)\b", t, re.I)
    prior_s = re.sub(r"\s+", "", prior.group(1)).upper() if prior else "the prior RFI"
    question = (
        t.strip()
        or f"Please confirm the Engineer response to {prior_s}."
    )
    return {
        "status": "success",
        "action": "rfi_generator",
        "rfi_number": "DRAFT-RFI",
        "question": question,
        "rfis": [{
            "rfi_number": "DRAFT-RFI",
            "subject": f"Follow-on to {prior_s}",
            "question": question,
            "status": "Open",
            "priority": "high",
        }],
        "total_rfis": 1,
        "note": (
            "Draft follow-on RFI from operator facts — not a submitted RFI. "
            "Do not rewrite this as a Work Inspection Request."
        ),
    }


def _cashflow_figures_from_message(text: str) -> Dict[str, float]:
    """Parse contract value + duration out of a cash-flow request.

    Same contract as _payment_figures_from_message: conservative, fills only
    gaps, callers let explicit params win. A currency-prefixed amount
    ("SAR 60M", "USD 24,000,000") counts as the contract value only when the
    message contains exactly ONE such amount — two amounts would be a guess.
    "18-month" / "18 months" supplies duration_months.
    """
    out: Dict[str, float] = {}
    if not text:
        return out
    t = str(text).lower()
    amounts = re.findall(
        rf"{_CCY}\s*([0-9][0-9,.]*)\s*(million|billion|mn|bn|m|b)?\b", t
    )
    if len(amounts) == 1:
        base = _parse_money_str(amounts[0][0])
        if base:
            out["contract_value"] = base * _MONEY_SUFFIX.get(amounts[0][1], 1)
    d = re.search(r"\b([0-9]{1,3})\s*[- ]\s*month", t)
    if d:
        out["duration_months"] = float(d.group(1))
    else:
        _word_months = {
            "three": 3, "six": 6, "nine": 9, "twelve": 12,
            "eighteen": 18, "twenty-four": 24, "twenty four": 24,
        }
        for word, months in _word_months.items():
            if re.search(rf"\b{re.escape(word)}[- ]months?\b", t):
                out["duration_months"] = float(months)
                break
    return out


def _carbon_quantities_from_message(text: str) -> Dict[str, float]:
    """Parse material quantities out of a carbon-footprint request.

    Predefined dispatch sends empty params, so "calculate the embodied
    carbon for 12,000 m3 of C40 concrete and 900t of rebar" reached the
    tool with no quantities and returned ZERO. Conservative, unit-aware:
    concrete/blinding in m3 -> concrete_m3; rebar/steel by mass (t->kg)
    -> rebar_kg / steel_kg; timber m3; glass/block m2. Returns only what
    it can label; callers let explicit params win.
    """
    out: Dict[str, float] = {}
    if not text:
        return out
    t = str(text).lower()

    def _num(m) -> float:
        return _parse_money_str(m.group(1)) or 0.0

    m = re.search(r"([0-9][0-9,.]*)\s*m\s*(?:3|³|\^3|cubic)\b.{0,30}?concret", t) \
        or re.search(r"concret.{0,30}?([0-9][0-9,.]*)\s*m\s*(?:3|³|\^3|cubic)", t)
    if m:
        out["concrete_m3"] = _num(m)
    # rebar / reinforcement by mass; tonnes -> kg
    m = re.search(r"([0-9][0-9,.]*)\s*(t|tonne|tonnes|ton|kg)\b.{0,30}?(?:rebar|reinforc)", t) \
        or re.search(r"(?:rebar|reinforc).{0,30}?([0-9][0-9,.]*)\s*(t|tonne|tonnes|ton|kg)", t)
    if m:
        v = _parse_money_str(m.group(1)) or 0.0
        out["rebar_kg"] = v if m.group(2) == "kg" else v * 1000
    # structural steel by mass
    m = re.search(r"([0-9][0-9,.]*)\s*(t|tonne|tonnes|ton|kg)\b.{0,30}?(?:structural\s+)?steel", t) \
        or re.search(r"(?:structural\s+)?steel.{0,30}?([0-9][0-9,.]*)\s*(t|tonne|tonnes|ton|kg)", t)
    if m:
        v = _parse_money_str(m.group(1)) or 0.0
        out["steel_kg"] = v if m.group(2) == "kg" else v * 1000
    return out


def _payment_figures_from_message(text: str) -> Dict[str, float]:
    """Deterministically parse IPC figures out of a chat message.

    The predefined dispatch path sends the raw chat envelope as ``input_data``
    with empty ``params`` — the router does not extract numbers — so a request
    like "gross valuation SAR 10,000,000, retention 10%" arrives with its
    figures only in ``message``. Labels are matched conservatively: a figure is
    used only when its label is explicit, and nothing is guessed. Callers must
    let explicit params/data override anything returned here.
    """
    out: Dict[str, float] = {}
    if not text:
        return out
    t = str(text).lower()

    def _amount(pattern: str) -> Optional[float]:
        m = re.search(pattern, t)
        if not m:
            return None
        # A % right after the number means it is a rate, not an amount.
        if t[m.end(1):].lstrip().startswith(("%", "percent")):
            return None
        base = _parse_money_str(m.group(1))
        if base is None:
            return None
        # Optional suffix group: "25M" / "25 million" on labelled amounts.
        suffix = ""
        if m.lastindex and m.lastindex >= 2:
            suffix = (m.group(2) or "").lower()
        return base * _MONEY_SUFFIX.get(suffix, 1)

    def _percent(pattern: str) -> Optional[float]:
        m = re.search(pattern, t)
        return float(m.group(1)) if m else None

    _SFX = r"(million|billion|mn|bn|m|b)?"

    v = (_amount(rf"gross\s+(?:valuation|value|work\s+done(?:\s+to\s+date)?){_FIG_GAP}{_FIG_NUM}")
         # "gross 750000" — live F5; label without the word valuation.
         or _amount(rf"gross{_FIG_GAP}{_FIG_NUM}")
         # "work valued at 900,000" / "work to the value of 900,000" — the
         # phrasing the live Q12 battery prompt uses (verified failing
         # 2026-08-02: the certificate refused for lack of a valuation).
         or _amount(rf"work\s+valued\s+at{_FIG_GAP}{_FIG_NUM}")
         or _amount(rf"value\s+of\s+(?:the\s+)?work(?:\s+done)?{_FIG_GAP}{_FIG_NUM}")
         or _amount(rf"(?:this\s+period\s+)?certified\s+work{_FIG_GAP}{_FIG_NUM}"))
    if v:
        out["gross_valuation"] = v
    # Live Phase 2 ask2: "measured works SAR 1,200,000" is the QS label
    # for this-period executed work. Not a synonym of "gross valuation".
    v = (
        _amount(rf"measured\s+works?{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
        or _amount(rf"value\s+of\s+measured\s+works?{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
    )
    if v:
        out["measured_works"] = v
        out.setdefault("gross_valuation", v)
    v = (
        _amount(rf"materials?\s+on\s+site{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
        or _amount(rf"\bmos\b{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
    )
    if v:
        out["materials_on_site"] = v
    v = (
        _amount(rf"(?:accepted\s+)?contract\s+(?:amount|value|sum|price)(?:\s+of)?{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
        or _amount(rf"accepted\s+contract\s+amount{_FIG_GAP}{_FIG_NUM}\s*{_SFX}")
        # "SAR 25M contract" / "25 million contract" — require currency or a
        # scale word so "Package 1 contract" is not a contract_value.
        or _amount(rf"{_CCY}\s*{_FIG_NUM}\s*{_SFX}\s+contract")
        or _amount(rf"{_FIG_NUM}\s*(million|billion|mn|bn|m|b)\s+contract")
    )
    if v:
        out["contract_value"] = v
    v = (_percent(rf"{_FIG_PCT}\s*(?:of\s+)?(?:work\s+done|complete|completed|progress)")
         or _percent(rf"work\s+done{_FIG_GAP}{_FIG_PCT}"))
    if v:
        out["work_done_percent"] = v
    v = (_percent(rf"retention{_FIG_GAP}{_FIG_PCT}")
         # reversed order: "10 percent retention" / "10% retention"
         or _percent(rf"{_FIG_PCT}\s+retention"))
    if v:
        out["retention_percent"] = v
    v = (
        _percent(rf"advance(?:\s+payment)?\s+recover\w*{_FIG_GAP}{_FIG_PCT}")
        or _percent(rf"amortis[ae]tion{_FIG_GAP}{_FIG_PCT}")
        or _percent(rf"{_FIG_PCT}\s+amortis")
    )
    if v is not None:
        out["advance_recovery_percent"] = v
    else:
        v = _amount(rf"advance(?:\s+payment)?(?:\s+recover\w*)?{_FIG_GAP}{_FIG_NUM}")
        if v:
            out["advance_payment"] = v
    v = (
        _percent(rf"advance(?:\s+payment)?{_FIG_GAP}{_FIG_PCT}")
        or _percent(rf"{_FIG_PCT}\s+advance(?:\s+payment)?")
    )
    if v is not None:
        out["advance_percent"] = v
    v = _amount(rf"previous\s+certif\w*(?:\s+total)?{_FIG_GAP}{_FIG_NUM}")
    if v:
        out["previous_certified"] = v
    return out


_IPC_EMPTY = (None, "", 0, 0.0)


def ipc_args_from_ask(text: str, existing: Optional[Dict] = None) -> Dict[str, Any]:
    """Fill an IPC tool-args dict from the operator ask.

    Forced tool_choice and the generic ``construction`` envelope often arrive
    as ``{}`` / ``{action: payment_certificate}``. The figures live in the
    user message. Merge them onto the invocation so ``payment_certificate``
    is never called empty when the ask supplied measured works, retention,
    MOS, or a contract sum. Explicit keys in ``existing`` win.
    """
    out: Dict[str, Any] = dict(existing or {})
    fig = _payment_figures_from_message(text)
    for key, val in fig.items():
        if out.get(key) in _IPC_EMPTY:
            out[key] = val
    if text:
        out.setdefault("message", text)
        out.setdefault("user_message", text)
        out.setdefault("text", text)
    return out


class ConstructionBoqMixin:
    async def _process_bill_of_materials(self, input_data: Any, params: Dict) -> Dict:
        """A BOM is a BOQ in everything but name — same shape (line items,
        quantities, units, rates). Delegate to boq_processor, which already
        knows how to parse Excel/CSV/PDF bills.

        Forwards ``project_id`` in params so boq_processor can resolve a bare
        filename to the project's stored file_path. Without this, the LLM
        passing just the document name (e.g. "Demolition BOQ.pdf") causes
        boq_processor's ``os.path.exists`` check to fail and surface "File
        not found" through the chat.
        """
        data = input_data if isinstance(input_data, dict) else {}
        p = dict(params or {})
        file_path = (
            data.get("file_path") if isinstance(data, dict) else None
        ) or p.get("file_path") or (input_data if isinstance(input_data, str) else None)
        block = self._resolve_block("boq_processor")
        if block is None:
            return {"status": "error", "doc_type": "bom", "error": "boq_processor block unavailable"}
        # Make sure project_id is in params (it normally is, but data may also
        # carry it from synthetic agent calls).
        if "project_id" not in p and isinstance(data, dict) and data.get("project_id"):
            p["project_id"] = data["project_id"]
        result = await block.process({"file_path": file_path} if file_path else data, p)
        if isinstance(result, dict):
            result.setdefault("doc_type", "bom")
        return result
    async def generate_cost_estimate(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        quantities = p.get("quantities", data.get("quantities", {}))
        location = p.get("location", "US National Average")
        project_type = p.get("project_type", "general_building")

        if not quantities:
            return {
                "status": "error",
                "action": "cost_estimate",
                "error": (
                    "No quantities supplied — provide quantities or a BOQ"
                ),
            }

        block = self._get_historical_benchmark_block()
        if block is None:
            return {
                "status": "error",
                "action": "cost_estimate",
                "error": (
                    "Benchmark rate source unavailable (construction kit not "
                    "loaded). Provide unit rates directly in the BOQ, or request "
                    "supplier quotes."
                ),
            }

        _UNIT_SUFFIXES = {"_m3": "m3", "_m2": "m2", "_kg": "kg", "_lm": "lm", "_ea": "ea", "_nr": "nr"}

        line_items = []
        unpriced_items = []
        for item_name, qty_data in quantities.items():
            if isinstance(qty_data, dict):
                quantity = _safe_float(qty_data.get("quantity", 0))
                unit = qty_data.get("unit", "ea")
            else:
                quantity = _safe_float(qty_data)
                unit = "ea"
                # extract unit from key suffix e.g. concrete_m3 → m3
                for suffix, u in _UNIT_SUFFIXES.items():
                    if item_name.endswith(suffix):
                        unit = u
                        break

            result = await block.process(
                {},
                {
                    "action": "lookup",
                    "item": item_name,
                    "unit": unit,
                    "location": location,
                    "project_type": project_type,
                },
            )
            if not isinstance(result, dict) or result.get("status") != "success":
                # No benchmark for this item — record honestly, do not fabricate a rate.
                unpriced_items.append(item_name)
                line_items.append({
                    "item": item_name,
                    "quantity": quantity,
                    "unit": unit,
                    "base_rate": None,
                    "adjusted_rate": None,
                    "location_factor": None,
                    "total": None,
                    "note": "no benchmark rate found — excluded from totals",
                })
                continue

            rates = result.get("rates", {})
            factors = result.get("factors", {})
            base_rate = rates.get("base_usd")
            adjusted_rate = rates.get("adjusted_usd")
            location_factor = factors.get("location_factor", 1.0)
            total = (quantity or 0) * (adjusted_rate or 0)

            line_items.append({
                "item": item_name,
                "quantity": quantity,
                "unit": unit,
                "base_rate": base_rate,
                "adjusted_rate": adjusted_rate,
                "location_factor": location_factor,
                "basis": result.get("basis"),
                "total": round(total, 2),
            })

        subtotal = sum(item["total"] for item in line_items if item["total"] is not None)
        overhead = subtotal * 0.10
        profit = subtotal * 0.08
        contingency = subtotal * 0.05
        total = subtotal + overhead + profit + contingency

        # Honesty: the benchmark rates carry different bases (material-only vs
        # supply-and-fix vs plant+labour). Summing a material-only line (e.g.
        # concrete supply, no placing) with an all-in line understates the
        # labour-heavy trades — so flag when the priced lines mix bases.
        priced_bases = {li.get("basis") for li in line_items
                        if li.get("total") is not None and li.get("basis")}
        mixed_basis = len(priced_bases) > 1
        basis_warning = (
            "Indicative benchmark rates mix cost bases "
            f"({sorted(priced_bases)}): material-only lines exclude their "
            "placing/fixing labour, so this total understates labour-heavy "
            "trades. Load your own project rates for a coherent total."
        ) if mixed_basis else None

        return {
            "status": "success",
            "action": "cost_estimate",
            "location": location,
            "project_type": project_type,
            "line_items": line_items,
            "unpriced_items": unpriced_items,
            "rate_source": "indicative_benchmark_fallback",
            "bases_used": sorted(priced_bases),
            "mixed_basis": mixed_basis,
            "basis_warning": basis_warning,
            "summary": {
                "subtotal": round(subtotal, 2),
                "overhead": round(overhead, 2),
                "profit": round(profit, 2),
                "contingency": round(contingency, 2),
                "total_estimate": round(total, 2)
            },
            "confidence": "low",
            "note": ("Priced from indicative general-knowledge benchmark rates — "
                     "replace with your project/supplier rates for a real estimate."),
        }
    async def _lookup_unit_cost(
        self, item_name: str, unit: str,
        location: str = "US National Average",
        project_type: str = "general_building",
    ):
        """Delegate per-item unit-rate lookup to the historical_benchmark block.

        Returns the location/project-adjusted USD rate (rates.adjusted_usd) as a
        float, or None when the block has no benchmark for the item. No fabricated
        fallback — an unknown item honestly yields None.
        """
        block = self._get_historical_benchmark_block()
        if block is None:
            return None

        result = await block.process(
            {},
            {
                "action": "lookup",
                "item": item_name,
                "unit": unit,
                "location": location,
                "project_type": project_type,
            },
        )
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        return result.get("rates", {}).get("adjusted_usd")
    @staticmethod
    def _normalize_measurements(measurements: Any) -> List[Dict]:
        """Normalise measurements into the list-of-dicts shape _calculate_quantities expects.

        Supports:
        - list of dicts (passed through, with numeric values preserved)
        - flat dict of numbers (e.g. {"area": 500, "volume": 120})
        - dict of dicts with unit/value (e.g. {"concrete_slab": {"value": 50, "unit": "m3"}})
        """
        if isinstance(measurements, dict):
            converted = []
            for key, value in measurements.items():
                if isinstance(value, (int, float)):
                    converted.append({"type": key, "value": value, "item": key})
                elif isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("item", key)
                    if not entry.get("type") and entry.get("unit"):
                        unit = str(entry["unit"]).lower()
                        if unit == "m3":
                            entry["type"] = "volume"
                        elif unit == "m2":
                            entry["type"] = "area"
                        elif unit in ("kg", "t", "tonne", "tonnes"):
                            entry["type"] = "weight"
                        elif unit in ("ea", "nr", "pcs", "each"):
                            entry["type"] = "count"
                    converted.append(entry)
                else:
                    converted.append({"type": key, "value": value, "item": key})
            measurements = converted

        if isinstance(measurements, list):
            normalized = []
            for m in measurements:
                if isinstance(m, dict):
                    normalized.append(m)
                elif isinstance(m, (int, float)):
                    normalized.append({"type": "count", "value": m})
                else:
                    normalized.append({"type": "unknown", "value": m})
            measurements = normalized

        return measurements or []

    async def extract_quantities(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        measurements = (
            data.get("measurements")
            or data.get("quantities")
            or p.get("measurements")
            or p.get("quantities")
            or []
        )
        if not measurements:
            return {
                "status": "error",
                "error": "No measurements found — upload a drawing or supply a measurements list",
                "quantities": {},
                "measurements": [],
            }
        measurements = self._normalize_measurements(measurements)
        quantities = self._calculate_quantities(measurements)
        return {"status": "success", "quantities": quantities, "measurements": measurements}
    async def estimate_costs(self, input_data: Any, params: Dict) -> Dict:
        """Public action: estimate costs from quantities, BOQ list, or process_document output."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        # Accept quantities from multiple upstream shapes
        quantities = p.get("quantities") or data.get("quantities") or {}

        # process_document output / raw measurements → derive quantities from measurements
        if not quantities and data.get("measurements"):
            measurements = self._normalize_measurements(data["measurements"])
            raw_q = self._calculate_quantities(measurements)
            quantities = {
                "Concrete Works": {"quantity": raw_q.get("concrete_volume_m3", 0), "unit": "m3"},
                "Steel / Rebar": {"quantity": raw_q.get("steel_weight_kg", 0), "unit": "kg"},
                "Formwork": {"quantity": raw_q.get("floor_area_m2", 0) * 2, "unit": "m2"},
            }

        # BOQ list → convert to quantities dict
        boq = p.get("boq") or data.get("boq") or data.get("line_items", [])
        if not quantities and isinstance(boq, list) and boq:
            quantities = {
                item.get("description", item.get("item", f"Item {i+1}")): {
                    "quantity": item.get("quantity", 0),
                    "unit": item.get("unit", "ea"),
                }
                for i, item in enumerate(boq)
            }

        if not quantities:
            return {
                "status": "error",
                "error": "No quantities found — extract quantities or supply a BOQ first",
                "summary": {},
                "line_items": [],
            }

        return await self.generate_cost_estimate(
            {"quantities": quantities},
            {
                "quantities": quantities,
                "location": p.get("location", data.get("location", "US National Average")),
                "project_type": p.get("project_type", data.get("project_type", "general_building")),
            },
        )
    async def payment_certificate(self, input_data: Any, params: Dict) -> Dict:
        """Generate Interim Payment Certificate (IPC) for contractor billing."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        # Figures parsed from the chat message fill ONLY the gaps left by
        # explicit params/data (predefined dispatch sends the envelope with
        # empty params, so the numbers often live in `message` alone).
        fig = _payment_figures_from_message(
            " ".join(
                str(x)
                for x in (
                    data.get("message"),
                    data.get("user_message"),
                    data.get("text"),
                    p.get("message"),
                    p.get("user_message"),
                    p.get("brief"),
                    input_data if isinstance(input_data, str) else "",
                )
                if x
            )
        )

        # Operator-stated Accepted Contract Amount wins over a stored
        # project fact (a stale remembered value must not beat the
        # amount the operator typed in the brief).
        if fig.get("contract_value"):
            contract_value = float(fig["contract_value"])
        else:
            contract_value = float(
                p.get("contract_value") or data.get("contract_value", 0) or 0
            )
        work_done_pct = float(p.get("work_done_percent") or data.get("work_done_percent", 0)
                              or fig.get("work_done_percent", 0)) / 100.0
        previous_certified = float(p.get("previous_certified") or data.get("previous_certified", 0)
                                   or fig.get("previous_certified", 0))
        # The stated retention wins wherever it was given (params or the
        # envelope); 0 is a stated value, not a missing one.
        stated_retention = next(
            (src[key] for src in (p, data) for key in ("retention_percent", "retention_rate")
             if isinstance(src, dict) and src.get(key) is not None and src.get(key) != ""),
            None,
        )
        retention_pct = float(stated_retention if stated_retention is not None
                              else fig.get("retention_percent", 10)) / 100.0
        advance_payment = float(p.get("advance_payment") or data.get("advance_paid", 0)
                                or data.get("advance_payment", 0) or fig.get("advance_payment", 0))
        advance_pct = float(
            p.get("advance_percent")
            or data.get("advance_percent")
            or fig.get("advance_percent")
            or 0
        )
        if advance_payment <= 0 and advance_pct > 0 and contract_value > 0:
            advance_payment = round(contract_value * (advance_pct / 100.0), 2)
        advance_recovery_pct = float(p.get("advance_recovery_percent",
                                           fig.get("advance_recovery_percent", 20))) / 100.0
        payment_period = p.get("payment_period", "Current Period")
        contractor = p.get("contractor_name", p.get("contractor", data.get("contractor_name", "Contractor")))

        measured_works = float(
            p.get("measured_works") or data.get("measured_works")
            or fig.get("measured_works") or 0
        )
        materials_on_site = float(
            p.get("materials_on_site") or data.get("materials_on_site")
            or fig.get("materials_on_site") or 0
        )
        # Accept gross_valuation directly if contract_value not provided.
        # QS ask2: measured works + MOS is the gross when no labelled gross.
        direct_gross = float(p.get("gross_valuation") or data.get("gross_valuation", 0)
                             or fig.get("gross_valuation", 0))
        if measured_works > 0:
            parts = measured_works + materials_on_site
            if direct_gross <= 0 or abs(direct_gross - measured_works) < 0.01:
                direct_gross = parts
        if contract_value <= 0:
            if direct_gross > 0:
                gross_valuation = round(direct_gross, 2)
                contract_value = direct_gross
            else:
                return {
                    "status": "error",
                    "error": (
                        "No contract value or gross valuation supplied — "
                        "provide 'contract_value' (with 'work_done_percent') "
                        "or 'gross_valuation' to issue a payment certificate"
                    ),
                }
        elif direct_gross > 0:
            # Operator stated this-period certified work / gross. Do not
            # wipe it with contract_value × 0% when work_done_percent is
            # absent (live M7: ACA + SAR 42,800,000 this period).
            gross_valuation = round(direct_gross, 2)
        else:
            gross_valuation = round(contract_value * work_done_pct, 2)
        retention_held = round(gross_valuation * retention_pct, 2)
        advance_recovered = round(
            min(advance_payment, gross_valuation * advance_recovery_pct), 2
        )
        net_this_period = round(
            gross_valuation - retention_held - advance_recovered - previous_certified, 2
        )
        cumulative_certified = round(previous_certified + net_this_period, 2)
        remaining_balance = round(contract_value - cumulative_certified, 2)

        return {
            "status": "success",
            "action": "payment_certificate",
            "certificate": {
                "period": payment_period,
                "contractor": contractor,
                "date_issued": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
            "valuation": {
                "contract_value": contract_value,
                "work_completed_percent": round(work_done_pct * 100, 1),
                "gross_valuation": gross_valuation,
                "measured_works": measured_works or None,
                "materials_on_site": materials_on_site or None,
            },
            "deductions": {
                "retention_percent": retention_pct * 100,
                "retention_held": retention_held,
                "advance_recovery": advance_recovered,
                "previous_payments": previous_certified,
                "total_deductions": round(
                    retention_held + advance_recovered + previous_certified, 2
                ),
            },
            "payment": {
                "net_due_this_period": net_this_period,
                "cumulative_certified": cumulative_certified,
                "remaining_contract_balance": remaining_balance,
            },
            "certificate_summary": (
                f"IPC – {payment_period}: {round(work_done_pct * 100, 1)}% complete. "
                f"Gross: {gross_valuation:,.2f}. Net due: {net_this_period:,.2f}."
            ),
        }
    async def procurement_list_generator(self, input_data: Any, params: Dict) -> Dict:
        """Generate a prioritised procurement list from quantities or estimate_costs output."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        quantities = p.get("quantities") or data.get("quantities") or {}
        boq = p.get("boq") or data.get("boq") or data.get("line_items", [])
        budget = _safe_float(p.get("budget") or data.get("summary", {}).get("total_estimate", 0))
        schedule_start = p.get("schedule_start_date") or data.get("schedule_start_date")
        location = p.get("location") or data.get("location") or "US National Average"
        project_type = p.get("project_type") or data.get("project_type") or "general_building"

        procurement_items: List[Dict] = []

        # From estimate_costs line_items
        if isinstance(boq, list) and boq and isinstance(boq[0], dict) and "adjusted_rate" in boq[0]:
            for item in boq:
                name = item.get("item", item.get("description", "Unknown"))
                qty = item.get("quantity", 0)
                unit = item.get("unit", "ea")
                unit_cost_raw = item.get("adjusted_rate") or item.get("base_rate") or 0
                unit_cost = (
                    _parse_money_str(unit_cost_raw)
                    if isinstance(unit_cost_raw, str)
                    else _safe_float(unit_cost_raw)
                )
                total = item.get("total") or (qty * unit_cost)
                cat, lead, supplier = self._classify_procurement_item(name)
                procurement_items.append(self._build_procurement_item(
                    name, qty, unit, unit_cost, total, cat, lead, supplier, schedule_start
                ))

        # From BOQ list. Rows arrive under the names the sheet, the ingest
        # and the indexed line use; read them through one alias table.
        elif isinstance(boq, list) and boq:
            from app.agents.hats.procurement.tools import _normalise_boq_row

            for item in boq:
                if isinstance(item, dict):
                    item = _normalise_boq_row(item)
                name = item.get("description", item.get("item", "Unknown"))
                qty = item.get("quantity", 0)
                if isinstance(qty, bool) or not isinstance(qty, (int, float)):
                    qty = 0
                unit = item.get("unit") or "ea"
                unit_cost = item.get("unit_price")
                if isinstance(unit_cost, bool) or not isinstance(unit_cost, (int, float)):
                    unit_cost = None
                stated_total = item.get("total")
                if isinstance(stated_total, bool) or not isinstance(stated_total, (int, float)):
                    stated_total = None
                if unit_cost is None and stated_total is not None and qty:
                    unit_cost = stated_total / qty
                if unit_cost is None:
                    unit_cost = await self._lookup_unit_cost(name, unit, location, project_type)
                if unit_cost is None and stated_total is not None:
                    total = stated_total
                else:
                    total = qty * (unit_cost or 0)
                cat, lead, supplier = self._classify_procurement_item(name)
                procurement_items.append(self._build_procurement_item(
                    name, qty, unit, unit_cost, total, cat, lead, supplier, schedule_start
                ))

        # From quantities dict
        elif quantities:
            # Aggregate metrics already covered by the cost panel — skip them as
            # individual procurement items so the list reflects discrete trades.
            aggregate_keys = {"floor_area_m2", "concrete_volume_m3", "steel_weight_kg", "rebar_length_m"}
            for item_name, qty_data in quantities.items():
                if item_name in aggregate_keys:
                    continue
                if isinstance(qty_data, dict):
                    qty = _safe_float(qty_data.get("quantity", 0))
                    unit = qty_data.get("unit", "ea")
                else:
                    qty = _safe_float(qty_data)
                    unit = "ea"
                if qty <= 0:
                    continue
                clean_name = " ".join(str(item_name).split())  # collapse whitespace + newlines
                unit_cost = await self._lookup_unit_cost(clean_name, unit, location, project_type)
                total = qty * (unit_cost or 0)
                cat, lead, supplier = self._classify_procurement_item(clean_name)
                procurement_items.append(self._build_procurement_item(
                    clean_name, qty, unit, unit_cost, total, cat, lead, supplier, schedule_start
                ))

        procurement_items.sort(key=lambda x: x["lead_time_weeks"], reverse=True)
        critical = [i for i in procurement_items if i["priority"] == "critical"]
        total_cost = round(sum(i["total_cost"] for i in procurement_items), 2)

        if not procurement_items:
            return {
                "status": "error",
                "action": "procurement_list",
                "error": (
                    "No BOQ line items or discrete quantities supplied — "
                    "cannot generate a procurement list"
                ),
                "total_items": 0,
                "procurement_list": [],
            }

        return {
            "status": "success",
            "action": "procurement_list",
            "total_items": len(procurement_items),
            "total_procurement_cost": total_cost,
            "budget": budget or None,
            "budget_variance": round(budget - total_cost, 2) if budget else None,
            "critical_long_lead_items": len(critical),
            "procurement_list": procurement_items,
            "by_category": self._group_by_category(procurement_items),
            "action_required": [
                f"Issue RFQ for '{i['item']}' immediately — lead time {i['lead_time_weeks']} weeks"
                for i in critical[:5]
            ],
            "recommendations": self._generate_procurement_recommendations(procurement_items),
        }
    def _build_procurement_item(
        self, name: str, qty: float, unit: str, unit_cost: float,
        total: float, category: str, lead: int, supplier: str,
        schedule_start: Optional[str],
    ) -> Dict:
        priority = "critical" if lead >= 12 else "high" if lead >= 6 else "normal"
        return {
            "item": name,
            "quantity": qty,
            "unit": unit,
            "unit_cost": round(unit_cost or 0, 2),
            "total_cost": round(total or 0, 2),
            "category": category,
            "lead_time_weeks": lead,
            "supplier_type": supplier,
            "order_by": self._calculate_order_date(schedule_start, lead),
            "priority": priority,
        }
    def _classify_procurement_item(self, name: str):
        n = name.lower()
        if any(k in n for k in ["structural steel", "steel frame", "steel beam", "steel column"]):
            return "Structural Steel", 16, "Steel Fabricator"
        if any(k in n for k in ["curtain wall", "facade", "curtain_wall"]):
            return "Glazing / Facades", 22, "Specialist Glazier"
        if any(k in n for k in ["glass", "glazing"]):
            return "Glazing", 18, "Glazing Supplier"
        if any(k in n for k in ["lift", "elevator", "escalator"]):
            return "Vertical Transport", 28, "OEM / Specialist"
        if any(k in n for k in ["hvac", "ductwork", "air handling", "chiller", "cooling"]):
            return "Mechanical / HVAC", 16, "MEP Contractor"
        if any(k in n for k in ["switchgear", "transformer", "generator", "hv cable"]):
            return "HV Electrical", 20, "Electrical Contractor"
        if any(k in n for k in ["electrical", "panel", "cable", "lighting", "power"]):
            return "Electrical", 10, "Electrical Contractor"
        if any(k in n for k in ["pump", "chilled water", "fire suppression"]):
            return "Mechanical Plant", 14, "MEP Contractor"
        if any(k in n for k in ["plumbing", "pipe", "sanitary", "drain"]):
            return "Plumbing", 8, "Plumbing Contractor"
        if any(k in n for k in ["stone", "marble", "granite", "cladding"]):
            return "Stone / Cladding", 20, "Stone Supplier"
        if any(k in n for k in ["rebar", "reinforcement"]):
            return "Rebar / Steel", 6, "Steel Stockholder"
        if any(k in n for k in ["concrete", "cement"]):
            return "Concrete", 2, "Ready-Mix Supplier"
        if any(k in n for k in ["steel", "structural"]):
            return "Structural Steel", 14, "Steel Fabricator"
        if any(k in n for k in ["pile", "piling", "foundation"]):
            return "Groundworks", 8, "Specialist Piling"
        if any(k in n for k in ["door", "window", "joinery", "frame"]):
            return "Joinery / Openings", 10, "Joinery Supplier"
        if any(k in n for k in ["tile", "floor", "finish", "paint", "plaster", "ceiling"]):
            return "Finishes", 6, "Finishing Contractor"
        if any(k in n for k in ["formwork", "shuttering", "scaffold"]):
            return "Temporary Works", 3, "Plant Hire"
        if any(k in n for k in ["waterproof", "membrane", "roof"]):
            return "Waterproofing / Roofing", 8, "Specialist Subcontractor"
        if any(k in n for k in ["insulation"]):
            return "Insulation", 6, "Insulation Supplier"
        return "General Materials", 4, "General Supplier"
    def _calculate_order_date(self, schedule_start: Optional[str], lead_time_weeks: int) -> Optional[str]:
        if not schedule_start:
            return None
        try:
            from datetime import timedelta
            start = datetime.strptime(str(schedule_start)[:10], "%Y-%m-%d")
            return (start - timedelta(weeks=lead_time_weeks)).strftime("%Y-%m-%d")
        except Exception:
            return None
    def _group_by_category(self, items: List[Dict]) -> Dict:
        grouped: Dict = {}
        for item in items:
            cat = item.get("category", "General")
            if cat not in grouped:
                grouped[cat] = {"items": [], "total": 0.0}
            grouped[cat]["items"].append(item["item"])
            grouped[cat]["total"] = round(grouped[cat]["total"] + item["total_cost"], 2)
        return grouped
    def _generate_procurement_recommendations(self, items: List[Dict]) -> List[str]:
        recs = []
        critical = [i for i in items if i["priority"] == "critical"]
        if critical:
            recs.append(
                f"Immediate action: {len(critical)} items have lead times ≥ 12 weeks — "
                "issue RFQs and appoint suppliers now"
            )
        categories = {i["category"] for i in items}
        if "Mechanical / HVAC" in categories and "Electrical" in categories:
            recs.append(
                "Consider combined MEP package tender to reduce procurement cost and interface risk"
            )
        total = sum(i["total_cost"] for i in items)
        if total > 5_000_000:
            recs.append(
                "Spend > $5M — pre-qualify all major suppliers and consider framework agreements"
            )
        elif total > 1_000_000:
            recs.append(
                "Spend > $1M — obtain minimum 3 quotes per major category"
            )
        long_lead = [i for i in items if i["lead_time_weeks"] >= 20]
        if long_lead:
            recs.append(
                f"{len(long_lead)} items have lead times ≥ 20 weeks — "
                "consider early letters of intent to secure slots"
            )
        return recs
    async def generate_carbon_report(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        quantities = p.get("quantities", data.get("quantities", {}))
        # Predefined dispatch sends empty params — pull material quantities
        # from the chat message so "12,000 m3 concrete and 900t rebar" isn't
        # computed as zero carbon. Explicit quantities always win.
        if not quantities:
            quantities = _carbon_quantities_from_message(
                data.get("message") or (input_data if isinstance(input_data, str) else "")
            )

        if not quantities:
            return {
                "status": "error",
                "action": "carbon_report",
                "error": (
                    "No material quantities supplied — provide quantities "
                    "or a message that names quantities"
                ),
            }

        carbon_factors = {
            "concrete_m3": 250.0,
            "steel_kg": 2.3,
            "rebar_kg": 1.9,
            "timber_m3": -500.0,
            "block_m2": 45.0,
            "aluminum_kg": 11.0,
            "glass_m2": 35.0
        }
    
        total_carbon = 0
        breakdown = []
    
        for material, qty_data in quantities.items():
            if isinstance(qty_data, dict):
                quantity = qty_data.get("quantity", 0)
            else:
                quantity = qty_data
        
            factor = carbon_factors.get(material, 100.0)
            carbon = quantity * factor
            total_carbon += carbon
        
            breakdown.append({
                "material": material,
                "quantity": quantity,
                "factor_kg_co2_per_unit": factor,
                "total_kg_co2": round(carbon, 2)
            })
    
        return {
            "status": "success",
            "action": "carbon_report",
            "total_embodied_carbon_kg": round(total_carbon, 2),
            "total_tonnes_co2": round(total_carbon / 1000, 2),
            "breakdown": breakdown,
            "benchmark": "Typical office building: 350-500 kg CO2/m²",
            "recommendations": [
                "Consider low-carbon concrete mixes",
                "Optimize steel tonnage through efficient design",
                "Specify recycled content where possible"
            ]
        }
    async def submittal_log_generator(self, input_data: Any, params: Dict) -> Dict:
        """Generate a submittal register from specification or BOQ data."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        spec_sections = p.get("spec_sections") or data.get("specifications") or data.get("spec_sections", [])
        boq = p.get("boq") or data.get("boq") or data.get("line_items", [])
        project_name = p.get("project_name", data.get("project_name", "Project"))
        contract_start = p.get("contract_start_date")

        submittals = []

        # From spec sections
        for section in spec_sections[:40]:
            item = section.get("value") or section.get("description") or str(section)
            submittals.append(self._create_submittal_item(item, "Material Submittal", contract_start))

        # From BOQ
        for i, item in enumerate(boq[:30]):
            name = item.get("description") or item.get("item") or f"Item {i+1}"
            submittals.append(self._create_submittal_item(name, "Shop Drawing", contract_start))
            if any(k in name.lower() for k in ["steel", "concrete", "pipe", "cable"]):
                submittals.append(self._create_submittal_item(name + " — Test Certificate", "Inspection & Test Plan", contract_start))

        if not submittals:
            return {
                "status": "error",
                "action": "submittal_log",
                "error": (
                    "No specification sections or BOQ items supplied — "
                    "cannot invent a submittal register"
                ),
                "submittal_register": [],
            }

        return {
            "status": "success",
            "action": "submittal_log",
            "project": project_name,
            "total_submittals": len(submittals),
            "by_type": self._group_submittals_by_type(submittals),
            "submittal_register": submittals,
            "recommendations": [
                f"Submit all pre-construction documents within 21 days of contract award",
                f"Allow minimum 14 days for Engineer review per contract",
                f"{len([s for s in submittals if s['type'] == 'Shop Drawing'])} shop drawings required — appoint drafting resource immediately",
            ],
        }
    def _create_submittal_item(self, name: str, sub_type: str, contract_start: Optional[str]) -> Dict:
        from datetime import timedelta
        import hashlib
        # Python's built-in `hash()` is PYTHONHASHSEED-randomized, so the same
        # submittal name produced a different ref number on every process
        # restart, making submittal registers unreproducible. Use md5 over the
        # canonical (lowercased) name for stable IDs across runs.
        seed = hashlib.md5(name.lower().encode("utf-8"), usedforsecurity=False).hexdigest()
        ref_num = f"SUB-{int(seed[:8], 16) % 9000 + 1000:04d}"
        due_offset = {"Method Statement": 14, "Material Submittal": 28, "Shop Drawing": 42,
                      "Inspection & Test Plan": 35, "Quality Document": 7, "Safety Document": 7,
                      "Logistics Document": 14}.get(sub_type, 21)
        due_date = None
        if contract_start:
            try:
                due_date = (
                    datetime.strptime(contract_start[:10], "%Y-%m-%d") + timedelta(days=due_offset)
                ).strftime("%Y-%m-%d")
            except Exception:
                logger.warning(
                    "swallowed %s in _create_submittal_item() — continuing",
                    "Exception", exc_info=True,
                )
        return {
            "ref": ref_num,
            "description": name,
            "type": sub_type,
            "status": "Not Submitted",
            "due_date": due_date,
            "review_days": 14,
        }
    async def risk_register_auto_populate(self, input_data: Any, params: Dict) -> Dict:
        """Auto-populate a risk register from document content, specs, or schedule."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        source_risks = (
            data.get("auto_risks")
            or data.get("risks")
            or p.get("risks")
            or []
        )
        # Also pull from document_engine downstream feed
        doc_risks = data.get("downstream", {}).get("risk_engine", {}).get("identified_risks", [])

        risks: List[Dict] = []

        for r in source_risks + doc_risks:
            severity = r.get("severity", "medium")
            prob = {"high": 0.7, "medium": 0.4, "low": 0.2}.get(severity, 0.4)
            impact = {"high": 0.8, "medium": 0.5, "low": 0.3}.get(severity, 0.5)
            risks.append({
                "id": f"RISK-{len(risks)+1:03d}",
                "category": r.get("category", r.get("type", "General")),
                "description": r.get("description", r.get("context", ""))[:200],
                "probability": prob,
                "impact": impact,
                "risk_score": round(prob * impact * 100, 1),
                "severity": severity,
                "mitigation": r.get("mitigation", "Review and action as required"),
                "owner": p.get("default_owner", "Project Manager"),
                "status": "Open",
                "source": "auto",
            })

        if not risks:
            return {
                "status": "error",
                "action": "risk_register",
                "error": (
                    "No risks supplied — provide risks / auto_risks from a "
                    "document. Catalogue risks are not invented."
                ),
                "risk_register": [],
            }

        risks.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "status": "success",
            "action": "risk_register",
            "total_risks": len(risks),
            "high_risks": len([r for r in risks if r["severity"] == "high"]),
            "medium_risks": len([r for r in risks if r["severity"] == "medium"]),
            "low_risks": len([r for r in risks if r["severity"] == "low"]),
            "top_risks": risks[:5],
            "risk_register": risks,
            "recommendations": [
                f"Top risk: {risks[0]['description'][:80]} — assign owner and review weekly"
            ] if risks else [],
        }
    async def rfi_generator(self, input_data: Any, params: Dict) -> Dict:
        """Generate Request for Information (RFI) documents from drawing or spec issues."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        joined = " ".join(
            str(x)
            for x in (
                p.get("text"),
                p.get("user_message"),
                p.get("message"),
                data.get("text"),
                data.get("user_message"),
                data.get("message"),
                input_data if isinstance(input_data, str) else "",
            )
            if x
        )
        drafted = _follow_on_rfi_from_text(joined)
        if drafted and not (
            p.get("issues") or data.get("issues") or data.get("auto_risks")
        ):
            return drafted

        issues = (
            p.get("issues")
            or data.get("issues")
            or data.get("auto_risks")
            or []
        )
        project_name = p.get("project_name", data.get("project_name", "Project"))
        contractor = p.get("contractor_name", "Contractor")
        engineer = p.get("engineer_name", "Engineer of Record")
        drawing_ref = p.get("drawing_ref") or data.get("file_name") or data.get("drawing_number", "")

        rfis = []
        rfi_num = p.get("start_number", 1)

        for issue in issues[:20]:
            desc = issue.get("description", issue.get("context", str(issue)))[:300]
            category = issue.get("type", issue.get("category", "Design Clarification"))
            rfis.append({
                "rfi_number": f"RFI-{rfi_num:04d}",
                "project": project_name,
                "subject": f"{category} — {desc[:60]}",
                "question": desc,
                "drawing_reference": drawing_ref,
                "discipline": self._map_rfi_discipline(category),
                "priority": issue.get("severity", "medium"),
                "date_issued": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "response_required_by": self._add_days(
                    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    14 if issue.get("severity") == "high" else 21,
                ),
                "issued_by": contractor,
                "addressed_to": engineer,
                "status": "Open",
            })
            rfi_num += 1

        if not rfis:
            return {
                "status": "error",
                "action": "rfi_generator",
                "error": (
                    "No issues found to generate RFIs from. Provide an "
                    "'issues' list or a question in the message."
                ),
                "rfis": [],
            }

        return {
            "status": "success",
            "action": "rfi_generator",
            "project": project_name,
            "total_rfis": len(rfis),
            "open_rfis": len(rfis),
            "rfis": rfis,
            "recommendations": [
                f"{len([r for r in rfis if r['priority'] == 'high'])} high-priority RFIs — expedite responses to protect programme",
                "Log all RFIs in contract admin system and track response times",
            ],
        }
    def _map_rfi_discipline(self, category: str) -> str:
        mapping = {
            "structural": "Structural",
            "specification": "Architecture",
            "data_quality": "Architecture",
            "coordination": "MEP Coordination",
            "procurement": "Procurement",
            "safety": "Health & Safety",
            "design": "Architecture",
        }
        return mapping.get(category.lower(), "Architecture")
    async def procurement_analysis(self, input_data: Any, params: Dict) -> Dict:
        """Procurement analysis = generate the procurement list + run the
        supplier optimiser on top. Both methods exist on this container — wire
        them in sequence so 'procurement_analysis' delivers a real artefact
        (grouped material list + supplier comparison) instead of a stub error.
        """
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        # Step 1 — build the procurement list from the quantities / line items
        # in input_data (procurement_list_generator handles the grouping).
        list_result = await self.procurement_list_generator(data, p)
        if isinstance(list_result, dict) and list_result.get("status") == "error":
            return {
                "status": "error",
                "action": "procurement_analysis",
                "stage": "list_generation",
                "error": list_result.get("error", "procurement list generation failed"),
            }

        # Step 2 — optimisation. The optimiser reads procurement_list / items
        # off input_data, so merge the list result back in before calling.
        opt_input = dict(data)
        if isinstance(list_result, dict):
            opt_input.setdefault("procurement_list", list_result.get("procurement_list") or list_result.get("items"))
            opt_input.setdefault("items", list_result.get("items"))
        opt_result = await self.procurement_optimizer(opt_input, p)

        return {
            "status": "success",
            "action": "procurement_analysis",
            "procurement_list": list_result if isinstance(list_result, dict) else {"raw": list_result},
            "optimization": opt_result if isinstance(opt_result, dict) else {"raw": opt_result},
        }
    async def change_order_impact(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
    
        co_type = p.get("change_type", data.get("change_type"))
        direct_cost = p.get("direct_cost", data.get("direct_cost", 0))
        if not co_type and not direct_cost:
            return {
                "status": "error",
                "action": "change_order_analysis",
                "error": (
                    "Provide change_type / description and/or direct_cost — "
                    "cannot analyse a variation with no scope and no cost"
                ),
            }
        if not co_type:
            co_type = "general"

        analysis = self._analyze_change_type(co_type, params)
        cost_impact = self._calculate_co_cost_impact(direct_cost, analysis)
    
        return {
            "status": "success",
            "action": "change_order_analysis",
            "change_type": co_type,
            "category": analysis.get("category"),
            "complexity": analysis.get("complexity"),
            "cost_impact": cost_impact,
            "schedule_impact_days": analysis.get("typical_delay_days", 0),
            "trade_involved": analysis.get("trade_involved"),
            "risk_level": analysis.get("risk_level"),
            "approvals_required": analysis.get("approvals", ["PM", "QS"]),
            "recommendation": "Approve with conditions" if analysis.get("category") != "major" else "Escalate to senior management"
        }
    def _analyze_change_type(self, co_type: str, params: Dict) -> Dict:
        categories = {
            "scope_addition": ["add", "extra", "additional", "new work", "extra work"],
            "scope_omission": ["delete", "remove", "omit", "deduct"],
            "design_change": ["redesign", "change spec", "substitution"],
            "site_condition": [" differing site", "unforeseen", "latent", "ground condition"],
            "delay_claim": ["delay", "acceleration", "time extension", "EOT"]
        }
        text_lower = co_type.lower()
        detected_category = "general"
        confidence = 0
        for cat, keywords in categories.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > confidence:
                detected_category = cat
                confidence = matches
        return {
            "category": detected_category,
            "confidence": min(confidence / 3, 1.0),
            "complexity": "high" if len(co_type) > 500 else "medium" if len(co_type) > 200 else "low",
            "trade_involved": self._detect_trade_from_text(co_type)
        }
    def _calculate_co_cost_impact(self, direct_cost: float, analysis: Dict) -> Dict:
        direct = float(direct_cost) if direct_cost else 0
        overhead = direct * 0.20
        profit = direct * 0.10 if analysis.get("category") == "scope_addition" else 0
        complexity = analysis.get("complexity", "medium")
        risk_rates = {"low": 0.05, "medium": 0.10, "high": 0.20}
        risk_allowance = direct * risk_rates.get(complexity, 0.10)
        total = direct + overhead + profit + risk_allowance
        return {
            "direct_cost": direct,
            "overhead": overhead,
            "profit": profit,
            "risk_allowance": risk_allowance,
            "total": total,
            "breakdown_percentages": {
                "direct": f"{(direct/total*100):.1f}%" if total else "0%",
                "overhead": f"{(overhead/total*100):.1f}%" if total else "0%",
                "risk": f"{(risk_allowance/total*100):.1f}%" if total else "0%"
            }
        }
    def _create_risk_item(self, category: str, description: str, probability: str, impact: str, mitigation: str, source: str) -> Dict:
        return {
            "category": category,
            "description": description,
            "probability": probability,
            "impact": impact,
            "mitigation": mitigation,
            "source": source,
            "id": f"RISK-{hash(description) % 10000:04d}"
        }
    def _calculate_quantities(self, measurements: List[Dict]) -> Dict:
        # Floor area derives from measurements explicitly tagged `type=area`
        # (or `type=floor_area`). Previously this summed every `type=dimension`
        # value — which captured wall heights, room widths, beam spans, etc.
        # — and any stray dimension cascaded into concrete_volume and
        # steel_weight_kg below. Stricter type filter prevents the cascade.
        total_area = sum(
            m.get("value", 0)
            for m in measurements
            if m.get("type") in ("area", "floor_area")
        )
        direct_volume = sum(m.get("value", 0) for m in measurements if m.get("type") == "volume")
        counts = {m.get("item", "unknown"): m.get("value", 0) for m in measurements if m.get("type") == "count"}

        # Sanity cap — largest buildings in the world are ~500k m²
        total_area = min(total_area, 500_000)
        # Default slab thickness and rebar-per-m³ ratio come from the central
        # construction constants module — see app/core/construction_constants.py
        # for the values and a discussion of when they're appropriate.
        from app.core.construction_constants import (
            DEFAULT_SLAB_THICKNESS_M,
            DEFAULT_REBAR_RATIO_KG_PER_M3,
        )
        concrete_volume = (
            direct_volume if direct_volume > 0
            else total_area * DEFAULT_SLAB_THICKNESS_M
        )
        concrete_volume = min(concrete_volume, 750_000)
        # Only keep steel weight — rebar_length is redundant (same material,
        # causes double-counting). This is a quantity ROUGH estimate; a real
        # rebar takeoff comes from bar schedules or BIM, not concrete volume.
        steel_weight_kg = round(concrete_volume * DEFAULT_REBAR_RATIO_KG_PER_M3, 2)

        result = {
            "floor_area_m2": round(total_area, 2),
            "concrete_volume_m3": round(concrete_volume, 2),
            "steel_weight_kg": steel_weight_kg,
        }
        # Whitelist of construction-material substrings — anything outside this is
        # noise (e.g. "Server hall", "Purpose and Structure"). Match on lowercase
        # substring so plurals + adjectives still hit (e.g. "fire door" → door).
        material_whitelist = (
            "door", "window", "column", "beam", "slab", "wall", "panel",
            "glazing", "lintel", "lift", "elevator", "stair", "balustrade",
            "louvre", "louver", "screen", "cladding", "roof", "rebar",
            "anchor", "bolt", "fixture", "fitting", "valve", "duct",
            "pipe", "cable", "luminaire", "lamp", "switch", "socket",
            "outlet", "tile", "block", "brick", "kerb", "curb", "manhole",
            "bollard", "gate", "fence", "railing", "handrail",
            "pump", "fan", "tank", "boiler", "chiller", "ahu", "vav",
            "fcu", "diffuser", "grille", "extinguisher", "sprinkler",
            "hydrant", "detector", "sensor", "transformer", "generator",
            "panelboard", "switchboard", "busbar",
        )
        for item_name, count in counts.items():
            if not item_name or item_name == "unknown":
                continue
            # Collapse all whitespace (including embedded newlines from regex matches)
            clean = " ".join(str(item_name).split()).lower()
            if not clean or not any(m in clean for m in material_whitelist):
                continue
            key = clean.replace(" ", "_")[:25] + "_count"
            result[key] = int(count)
        return result
    def _estimate_costs(self, quantities: Dict, rates: Optional[Dict] = None) -> Dict:
        """Quick cost estimate from a quantities dict.

        Rates ($/m³ concrete, $/kg steel, etc.) MUST be supplied by the
        caller — this method has no hardcoded $150/m³ default and does not
        consult the benchmark. (For an INDICATIVE benchmark-priced estimate,
        use `cost_estimate`, which delegates to the historical_benchmark block
        and clearly flags the rates as indicative fallbacks.) Returns an error
        dict when no rates are provided.

        Note: the previous version also tried to read a `rebar_length_m`
        quantity key that `_calculate_quantities` never emits — the rebar
        cost line was always 0. Rebar weight is rolled into `steel_weight_kg`
        by `_calculate_quantities`, so there's no separate rebar line here.
        """
        rates = rates or {}
        required = ("concrete_usd_per_m3", "steel_usd_per_kg")
        missing = [k for k in required if not rates.get(k)]
        if missing:
            return {
                "status": "error",
                "error": (
                    "No rates supplied. Pass `rates={'concrete_usd_per_m3': X, "
                    "'steel_usd_per_kg': Y, ...}` from supplier quotes, the BOQ, "
                    "or learning_engine. Missing: " + ", ".join(missing)
                ),
            }
        # Optional overhead pct; default 0 so the breakdown stays additive
        # and the caller decides whether to apply OH/P/contingency.
        overhead_pct = float(rates.get("overhead_pct", 0))
        concrete_cost = quantities.get("concrete_volume_m3", 0) * rates["concrete_usd_per_m3"]
        steel_cost = quantities.get("steel_weight_kg", 0) * rates["steel_usd_per_kg"]
        subtotal = concrete_cost + steel_cost
        overhead = subtotal * overhead_pct
        return {
            "concrete_cost": round(concrete_cost, 2),
            "steel_cost": round(steel_cost, 2),
            "subtotal": round(subtotal, 2),
            "overhead_pct": overhead_pct,
            "overhead": round(overhead, 2),
            "total": round(subtotal + overhead, 2),
            "rates_used": rates,
        }
    def _estimate_carbon(self, quantities: Dict) -> Dict:
        # Default kgCO2e factors live in the central construction_constants
        # module. These are rule-of-thumb values and vary substantially by
        # mix design / steel manufacturing route — production work should
        # override with project-specific EPDs.
        from app.core.construction_constants import (
            DEFAULT_CONCRETE_KGCO2_PER_M3,
            DEFAULT_STEEL_KGCO2_PER_KG,
        )
        concrete_carbon = quantities.get("concrete_volume_m3", 0) * DEFAULT_CONCRETE_KGCO2_PER_M3
        steel_carbon = quantities.get("steel_weight_kg", 0) * DEFAULT_STEEL_KGCO2_PER_KG
    
        return {
            "concrete_co2_kg": round(concrete_carbon, 2),
            "steel_co2_kg": round(steel_carbon, 2),
            "total_embodied_carbon_kg": round(concrete_carbon + steel_carbon, 2)
        }
    async def value_engineering(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        current_boq = data.get("boq") or p.get("boq", [])
        cost_overrun_threshold = p.get("overrun_threshold", 0.10)
        target_reduction = p.get("target_reduction", 0.15)
        carbon_priority = p.get("carbon_priority", False)

        if not current_boq:
            return {
                "status": "error",
                "action": "value_engineering_analysis",
                "error": (
                    "Provide a BOQ / priced items — cannot invent "
                    "value-engineering alternatives"
                ),
            }

        alternatives = []
        for item in current_boq:
            item_alts = self._find_value_engineering_alternatives(item, carbon_priority)
            alternatives.extend(item_alts)
    
        viable_alternatives = [a for a in alternatives if a.get("viability_score", 0) > 0.7]
        scenarios = self._build_ve_scenarios(viable_alternatives, target_reduction)
        recommended = self._select_optimal_scenario(scenarios, cost_priority=not carbon_priority)
    
        return {
            "status": "success",
            "action": "value_engineering_analysis",
            "current_project_cost": sum(i.get("total_cost", 0) for i in current_boq),
            "analysis_parameters": {
                "cost_overrun_threshold": f"{cost_overrun_threshold*100}%",
                "target_reduction": f"{target_reduction*100}%",
                "carbon_priority": carbon_priority
            },
            "alternatives_identified": len(alternatives),
            "viable_alternatives": len(viable_alternatives),
            "by_category": self._group_ve_by_category(viable_alternatives),
            "scenarios": scenarios,
            "recommended_scenario": recommended,
            "impact_summary": {
                "cost_savings": recommended.get("cost_savings", 0),
                "cost_savings_percent": recommended.get("savings_percent", 0),
                "carbon_impact": recommended.get("carbon_delta", 0),
                "schedule_impact_days": recommended.get("schedule_impact", 0),
                "quality_impact": recommended.get("quality_impact", "neutral"),
                "risk_level": recommended.get("risk_level", "low")
            },
            "implementation_roadmap": self._generate_ve_roadmap(recommended),
            "approvals_required": self._identify_ve_approvals(recommended)
        }
    def _find_value_engineering_alternatives(self, boq_item: Dict, carbon_priority: bool) -> List[Dict]:
        material = boq_item.get("material_type", "concrete_c30")
        quantity = boq_item.get("quantity", 0)
        current_cost = boq_item.get("total_cost", 0)
        alternatives = []
    
        if "concrete" in material:
            alternatives.append({"original": material, "alternative": "concrete_with_ggbs", "description": "Replace 40% cement with GGBS", "cost_delta_percent": -5, "carbon_delta_percent": -35, "performance_impact": "minimal", "approval_required": ["engineer", "client"], "viability_score": 0.9})
            alternatives.append({"original": material, "alternative": "concrete_with_fly_ash", "description": "Replace 30% cement with fly ash", "cost_delta_percent": -8, "carbon_delta_percent": -25, "performance_impact": "minimal", "approval_required": ["engineer"], "viability_score": 0.85})
        elif "steel" in material:
            alternatives.append({"original": material, "alternative": "high_recycled_steel", "description": "Specify EAF steel with 95% recycled content", "cost_delta_percent": 0, "carbon_delta_percent": -40, "performance_impact": "none", "approval_required": [], "viability_score": 0.95})
        elif "block" in material:
            alternatives.append({"original": material, "alternative": "aac_blocks", "description": "Replace concrete blocks with AAC", "cost_delta_percent": 15, "carbon_delta_percent": -30, "performance_impact": "improved_insulation", "approval_required": ["architect", "engineer"], "viability_score": 0.8})
        elif "formwork" in material:
            alternatives.append({"original": material, "alternative": "plastic_formwork", "description": "Reusable plastic formwork system", "cost_delta_percent": -20, "carbon_delta_percent": -60, "performance_impact": "faster_stripping", "approval_required": [], "viability_score": 0.75, "note": "Requires minimum 10 reuses to break even"})
    
        for alt in alternatives:
            alt["cost_delta_amount"] = current_cost * alt["cost_delta_percent"] / 100
            alt["carbon_delta_amount"] = (boq_item.get("carbon_impact", 0) * alt["carbon_delta_percent"] / 100)
            alt["applies_to_boq_item"] = boq_item.get("id")
        return alternatives
    def _build_ve_scenarios(self, alternatives: List[Dict], target_reduction: float) -> Dict:
        total_savings = sum(a.get("cost_delta_amount", 0) for a in alternatives if a.get("cost_delta_amount", 0) < 0)
        total_carbon_savings = sum(a.get("carbon_delta_amount", 0) for a in alternatives if a.get("carbon_delta_amount", 0) < 0)
        return {
            "conservative": {"name": "conservative", "cost_savings": abs(total_savings) * 0.5, "savings_percent": 5, "carbon_delta": abs(total_carbon_savings) * 0.5, "schedule_impact": 0, "quality_impact": "neutral", "risk_level": "low"},
            "aggressive": {"name": "aggressive", "cost_savings": abs(total_savings), "savings_percent": min(abs(total_savings) / 100000 * 100, 20), "carbon_delta": abs(total_carbon_savings), "schedule_impact": 7, "quality_impact": "neutral", "risk_level": "medium"},
            "carbon_optimized": {"name": "carbon_optimized", "cost_savings": 0, "savings_percent": 0, "carbon_delta": abs(total_carbon_savings), "schedule_impact": 0, "quality_impact": "neutral", "risk_level": "low"}
        }
    async def tender_bid_analysis(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        bids = data.get("bids") or p.get("bids", [])
        evaluation_criteria = p.get("criteria", ["price", "schedule", "experience", "financial", "safety", "quality", "innovation"])
        project_type = p.get("project_type", "general_construction")
        weights = p.get("weights", {"price": 0.30, "schedule": 0.20, "experience": 0.15, "financial": 0.15, "safety": 0.10, "quality": 0.10})
    
        if not bids or len(bids) < 2:
            # Honest gate — NEVER fabricate bidders. A tender analysis that
            # invents contractors and prices could drive a real award decision.
            return {
                "status": "error",
                "action": "tender_bid_analysis",
                "error": (
                    "Tender bid analysis requires at least two real bids. Provide "
                    "bids as a list of {contractor_name, total_price, duration_days, "
                    "experience_score, financial_stability, safety_rating, quality_score}."
                ),
                "bids_supplied": len(bids),
            }

        # Normalize common aliases so callers can pass either total_price/amount
        # or duration_days/duration without KeyError crashes.
        normalized_bids = []
        for bid in bids:
            normalized_bids.append({
                **bid,
                "total_price": bid.get("total_price") if bid.get("total_price") is not None else bid.get("amount", 0),
                "duration_days": bid.get("duration_days") if bid.get("duration_days") is not None else bid.get("duration", 0),
            })
        bids = normalized_bids

        analyzed_bids = []
        for bid in bids:
            bidder_name = bid.get("contractor_name", "Unknown")
            bid_price = bid.get("total_price", 0)
            bid_duration = bid.get("duration_days", 0)
            all_prices = [b.get("total_price", 0) for b in bids]
            all_durations = [b.get("duration_days", 0) for b in bids]
        
            scores = {
                "price": self._score_price(bid_price, all_prices),
                "schedule": self._score_schedule(bid_duration, all_durations),
                "experience": bid.get("experience_score", 70),
                "financial": bid.get("financial_stability", 80),
                "safety": bid.get("safety_rating", 75),
                "quality": bid.get("quality_score", 75),
                "innovation": bid.get("innovation_score", 60)
            }
            weighted_score = sum(scores[k] * weights.get(k, 0.1) for k in scores)
            risks = self._assess_bidder_risk(bid, scores)
        
            analyzed_bids.append({
                "contractor": bidder_name,
                "bid_amount": bid_price,
                "duration_days": bid_duration,
                "unit_price_analysis": self._analyze_unit_prices(bid.get("boq", [])),
                "scores": scores,
                "weighted_score": round(weighted_score, 2),
                "rank": 0,
                "risk_level": risks["level"],
                "risk_factors": risks["factors"],
                "qualification_gaps": self._identify_qualification_gaps(bid),
                "alternatives_proposed": bid.get("alternatives", []),
                "clarifications_required": self._identify_bid_clarifications(bid)
            })
    
        analyzed_bids.sort(key=lambda x: x["weighted_score"], reverse=True)
        for i, bid in enumerate(analyzed_bids):
            bid["rank"] = i + 1
    
        best_value = analyzed_bids[0] if analyzed_bids else None
        lowest_price = min(analyzed_bids, key=lambda x: x["bid_amount"]) if analyzed_bids else None
        negotiation = self._generate_negotiation_strategy(analyzed_bids)
    
        return {
            "status": "success",
            "action": "tender_bid_analysis",
            # Sub-analyses that currently return empty results by construction
            # (_identify_qualification_gaps / _identify_bid_clarifications are
            # not implemented). Named here so the deliverable cannot read as a
            # full qualification review.
            "analysis_gaps": ["qualification_gaps", "clarifications_required"],
            "project_type": project_type,
            "bids_received": len(bids),
            "evaluation_criteria": evaluation_criteria,
            "weighting_applied": weights,
            "bid_comparison_matrix": analyzed_bids,
            "ranking": {
                "first": analyzed_bids[0] if len(analyzed_bids) > 0 else None,
                "second": analyzed_bids[1] if len(analyzed_bids) > 1 else None,
                "third": analyzed_bids[2] if len(analyzed_bids) > 2 else None
            },
            "price_analysis": {
                "lowest_bid": lowest_price["bid_amount"] if lowest_price else 0,
                "highest_bid": max(analyzed_bids, key=lambda x: x["bid_amount"])["bid_amount"] if analyzed_bids else 0,
                "average_bid": sum(b["bid_amount"] for b in analyzed_bids) / len(analyzed_bids) if analyzed_bids else 0,
                "best_value_bid": best_value["bid_amount"] if best_value else 0,
                "price_spread_percent": ((max(analyzed_bids, key=lambda x: x["bid_amount"])["bid_amount"] / min(analyzed_bids, key=lambda x: x["bid_amount"])["bid_amount"] - 1) * 100) if analyzed_bids and min(analyzed_bids, key=lambda x: x["bid_amount"])["bid_amount"] > 0 else 0
            },
            "risk_assessment": {
                "high_risk_bidders": [b["contractor"] for b in analyzed_bids if b["risk_level"] == "high"],
                "mitigation_required": any(b["risk_level"] == "high" for b in analyzed_bids)
            },
            "recommendation": {
                "award_to": best_value["contractor"] if best_value else None,
                "confidence": "high" if best_value and best_value["weighted_score"] > 80 else "medium",
                "negotiation_strategy": negotiation,
                "clarifications_needed": sum(len(b["clarifications_required"]) for b in analyzed_bids)
            },
            "award_summary": f"Recommend award to {best_value['contractor']} at {best_value['bid_amount']}" if best_value else "No recommendation possible"
        }
    def _score_price(self, price: float, all_prices: List[float]) -> float:
        if not all_prices or price <= 0:
            return 50
        avg = sum(all_prices) / len(all_prices)
        min_p = min(all_prices)
        if price == min_p:
            return 100
        elif price <= avg:
            return 80
        elif price <= avg * 1.1:
            return 60
        return 40
    def _assess_bidder_risk(self, bid: Dict, scores: Dict) -> Dict:
        factors = []
        if scores["financial"] < 60:
            factors.append("Financial stability concerns")
        if scores["safety"] < 70:
            factors.append("Below average safety record")
        if scores["experience"] < 50:
            factors.append("Limited relevant experience")
        boq = bid.get("boq", [])
        if boq:
            unit_prices = [item.get("unit_price", 0) for item in boq if item.get("unit_price", 0) > 0]
            if unit_prices:
                avg_price = sum(unit_prices) / len(unit_prices)
                high_items = [i for i in boq if i.get("unit_price", 0) > avg_price * 3]
                if len(high_items) > len(boq) * 0.1:
                    factors.append("Unbalanced bid detected - front loading")
        level = "high" if len(factors) >= 2 else "medium" if len(factors) == 1 else "low"
        return {"level": level, "factors": factors}
    def _analyze_unit_prices(self, boq: List[Dict]) -> Dict:
        if not boq:
            return {}
        prices = [i.get("unit_price", 0) for i in boq]
        return {
            "total_items": len(boq),
            "price_range": {"min": min(prices), "max": max(prices)} if prices else {},
            "average_unit_price": sum(prices) / len(prices) if prices else 0,
            "high_value_items": sorted(boq, key=lambda x: x.get("quantity", 0) * x.get("unit_price", 0), reverse=True)[:5]
        }
    def _generate_negotiation_strategy(self, bids: List[Dict]) -> List[Dict]:
        if len(bids) < 2:
            return []
        best = bids[0]
        second = bids[1]
        strategies = []
        price_gap = second["weighted_score"] - best["weighted_score"]
        if price_gap < 10:
            strategies.append({"tactic": "competitive dialogue", "target": second["contractor"], "approach": "Request best and final offer"})
        if best["risk_level"] == "medium":
            strategies.append({"tactic": "risk mitigation", "target": best["contractor"], "approach": "Request parent company guarantee"})
        return strategies
    async def variation_order_manager(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        vo_data = data.get("variation_data") or p.get("variation_data", {})
        existing_vos = data.get("existing_vos") or p.get("existing_vos", [])
        contract_file = data.get("contract_file") or p.get("contract_file")
        contract_value = data.get("contract_value") or p.get("contract_value") or 0
        ask_text = " ".join(
            str(x)
            for x in (
                data.get("message"),
                data.get("text"),
                data.get("user_message"),
                p.get("user_message"),
                p.get("text"),
            )
            if x
        )

        if not vo_data and not ask_text:
            return {"status": "error", "error": "Provide vo_data with at least one variation"}
        if not _variation_has_draft_facts(vo_data, ask_text):
            return {
                "status": "error",
                "action": "variation_order_manager",
                "error": (
                    "Cannot draft a variation order with no scope and no cost"
                ),
            }

        priced_lines = list(vo_data.get("lines") or []) if isinstance(vo_data.get("lines"), list) else []
        if not priced_lines:
            priced_lines = _parse_vo_priced_lines(
                " ".join(
                    x for x in (
                        vo_data.get("description"),
                        ask_text,
                    ) if x
                )
            )
        priced_net = (
            round(sum(float(l.get("amount") or 0) for l in priced_lines), 2)
            if priced_lines else None
        )

        vo_description = str(vo_data.get("description") or "").strip()
        if priced_lines and (
            not vo_description or _DRAFT_VO_ASK_RE.search(vo_description)
        ):
            vo_description = "; ".join(
                f"{l.get('kind')} {l.get('quantity')} {l.get('unit')} "
                f"{l.get('description')}".strip()
                for l in priced_lines
            )
        if not vo_description:
            vo_description = _vo_scope_after_ask(ask_text)

        if priced_lines:
            kinds = {str(l.get("kind") or "").upper() for l in priced_lines}
            if kinds == {"OMIT"}:
                default_type = "omission"
            elif kinds == {"ADD"}:
                default_type = "addition"
            else:
                default_type = "mixed"
        else:
            default_type = "addition"
        vo_type = vo_data.get("type", default_type)
        vo_number = vo_data.get(
            "vo_number",
            f"VO-D-{len(existing_vos)+1:03d}" if priced_lines else f"VO-{len(existing_vos)+1:03d}",
        )

        contract_terms = {}
        if contract_file:
            contract_data = await self.process_contract({"file_path": contract_file}, {})
            contract_terms = self._extract_variation_clauses(contract_data)

        category = self._categorize_variation(vo_description)
        if priced_lines and priced_net is not None:
            # Operator-stated ADD/OMIT rates are the draft; do not reload
            # prelim / OH / profit on top (live MATCH is the line net).
            pricing = {
                "direct": priced_net,
                "indirect": 0.0,
                "overhead": 0.0,
                "profit": 0.0,
                "total": priced_net,
                "breakdown": vo_data.get("resource_breakdown", {}),
            }
        else:
            pricing = self._calculate_variation_price(vo_data, vo_type)
        if contract_value:
            cumulative = self._calculate_cumulative_variations(
                existing_vos, pricing["total"], contract_value,
            )
            if isinstance(cumulative, dict) and cumulative.get("status") == "error":
                return cumulative
        elif priced_lines:
            # First-ask priced draft is self-contained; percent-of-contract
            # stays unset until a contract value is supplied.
            cumulative = {
                "previous_vo_count": len(existing_vos),
                "previous_vo_value": sum(
                    v.get("total", v.get("value", 0)) for v in existing_vos
                ),
                "this_vo_value": pricing["total"],
                "cumulative_value": sum(
                    v.get("total", v.get("value", 0)) for v in existing_vos
                ) + pricing["total"],
                "percent_of_contract": None,
                "approaching_cap": False,
            }
        else:
            cumulative = self._calculate_cumulative_variations(
                existing_vos, pricing["total"], contract_value,
            )
            if isinstance(cumulative, dict) and cumulative.get("status") == "error":
                return cumulative
        workflow = self._determine_approval_workflow(
            pricing["total"],
            float(cumulative.get("percent_of_contract") or 0),
            vo_type,
        )
        schedule_impact = vo_data.get("schedule_impact_days", 0)
        vo_document = self._generate_vo_document(
            vo_number, vo_description, pricing, vo_type, lines=priced_lines,
        )
    
        return {
            "status": "success",
            "action": "variation_order_processed",
            "vo_number": vo_number,
            "vo_type": vo_type,
            "category": category,
            "description": vo_description[:200] if vo_description else "",
            "lines": priced_lines,
            "net": priced_net if priced_net is not None else pricing["total"],
            "pricing": {
                "direct_costs": pricing["direct"],
                "indirect_costs": pricing["indirect"],
                "overhead": pricing["overhead"],
                "profit": pricing["profit"],
                "total_value": pricing["total"],
                "breakdown_by_resource": pricing["breakdown"]
            },
            "cumulative_impact": cumulative,
            "approval_workflow": workflow,
            "schedule_impact": {
                "days": schedule_impact,
                "critical_path": vo_data.get("critical_path", False),
                "justification": vo_data.get("delay_justification", "")
            },
            "contract_compliance": {
                "variation_clause": (
                    contract_terms.get("clause_reference")
                    or vo_data.get("clause_reference")
                ),
                "entitlement_clear": (
                    contract_terms["clear_entitlement"]
                    if "clear_entitlement" in contract_terms
                    else vo_data.get("entitlement_clear")
                ),
                "pricing_methodology": (
                    contract_terms.get("pricing_method")
                    or vo_data.get("pricing_method")
                ),
                "notice_requirements_met": (
                    vo_data["notice_given"] if "notice_given" in vo_data else None
                ),
                "time_bar_risk": self._check_time_bar(existing_vos, vo_data),
                "note": contract_terms.get("note"),
            },
            "supporting_documents": self._list_vo_documents(vo_data),
            "document_content": vo_document,
            "recommended_action": "approve" if pricing["total"] < 50000 and workflow["level"] == "project_manager" else "escalate",
            "risk_flags": self._identify_vo_risks(vo_data, cumulative)
        }
    def _categorize_variation(self, description: str) -> str:
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["drawing", "spec", "design", "architect"]):
            return "design_change"
        elif any(w in desc_lower for w in ["unforeseen", "ground", "condition", "rock"]):
            return "unforeseen_condition"
        elif any(w in desc_lower for w in ["accelerate", "crash", "fast", "speed"]):
            return "acceleration"
        elif any(w in desc_lower for w in ["omission", "delete", "remove", "reduce"]):
            return "scope_reduction"
        elif any(w in desc_lower for w in ["delay", "disruption", "waiting", "standby"]):
            return "prolongation"
        return "scope_addition"
    def _calculate_variation_price(self, vo_data: Dict, vo_type: str) -> Dict:
        base_cost = vo_data.get("direct_cost", 0)
        quantity = vo_data.get("quantity", 1)
        direct = base_cost * quantity
        prelim_percent = 0.15 if vo_type != "omission" else 0
        indirect = direct * prelim_percent
        oh_percent = vo_data.get("overhead_percent", 0.10)
        profit_percent = vo_data.get("profit_percent", 0.08)
        overhead = (direct + indirect) * oh_percent if vo_type != "omission" else -(direct * oh_percent)
        profit = (direct + indirect) * profit_percent if vo_type != "omission" else -(direct * profit_percent)
        total = direct + indirect + overhead + profit
        return {"direct": round(direct, 2), "indirect": round(indirect, 2), "overhead": round(overhead, 2), "profit": round(profit, 2), "total": round(total, 2), "breakdown": vo_data.get("resource_breakdown", {})}
    def _calculate_cumulative_variations(self, existing: List[Dict], new_amount: float, contract_value: float = 0) -> Dict:
        if not contract_value or contract_value <= 0:
            return {"status": "error", "error": "contract_value required for variation calculations"}
        # Each prior VO was emitted by `_calculate_variation_price` (line 3737)
        # under the key "total". The legacy reader looked for "value" which
        # this code never wrote, so every existing VO contributed 0 and the
        # cumulative figure was wrong for every multi-VO project. Read "total"
        # with a fallback to "value" for any legacy persisted data.
        current_total = sum(
            v.get("total", v.get("value", 0)) for v in existing
        )
        new_total = current_total + new_amount
        return {
            "previous_vo_count": len(existing),
            "previous_vo_value": current_total,
            "this_vo_value": new_amount,
            "cumulative_value": new_total,
            "percent_of_contract": (new_total / contract_value * 100),
            "approaching_cap": new_total > contract_value * 0.2
        }
    def _determine_approval_workflow(self, value: float, percent: float, vo_type: str) -> Dict:
        if value < 10000:
            level = "project_manager"
            approvers = ["Project Manager"]
        elif value < 50000:
            level = "contracts_manager"
            approvers = ["Project Manager", "Contracts Manager"]
        elif value < 100000:
            level = "director"
            approvers = ["Project Manager", "Contracts Manager", "Director"]
        else:
            level = "board_client"
            approvers = ["Project Manager", "Contracts Manager", "Director", "Client"]
        if percent > 15:
            approvers.append("Client (Major Change)")
        return {"level": level, "required_approvers": approvers, "estimated_approval_days": len(approvers) * 2}
    async def cash_flow_forecast(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        schedule_file = data.get("schedule_file") or p.get("schedule_file")
        boq = data.get("boq") or p.get("boq", [])
        # Predefined dispatch sends the chat envelope with empty params — the
        # figures ("SAR 60M, 18-month project") then live only in `message`.
        # Parsed values fill gaps ONLY; explicit params/data always win.
        seen: list[str] = []
        for x in (
            data.get("message"),
            data.get("text"),
            data.get("user_message"),
            p.get("user_message"),
            p.get("brief"),
            input_data if isinstance(input_data, str) else "",
        ):
            s = str(x).strip() if x else ""
            if s and s not in seen:
                seen.append(s)
        fig = _cashflow_figures_from_message(" ".join(seen))
        contract_value = (data.get("contract_value") or p.get("contract_value", 0)
                          or fig.get("contract_value", 0))
        payment_terms = p.get("payment_terms", {"advance_payment": 0.10, "retention": 0.10, "payment_delay_days": 30, "mobilization_duration": 2})
        project_start = p.get("project_start_date", datetime.now(timezone.utc).isoformat())

        if not contract_value or contract_value <= 0:
            return {"status": "error", "error": "contract_value required for cash flow forecast"}

        activities = []
        if schedule_file:
            schedule_data = await self._parse_xer_file(schedule_file)
            if schedule_data.get("status") == "error":
                return schedule_data
            activities = schedule_data.get("activities", [])

        project_duration_months = (
            max(6, int(len(activities) / 20)) if activities
            else int(p.get("duration_months") or fig.get("duration_months") or 18)
        )
        monthly_forecast = []
        cumulative_percent = 0.0

        # Total amounts to track properly:
        # - advance_payment_total: lump sum paid in month 0
        # - advance_recovered_so_far: running deduction across subsequent
        #   months until the full advance is recouped
        # - retention_held_so_far: running balance of retention deducted
        # - retention_released_total: one-time release at substantial
        #   completion (progress crosses 0.95 threshold); flagged with a
        #   boolean to prevent the previous bug where retention was
        #   re-released every month at >= 95%.
        advance_payment_total = contract_value * payment_terms["advance_payment"]
        # Recover the advance over the first 80% of the project duration so
        # the contractor isn't repaying during early peak cash burn.
        recovery_window = max(1, int(project_duration_months * 0.8))
        advance_recovery_per_month = advance_payment_total / recovery_window
        advance_recovered_so_far = 0.0

        retention_held_so_far = 0.0
        retention_released_total = 0.0
        retention_already_released = False

        for month in range(project_duration_months):
            time_percent = (month + 1) / project_duration_months
            if time_percent <= 0.25:
                progress = time_percent * 0.8
            elif time_percent <= 0.5:
                progress = 0.2 + (time_percent - 0.25) * 1.2
            elif time_percent <= 0.75:
                progress = 0.5 + (time_percent - 0.5) * 1.2
            else:
                progress = min(0.95, 0.8 + (time_percent - 0.75) * 0.6)

            monthly_value = (progress - cumulative_percent) * contract_value
            cumulative_percent = progress

            # Monthly retention deducted from the contractor's payment.
            retention_deduction = monthly_value * payment_terms["retention"]
            retention_held_so_far += retention_deduction

            # Retention release fires ONCE when progress crosses 0.95.
            # The previous code emitted the full release every month >= 0.95,
            # triple-counting it across multiple qualifying months.
            if progress >= 0.95 and not retention_already_released:
                retention_release = round(retention_held_so_far, 2)
                retention_released_total = retention_release
                retention_held_so_far = 0.0
                retention_already_released = True
            else:
                retention_release = 0.0

            # Cash in from the certification, BEFORE adjustments.
            cash_in = monthly_value - retention_deduction

            # Lump-sum advance paid in month 0.
            if month == 0:
                cash_in += advance_payment_total

            # Advance recovery: subtract a fixed slice in each month within
            # the recovery window. Previously this field was computed but
            # never applied, so the contractor "kept" the full advance forever.
            if month < recovery_window and advance_recovered_so_far < advance_payment_total:
                this_month_recovery = min(
                    advance_recovery_per_month,
                    advance_payment_total - advance_recovered_so_far,
                )
                advance_recovered_so_far += this_month_recovery
                cash_in -= this_month_recovery
            else:
                this_month_recovery = 0.0

            # Add back the retention release in the qualifying month.
            cash_in += retention_release

            monthly_forecast.append({
                "month": month + 1,
                "period": self._add_months(project_start, month),
                "planned_progress_percent": round(progress * 100, 2),
                "monthly_value": round(monthly_value, 2),
                "cumulative_value": round(progress * contract_value, 2),
                "advance_recovery": round(this_month_recovery, 2),
                "advance_recovered_to_date": round(advance_recovered_so_far, 2),
                "retention_deduction": round(retention_deduction, 2),
                "retention_held_to_date": round(retention_held_so_far + retention_released_total, 2),
                "retention_release": round(retention_release, 2),
                "net_cash_in": round(cash_in, 2),
                "cumulative_cash": round(sum(m["net_cash_in"] for m in monthly_forecast) + cash_in, 2),
            })
    
        total_revenue = sum(m["monthly_value"] for m in monthly_forecast)
        peak_month = max(monthly_forecast, key=lambda x: x["monthly_value"]) if monthly_forecast else None
        avg_monthly = total_revenue / project_duration_months if project_duration_months > 0 else 0
    
        return {
            "status": "success",
            "action": "cash_flow_forecast",
            "project_parameters": {
                "contract_value": contract_value,
                "duration_months": project_duration_months,
                "start_date": project_start,
                "payment_terms": payment_terms
            },
            "s_curve_data": monthly_forecast,
            "summary_metrics": {
                "total_planned_revenue": round(total_revenue, 2),
                "peak_monthly_billing": round(peak_month["monthly_value"], 2) if peak_month else 0,
                "peak_month": peak_month["month"] if peak_month else None,
                "average_monthly_billing": round(avg_monthly, 2),
                # Cumulative retention held minus released. Previously this
                # reported the last month's single deduction, which is not a
                # balance figure. Use the latest "retention_held_to_date".
                "final_retention_balance": round(
                    monthly_forecast[-1]["retention_held_to_date"]
                    - retention_released_total
                    if monthly_forecast else 0, 2
                ),
                "advance_recovered_total": round(advance_recovered_so_far, 2),
                "retention_released_total": round(retention_released_total, 2),
                "cash_flow_peak_month": peak_month["month"] if peak_month else None
            },
            "funding_requirements": {
                "working_capital_peak": round(peak_month["monthly_value"] * 0.3 if peak_month else 0, 2),
                "mobilization_costs": round(contract_value * 0.05, 2)
            },
            "risk_adjusted_scenarios": {
                "optimistic": [{"month": m["month"], "value": m["monthly_value"] * 1.1} for m in monthly_forecast],
                "pessimistic": [{"month": m["month"], "value": m["monthly_value"] * 0.85} for m in monthly_forecast],
                "delayed_start": [{"month": m["month"], "value": m["monthly_value"]} for m in [{"month": 1, "monthly_value": 0}] + monthly_forecast[:-1]]
            },
            "chart_data": {
                "labels": [f"Month {m['month']}" for m in monthly_forecast],
                "planned_value": [m["cumulative_value"] for m in monthly_forecast],
                # Earned Value (EV) and Actual Cost (AC) require real progress
                # measurement data — they cannot be derived from the planned
                # forecast. Previously this code emitted EV = PV * 0.95 and
                # AC = PV * 1.02, which are fabricated multipliers, not
                # measurements. EV/AC are now nulls; callers must supply
                # actual progress + cost data via a future progress-tracking
                # endpoint to populate them.
                "earned_value": [None] * len(monthly_forecast),
                "actual_cost": [None] * len(monthly_forecast),
                "_ev_ac_note": (
                    "Earned Value and Actual Cost require measured progress "
                    "and incurred-cost data; they are nulled here so the "
                    "chart does not silently display fabricated curves."
                ),
            }
        }
    async def procurement_optimizer(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        boq = data.get("boq") or p.get("boq", [])
        suppliers = data.get("suppliers") or p.get("suppliers", [])
        constraints = p.get("constraints", {"max_suppliers": 5, "geographic_limit": None, "quality_threshold": 80, "payment_terms_preference": "net_30"})
    
        # Score ONLY what the caller supplied.
        #
        # These six used to default to 70/75/80/80/60/70. Nothing in the
        # platform produces `price_score`, `delivery_score`, `quality_score`,
        # `financial_score`, `esg_score` or `support_score` — grep them — so a
        # supplier list without them (which is every list the platform can
        # build itself) gave every supplier the identical invented scorecard
        # and an identical total of 73.8. The action then sorted that and
        # presented a ranking and a recommended supplier. The output looked
        # like an assessment and was an artefact of the defaults; worse, the
        # visible result was a confident ordering rather than the tie that
        # would have exposed it.
        #
        # Unscored suppliers now carry total_score None and are listed after
        # the scored ones, with the gap named in `scoring_basis`.
        _SCORE_FIELDS = {
            "price_competitiveness": ("price_score", 0.25),
            "delivery_reliability": ("delivery_score", 0.25),
            "quality_rating": ("quality_score", 0.20),
            "financial_stability": ("financial_score", 0.15),
            "sustainability": ("esg_score", 0.10),
            "technical_support": ("support_score", 0.05),
        }
        scored_suppliers = []
        unscored_count = 0
        for supplier in suppliers:
            scores = {
                label: supplier[key]
                for label, (key, _w) in _SCORE_FIELDS.items()
                if isinstance(supplier.get(key), (int, float))
                and not isinstance(supplier.get(key), bool)
            }
            supplied_weight = sum(
                w for label, (_k, w) in _SCORE_FIELDS.items() if label in scores
            )
            if supplied_weight:
                # Re-normalise over the criteria actually provided, so a
                # supplier scored on three criteria is not penalised against
                # one scored on six.
                total_score = round(sum(
                    scores[label] * _SCORE_FIELDS[label][1] for label in scores
                ) / supplied_weight, 1)
                basis = (
                    f"{len(scores)} of {len(_SCORE_FIELDS)} criteria supplied "
                    f"({', '.join(sorted(scores))})"
                )
            else:
                total_score = None
                basis = "No supplier scores were supplied — not assessed"
                unscored_count += 1
            scored_suppliers.append({
                "name": supplier.get("name"),
                "scores": scores,
                "total_score": total_score,
                "scoring_basis": basis,
                "lead_time_weeks": supplier.get("lead_time", 4),
                "payment_terms": supplier.get("payment_terms", "net_30"),
                "certifications": supplier.get("certifications", []),
                "geographic_location": supplier.get("location"),
                "capabilities": supplier.get("capabilities", []),
                "recommended_for": []
            })

        # Unscored suppliers sort last and keep their None rather than being
        # coerced to 0 — a 0 would read as "assessed and terrible".
        scored_suppliers.sort(
            key=lambda x: (x["total_score"] is not None, x["total_score"] or 0),
            reverse=True,
        )
    
        procurement_plan = []
        for item in boq:
            material = item.get("material_type", "general")
            qty = item.get("quantity", 0)
            required_date = item.get("required_date")
            capable_suppliers = [s for s in scored_suppliers if material in s.get("capabilities", []) or not s.get("capabilities")]
            if capable_suppliers:
                best = capable_suppliers[0]
                order_date = self._subtract_weeks(required_date, best["lead_time_weeks"]) if required_date else "ASAP"
                procurement_plan.append({
                    "material": material,
                    "boq_item": item.get("id"),
                    "quantity": qty,
                    "unit": item.get("unit"),
                    "required_date": required_date,
                    "recommended_supplier": best["name"],
                    "supplier_score": best["total_score"],
                    "order_date": order_date,
                    "order_lead_time": best["lead_time_weeks"],
                    "buffer_weeks": 2,
                    "packaging_strategy": "bulk" if qty > 100 else "standard",
                    "inspection_required": item.get("quality_critical", False),
                    "alternative_suppliers": [s["name"] for s in capable_suppliers[1:3]]
                })
    
        insights = self._generate_procurement_insights(procurement_plan, scored_suppliers)
        risks = self._identify_procurement_risks(procurement_plan)
    
        return {
            "status": "success",
            "action": "procurement_optimization",
            "suppliers_evaluated": len(suppliers),
            "top_suppliers": scored_suppliers[:constraints["max_suppliers"]],
            "procurement_plan": {
                "total_items": len(procurement_plan),
                "total_value": sum(item.get("value", 0) for item in boq),
                "critical_path_items": len([p for p in procurement_plan if p["inspection_required"]]),
                "plan": procurement_plan
            },
            "optimization_insights": insights,
            "consolidation_opportunities": self._identify_consolidation(procurement_plan),
            "bundle_recommendations": self._suggest_bundling(procurement_plan, scored_suppliers),
            "risk_mitigation": risks,
            "timeline": {
                "earliest_order": min((p["order_date"] for p in procurement_plan if p["order_date"] != "ASAP"), default="N/A"),
                "latest_order": max((p["order_date"] for p in procurement_plan if p["order_date"] != "ASAP"), default="N/A")
            }
        }
    def _generate_procurement_insights(self, plan: List[Dict], suppliers: List[Dict]) -> List[str]:
        insights = []
        long_lead_items = [p for p in plan if p.get("order_lead_time", 0) > 8]
        if long_lead_items:
            insights.append(f"Attention: {len(long_lead_items)} long-lead items require immediate ordering")
        single_source = [p for p in plan if len(p.get("alternative_suppliers", [])) == 0]
        if single_source:
            insights.append(f"Risk: {len(single_source)} items have single-source dependency")
        # `supplier_score` is None for suppliers the caller gave no scores for
        # (see procurement_optimizer). Averaging over those would have been a
        # TypeError before; treating them as 0 would be worse — it would drag
        # the average under 75 and emit a re-tender recommendation caused by
        # missing data rather than by weak suppliers.
        scored = [p["supplier_score"] for p in plan
                  if isinstance(p.get("supplier_score"), (int, float))]
        if not scored:
            if plan:
                insights.append(
                    "No supplier scores were supplied, so supplier quality was not "
                    "assessed — this plan ranks on lead time and price only."
                )
        else:
            avg_score = sum(scored) / len(scored)
            if avg_score < 75:
                insights.append(
                    f"Consider re-tendering: average supplier score {avg_score:.0f} "
                    f"is below 75 (across {len(scored)} of {len(plan)} scored items)"
                )
        return insights
    async def esg_sustainability_report(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        project_data = data.get("project_data") or p.get("project_data", {})
        boq = data.get("boq") or p.get("boq", [])
        manpower_data = data.get("manpower") or p.get("manpower", {})
        safety_records = data.get("safety_records") or p.get("safety_records", [])
        reporting_period = p.get("period", "annual")
    
        env_metrics = await self._calculate_environmental_metrics(boq, project_data)
        social_metrics = self._calculate_social_metrics(manpower_data, safety_records)
        gov_metrics = self._calculate_governance_metrics(project_data)

        # Score a pillar ONLY when its inputs were supplied. Social/governance
        # metrics are placeholder zeros without project data; scoring them anyway
        # invents a rating — _score_social returns 80 for ltifr==0 (which means
        # "no data", not "zero injuries") and _score_governance a flat 70 — and
        # folds that fiction into the composite + letter grade. (V2)
        env_available = bool(boq)
        social_available = bool(manpower_data or safety_records)
        gov_available = bool(project_data)

        pillar_scores = {
            "environmental": self._score_environmental(env_metrics) if env_available else None,
            "social": self._score_social(social_metrics) if social_available else None,
            "governance": self._score_governance(gov_metrics) if gov_available else None,
        }
        assessed = {k: v for k, v in pillar_scores.items() if v is not None}
        overall = (sum(assessed.values()) / len(assessed)) if assessed else None
        complete = len(assessed) == 3

        # None-safe view for the eligibility / recommendation / narrative helpers:
        # a pillar with no data counts as 0 there (never grants false credit).
        scores = {k: (v if v is not None else 0) for k, v in pillar_scores.items()}
        scores["overall"] = overall if overall is not None else 0

        benchmarks = {"industry_average": 65, "best_practice": 85, "your_score": scores["overall"]}
        certifications = self._check_certification_eligibility(scores, env_metrics)
        sdg_alignment = self._map_to_sdgs(env_metrics, social_metrics)
    
        return {
            "status": "success",
            "action": "esg_sustainability_report",
            "reporting_period": reporting_period,
            "esg_scores": {
                "environmental": round(pillar_scores["environmental"], 1) if pillar_scores["environmental"] is not None else None,
                "social": round(pillar_scores["social"], 1) if pillar_scores["social"] is not None else None,
                "governance": round(pillar_scores["governance"], 1) if pillar_scores["governance"] is not None else None,
                "overall": round(overall, 1) if overall is not None else None,
                "rating": (
                    None if overall is None
                    else ("A" if overall >= 80 else "B" if overall >= 65 else "C" if overall >= 50 else "D")
                    if complete
                    else f"partial ({len(assessed)}/3 pillars: {', '.join(sorted(assessed))})"
                ),
                "data_status": {
                    "pillars_assessed": sorted(assessed.keys()),
                    "pillars_missing_data": sorted(k for k, v in pillar_scores.items() if v is None),
                    "complete": complete,
                    "note": None if complete else (
                        "Overall score and rating reflect only pillars with supplied data; "
                        "pillars without data are excluded, not scored as zero. Provide "
                        "manpower/safety_records (social) and project_data (governance) for a full ESG rating."
                    ),
                },
            },
            "environmental": {
                "carbon_emissions_tons": env_metrics.get("total_carbon", 0),
                "carbon_intensity": env_metrics.get("carbon_per_value", 0),
                "energy_consumption_mwh": env_metrics.get("energy", 0),
                "water_usage_m3": env_metrics.get("water", 0),
                "waste_generated_tons": env_metrics.get("waste", 0),
                "waste_diversion_percent": env_metrics.get("waste_diversion", 0),
                "recycled_materials_percent": env_metrics.get("recycled_content", 0),
                "local_materials_percent": env_metrics.get("local_content", 0)
            },
            "social": {
                "total_workforce": social_metrics.get("total_workers", 0),
                "local_hire_percent": social_metrics.get("local_percent", 0),
                "safety_incidents": social_metrics.get("incidents", 0),
                "lost_time_injury_rate": social_metrics.get("ltifr", 0),
                "training_hours": social_metrics.get("training_hours", 0),
                "community_investment": social_metrics.get("community_spend", 0),
                "gender_diversity_percent": social_metrics.get("gender_diversity", 0),
                "local_business_engagement_percent": social_metrics.get("local_procurement", 0)
            },
            "governance": {
                "ethics_training_compliance": gov_metrics.get("ethics_training", 0),
                "anti_corruption_policies": gov_metrics.get("anti_corruption", True),
                "supply_chain_audit_percent": gov_metrics.get("supplier_audits", 0),
                "transparency_score": gov_metrics.get("transparency", 70)
            },
            "benchmarking": benchmarks,
            "certification_eligibility": certifications,
            "sdg_alignment": sdg_alignment,
            "recommendations": self._generate_esg_recommendations(scores, env_metrics, social_metrics),
            "improvement_targets": {
                "carbon_reduction_target_2030": "50% reduction",
                "net_zero_target": "2050",
                "zero_incident_target": "Ongoing"
            },
            "stakeholder_disclosure": self._generate_stakeholder_narrative(scores, env_metrics, social_metrics)
        }
    async def _calculate_environmental_metrics(self, boq: List[Dict], project: Dict) -> Dict:
        carbon_data = await self.carbon_footprint_calculator({"boq": boq}, {})
        total_carbon = carbon_data.get("summary", {}).get("total_embodied_carbon_kg", 0) / 1000
        total_value = sum(i.get("total_cost", 0) for i in boq)
        return {
            "total_carbon": total_carbon,
            "carbon_per_value": total_carbon / total_value if total_value else 0,
            "energy": total_value * 0.0005,
            "water": total_value * 0.5,
            "waste": total_carbon * 0.1,
            "waste_diversion": 60,
            "recycled_content": 15,
            "local_content": 70
        }
    def _calculate_social_metrics(self, manpower: Dict, safety: List) -> Dict:
        # Return numeric defaults so downstream scoring comparisons never
        # fail with "'<' not supported between instances of 'str' and 'int'".
        return {
            "total_workers": 0,
            "local_percent": 0,
            "incidents": 0,
            "ltifr": 0,
            "training_hours": 0,
            "community_spend": 0,
            "gender_diversity": 0,
            "local_procurement": 0,
            "note": "Field requires project-specific data; not auto-computed.",
        }
    def _calculate_governance_metrics(self, project: Dict) -> Dict:
        return {
            "ethics_training": 0,
            "anti_corruption": False,
            "supplier_audits": 0,
            "transparency": 50,
            "note": "Field requires project-specific data; not auto-computed.",
        }
    def _score_environmental(self, metrics: Dict) -> float:
        score = 50
        ci = metrics.get("carbon_per_value", 0)
        if ci < 0.1:
            score += 20
        elif ci < 0.2:
            score += 10
        if metrics.get("waste_diversion", 0) > 70:
            score += 10
        if metrics.get("recycled_content", 0) > 20:
            score += 10
        return min(100, score)
    def _score_social(self, metrics: Dict) -> float:
        score = 60
        ltifr = metrics.get("ltifr", 0)
        if ltifr == 0:
            score += 20
        elif ltifr < 2:
            score += 10
        if metrics.get("local_percent", 0) > 80:
            score += 10
        return min(100, score)
    def _check_certification_eligibility(self, scores: Dict, env: Dict) -> List[Dict]:
        certs = []
        if scores["environmental"] >= 75:
            certs.append({"certification": "LEED Gold", "eligible": scores["overall"] >= 70, "next_steps": "Submit for review" if scores["overall"] >= 70 else "Improve energy metrics"})
        if env.get("carbon_per_value", 999) < 0.15:
            certs.append({"certification": "BREEAM Excellent", "eligible": True, "next_steps": "Engage BREEAM assessor"})
        if scores["overall"] >= 80:
            certs.append({"certification": "WELL Building", "eligible": True, "next_steps": "Focus on occupant wellness features"})
        return certs
    def _map_to_sdgs(self, env: Dict, social: Dict) -> List[Dict]:
        sdgs = []
        if env.get("carbon_per_value", 0) < 0.2:
            sdgs.append({"goal": 13, "name": "Climate Action", "contribution": "Low carbon construction"})
        if social.get("local_percent", 0) > 70:
            sdgs.append({"goal": 8, "name": "Decent Work", "contribution": "Local employment"})
        if env.get("waste_diversion", 0) > 50:
            sdgs.append({"goal": 12, "name": "Responsible Consumption", "contribution": "Waste reduction"})
        return sdgs
    async def extract_measurements(self, input_data: Any, params: Dict) -> Dict:
        """Extract measurements from construction drawings"""
        if self._looks_like_file(input_data, params):
            result = await self.process_document(input_data, params)
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "measurements": result.get("measurements", []),
                    "specifications": result.get("specifications", []),
                    "count": len(result.get("measurements", [])),
                    # Pass the measured confidence through unchanged.
                    #
                    # This used to read `.get("measurement_extraction", 0)`.
                    # `measurement_extraction` is not a key `process_document`
                    # has ever produced — that line was its only occurrence in
                    # the codebase — so every successful extraction reported
                    # confidence 0, which reads as "the extractor is not sure"
                    # rather than "the wrong key was asked for". The real
                    # contract is a dict: {"overall": 0.857, "signals": {...},
                    # "measured": true, "caveats": [...]}. Forwarded whole
                    # rather than reduced to `overall`, because the caveats say
                    # WHY it is not 1.0 and dropping them would be a quieter
                    # version of the same defect. None when the document path
                    # genuinely computes no confidence.
                    "confidence": result.get("confidence")
                }
            return result
    
        # Fallback: non-file requests — extract from PDF text if a PDF block is available
        pdf_block = self.get_dep("pdf")
        if pdf_block and input_data:
            pdf_result = await pdf_block.process(input_data, {"extract_tables": True})
            if pdf_result.get("status") == "success":
                text = pdf_result.get("result", {}).get("text", "")
                measurements = self._extract_measurements_advanced(text, {})
                return {
                    "status": "success",
                    "source": "pdf_extraction",
                    "measurements": measurements,
                    "count": len(measurements),
                    "extracted_text": text[:500],
                }
            return pdf_result

        return {
            "status": "error",
            "error": (
                "Nothing to extract — supply a construction drawing/document "
                "(file path or PDF input) for measurement extraction"
            ),
        }
    async def boq_process(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to BOQProcessorBlock: parse Excel/CSV BOQs."""
        block = self._resolve_block("boq_processor")
        if block is None:
            return {"status": "error", "error": "boq_processor block unavailable"}
        return await block.process(input_data, params)
    async def benchmark_lookup(self, input_data: Any, params: Dict) -> Dict:
        """Historical benchmark lookup — delegates to the historical_benchmark block.

        The block ships a small, conservative, region-adjusted rate book for
        common construction items. Unknown items return an honest error so the
        caller can supply supplier quotes or add a custom rate.
        """
        block = self._get_historical_benchmark_block()
        if block is None:
            return {
                "status": "error",
                "error": "historical_benchmark block unavailable",
            }
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        lookup_params = {
            "action": "lookup",
            "item": p.get("item") or data.get("item", ""),
            "unit": p.get("unit") or data.get("unit", ""),
            "location": p.get("location") or data.get("location", "US National Average"),
            "project_type": p.get("project_type") or data.get("project_type", "general_building"),
        }
        return await block.process(data, lookup_params)
