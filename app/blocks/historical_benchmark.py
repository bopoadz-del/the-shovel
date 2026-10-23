"""Historical benchmark block — lightweight, extensible unit-rate source.

Provides plausible, clearly-labeled construction unit rates for common items
(concrete, steel, formwork, etc.) with regional location factors. The block
is intentionally simple: no drifting hardcoded RS-Means snapshot, no ML model.
Rates can be overridden per-item via the learning_engine / record action so
the platform accumulates real project data over time.
"""

from typing import Any, Dict, Optional

from app.core.universal_base import UniversalBlock


# Base USD rates for frequently-estimated construction items.
# INDICATIVE US-national-average general-knowledge figures — a day-one FALLBACK,
# not a priced rate book. Every response flags them as indicative and recommends
# loading the project's own rates / supplier quotes.
#
# `basis` states what each rate INCLUDES so a caller never sums inconsistent
# bases blindly:
#   "material"     — supply only (no placing/fixing labour)
#   "all_in"       — supply-and-fix / in-place (material + labour)
#   "plant_labour" — plant + labour only (no material)
_BASIS_NOTE = {
    "material": "supply only — placing/fixing labour NOT included",
    "all_in": "supply-and-fix (material + labour) in place",
    "plant_labour": "plant + labour only — no material",
}
_BASE_RATES: Dict[str, Dict[str, Any]] = {
    "concrete": {"unit": "m3", "base_usd": 150.0, "basis": "material",
                 "note": "Ready-mix concrete supply, ~30 MPa (excl. place/pour/finish)."},
    "ready-mix concrete": {"unit": "m3", "base_usd": 150.0, "basis": "material",
                 "note": "Ready-mix concrete supply, ~30 MPa (excl. place/pour/finish)."},
    "steel": {"unit": "kg", "base_usd": 1.0, "basis": "material",
              "note": "Reinforcing steel / rebar supply (excl. cut/bend/fix)."},
    "rebar": {"unit": "kg", "base_usd": 1.0, "basis": "material",
              "note": "Reinforcing steel / rebar supply (excl. cut/bend/fix)."},
    "reinforcing steel": {"unit": "kg", "base_usd": 1.0, "basis": "material",
              "note": "Reinforcing steel / rebar supply (excl. cut/bend/fix)."},
    "structural steel": {"unit": "kg", "base_usd": 2.5, "basis": "material",
              "note": "Fabricated structural steel sections supply (excl. erection)."},
    "formwork": {"unit": "m2", "base_usd": 35.0, "basis": "all_in",
              "note": "Traditional formwork, incl. labour for erection/stripping."},
    "blockwork": {"unit": "m2", "base_usd": 45.0, "basis": "all_in",
              "note": "CMU blockwork, supply and lay."},
    "plaster": {"unit": "m2", "base_usd": 18.0, "basis": "all_in",
              "note": "Internal plaster finish, applied."},
    "paint": {"unit": "m2", "base_usd": 12.0, "basis": "all_in",
              "note": "Emulsion paint, two coats, applied."},
    "tiles": {"unit": "m2", "base_usd": 60.0, "basis": "all_in",
              "note": "Ceramic floor/wall tiles, incl. fixing."},
    "backfill": {"unit": "m3", "base_usd": 20.0, "basis": "all_in",
              "note": "Imported granular backfill and compaction."},
    "asphalt": {"unit": "m2", "base_usd": 40.0, "basis": "all_in",
              "note": "Asphalt paving, 50 mm, laid."},
    "excavation": {"unit": "m3", "base_usd": 15.0, "basis": "plant_labour",
              "note": "Bulk excavation, machine (plant + operator)."},
    "earthworks": {"unit": "m3", "base_usd": 15.0, "basis": "plant_labour",
              "note": "Bulk earthworks (plant + labour)."},
}

# Regional location cost factors (multipliers on the US-national-average base).
# INDICATIVE rough construction-cost indices from general knowledge — NOT a
# sourced index (Turner / RLB-style ballparks). Unknown location -> factor 1.0
# with `location_matched=False` so the caller knows it fell back to the US base.
_LOCATION_FACTORS: Dict[str, float] = {
    # North America
    "us national average": 1.0, "united states": 1.0, "usa": 1.0, "us": 1.0,
    "us nyc": 1.35, "new york": 1.35, "us sf": 1.40, "san francisco": 1.40,
    "canada": 1.15, "toronto": 1.20, "vancouver": 1.22,
    # Middle East / GCC
    "saudi arabia": 0.85, "saudi": 0.85, "ksa": 0.85, "riyadh": 0.85,
    "jeddah": 0.88, "dammam": 0.84, "neom": 0.95,
    "uae": 0.95, "united arab emirates": 0.95, "dubai": 0.95,
    "abudhabi": 0.95, "abu dhabi": 0.95, "sharjah": 0.92,
    "qatar": 0.98, "doha": 0.98, "kuwait": 0.92, "bahrain": 0.90,
    "oman": 0.82, "muscat": 0.82, "jordan": 0.70, "amman": 0.70,
    "lebanon": 0.72, "iraq": 0.80, "baghdad": 0.80,
    # North Africa
    "egypt": 0.55, "cairo": 0.55, "morocco": 0.60, "casablanca": 0.60,
    "algeria": 0.58, "tunisia": 0.55, "libya": 0.62,
    # Sub-Saharan Africa
    "south africa": 0.65, "johannesburg": 0.65, "cape town": 0.66,
    "nigeria": 0.70, "lagos": 0.70, "kenya": 0.60, "nairobi": 0.60,
    "ghana": 0.65, "ethiopia": 0.55, "tanzania": 0.58,
    # Europe
    "uk": 1.25, "united kingdom": 1.25, "london": 1.35, "ireland": 1.20,
    "dublin": 1.22, "germany": 1.30, "france": 1.25, "paris": 1.32,
    "netherlands": 1.30, "spain": 1.00, "madrid": 1.02, "italy": 1.05,
    "poland": 0.80, "switzerland": 1.70, "zurich": 1.75, "sweden": 1.35,
    "norway": 1.55, "portugal": 0.95, "turkey": 0.65, "istanbul": 0.65,
    # Asia-Pacific
    "australia": 1.30, "sydney": 1.35, "melbourne": 1.30,
    "new zealand": 1.20, "singapore": 1.20, "japan": 1.25, "tokyo": 1.35,
    "south korea": 1.00, "seoul": 1.02, "hong kong": 1.45,
    "china": 0.60, "shanghai": 0.70, "beijing": 0.68,
    "malaysia": 0.55, "kuala lumpur": 0.55, "indonesia": 0.50, "jakarta": 0.50,
    "philippines": 0.50, "manila": 0.50, "thailand": 0.55, "bangkok": 0.55,
    "vietnam": 0.45, "india": 0.35, "mumbai": 0.35, "delhi": 0.35,
    "pakistan": 0.35, "bangladesh": 0.35, "sri lanka": 0.40,
    # Latin America
    "brazil": 0.65, "sao paulo": 0.68, "mexico": 0.60, "mexico city": 0.62,
    "argentina": 0.55, "chile": 0.75, "colombia": 0.58, "peru": 0.55,
}

# Project-type factors.
_PROJECT_TYPE_FACTORS: Dict[str, float] = {
    "general_building": 1.0,
    "building": 1.0,
    "data_center": 1.15,
    "solar_plant": 0.95,
    "wind_farm": 1.05,
    "infrastructure": 0.95,
    "industrial": 1.10,
    "residential": 0.95,
    "commercial": 1.0,
    "healthcare": 1.20,
}


def _normalize(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
    )


def _match_item(name: str) -> Optional[str]:
    norm = _normalize(name)
    # Exact key match first.
    if norm in _BASE_RATES:
        return norm
    # Substring match: item name contains a known key.
    for key in _BASE_RATES:
        if key in norm or norm in key:
            return key
    return None


def _location_factor(location: str) -> tuple[float, bool]:
    """Return (factor, matched). Unknown location -> (1.0, False) so the caller
    can see it fell back to the US-national-average base rather than silently
    pricing an unlisted country at 1.0."""
    norm = _normalize(location)
    if norm in _LOCATION_FACTORS:
        return _LOCATION_FACTORS[norm], True
    return 1.0, False


def _project_factor(project_type: str) -> float:
    return _PROJECT_TYPE_FACTORS.get(_normalize(project_type), 1.0)


def supported_locations() -> list:
    """Distinct location keys the benchmark can price (deduped, sorted)."""
    return sorted(set(_LOCATION_FACTORS))


class HistoricalBenchmarkBlock(UniversalBlock):
    name = "historical_benchmark"
    version = "1.0.0"
    description = "Indicative multi-country construction unit-rate benchmark (day-one fallback)"
    layer = 2
    tags = ["construction", "cost", "benchmark", "unit-rates"]
    requires = []

    default_config = {
        "confidence": "low",
        "source_note": ("Indicative general-knowledge rates (multi-country cost "
                        "factors) — a day-one fallback. Replace with your project "
                        "rates / supplier quotes for pricing."),
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        action = params.get("action") or input_data.get("action") if isinstance(input_data, dict) else None

        if action == "record" and isinstance(input_data, dict):
            # Future: persist user-supplied rates to learning_engine.
            return {
                "status": "success",
                "action": "record",
                "message": "Rate sample recorded (stub — learning_engine integration pending).",
            }

        # Enumerate the countries/locations this benchmark can price.
        if action == "list_locations":
            return {
                "status": "success",
                "action": "list_locations",
                "count": len(supported_locations()),
                "locations": supported_locations(),
            }

        item = params.get("item") or (input_data.get("item") if isinstance(input_data, dict) else "")
        unit = params.get("unit") or (input_data.get("unit") if isinstance(input_data, dict) else "")
        location = params.get("location") or (input_data.get("location") if isinstance(input_data, dict) else "US National Average")
        project_type = params.get("project_type") or (input_data.get("project_type") if isinstance(input_data, dict) else "general_building")

        key = _match_item(item)
        if not key:
            return {
                "status": "error",
                "action": "lookup",
                "item": item,
                "error": f"No benchmark rate available for '{item}'. Provide a supplier quote or add a custom rate.",
            }

        base = _BASE_RATES[key]
        loc_factor, loc_matched = _location_factor(location)
        proj_factor = _project_factor(project_type)
        adjusted = round(base["base_usd"] * loc_factor * proj_factor, 2)
        basis = base.get("basis", "all_in")

        return {
            "status": "success",
            "action": "lookup",
            "item": item,
            "matched_key": key,
            "basis": basis,
            "basis_note": _BASIS_NOTE.get(basis, ""),
            "location_matched": loc_matched,
            "unit": unit or base.get("unit"),
            "rates": {
                "base_usd": base["base_usd"],
                "adjusted_usd": adjusted,
            },
            "factors": {
                "location_factor": loc_factor,
                "location_matched": loc_matched,
                "project_type_factor": proj_factor,
                "location": location,
                "project_type": project_type,
            },
            "note": base.get("note", ""),
            "source_note": (
                self.config.get("source_note")
                + ("" if loc_matched else
                   f" NOTE: '{location}' is not in the location index — priced at "
                   f"the US-national-average base (factor 1.0).")
            ),
            "confidence": self.config.get("confidence"),
        }
