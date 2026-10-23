"""
Unified Construction Catalog — Materials, Labour, and Equipment for KSA.

Instead of hardcoding every item, this module uses keyword-driven groups:

  1. Material groups   — e.g. "envelope" catches cladding, roofing, flashing
  2. Labour trades     — e.g. "electrician" matches all electrician variants
  3. Equipment types   — e.g. "excavator" matches all excavator variants
  4. Rate tiers        — low / medium / high based on spec or skill level
  5. Regional rates    — KSA-specific from Turner & Townsend / SCA / AECOM

Usage:
    catalog = UnifiedCatalog(region="ksa")

    # Materials — same as before
    result = catalog.lookup("aluminium curtain wall cladding")
    # → MaterialGroup "envelope_cladding" at 650-1400 SAR/m2

    # Labour — by trade name or craft description
    result = catalog.lookup(" certified welder for structural steel")
    # → LabourGroup "welder" at 395 SAR/day

    # Equipment — by machine type or function
    result = catalog.lookup("20-ton hydraulic excavator with breaker")
    # → EquipmentGroup "excavator" at 800 SAR/day rental

    # Bulk classify a mixed BOQ + labour + equipment list
    results = catalog.classify_items([
        "Supply & install GI earthing strip 25x3mm",
        "Supply & install 20-ton excavator with operator",
        "Concreter for raft foundation pour",
        "Galvanised steel guardrail 1.2m high",
    ])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════
#  BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialGroup:
    """A material category with keyword patterns and rate tiers."""
    name: str
    keywords: List[str]
    unit: str
    rate_low: float      # SAR per unit — basic spec
    rate_mid: float      # SAR per unit — standard spec
    rate_high: float     # SAR per unit — premium spec
    source: str = ""

    @property
    def group_type(self) -> str:
        return "material"


@dataclass
class LabourGroup:
    """A labour trade with keyword patterns and daily rate tiers.

    Rates are SAR/day for a 9.5-hour shift (standard KSA working day).
    Tiers: helper/apprentice → skilled/journeyman → specialist/master.
    """
    name: str
    keywords: List[str]
    rate_helper: float   # Apprentice / helper / unskilled within trade
    rate_skilled: float  # Qualified journeyman
    rate_master: float   # Specialist / master / certified
    source: str = ""
    unit: str = "day"    # Daily rate unit

    @property
    def group_type(self) -> str:
        return "labour"

    @property
    def rate_low(self) -> float:
        return self.rate_helper

    @property
    def rate_mid(self) -> float:
        return self.rate_skilled

    @property
    def rate_high(self) -> float:
        return self.rate_master


@dataclass
class EquipmentGroup:
    """A construction equipment category with rental rate tiers.

    Rates are SAR per day / week / month for dry hire (without operator).
    Add ~15% for wet hire (with operator) — roughly one labour rate.
    """
    name: str
    keywords: List[str]
    rate_day: float
    rate_week: float
    rate_month: float
    source: str = ""

    @property
    def group_type(self) -> str:
        return "equipment"

    @property
    def rate_low(self) -> float:
        return self.rate_day

    @property
    def rate_mid(self) -> float:
        return self.rate_week

    @property
    def rate_high(self) -> float:
        return self.rate_month

    @property
    def unit(self) -> str:
        return "day"  # default display unit; week/month available on group


# Union type for all group kinds
Group = Union[MaterialGroup, LabourGroup, EquipmentGroup]


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT DATA — KSA RATES
# ═══════════════════════════════════════════════════════════════════════════

# ── Material Groups ───────────────────────────────────────────────────────
# (29 groups — same as v1, rates from Turner & Townsend KSAMI 2025, SCA, AECOM)

DEFAULT_MATERIALS: List[MaterialGroup] = [
    # ── Structural ────────────────────────────────────────────────────────
    MaterialGroup("concrete_cast_in_situ",
        ["concrete", "readymix", "ready-mix", "rcc", "reinforced concrete",
         "raft", "pile cap", "grade beam", "slab", "column", "shear wall"],
        "m3", 280, 320, 420,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("concrete_precast",
        ["precast", "hollowcore", "t-beam", "double tee", "prestressed",
         "precast panel", "precast plank"],
        "m2", 450, 750, 1200,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("rebar",
        ["rebar", "reinforcement", "reinforcing steel", "tmt", "deformed bar",
         "mild steel bar", "mesh", "fabric", " BRC"],
        "ton", 2800, 3200, 3900,
        "SCA Jan 2025 / msamer.com"),

    MaterialGroup("structural_steel",
        ["structural steel", "steel beam", "steel column", "truss",
         "portal frame", "I-beam", "H-beam", "universal beam", "ub", "uc",
         "built-up section", "plate girder", "purlin", "girt", "bracing"],
        "ton", 12500, 15500, 19500,
        "Turner & Townsend KSAMI 2025"),

    # ── Envelope ──────────────────────────────────────────────────────────
    MaterialGroup("envelope_cladding",
        ["cladding", "curtain wall", "alucobond", "composite panel",
         "aluminium panel", "metal cladding", "rainscreen", "façade",
         "facade", "exterior wall", "glass curtain", "unitised", "stick system"],
        "m2", 650, 950, 1400,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("roofing",
        ["roof", "roofing", "roof tile", "metal deck", "insulated panel",
         "standing seam", "membrane", "waterproofing", "bitumen", "tpo",
         "epdm", "liquid applied", "roof slab"],
        "m2", 180, 320, 550,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("glazing_windows",
        ["window", "glazing", "glass", "double glazed", "single glazed",
         "low-e", "toughened glass", "laminated glass", "spandrel",
         "vision glass", "skylight", "louvre"],
        "m2", 850, 1300, 2200,
        "Turner & Townsend KSAMI 2025"),

    # ── Doors & Ironmongery ───────────────────────────────────────────────
    MaterialGroup("doors_wooden",
        ["wooden door", "timber door", "solid core", "flush door",
         "veneered door", "hardwood door", "mdf door", "plywood door",
         "louvre door", "panel door"],
        "no", 800, 1500, 3500,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("doors_steel",
        ["steel door", "metal door", "hollow metal", "hm door",
         "fire rated door", "fire door", "pressurised door",
         "blast door", "security door", "roller shutter", "rolling shutter"],
        "no", 2200, 4500, 8500,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("doors_aluminium_glass",
        ["aluminium door", "glass door", "frameless door",
         "shopfront", "sliding door", "folding door", "revolving door",
         "automatic door", "sliding folding"],
        "no", 3500, 6500, 12000,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("ironmongery_hardware",
        ["ironmongery", "hardware", "handle", "lock", "hinge", "closer",
         "panic bar", "floor spring", "mortice lock", "cylinder",
         "door stop", "coat hook", "towel rail", "toilet partition"],
        "no", 450, 900, 1800,
        "AECOM Middle East Handbook 2024"),

    # ── Finishes ──────────────────────────────────────────────────────────
    MaterialGroup("tiles_ceramic",
        ["tile", "ceramic", "porcelain", "granite tile", "marble tile",
         "mosaic", "terrazzo", "vitrified"],
        "m2", 120, 250, 600,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("paint_wallpaper",
        ["paint", "emulsion", "oil paint", "epoxy paint", "wallpaper",
         "vinyl wallpaper", "texture paint", "lacquer", "varnish",
         "anti-mould", "anti-bacterial"],
        "m2", 35, 65, 140,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("ceiling_suspended",
        ["suspended ceiling", "false ceiling", "gypsum board",
         "plasterboard", "acoustic ceiling", "metal ceiling", "grid",
         "mineral fibre", "linear ceiling", "baffle", "metal pan",
         "drop ceiling"],
        "m2", 120, 220, 450,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("flooring_resilient",
        ["vinyl", "linoleum", "rubber flooring", "carpet", "carpet tile",
         "epoxy floor", "resin floor", "raised floor", "access floor",
         "sports flooring", "polyurethane"],
        "m2", 150, 350, 750,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("screed_plaster",
        ["screed", "plaster", "render", "stucco", "skim coat",
         "rough cast", "pebble dash", "self-levelling"],
        "m2", 55, 85, 150,
        "Turner & Townsend KSAMI 2025"),

    # ── MEP ───────────────────────────────────────────────────────────────
    MaterialGroup("electrical_containment",
        ["conduit", "trunking", "cable tray", "cable ladder", "cable basket",
         "GI conduit", "PVC conduit", "emt", "imc", "wireway",
         "grommet", "escutcheon"],
        "m", 45, 85, 160,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("earthing_lightning",
        ["earthing", "grounding", "earth electrode", "earthing electrode",
         "ground electrode", "earth pit", "earth strip", "earth rod",
         "ground rod", "earth bus", "copper tape", "GI strip",
         "lightning rod", "lightning protection", "air terminal",
         "surge protector", "spd", "equipotential", "bonding", "down conductor"],
        "ls", 1200, 2800, 6500,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("fire_detection_alarm",
        ["fire alarm", "smoke detector", "heat detector", "beam detector",
         "call point", "sounder", "strobe", "aspirating", "vesda",
         "fire panel", "repeater panel", "module", "sprinkler",
         "hose reel", "fire blanket", "extinguisher"],
        "no", 180, 450, 850,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("hvac_ductwork",
        ["duct", "ductwork", "ahu", "fcu", "rtu", "chiller", "fan coil",
         "air handling", "vav", "diffuser", "grille", "louvre",
         "damper", "fire damper", "extract fan", "supply fan",
         "chilled beam", "active beam", "chilled water", "fume hood",
         "exhaust", "cooling coil", "heating coil", "humidifier"],
        "m2_sa", 650, 1200, 2200,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("plumbing_sanitary",
        ["pipe", "fitting", "valve", "tap", "mixer", "wc", "basin",
         "urinal", "shower", "bathtub", "floor drain", "gully",
         "grease trap", "manhole", "inspection chamber",
         "pump", "booster set", "water tank", "septic tank",
         "upvc", "pvc", "ppr", "hdpe", "gi pipe",
         "rainwater", "stormwater", "soil pipe", "waste pipe"],
        "m2_ba", 350, 650, 1200,
        "AECOM Middle East Handbook 2024"),

    # ── Civil Works ───────────────────────────────────────────────────────
    MaterialGroup("earthworks_excavation",
        ["excavation", "earthwork", "backfill", "compaction", "subgrade",
         "trench", "cut and fill", "soil improvement", "dewatering",
         "shoring", "sheet pile", "strut"],
        "m3", 55, 95, 180,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("asphalt_paving",
        ["asphalt", "tarmac", "bitumen", "paving", "road", "car park",
         "interlock", "paver", "kerb", "curb", "edging", "road marking",
         "thermoplastic"],
        "m2", 120, 220, 450,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("guardrail_barrier",
        ["guardrail", "guard rail", "w-beam", "thrie beam", "crash barrier",
         "pedestrian barrier", "barrier post", "stanchion", "balustrade",
         "handrail system", "fence", "bollard", "wheel stop", "chain link",
         "steel railing", "glass balustrade", "steel guardrail",
         "galvanised guardrail", "galvanized guardrail"],
        "m", 280, 550, 1200,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("landscaping_hardscape",
        ["landscape", "softscape", "irrigation", "sprinkler", "palm tree",
         "shrub", "turf", "sod", "planter", "pergola", "gazebo",
         "fountain", "water feature", "outdoor furniture", "bench"],
        "m2", 250, 550, 1200,
        "Turner & Townsend KSAMI 2025"),

    # ── Specialised ───────────────────────────────────────────────────────
    MaterialGroup("lift_elevator",
        ["lift", "elevator", "escalator", "travelator", "dumbwaiter",
         "goods lift", "passenger lift", "stretcher lift", "freight",
         "hydraulic lift", "traction lift", "mrl"],
        "no", 95000, 250000, 650000,
        "Turner & Townsend KSAMI 2025"),

    MaterialGroup("kitchen_equipment",
        ["kitchen", "cooking range", "oven", "hood", "extract canopy",
         "cold room", "freezer", "dishwasher", "prep table", "sink",
         "grease duct", "stainless steel"],
        "m2_kitchen", 3500, 6500, 15000,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("security_access_control",
        ["cctv", "camera", "access control", "turnstile", "boom barrier",
         "bollard", "intruder alarm", "pabx", "intercom", "public address",
         "av system", "matv", "iptv", "data cabling", "structured cabling",
         "cat6", "fiber optic"],
        "m2", 85, 180, 450,
        "AECOM Middle East Handbook 2024"),

    MaterialGroup("insulation_waterproofing",
        ["insulation", "thermal insulation", "rockwool", "glasswool",
         "xps", "eps", "pu foam", "fireproofing", "intumescent",
         "waterproofing", "damp proof", "vapour barrier", "dpm"],
        "m2", 65, 120, 280,
        "Turner & Townsend KSAMI 2025"),
]


# ── Labour Groups ─────────────────────────────────────────────────────────
# Rates are SAR/day for a 9.5-hour shift.
# Sources: AECOM Middle East Handbook 2024, Riyadh market.
# Skill levels: helper → skilled → master/specialist.

DEFAULT_LABOUR: List[LabourGroup] = [
    LabourGroup("concreter",
        ["concreter", "concrete worker", "concrete finisher",
         "concrete pourer", "concrete labourer", "shutter man",
         "shuttering carpenter", "formwork carpenter"],
        200, 280, 380,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("steel_fixer",
        ["steel fixer", "rebar worker", "bar bender", "bar benders",
         "reinforcement fixer", "steel reinforcement worker",
         "mesh fixer", " BRC fixer"],
        210, 295, 400,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("bricklayer",
        ["bricklayer", "block layer", "mason", "block mason",
         "brick mason", "stone mason", "block work", "blockwork",
         "masonry worker"],
        220, 310, 420,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("carpenter",
        ["carpenter", "joiner", "woodworker", "timber worker",
         "rough carpenter", "finish carpenter", "wood framing",
         "door carpenter"],
        220, 310, 420,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("electrician",
        ["electrician", "electrical technician", "electrical fitter",
         "wireman", "panel builder", "electrical installer",
         "HV electrician", "MV electrician", "LV electrician"],
        260, 370, 520,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("plumber",
        ["plumber", "pipe fitter", "piping technician", "plumbing installer",
         "sanitary installer", "drainage worker", "water fitter"],
        240, 345, 470,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("welder",
        ["welder", "fabricator", "steel fabricator",
         "mig welder", "tig welder", "arc welder", "stick welder",
         "certified welder", "structural welder", "pipe welder",
         "combination welder", "welding operator"],
        280, 395, 550,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("plasterer",
        ["plasterer", "drywall installer", "gypsum installer",
         "plastering worker", "render worker", "stucco worker",
         "wall finisher"],
        200, 280, 380,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("painter",
        ["painter", "spray painter", "decorative painter",
         "wall painter", "coating applicator", "epoxy painter"],
        185, 265, 360,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("tiler",
        ["tiler", "tile fixer", "tile setter", "stone fixer",
         "marble worker", "granite installer", "ceramic fixer"],
        195, 275, 380,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("rigger",
        ["rigger", "slinger", "slinging worker",
         "crane rigger", "lifting supervisor", "lifting rigger",
         "dogman"],
        230, 325, 450,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("crane_operator",
        ["crane operator", "crane driver", "crane man",
         "tower crane operator", "mobile crane operator",
         "crawler crane operator", "overhead crane operator",
         "hoist operator"],
        330, 465, 650,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("scaffolder",
        ["scaffolder", "scaffold builder", "scaffold erector",
         "scaffold worker", "tube and fitting scaffolder",
         "system scaffolder", "shoring erector"],
        205, 290, 400,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("general_labourer",
        ["labourer", "general labourer", "helper", "construction worker",
         "unskilled worker", "site cleaner", "material handler",
         "load carrier", "skilled labourer"],
        140, 200, 280,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("foreman",
        ["foreman", "chargehand", "team leader", "gang boss",
         "trade foreman", "site foreman", "working foreman",
         "supervisor", "general foreman"],
        380, 560, 850,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("safety_officer",
        ["safety officer", "hse officer", "safety inspector",
         "safety supervisor", "safety watchman", "hse advisor",
         "safety coordinator", "fire watch"],
        320, 480, 720,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("qaqc_inspector",
        ["qaqc inspector", "quality inspector", "qc inspector",
         "welding inspector", "coating inspector", "ndt inspector",
         "quality engineer", "quality controller",
         "material inspector", "structural inspector"],
        290, 440, 650,
        "AECOM Middle East Handbook 2024"),

    LabourGroup("surveyor",
        ["surveyor", "land surveyor", "quantity surveyor",
         "setting out engineer", "topographic surveyor",
         "marine surveyor", "building surveyor"],
        300, 450, 680,
        "AECOM Middle East Handbook 2024"),
]


# ── Equipment Groups ──────────────────────────────────────────────────────
# Rates are SAR per day / week / month for dry hire.
# Sources: Saudi equipment rental market data 2025, major rental firms.

DEFAULT_EQUIPMENT: List[EquipmentGroup] = [
    EquipmentGroup("excavator",
        ["excavator", "hydraulic excavator", "track excavator",
         "crawler excavator", "mini excavator", "digger", "backhoe excavator",
         "long reach excavator", "demolition excavator"],
        800, 4000, 12000,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("bulldozer",
        ["bulldozer", "dozer", "crawler dozer", "wheel dozer",
         "track dozer", "角度 blade dozer"],
        950, 4750, 14250,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("wheel_loader",
        ["wheel loader", "front loader", "front end loader",
         "shovel loader", "bucket loader", "payloader",
         "skid steer", "bobcat", "compact loader"],
        750, 3750, 11250,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("crane_mobile",
        ["mobile crane", "truck crane", "telescopic crane",
         "all terrain crane", "rough terrain crane", "pick and carry",
         "crane truck"],
        2500, 12500, 37500,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("crane_crawler",
        ["crawler crane", "lattice crane", "crawler lattice",
         "heavy lift crane", "crawler mounted crane"],
        8000, 40000, 120000,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("tower_crane",
        ["tower crane", "building crane", "construction tower crane",
         "topkit crane", "flat top crane", "luffing crane"],
        3500, 17500, 52500,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("dumper",
        ["dumper", "dump truck", "tipper", "tipper truck",
         "articulated dumper", "rigid dumper", "site dumper",
         "mini dumper", "tracked dumper"],
        500, 2500, 7500,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("generator",
        ["generator", "genset", "diesel generator", "power generator",
         "portable generator", "silent generator", "standby generator",
         "transformer", "distribution board", "switchgear"],
        400, 2000, 6000,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("compressor",
        ["compressor", "air compressor", "diesel compressor",
         "portable compressor", "screw compressor", "piston compressor"],
        200, 1000, 3000,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("forklift",
        ["forklift", "fork lift", "telehandler", "telescopic handler",
         "rough terrain forklift", "warehouse forklift", "reach truck",
         "material handler", "manlift", "boom lift", "scissor lift"],
        300, 1500, 4500,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("backhoe_loader",
        ["backhoe", "backhoe loader", "tractor loader backhoe",
         "tlb", "jcb", "loader backhoe"],
        450, 2250, 6750,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("motor_grader",
        ["grader", "motor grader", "road grader", "blade grader",
         "civil grader", "finish grader"],
        850, 4250, 12750,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("concrete_pump",
        ["concrete pump", "concrete boom pump", "line pump",
         "truck mounted pump", "stationary pump", "placing boom",
         "concrete mixer truck", "transit mixer"],
        1200, 6000, 18000,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("welding_machine",
        ["welding machine", "welder generator", "welding set",
         "mig machine", "tig machine", "arc welder", "spot welder"],
        120, 600, 1800,
        "Saudi Equipment Rental Market 2025"),

    EquipmentGroup("scaffolding_system",
        ["scaffolding", "scaffold system", "frame scaffold",
         "system scaffold", "cuplock", "ringlock", "kwikstage",
         "aluminium scaffold", "mobile scaffold", "shoring tower",
         "falsework", "formwork system"],
        15, 75, 220,
        "Saudi Equipment Rental Market 2025"),  # per m2/month
]


# ═══════════════════════════════════════════════════════════════════════════
#  STOP WORDS
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS = {"the", "and", "for", "with", "from", "into", "onto", "upon", "over",
                "under", "through", "between", "among", "within", "without", "against",
                "during", "before", "after", "above", "below", "near", "off", "out",
                "up", "down", "in", "on", "at", "to", "of", "by", "as", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had", "do",
                "does", "did", "will", "would", "shall", "should", "may", "might",
                "can", "could", "must", "all", "any", "both", "each", "few", "more",
                "most", "other", "some", "such", "nor", "not", "only", "own", "same",
                "so", "than", "too", "very", "just", "now", "also", "mm", "mtr",
                "nos", "dia", "supply", "install", "type", "types", "grade", "high",
                "low", "medium", "standard", "premium", "basic", "size", "sizes",
                "length", "width", "thick", "thickness", "diameter", "serial",
                "number", "item", "code", "per", "unit", "total", "sub", "main",
                "auxiliary", "temporary", "day", "week", "month", "hour", "daily",
                "weekly", "monthly", "rent", "rental", "hire", "wet", "dry",
                "operator", "driver", "include", "excluding", "including",
                "floor", "wall", "ceiling", "roof", "work", "worker", "building",
                "construction", "general", "certified", "skilled", "qualified"}


# ═══════════════════════════════════════════════════════════════════════════
#  LOOKUP RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LookupResult:
    """Result of a catalog lookup — works for material, labour, or equipment."""
    input_text: str
    matched_group: str
    unit: str
    rate_low: float
    rate_mid: float
    rate_high: float
    source: str
    confidence: str              # high | medium | low
    matched_keywords: Tuple[str, ...] = ()
    group_type: str = "material"  # material | labour | equipment


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED CATALOG
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedCatalog:
    """Keyword-driven catalog covering materials, labour, and equipment."""

    def __init__(
        self,
        materials: Optional[List[MaterialGroup]] = None,
        labour: Optional[List[LabourGroup]] = None,
        equipment: Optional[List[EquipmentGroup]] = None,
        region: str = "ksa",
    ) -> None:
        self.materials = materials or list(DEFAULT_MATERIALS)
        self.labour = labour or list(DEFAULT_LABOUR)
        self.equipment = equipment or list(DEFAULT_EQUIPMENT)
        self.region = region

        # Flatten all groups for unified lookup
        self._all_groups: List[Group] = []
        self._all_groups.extend(self.materials)
        self._all_groups.extend(self.labour)
        self._all_groups.extend(self.equipment)

        self._keyword_index: Dict[str, List[Group]] = {}
        self._build_index()

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Inverted index: keyword → list of groups (any type)."""
        self._keyword_index.clear()
        for group in self._all_groups:
            for kw in group.keywords:
                self._keyword_index.setdefault(kw.lower(), []).append(group)

    def _score_groups(self, text: str) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, Group]]:
        """Score every group against the input text. Returns (scores, matched_kws, group_map)."""
        words = set(re.findall(r"[a-z]+", text))
        content_words = {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

        scores: Dict[str, int] = {}
        matched_kws: Dict[str, List[str]] = {}
        group_map: Dict[str, Group] = {}

        for kw, groups in self._keyword_index.items():
            for g in groups:
                key = f"{g.group_type}:{g.name}"
                group_map[key] = g
                if kw in text:
                    # Multi-word exact matches score much higher than single-word
                    bonus = 6 if " " in kw else 2
                    scores[key] = scores.get(key, 0) + bonus
                    matched_kws.setdefault(key, []).append(kw)
                else:
                    for w in content_words:
                        if kw in w or w in kw:
                            scores[key] = scores.get(key, 0) + 1
                            matched_kws.setdefault(key, []).append(f"{kw}~{w}")
                            break

        return scores, matched_kws, group_map

    # ── Public API ────────────────────────────────────────────────────────

    def lookup(self, description: str, group_type: Optional[str] = None) -> Optional[LookupResult]:
        """Find the best-matching group for a description string.

        Args:
            description: Free-text item description (BOQ line, requisition, etc.)
            group_type: Filter by "material", "labour", or "equipment" (optional)

        Returns:
            LookupResult with rate tiers and confidence level.
        """
        text = description.lower()
        scores, matched_kws, group_map = self._score_groups(text)

        if not scores:
            return None

        # Optional type filter
        if group_type:
            scores = {k: v for k, v in scores.items() if k.startswith(f"{group_type}:")}
            if not scores:
                return None

        best_key = max(scores, key=lambda k: scores[k])
        best = group_map[best_key]
        score = scores[best_key]
        max_possible = len(best.keywords) * 2

        confidence = (
            "high" if score >= max_possible * 0.35 else
            "medium" if score >= max_possible * 0.18 else
            "low"
        )

        return LookupResult(
            input_text=description,
            matched_group=best.name,
            unit=best.unit,
            rate_low=best.rate_low,
            rate_mid=best.rate_mid,
            rate_high=best.rate_high,
            source=best.source,
            confidence=confidence,
            matched_keywords=tuple(set(matched_kws.get(best_key, []))),
            group_type=best.group_type,
        )

    def classify_items(self, descriptions: List[str]) -> List[LookupResult]:
        """Classify a list of mixed descriptions (materials, labour, equipment)."""
        results: List[LookupResult] = []
        for desc in descriptions:
            r = self.lookup(desc)
            if r:
                results.append(r)
            else:
                results.append(LookupResult(
                    input_text=desc,
                    matched_group="UNKNOWN",
                    unit="-",
                    rate_low=0, rate_mid=0, rate_high=0,
                    source="",
                    confidence="none",
                    matched_keywords=(),
                    group_type="unknown",
                ))
        return results

    def estimate_cost(
        self,
        description: str,
        quantity: float,
        spec_level: str = "mid",
        period: str = "day",
    ) -> Dict[str, float]:
        """Quick cost estimate from a description + quantity.

        Args:
            description: Item or trade description
            quantity: Number of units (m2, m3, days, weeks, etc.)
            spec_level: "low" | "mid" | "high"
            period: For equipment — "day" | "week" | "month"
        """
        r = self.lookup(description)
        if not r:
            return {"rate": 0, "total": 0, "confidence": "none", "group": "UNKNOWN", "type": "unknown"}

        # For equipment, allow period override
        if r.group_type == "equipment":
            rate = {"day": r.rate_low, "week": r.rate_mid, "month": r.rate_high}.get(period, r.rate_low)
        else:
            rate = {"low": r.rate_low, "mid": r.rate_mid, "high": r.rate_high}.get(spec_level, r.rate_mid)

        return {
            "rate_sar": rate,
            "quantity": quantity,
            "total_sar": round(rate * quantity, 2),
            "group": r.matched_group,
            "type": r.group_type,
            "unit": r.unit,
            "confidence": r.confidence,
            "source": r.source,
        }

    def list_groups(self, group_type: Optional[str] = None) -> List[Dict]:
        """Return all groups, optionally filtered by type."""
        groups = self._all_groups
        if group_type:
            groups = [g for g in groups if g.group_type == group_type]
        return [
            {"group": g.name, "type": g.group_type,
             "rate_low": g.rate_low, "rate_mid": g.rate_mid, "rate_high": g.rate_high}
            for g in groups
        ]

    # ── Mutators ──────────────────────────────────────────────────────────

    def add_group(self, group: Group) -> None:
        """Add a new group at runtime (no code changes needed)."""
        if isinstance(group, MaterialGroup):
            self.materials.append(group)
        elif isinstance(group, LabourGroup):
            self.labour.append(group)
        elif isinstance(group, EquipmentGroup):
            self.equipment.append(group)
        else:
            raise TypeError(f"Unknown group type: {type(group)}")

        self._all_groups.append(group)
        for kw in group.keywords:
            self._keyword_index.setdefault(kw.lower(), []).append(group)


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    catalog = UnifiedCatalog()

    # ── Demo 1: Mixed lookup (materials + labour + equipment) ─────────────
    print("=" * 70)
    print("MIXED LOOKUP — Materials, Labour & Equipment")
    print("=" * 70)
    test_items = [
        # Materials
        ("Aluminium curtain wall cladding system", None),
        ("Copper earthing strip 25x3mm with earth pit", None),
        ("Galvanised steel guardrail 1.2m high with posts", None),
        ("Fire-rated steel door 120min with panic hardware", None),
        ("Solid core oak veneer wooden door 900x2100", None),
        ("Chilled beam ceiling-mounted", None),
        # Labour
        ("Certified welder for structural steel connection", None),
        ("Skilled electrician for LV distribution panel", None),
        ("Concreter for raft foundation pour", None),
        ("Bar bender for rebar 16mm", None),
        ("Site foreman with 10 years experience", None),
        ("Safety officer for hot work permit", None),
        ("Tower crane operator certified", None),
        # Equipment
        ("20-ton hydraulic excavator with breaker", None),
        ("50-ton mobile crane for steel erection", None),
        ("Concrete boom pump 36m reach", None),
        ("500kVA diesel generator standby", None),
        ("Scaffold system cuplock for facade work", None),
        ("Forklift 5-ton rough terrain", None),
    ]

    for desc, filt in test_items:
        r = catalog.lookup(desc, group_type=filt)
        if r:
            type_label = r.group_type.upper()[:3]
            if r.group_type == "labour":
                rate_str = f"{r.rate_mid:>6,.0f} SAR/day"
            elif r.group_type == "equipment":
                rate_str = f"{r.rate_low:>6,.0f} SAR/day  ({r.rate_mid:>8,.0f}/wk  {r.rate_high:>10,.0f}/mo)"
            else:
                rate_str = f"{r.rate_mid:>8,.0f} SAR/{r.unit}"
            print(f"  {type_label}  {desc[:45]:45s} → {r.matched_group:28s}  {rate_str}  ({r.confidence})")
        else:
            print(f"  ???  {desc[:45]:45s} → NOT FOUND")

    # ── Demo 2: Bulk classify a mixed BOQ ─────────────────────────────────
    print()
    print("=" * 70)
    print("MIXED BOQ CLASSIFICATION")
    print("=" * 70)
    mixed_boq = [
        "Supply & install aluminium composite panel cladding",
        "Supply & install 20-ton excavator with operator for 3 months",
        "Certified welder for structural steel connections",
        "GI earthing electrode with inspection chamber",
        "Tower crane operator for high-rise construction",
        "W-beam galvanised guardrail with terminal section",
        "Concrete boom pump for slab pour",
        "Solid core timber door with ironmongery set",
        "Skilled concreter for grade beam pour",
        "500kVA diesel generator standby power",
        "Fire rated steel door 2hr with vision panel",
        "Double glazed curtain wall stick system",
        "Scaffolding system ringlock for facade access",
        "Structural steel beam universal beam 305x165 UB",
        "Safety officer for confined space entry",
        "Readymix concrete grade C35",
        "Motor grader for road base finishing",
        "TMT reinforcement bar 16mm diameter",
        "Mobile crane 50-ton for steel lifting",
        "Ceramic floor tile 600x600mm",
        "QA/QC inspector for welding inspection",
    ]
    results = catalog.classify_items(mixed_boq)
    for desc, r in zip(mixed_boq, results):
        if r.matched_group != "UNKNOWN":
            type_label = r.group_type.upper()[:3]
            if r.group_type == "labour":
                rate_str = f"{r.rate_mid:>6,.0f} SAR/day"
            elif r.group_type == "equipment":
                rate_str = f"{r.rate_low:>6,.0f} SAR/day"
            else:
                rate_str = f"{r.rate_mid:>8,.0f} SAR/{r.unit}"
            print(f"  {type_label}  {desc[:42]:42s} → {r.matched_group:26s}  {rate_str}")
        else:
            print(f"  ???  {desc[:42]:42s} → NOT CLASSIFIED")

    # ── Demo 3: Cost estimates ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COST ESTIMATES")
    print("=" * 70)

    # Material estimate
    e = catalog.estimate_cost("aluminium cladding", 2500, "mid")
    print(f"  Material: {e['group']:25s}  qty={e['quantity']:>8,.1f}  "
          f"rate={e['rate_sar']:>10,.0f} SAR  total={e['total_sar']:>14,.0f} SAR")

    # Labour estimate
    e = catalog.estimate_cost("certified welder", 180, "high")
    print(f"  Labour:   {e['group']:25s}  qty={e['quantity']:>8,.1f} days  "
          f"rate={e['rate_sar']:>10,.0f} SAR  total={e['total_sar']:>14,.0f} SAR")

    # Equipment estimate — monthly
    e = catalog.estimate_cost("20-ton excavator", 6, period="month")
    print(f"  Equipment:{e['group']:25s}  qty={e['quantity']:>8,.1f} mo   "
          f"rate={e['rate_sar']:>10,.0f} SAR  total={e['total_sar']:>14,.0f} SAR")

    # Equipment estimate — weekly
    e = catalog.estimate_cost("concrete boom pump", 8, period="week")
    print(f"  Equipment:{e['group']:25s}  qty={e['quantity']:>8,.1f} wk   "
          f"rate={e['rate_sar']:>10,.0f} SAR  total={e['total_sar']:>14,.0f} SAR")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(f"Catalog summary:")
    print(f"  Materials:  {len(catalog.materials):2d} groups")
    print(f"  Labour:     {len(catalog.labour):2d} trades")
    print(f"  Equipment:  {len(catalog.equipment):2d} types")
    print(f"  ─────────────────────────")
    print(f"  Total:      {len(catalog._all_groups):2d} groups")
    print()
    print("All rates in SAR.  1 SAR ≈ 0.27 USD (2025).")


# ═══════════════════════════════════════════════════════════════════════════
#  GK KNOWLEDGE EXPORT — rates belong in RAG, not read straight off this table.
#  Emits the catalog's rates as a curated General-Knowledge markdown note so a
#  cost query RETRIEVES them (grounded, cited) instead of the LLM reading a
#  hardcoded oracle. Regenerate docs/knowledge/... from this on any rate change.
# ═══════════════════════════════════════════════════════════════════════════

def to_knowledge_markdown() -> str:
    """Render the default KSA catalog as a GK knowledge note (indicative rates,
    cited per row; the project's priced BOQ still takes precedence)."""
    lines: List[str] = []
    lines.append("# Construction Materials, Labour & Equipment Rates — Saudi Arabia (2025)")
    lines.append("")
    lines.append(
        "INDICATIVE KSA construction rates (materials, labour, equipment) for use "
        "as the GENERAL-KNOWLEDGE reference when the project has no priced BOQ of "
        "its own. All rates in SAR (~3.75 SAR/USD). A figure here is indicative — "
        "the project's own priced BOQ / rate schedule takes precedence, and a "
        "supplier quote gives the firm price. Cite the source and year."
    )
    lines.append("")
    lines.append("## Material supply rates (SAR per unit)")
    lines.append("")
    lines.append("| Material group | Unit | Low | Mid | High | Source |")
    lines.append("|---|---|---|---|---|---|")
    for m in DEFAULT_MATERIALS:
        lines.append(
            f"| {m.name.replace('_', ' ')} | {m.unit} | {m.rate_low:,.0f} | "
            f"{m.rate_mid:,.0f} | {m.rate_high:,.0f} | {m.source} |"
        )
    lines.append("")
    lines.append("## Labour rates (SAR per day, 9.5-hour KSA shift)")
    lines.append("")
    lines.append("| Trade | Helper | Skilled | Master/specialist | Source |")
    lines.append("|---|---|---|---|---|")
    for lb in DEFAULT_LABOUR:
        lines.append(
            f"| {lb.name.replace('_', ' ')} | {lb.rate_helper:,.0f} | "
            f"{lb.rate_skilled:,.0f} | {lb.rate_master:,.0f} | {lb.source} |"
        )
    lines.append("")
    lines.append("## Equipment dry-hire rates (SAR, without operator)")
    lines.append("")
    lines.append("| Equipment | Per day | Per week | Per month | Source |")
    lines.append("|---|---|---|---|---|")
    for eq in DEFAULT_EQUIPMENT:
        lines.append(
            f"| {eq.name.replace('_', ' ')} | {eq.rate_day:,.0f} | "
            f"{eq.rate_week:,.0f} | {eq.rate_month:,.0f} | {eq.source} |"
        )
    lines.append("")
    lines.append("## How to use")
    lines.append("")
    lines.append(
        "- These are INDICATIVE KSA rates — prefer them over a generic global "
        "estimate for a Saudi project, but the project's own priced BOQ wins."
    )
    lines.append(
        "- Quote the range/spec tier, cite the source + year, and recommend a "
        "supplier quote for a firm price. Do not present an indicative rate as a "
        "firm one."
    )
    lines.append("")
    return "\n".join(lines)
