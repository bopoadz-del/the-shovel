#!/usr/bin/env python3
"""Generate deterministic Q&A training pairs from app/prompts/construction_evm.md.

Every answer is grounded in the source document — formulas, thresholds,
numbers, and worked examples are copied verbatim from construction_evm.md.
No LLM calls. No paraphrasing. The source data is encoded inline as Python
constants so the operator can audit every fact against the markdown.

Output schema (matches scripts/generate_knowledge_scenarios.py):

    {"instruction": "<question>", "response": "<answer>", "source": "..."}

Sources are tagged ``construction_evm.md:<section-number>`` so any row can
be traced back to a section of the source markdown.

CLI:

    python scripts/generate_evm_scenarios.py \\
        --out data/learning/evm_scenarios.jsonl

    # Or merge straight into an existing document-driven file:
    python scripts/generate_evm_scenarios.py \\
        --out data/learning/training_scenarios.jsonl --append
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, Iterator, List, Tuple


# ── row helper ────────────────────────────────────────────────────────────

def _row(instruction: str, response: str, section: str) -> Dict[str, str]:
    return {
        "instruction": instruction.strip(),
        "response": response.strip(),
        "source": f"construction_evm.md:{section}",
    }


# ── SECTION 1 — Cost Management Fundamentals ──────────────────────────────

_LIFECYCLE_PHASES: List[Tuple[str, str]] = [
    ("Concept", "+/-30% estimate accuracy"),
    ("Planning", "+/-15% estimate accuracy"),
    ("Execution", "+/-5% estimate accuracy"),
    ("Close-Out", "+/-3% estimate accuracy"),
    ("Operations", "Variable"),
]

_PILLARS: List[Tuple[str, str]] = [
    ("Budget Planning", "Define scope, develop estimates, allocate by WBS/CBS, establish baseline"),
    ("Cost Tracking", "Capture actuals, track progress & quantities, monitor commitments"),
    ("Forecasting", "Analyze trends, predict EAC, identify potential overruns early"),
    ("Cost Reporting", "Prepare reports, communicate performance, support decisions"),
    ("Variance Analysis", "Identify variances, find root causes, take corrective actions"),
]

_COST_CATEGORIES: List[Tuple[str, str]] = [
    ("LABOUR", "Salaries, wages, benefits, overtime, incentives"),
    ("MATERIAL", "Construction materials, consumables, bulk items"),
    ("EQUIPMENT", "Owned/rented equipment, fuel, operators"),
    ("SUBCONTRACT", "Subcontractor packages and specialist works"),
    ("OTHER COSTS", "Permits, insurance, taxes, mobilization, misc"),
]

_DIRECT_INDIRECT: List[Tuple[str, str]] = [
    ("Direct (Traceable)", "Concrete for foundation, rebar for structure, equipment for excavation"),
    ("Indirect (Non-Traceable)", "Site office & utilities, PM team, security & safety, insurance & bonds"),
]


# ── SECTION 4 — EVM core formulas ─────────────────────────────────────────

# Source numbers from Section 20 are reused for worked examples so they
# always reconcile back to the document.
_S20_BAC = 50_000_000
_S20_AC = 32_000_000
_S20_EV = 28_500_000
_S20_PV = 30_500_000  # derived from SPI = EV/PV = 0.93 → PV = 28.5/0.93 ~ 30.5M (matches doc)
_S20_CV = -3_500_000
_S20_SV = -2_000_000
_S20_VAC = -6_500_000
_S20_EAC = 56_500_000
_S20_CPI = 0.89
_S20_SPI = 0.93
_S20_TCPI = 1.19

# Section 7 worked example.
_S7_BAC = 60_000_000
_S7_AC = 36_500_000
_S7_EV = 33_200_000
_S7_CPI = 0.91
_S7_EAC = 65_934_066
_S7_ETC = 29_434_066
_S7_VAC = -5_934_066

# Section 11 TCPI example.
_S11_BAC = 100_000_000
_S11_EV = 40_000_000
_S11_AC = 50_000_000
_S11_TCPI = 1.20


_FORMULAS: List[Tuple[str, str, str, str]] = [
    # (key, formula, plain description, section)
    ("CV", "CV = EV - AC", "Cost Variance: positive means under budget, negative means over budget", "4"),
    ("SV", "SV = EV - PV", "Schedule Variance: positive means ahead of schedule, negative means behind", "4"),
    ("CPI", "CPI = EV / AC", "Cost Performance Index: >1.0 means cost efficient, <1.0 means over budget", "4"),
    ("SPI", "SPI = EV / PV", "Schedule Performance Index: >1.0 means ahead of schedule, <1.0 means behind", "4"),
    ("EAC", "EAC = BAC / CPI", "Estimate at Completion: forecast total cost assuming current CPI continues", "4"),
    ("ETC", "ETC = EAC - AC", "Estimate to Complete: remaining cost to finish the project", "4"),
    ("VAC", "VAC = BAC - EAC", "Variance at Completion: expected over/under run at project end", "4"),
    ("TCPI", "TCPI = (BAC - EV) / (BAC - AC)", "To-Complete Performance Index: efficiency required for remaining work to finish within budget", "4/11"),
    ("BAC", "BAC = Total approved budget", "Budget at Completion: total approved project budget", "4"),
    ("Exposure", "Exposure = AC + Commitments", "Total cost exposure including signed commitments not yet invoiced", "14"),
    ("True Remaining Budget", "True Remaining Budget = BAC - AC - Commitments", "Truly unencumbered budget after subtracting actuals and commitments", "14"),
]


# ── SECTION 5 — Traffic-light thresholds ──────────────────────────────────

_CPI_BANDS: List[Tuple[str, str, str, str]] = [
    ("GREEN", "CPI > 1.00", "Under Budget, Cost Efficient", "Keep Doing What Works"),
    ("AMBER", "0.90 <= CPI <= 1.00", "Slightly Over Budget", "Investigate & Take Action"),
    ("RED", "CPI < 0.90", "Over Budget, Cost Inefficient", "Take Corrective Action NOW"),
]

_SPI_BANDS: List[Tuple[str, str, str, str]] = [
    ("GREEN", "SPI > 1.00", "Ahead of Schedule, Very Good", "Maintain Momentum"),
    ("AMBER", "0.90 <= SPI <= 1.00", "Slightly Behind", "Review Plan & Recover"),
    ("RED", "SPI < 0.90", "Behind Schedule, At Risk", "Take Corrective Action NOW"),
]

_COMBINED_STATUS: List[Tuple[str, str]] = [
    ("CPI > 1.0 AND SPI > 1.0", "Ahead & Under Budget - Great Performance"),
    ("CPI < 1.0 AND SPI > 1.0", "Ahead But Over Budget - Monitor Costs"),
    ("CPI > 1.0 AND SPI < 1.0", "On Budget But Behind - Recover Schedule"),
    ("CPI < 1.0 AND SPI < 1.0", "Behind & Over Budget - TAKE CORRECTIVE ACTION"),
]


# ── SECTION 11 — TCPI interpretation table ────────────────────────────────

_TCPI_BANDS: List[Tuple[str, str, str]] = [
    ("TCPI < 1.0", "Remaining work needs LESS efficiency than planned", "Achievable"),
    ("TCPI = 1.0", "Remaining work needs EXACTLY planned efficiency", "On Track"),
    ("TCPI > 1.0", "Remaining work needs MORE efficiency than planned", "Difficult"),
    ("TCPI > 1.10", "Remaining work needs 10%+ more efficiency", "Very Unlikely"),
    ("TCPI > 1.20", "Virtually impossible without scope reduction or budget increase", "RED"),
]

_TCPI_RECOMMENDATIONS = [
    "Request budget increase (EAC revision to management)",
    "Scope reduction options",
    "Productivity improvement plan",
    "Re-baseline schedule and cost",
]


# ── SECTION 18 — Three-scenario weighted EAC example ──────────────────────

_FORECAST_SCENARIOS: List[Tuple[str, float, str, str, str]] = [
    # (name, CPI, EAC, VAC, probability)
    ("Optimistic", 1.05, "$95.2M", "+$4.8M", "20%"),
    ("Expected", 0.91, "$109.9M", "-$9.9M", "60%"),
    ("Pessimistic", 0.80, "$125.0M", "-$25.0M", "20%"),
]

_WEIGHTED_EAC = "$110.0M"
_WEIGHTED_EAC_CALC = "(0.20 x $95.2M) + (0.60 x $109.9M) + (0.20 x $125.0M)"


# ── SECTION 14 — Commitment tracking line items ───────────────────────────

_COMMITMENT_LINES: List[Tuple[str, str]] = [
    ("BAC", "$48,000,000"),
    ("Work Completed (AC)", "$32,000,000"),
    ("Commitments Outstanding", "$12,500,000"),
    ("True Exposure", "$44,500,000"),
    ("True Remaining Unencumbered Budget", "$3,500,000"),
    ("Risk", "Only 7.3% of budget truly unencumbered"),
]


# ── SECTION 16 — Common reporting mistakes ────────────────────────────────

_MISTAKES: List[Tuple[str, str, str]] = [
    (
        "Reporting without analysis",
        "CPI is 0.87",
        "CPI is 0.87 - we are getting only $0.87 of value for every dollar spent. "
        "Primary cause is MEP rework. Immediate action: re-inspect all Level 3 MEP "
        "before proceeding to Level 4.",
    ),
    (
        "Focusing only on past performance",
        "Reporting only what happened last month",
        "Always pair actuals with forecast - 'We spent $X last month AND we forecast $Y at completion.'",
    ),
    (
        "Ignoring forecasts and trends",
        "Monthly snapshot reporting only",
        "Show the trend line. Three consecutive months of declining CPI is an emergency regardless of current CPI value.",
    ),
    (
        "Overloading with too much data",
        "50-page cost report with every line item",
        "Exception-based reporting. Show only RED items at executive level. Detail available on request.",
    ),
    (
        "No clear actions or ownership",
        "Costs are over budget.",
        "Costs are over budget. [NAME] will re-baseline the mechanical subcontract by [DATE]. "
        "Target: recover 0.05 CPI points by end of next month.",
    ),
]

_REPORT_THREE_QUESTIONS = [
    "Where are we? (Current status)",
    "Where are we going? (Forecast)",
    "What are we doing about it? (Actions with owners and dates)",
]


# ── SECTION 17 — Top 8 root causes ────────────────────────────────────────

_ROOT_CAUSES: List[Tuple[int, str, str]] = [
    (1, "Poor productivity", "crew inefficiency, supervision gaps, learning curve"),
    (2, "Scope changes", "uncontrolled variations, late client changes, design gaps"),
    (3, "Design delays", "late drawings, RFI backlog, incomplete specifications"),
    (4, "Material shortages", "supply chain disruptions, procurement lead times"),
    (5, "Rework", "quality defects, non-conformances, incorrect installations"),
    (6, "Poor planning", "unrealistic schedules, missed sequencing, interface failures"),
    (7, "Price escalation", "material price increases beyond allowances"),
    (8, "Weather and site conditions", "unforeseen ground conditions, extreme weather"),
]

_FISHBONE: List[Tuple[str, str]] = [
    ("PEOPLE", "Lack of training, high turnover, low motivation"),
    ("MATERIALS", "Price increase, material waste, late deliveries"),
    ("METHODS", "Poor planning, rework, inefficient process"),
    ("EQUIPMENT", "Breakdowns, wrong equipment, low availability"),
    ("ENVIRONMENT", "Weather delays, site conditions"),
    ("MANAGEMENT", "Poor communication, scope changes, unrealistic targets"),
]

_DECLINING_DIAGNOSIS: List[Tuple[str, List[str]]] = [
    ("SPI is declining but CPI is acceptable", [
        "Check critical path activities specifically",
        "Look for sequencing problems",
        "Review resource allocation to critical activities",
        "Check for approval bottlenecks (RFIs, submittals, inspections)",
    ]),
    ("CPI is declining but SPI is acceptable", [
        "Productivity is the primary suspect",
        "Check labour hours vs quantities installed",
        "Review subcontractor performance",
        "Check material wastage rates",
    ]),
    ("BOTH CPI and SPI are declining", [
        "Systemic problem - management intervention required",
        "Consider re-baseline with realistic recovery plan",
        "Escalate to executive level immediately",
    ]),
]


# ── SECTION 8 — Executive dashboard ───────────────────────────────────────

_EXEC_KPIS: List[Tuple[int, str, str]] = [
    (1, "CPI", "Cost Performance Index"),
    (2, "SPI", "Schedule Performance Index"),
    (3, "CV", "Cost Variance ($)"),
    (4, "SV", "Schedule Variance ($)"),
    (5, "EAC", "Estimate at Completion ($)"),
    (6, "VAC", "Variance at Completion ($)"),
]

_TRAFFIC_LIGHT_REPORTING: List[Tuple[str, str]] = [
    ("CPI < 1.0", "RED - Over Budget"),
    ("SPI < 1.0", "RED - Behind Schedule"),
    ("CV < 0", "RED - Cost Overrun"),
    ("SV < 0", "RED - Schedule Delay"),
    ("EAC > BAC", "RED - Forecast Over Budget"),
    ("VAC < 0", "RED - Negative Variance"),
]

_MGMT_SUMMARY: List[Tuple[int, str]] = [
    (1, "Performance summary (CPI/SPI status)"),
    (2, "Key drivers (root causes of variance)"),
    (3, "Top risks (upcoming threats)"),
    (4, "Actions taken (corrective measures)"),
    (5, "Next steps (recovery plan)"),
]

_REPORTING_BEST_PRACTICES: List[Tuple[int, str]] = [
    (1, "Use consistent data and definitions"),
    (2, "Report on leading AND lagging indicators"),
    (3, "Highlight exceptions, not everything"),
    (4, "Keep dashboards simple and visual"),
    (5, "Tell the story behind the numbers"),
    (6, "Update regularly and on time"),
    (7, "Drive actions, not just reporting"),
    (8, "Focus on trends, not single data points"),
    (9, "Communicate to the right audience"),
]

_S_CURVE_LINES: List[Tuple[str, str]] = [
    ("PV curve", "Planned expenditure over time"),
    ("EV curve", "Earned value over time"),
    ("AC curve", "Actual expenditure over time"),
    ("If AC > EV", "Over budget for work done"),
    ("If EV < PV", "Behind schedule"),
]


# ── SECTION 9 — Budget distribution benchmarks ────────────────────────────

_BUDGET_SPLIT: List[Tuple[str, str]] = [
    ("Labour", "35%"),
    ("Material", "40%"),
    ("Equipment", "15%"),
    ("Subcontract", "7%"),
    ("Other Costs", "3%"),
]

_S_CURVE_PHASES: List[Tuple[str, str]] = [
    ("Slow start (mobilization)", "0-15%"),
    ("Acceleration phase", "15-70%"),
    ("Peak expenditure", "50-85%"),
    ("Slowdown (commissioning)", "85-100%"),
]


# ── SECTION 2 — CBS hierarchy ─────────────────────────────────────────────

_CBS_LEVELS: List[Tuple[int, str, str]] = [
    (1, "Total Project Cost", "Top of the hierarchy: the entire project budget rolled up"),
    (2, "Area", "e.g., Area 1 Site Prep, Area 2 Process Plant, Area 3 Utilities"),
    (3, "Discipline", "Civil, Mechanical, Electrical, Piping, etc."),
    (4, "Work Package", "WP-101, WP-102, etc."),
    (5, "Cost Account", "Labor, Material, Equipment, Subcontract"),
]

_CBS_CODE_PARTS: List[Tuple[str, str]] = [
    ("02", "Area"),
    ("20", "Discipline"),
    ("ME", "System (Mechanical)"),
    ("P", "Work Package"),
    ("101A", "Cost Account Identifier"),
]


# ── SECTION 3 — Build steps + unit rates ──────────────────────────────────

_BUILD_STEPS: List[Tuple[int, str, str]] = [
    (1, "Estimate", "Define scope, quantify items, apply unit rates"),
    (2, "Review", "Check quantities, validate rates, assess risks, value engineering"),
    (3, "Approve", "Management review, budget approval, baseline creation, authorization"),
    (4, "Load Budget", "Assign resources to schedule, load costs, distribute over time, create cash flow"),
    (5, "Track Performance", "Monitor actuals, compare vs plan, analyze variances, forecast outcome"),
]

_UNIT_RATES: List[Tuple[str, str]] = [
    ("Skilled Worker", "$25/HR"),
    ("Foreman", "$35/HR"),
    ("Engineer", "$60/HR"),
    ("Concrete (m3)", "$120/m3"),
    ("Rebar (kg)", "$1.50/kg"),
    ("Steel (ton)", "$900/ton"),
    ("Excavator", "$150/HR"),
    ("Crane (50T)", "$200/HR"),
    ("Generator", "$80/HR"),
]


# ── SECTION 6 — Variance formulas ─────────────────────────────────────────

_VARIANCE_FORMULAS: List[Tuple[str, str, str]] = [
    ("LABOUR VARIANCE (LV)", "LV = (AH x (AR - SR)) + (SH x (AH - SH))",
     "Wage rate changes, unplanned overtime, inefficient crew mix, learning curve"),
    ("MATERIAL VARIANCE (MV)", "MV = (AQ x (AP - SP)) + (SQ x (AQ - SQ))",
     "Price fluctuations, quantity waste, substitution, material damage"),
    ("PRODUCTIVITY VARIANCE (PV)", "PV = (SH x SR) - (AH x SR)",
     "Poor planning, rework & defects, equipment slowdown, inefficient methods"),
    ("PROCUREMENT VARIANCE (PrV)", "PrV = (Actual Cost) - (Planned Cost)",
     "Poor vendor performance, contract changes, late deliveries"),
    ("SCOPE CHANGE IMPACT", "Change Impact = Additional Cost - Approved Budget",
     "Design changes, scope additions, client changes, unplanned work"),
]

_VARIANCE_BEST_PRACTICES = [
    "Analyze variances weekly",
    "Don't accept unfavorable trends",
    "Look beyond the numbers",
    "Document lessons learned",
    "Act early, not late",
]

_HEATMAP: List[Tuple[str, str, str, str]] = [
    ("Civil", "-5%", "+8%", "0%"),
    ("Structural", "-2%", "+5%", "-3%"),
    ("Mechanical", "-3%", "+12%", "+5%"),
    ("Electrical", "-1%", "+6%", "-2%"),
]


# ── SECTION 13 — Milestone gates ──────────────────────────────────────────

_MILESTONES: List[Tuple[int, str, str]] = [
    (1, "End of Mobilization", "Initial EAC established, baseline confirmed"),
    (2, "End of Foundations", "First major cost data, early CPI trend forming"),
    (3, "End of Structure", "CPI trend reliable, EAC revision if needed"),
    (4, "MEP Rough-in Complete", "High-risk phase complete, forecast stabilizes"),
    (5, "Substantial Completion", "Final EAC, VAC calculated, lessons documented"),
    (6, "Project Completion", "BAC vs AC final closeout, archive for benchmarking"),
]

_MILESTONE_RANGES: List[Tuple[str, str, str]] = [
    ("Early milestones (0-30%)", "CPI is volatile", "forecast range +/-15%"),
    ("Mid-project (30-70%)", "CPI stabilizes", "forecast range +/-5-10%"),
    ("Late project (70-100%)", "CPI very stable", "forecast range +/-2-3%"),
]

_MILESTONE_TABLE: List[Tuple[str, str, str, str]] = [
    ("End of Foundation", "15-MAR-24", "$48.0M", "$18.0M"),
    ("End of Structure", "30-JUN-24", "$48.0M", "$32.5M"),
    ("MEP Rough-in", "31-AUG-24", "$48.0M", "$46.0M"),
    ("Substantial Comp.", "31-OCT-24", "$48.0M", "$48.0M"),
    ("Project Complete", "31-DEC-24", "$48.0M", "$54.0M"),
]


# ── SECTION 15 — Decision framework ───────────────────────────────────────

_DECISION_STEPS: List[Tuple[int, str, List[str]]] = [
    (1, "MEASURE", [
        "Collect accurate, consistent data",
        "Update progress weekly minimum",
        "Verify quantities independently",
        "Lock the data date",
    ]),
    (2, "ANALYZE", [
        "Calculate all EVM metrics",
        "Identify variances by discipline and cost type",
        "Compare against baseline and prior period",
        "Build the S-curve",
    ]),
    (3, "DIAGNOSE", [
        "Find root causes (not symptoms)",
        "Use Fishbone diagram for complex variances",
        "Separate controllable from uncontrollable causes",
        "Quantify impact of each root cause",
    ]),
    (4, "DECIDE", [
        "Select the best corrective action",
        "Evaluate cost vs benefit of each option",
        "Get stakeholder alignment",
        "Document decision and rationale",
    ]),
    (5, "ACT & MONITOR", [
        "Implement corrective action immediately",
        'Set measurable targets (e.g., "CPI must reach 0.95 by Month 6")',
        "Review effectiveness weekly",
        "Adjust if target not being met",
    ]),
]


# ── SECTION 19 — Structure works cost breakdown ───────────────────────────

_STRUCT_LABOUR: List[Tuple[str, int, int, int]] = [
    ("Skilled Worker", 1000, 25, 25000),
    ("Carpenter", 800, 22, 17600),
    ("Steel Fixer", 900, 24, 21600),
    ("Foreman", 400, 40, 16000),
    ("Engineer", 200, 60, 12000),
]
_STRUCT_LABOUR_TOTAL = 92200

_STRUCT_MATERIAL: List[Tuple[str, int, str, float, int]] = [
    ("Concrete", 500, "m3", 120.0, 60000),
    ("Rebar", 20000, "kg", 1.50, 30000),
    ("Formwork", 1000, "m2", 25.0, 25000),
    ("Steel Section", 50, "ton", 900.0, 45000),
]
_STRUCT_MATERIAL_TOTAL = 160000

_STRUCT_EQUIPMENT: List[Tuple[str, int, int, int]] = [
    ("Tower Crane", 240, 200, 48000),
    ("Excavator", 160, 150, 24000),
    ("Concrete Pump", 80, 180, 14400),
    ("Generator", 160, 80, 12800),
]
_STRUCT_EQUIPMENT_TOTAL = 99200
_STRUCT_GRAND_TOTAL = 351400


# ── SECTION "Critical Rules" final block ──────────────────────────────────

_CRITICAL_RULES_FINAL: List[Tuple[int, str]] = [
    (1, "Always calculate EAC and VAC when given project data - never skip this"),
    (2, "Always apply the traffic light system - Green/Amber/Red on every metric"),
    (3, "Never just report numbers - always interpret what they mean"),
    (4, "Always recommend specific actions - never vague advice"),
    (5, "Use construction industry terminology correctly"),
    (6, "When CPI < 0.90 - escalate immediately, this is a crisis"),
    (7, "When SPI < 0.85 - critical path is at risk, recovery plan required"),
    (8, "Always distinguish between cost variance (budget problem) and schedule variance (time problem)"),
    (9, "Remember: EV is the bridge between cost and schedule - it measures both"),
]


# ── SECTION 1 — section headlines for concept Q&A ─────────────────────────

# (section_number, section_title, list of (question, answer) tuples grounded in the source).
_SECTION_CONCEPTS: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    ("1", "COST MANAGEMENT FUNDAMENTALS", [
        ("What does the Project Lifecycle consist of in cost management terms?",
         "5 phases with decreasing cost estimate ranges: Concept (+/-30%), Planning (+/-15%), Execution (+/-5%), Close-Out (+/-3%), and Operations (Variable)."),
        ("What is the key principle for cost management given in the source?",
         '"Projects fail more often from poor cost control than poor planning."'),
        ("Name the 5 pillars of cost management.",
         "Budget Planning, Cost Tracking, Forecasting, Cost Reporting, and Variance Analysis."),
        ("List the 5 cost categories used in construction projects.",
         "LABOUR, MATERIAL, EQUIPMENT, SUBCONTRACT, and OTHER COSTS."),
        ("Give an example of a direct (traceable) cost.",
         "Concrete for foundation, rebar for structure, equipment for excavation."),
        ("Give an example of an indirect (non-traceable) cost.",
         "Site office & utilities, PM team, security & safety, insurance & bonds."),
    ]),
    ("2", "COST BREAKDOWN STRUCTURE (CBS)", [
        ("What is the Cost Breakdown Structure (CBS)?",
         "A hierarchical framework organizing all project costs into a logical structure. It links the budget to the scope and provides the foundation for cost control."),
        ("How many levels does the CBS hierarchy have?",
         "5 levels: Level 1 Total Project Cost, Level 2 Area, Level 3 Discipline, Level 4 Work Package, Level 5 Cost Account."),
        ("What is the difference between WBS and CBS?",
         "WBS defines WHAT will be done (scope & deliverables, owned by PM); CBS defines HOW MUCH it will cost (costs & budget, owned by Cost Manager)."),
        ("What is the CBS rule about cost structure?",
         '"If the cost structure is wrong, the reporting is wrong."'),
        ("In the cost coding example 02.20.ME.P.101A, what does 02 represent?",
         "02 = Area."),
        ("In the cost coding example 02.20.ME.P.101A, what does ME represent?",
         "ME = System (Mechanical)."),
    ]),
    ("3", "BUDGET DEVELOPMENT & COST LOADING", [
        ("List the 5 steps to build a cost-loaded schedule.",
         "1. Estimate, 2. Review, 3. Approve, 4. Load Budget, 5. Track Performance."),
        ("What does step 1 (Estimate) involve when building a cost-loaded schedule?",
         "Define scope, quantify items, apply unit rates."),
        ("What does step 4 (Load Budget) involve when building a cost-loaded schedule?",
         "Assign resources to schedule, load costs, distribute over time, create cash flow."),
        ("What is the unit rate for an Engineer in the cost-loading example?",
         "$60/HR."),
        ("What is the unit rate for Concrete in the cost-loading example?",
         "$120/m3."),
        ("What is the unit rate for Rebar in the cost-loading example?",
         "$1.50/kg."),
        ("What is the rate for a 50T Crane in the cost-loading example?",
         "$200/HR."),
        ("Why does the source say 'A Schedule Without Cost Loading is Only Half the Story'?",
         "Without cost loading: only durations visible, no financial impact, weak cash flow visibility, poor forecasting, high risk of overrun. With cost loading: time + cost visibility, realistic cash flow, better forecasting, stronger control, higher chances of success."),
    ]),
    ("4", "EARNED VALUE MANAGEMENT (EVM)", [
        ("What does EVM stand for and why is it powerful?",
         "Earned Value Management. It is the most powerful project performance system because it integrates scope, schedule, and cost to measure performance and predict future outcomes."),
        ("What are the three key values in EVM?",
         "PV (Planned Value / BCWS), EV (Earned Value / BCWP), and AC (Actual Cost / ACWP)."),
        ("What does BCWS stand for?",
         "Budgeted Cost of Work Scheduled - what was PLANNED to happen."),
        ("What does BCWP stand for?",
         "Budgeted Cost of Work Performed - what has actually been EARNED."),
        ("What does ACWP stand for?",
         "Actual Cost of Work Performed - what it actually COST."),
        ("What does CPI > 1.0 mean?",
         "Cost Efficient - under budget. Interpretation: for every $1 spent, you get more than $1 worth of work."),
        ("What does SPI < 1.0 mean?",
         "Behind Schedule."),
        ("Define BAC.",
         "BAC (Budget at Completion): the total approved budget."),
    ]),
    ("5", "CPI & SPI TRAFFIC LIGHT SYSTEM", [
        ("What is the key message of the traffic light system section?",
         '"You cannot improve what you do not measure."'),
        ("What CPI band recommends 'Keep Doing What Works'?",
         "GREEN (CPI > 1.00) - Under Budget, Cost Efficient."),
        ("What CPI band recommends 'Investigate & Take Action'?",
         "AMBER (0.90 <= CPI <= 1.00) - Slightly Over Budget."),
        ("What CPI band recommends 'Take Corrective Action NOW'?",
         "RED (CPI < 0.90) - Over Budget, Cost Inefficient."),
        ("If CPI > 1.0 AND SPI < 1.0, what is the recommended action?",
         "On Budget But Behind - Recover Schedule."),
        ("If CPI < 1.0 AND SPI < 1.0, what is the recommended action?",
         "Behind & Over Budget - TAKE CORRECTIVE ACTION."),
    ]),
    ("6", "COST VARIANCE ANALYSIS", [
        ("How many types of variance are recognised in the source?",
         "5 types: Labour Variance, Material Variance, Productivity Variance, Procurement Variance, and Scope Change Impact."),
        ("What is the formula for Labour Variance?",
         "LV = (AH x (AR - SR)) + (SH x (AH - SH))."),
        ("What is the formula for Material Variance?",
         "MV = (AQ x (AP - SP)) + (SQ x (AQ - SQ))."),
        ("What is the formula for Productivity Variance?",
         "PV = (SH x SR) - (AH x SR)."),
        ("List the Fishbone categories used in root cause analysis.",
         "PEOPLE, MATERIALS, METHODS, EQUIPMENT, ENVIRONMENT, MANAGEMENT."),
        ("What is the variance analysis rule from the source?",
         '"Treat Causes, Not Symptoms."'),
        ("In the variance heatmap, which discipline shows the worst material variance?",
         "Mechanical at +12% material variance."),
    ]),
    ("7", "FORECASTING PROJECT OUTCOMES", [
        ("List the three EAC scenarios used for forecasting.",
         "Optimistic (CPI = 1.05), Expected (CPI = current), Pessimistic (CPI = 0.80)."),
        ("In the Section 7 example, what is the BAC?",
         "$60,000,000."),
        ("In the Section 7 example, what is the CPI and how is it computed?",
         "CPI = EV/AC = 33,200,000 / 36,500,000 = 0.91."),
        ("In the Section 7 example, what is the EAC?",
         "EAC = BAC/CPI = $60,000,000 / 0.91 = $65,934,066."),
        ("In the Section 7 example, what is the VAC and what does it mean?",
         "VAC = BAC - EAC = -$5,934,066 (OVER BUDGET)."),
        ("What are the two key messages of the forecasting section?",
         '"Forecasts provide time to act before problems become crises." and "The sooner you see it, the easier it is to fix it."'),
        ("List the cost drivers behind forecast changes.",
         "Productivity changes, material price fluctuations, scope changes, rework/quality issues, claims & delays impact, market & inflation."),
    ]),
    ("8", "COST REPORTING & EXECUTIVE DASHBOARDS", [
        ("How many KPIs are on the Executive KPI Dashboard?",
         "6 metrics: CPI, SPI, CV, SV, EAC, VAC."),
        ("What does CV < 0 indicate on the dashboard?",
         "RED - Cost Overrun."),
        ("What does EAC > BAC indicate on the dashboard?",
         "RED - Forecast Over Budget."),
        ("List the 5 items the Management Summary must include.",
         "1. Performance summary (CPI/SPI status), 2. Key drivers (root causes of variance), 3. Top risks (upcoming threats), 4. Actions taken (corrective measures), 5. Next steps (recovery plan)."),
        ("What is the monthly cost summary table format?",
         "Month | PV | EV | AC | CV | SV | CPI | SPI."),
        ("What is the closing quote on reporting?",
         '"Good Reporting Drives Good Decisions."'),
    ]),
    ("9", "BUDGET DISTRIBUTION BENCHMARKS", [
        ("What is the typical Labour share of the budget split?",
         "35%."),
        ("What is the typical Material share of the budget split?",
         "40%."),
        ("What is the typical Equipment share of the budget split?",
         "15%."),
        ("What is the typical Subcontract share of the budget split?",
         "7%."),
        ("What is the typical Other Costs share of the budget split?",
         "3%."),
        ("During the slow start (mobilization) phase, what cumulative % of budget is spent?",
         "0-15%."),
        ("During the acceleration phase, what cumulative % of budget is spent?",
         "15-70%."),
    ]),
    ("10", "HOW TO RESPOND TO PROJECT DATA", [
        ("What are the 7 steps when a user provides project cost data?",
         "1. Calculate CPI and SPI immediately, 2. Apply traffic light status, 3. Calculate EAC and VAC, 4. State what this means in plain English, 5. Identify the top 3 risks, 6. Recommend specific corrective actions, 7. Show the forecast scenario."),
        ("What are the three forecast scenarios in the response template?",
         "Optimistic (CPI improves), Most Likely (current trend), Pessimistic (CPI drops)."),
        ("What categories appear at the top of the standard response format?",
         "PROJECT HEALTH (GREEN/AMBER/RED), KEY METRICS, ROOT CAUSES, CORRECTIVE ACTIONS, FORECAST SCENARIOS."),
    ]),
    ("11", "TCPI - TO-COMPLETE PERFORMANCE INDEX", [
        ("What is the TCPI formula?",
         "TCPI = (BAC - EV) / (BAC - AC)."),
        ("Why is TCPI the most forward-looking EVM metric?",
         "It is the efficiency rate required for the remaining work to finish within the original budget."),
        ("In the Section 11 TCPI example, what are BAC, EV, and AC?",
         "BAC = $100M, EV = $40M, AC = $50M."),
        ("In the Section 11 TCPI example, what is the computed TCPI?",
         "TCPI = ($100M - $40M) / ($100M - $50M) = $60M / $50M = 1.20."),
        ("What does TCPI = 1.20 mean in the Section 11 example?",
         "Must work 20% more efficiently for the rest of the project. Action Required: Immediate budget revision or scope reduction."),
        ("When TCPI > 1.10, what four actions should always be recommended?",
         "1. Request budget increase (EAC revision to management), 2. Scope reduction options, 3. Productivity improvement plan, 4. Re-baseline schedule and cost."),
    ]),
    ("12", "THE EVM TRIANGLE", [
        ("What are the three elements of the EVM Triangle?",
         "SCOPE, SCHEDULE, COST."),
        ("What does SCOPE answer in the EVM Triangle?",
         '"What work gets done?" - defines the total work (BAC), drives the planned progress, changes must flow through change control.'),
        ("What does SCHEDULE answer in the EVM Triangle?",
         '"When work gets done?" - drives the planned value (PV) curve, determines when budget should be spent, controls the S-curve shape.'),
        ("What does COST answer in the EVM Triangle?",
         '"What work costs?" - measures resources used (AC), compared against earned value (EV), determines efficiency (CPI).'),
        ("Why can you not 'game' the EVM system?",
         "EVM integrates all three elements simultaneously - if scope, schedule, AND cost all look good, performance is genuinely good. One weak element exposes the others."),
        ("List the 6 ways EVM drives success.",
         "1. Provides early warning, 2. Identifies problems sooner, 3. Realistic forecasting, 4. Improves decision making, 5. Aligns team on performance, 6. Protects project objectives."),
    ]),
    ("13", "FORECAST MILESTONES", [
        ("How many standard milestone forecast gates does the source list?",
         "6 gates: End of Mobilization, End of Foundations, End of Structure, MEP Rough-in Complete, Substantial Completion, Project Completion."),
        ("What is the rule about milestone forecasts?",
         "Never skip a milestone forecast. Each gate is a decision point."),
        ("What is the forecast range during early milestones (0-30%)?",
         "+/-15% - CPI is volatile."),
        ("What is the forecast range during late project (70-100%)?",
         "+/-2-3% - CPI very stable."),
        ("In the milestone table example, what is the BAC?",
         "$48.0M."),
        ("In the milestone table example, what is the Forecast EAC at Project Complete?",
         "$54.0M."),
    ]),
    ("14", "COMMITMENT TRACKING", [
        ("Why does Section 14 say commitments are the most commonly missed cost element?",
         "Because they are contractual obligations not yet invoiced - they cause budget surprises when ignored."),
        ("Name the three cost states every cost manager must track.",
         "1. ACTUALS (AC) - Invoices received and approved, money paid. 2. COMMITMENTS - Purchase orders and contracts signed, not yet invoiced. 3. BUDGET REMAINING - BAC - AC - Commitments."),
        ("What is the Exposure formula in Section 14?",
         "Exposure = AC + Commitments."),
        ("What is the True Remaining Budget formula in Section 14?",
         "True Remaining Budget = BAC - AC - Commitments."),
        ("Why does ignoring commitments cause budget surprises?",
         "A subcontract signed for $5M is a $5M commitment even if $0 invoiced. Ignoring commitments gives a false sense of budget availability. Projects 'run out of money' when commitments exceed remaining budget."),
        ("What should always be reported in addition to spend?",
         "'Budget Spent + Commitments' reported as 'Total Exposure'."),
    ]),
    ("15", "DECISION-MAKING FRAMEWORK", [
        ("How many steps are in the Cost Control Decision Framework?",
         "5 steps: 1. MEASURE, 2. ANALYZE, 3. DIAGNOSE, 4. DECIDE, 5. ACT & MONITOR."),
        ("What is the key principle of the decision framework?",
         '"A decision without monitoring is just a hope."'),
        ("What does Step 3 (DIAGNOSE) require?",
         "Find root causes (not symptoms); use Fishbone diagram for complex variances; separate controllable from uncontrollable causes; quantify impact of each root cause."),
        ("What does Step 5 (ACT & MONITOR) require?",
         "Implement corrective action immediately; set measurable targets (e.g. 'CPI must reach 0.95 by Month 6'); review effectiveness weekly; adjust if target not being met."),
    ]),
    ("16", "COMMON REPORTING MISTAKES", [
        ("How many common cost-reporting mistakes does Section 16 list?",
         "5 dangerous mistakes."),
        ("What is Mistake 1?",
         "Reporting without analysis. Wrong: 'CPI is 0.87.' Right: explain what 0.87 means and what action to take."),
        ("What is Mistake 4?",
         "Overloading with too much data. Right answer: exception-based reporting - show only RED items at executive level."),
        ("What three questions must every cost report answer?",
         "1. Where are we? (Current status). 2. Where are we going? (Forecast). 3. What are we doing about it? (Actions with owners and dates)."),
    ]),
    ("17", "COMMON CAUSES OF POOR PERFORMANCE", [
        ("How many top root causes of cost overrun does Section 17 list?",
         "8 root causes."),
        ("What is the #1 root cause of cost overruns?",
         "Poor productivity - crew inefficiency, supervision gaps, learning curve."),
        ("What is the #5 root cause of cost overruns?",
         "Rework - quality defects, non-conformances, incorrect installations."),
        ("What should you check when SPI is declining but CPI is acceptable?",
         "Critical path activities specifically; sequencing problems; resource allocation to critical activities; approval bottlenecks (RFIs, submittals, inspections)."),
        ("What should you check when CPI is declining but SPI is acceptable?",
         "Productivity is the primary suspect; labour hours vs quantities installed; subcontractor performance; material wastage rates."),
        ("What should you do when both CPI and SPI are declining?",
         "Systemic problem - management intervention required; consider re-baseline with realistic recovery plan; escalate to executive level immediately."),
    ]),
    ("18", "COST GROWTH CURVES", [
        ("In the optimistic scenario, what CPI is assumed?",
         "CPI = 1.05."),
        ("In the pessimistic scenario, what change in CPI is assumed?",
         "CPI drops by 0.10."),
        ("In the Section 18 BAC = $100M example, what is the weighted EAC?",
         "$110.0M, computed as (0.20 x $95.2M) + (0.60 x $109.9M) + (0.20 x $125.0M)."),
        ("In the Section 18 example, what is the probability of the Expected scenario?",
         "60%."),
        ("In the Section 18 example, what is the EAC under the Pessimistic scenario?",
         "$125.0M with VAC of -$25.0M."),
    ]),
    ("19", "LABOUR, MATERIAL & EQUIPMENT COST EXAMPLES", [
        ("In the structure-works example, what is total LABOUR cost?",
         "$92,200."),
        ("In the structure-works example, what is total MATERIAL cost?",
         "$160,000."),
        ("In the structure-works example, what is total EQUIPMENT cost?",
         "$99,200."),
        ("In the structure-works example, what is the GRAND TOTAL?",
         "$351,400 = $92,200 + $160,000 + $99,200."),
        ("In the structure-works example, how many concrete cubic metres are estimated?",
         "500 m3 at $120/m3 = $60,000."),
        ("In the structure-works example, how many Engineer hours are estimated?",
         "200 hours at $60/HR = $12,000."),
    ]),
    ("20", "COST OVERRUN EXAMPLE", [
        ("In the Section 20 Commercial Building example, what is the Contract Value (BAC)?",
         "$50,000,000."),
        ("In the Section 20 example, what is the Actual Cost to Date?",
         "$32,000,000."),
        ("In the Section 20 example, what is the Earned Value?",
         "$28,500,000."),
        ("In the Section 20 example, what is the Cost Variance?",
         "CV = EV - AC = $28,500,000 - $32,000,000 = -$3,500,000 (OVER BUDGET)."),
        ("In the Section 20 example, what is the EAC?",
         "EAC = BAC/CPI = $56,500,000."),
        ("In the Section 20 example, what is the computed CPI?",
         "CPI = $28.5M / $32M = 0.89 (RED - Take Corrective Action NOW)."),
        ("In the Section 20 example, what is the computed SPI?",
         "SPI = $28.5M / $30.5M = 0.93 (AMBER - Monitor Closely)."),
        ("In the Section 20 example, what is the computed TCPI?",
         "TCPI = ($50M - $28.5M) / ($50M - $32M) = $21.5M / $18M = 1.19."),
    ]),
]


# ── Generators ────────────────────────────────────────────────────────────

def gen_section_concepts() -> Iterator[Dict[str, str]]:
    for section, _title, qa_pairs in _SECTION_CONCEPTS:
        for q, a in qa_pairs:
            yield _row(q, a, section)


def gen_formulas_with_worked_examples() -> Iterator[Dict[str, str]]:
    """Every formula in the source, paired with a worked example using
    real numbers from the document (mostly Section 20 + Section 7 + Section 11).
    """
    # CV
    yield _row(
        "What is the formula for Cost Variance (CV) and show a worked example.",
        f"CV = EV - AC. Using Section 20 numbers: CV = $28,500,000 - $32,000,000 = -$3,500,000 (OVER BUDGET).",
        "4/20",
    )
    yield _row(
        "Compute CV for the Section 20 Commercial Building project.",
        "CV = EV - AC = $28,500,000 - $32,000,000 = -$3,500,000. CV < 0 means Over Budget.",
        "20",
    )
    # SV
    yield _row(
        "What is the formula for Schedule Variance (SV) and show a worked example.",
        "SV = EV - PV. Using Section 20 numbers: SV = $28,500,000 - $30,500,000 = -$2,000,000 (BEHIND SCHEDULE).",
        "4/20",
    )
    yield _row(
        "Compute SV for the Section 20 Commercial Building project.",
        "SV = EV - PV = $28,500,000 - $30,500,000 = -$2,000,000. SV < 0 means Behind Schedule.",
        "20",
    )
    # CPI
    yield _row(
        "What is the formula for the Cost Performance Index (CPI) and show a worked example.",
        "CPI = EV / AC. Using Section 20 numbers: CPI = $28,500,000 / $32,000,000 = 0.89. CPI < 0.90 triggers RED.",
        "4/20",
    )
    yield _row(
        "Compute CPI for the Section 7 example.",
        "CPI = EV / AC = $33,200,000 / $36,500,000 = 0.91.",
        "7",
    )
    yield _row(
        "Compute CPI for the Section 20 Commercial Building project.",
        "CPI = $28,500,000 / $32,000,000 = 0.89.",
        "20",
    )
    # SPI
    yield _row(
        "What is the formula for the Schedule Performance Index (SPI) and show a worked example.",
        "SPI = EV / PV. Using Section 20 numbers: SPI = $28,500,000 / $30,500,000 = 0.93 (AMBER).",
        "4/20",
    )
    yield _row(
        "Compute SPI for the Section 20 Commercial Building project.",
        "SPI = $28,500,000 / $30,500,000 = 0.93. SPI between 0.90 and 1.00 is AMBER.",
        "20",
    )
    # EAC
    yield _row(
        "What is the formula for the Estimate at Completion (EAC) and show a worked example.",
        "EAC = BAC / CPI. Section 7 example: EAC = $60,000,000 / 0.91 = $65,934,066.",
        "4/7",
    )
    yield _row(
        "Compute EAC for the Section 20 Commercial Building project.",
        "EAC = BAC / CPI = $50,000,000 / 0.89 = $56,500,000.",
        "20",
    )
    yield _row(
        "Compute EAC for the Section 7 example.",
        "EAC = $60,000,000 / 0.91 = $65,934,066.",
        "7",
    )
    # ETC
    yield _row(
        "What is the formula for the Estimate to Complete (ETC) and show a worked example.",
        "ETC = EAC - AC. Section 7 example: ETC = $65,934,066 - $36,500,000 = $29,434,066.",
        "4/7",
    )
    yield _row(
        "Compute ETC for the Section 7 example.",
        "ETC = EAC - AC = $65,934,066 - $36,500,000 = $29,434,066.",
        "7",
    )
    yield _row(
        "Compute ETC for the Section 20 Commercial Building project.",
        "ETC = EAC - AC = $56,500,000 - $32,000,000 = $24,500,000 (remaining cost to finish).",
        "20",
    )
    # VAC
    yield _row(
        "What is the formula for the Variance at Completion (VAC) and show a worked example.",
        "VAC = BAC - EAC. Section 20 example: VAC = $50,000,000 - $56,500,000 = -$6,500,000 (over budget at completion).",
        "4/20",
    )
    yield _row(
        "Compute VAC for the Section 7 example.",
        "VAC = BAC - EAC = $60,000,000 - $65,934,066 = -$5,934,066 (OVER BUDGET).",
        "7",
    )
    yield _row(
        "Compute VAC for the Section 20 Commercial Building project.",
        "VAC = BAC - EAC = $50,000,000 - $56,500,000 = -$6,500,000.",
        "20",
    )
    # TCPI
    yield _row(
        "What is the formula for the To-Complete Performance Index (TCPI) and show a worked example.",
        "TCPI = (BAC - EV) / (BAC - AC). Section 11 example: TCPI = ($100M - $40M) / ($100M - $50M) = $60M / $50M = 1.20 (must work 20% more efficiently).",
        "11",
    )
    yield _row(
        "Compute TCPI for the Section 20 Commercial Building project.",
        "TCPI = ($50M - $28.5M) / ($50M - $32M) = $21.5M / $18M = 1.19. Need 19% efficiency improvement - very challenging.",
        "20",
    )
    yield _row(
        "Compute TCPI for the Section 11 example.",
        "TCPI = ($100M - $40M) / ($100M - $50M) = $60M / $50M = 1.20.",
        "11",
    )
    # Exposure / commitments
    yield _row(
        "What is the Exposure formula and show a worked example.",
        "Exposure = AC + Commitments. Section 14 example: Exposure = $32,000,000 + $12,500,000 = $44,500,000.",
        "14",
    )
    yield _row(
        "What is the True Remaining Budget formula and show a worked example.",
        "True Remaining Budget = BAC - AC - Commitments. Section 14 example: $48,000,000 - $32,000,000 - $12,500,000 = $3,500,000.",
        "14",
    )
    # Variance formulas with notation explanation
    yield _row(
        "What is the Labour Variance (LV) formula and explain its terms?",
        "LV = (AH x (AR - SR)) + (SH x (AH - SH)). Causes include wage rate changes, unplanned overtime, inefficient crew mix, learning curve.",
        "6",
    )
    yield _row(
        "What is the Material Variance (MV) formula and explain its causes?",
        "MV = (AQ x (AP - SP)) + (SQ x (AQ - SQ)). Causes: price fluctuations, quantity waste, substitution, material damage.",
        "6",
    )
    yield _row(
        "What is the Productivity Variance (PV) formula and explain its causes?",
        "PV = (SH x SR) - (AH x SR). Causes: poor planning, rework & defects, equipment slowdown, inefficient methods.",
        "6",
    )
    yield _row(
        "What is the Procurement Variance (PrV) formula?",
        "PrV = (Actual Cost) - (Planned Cost). Includes purchase price, logistics, expediting, penalties.",
        "6",
    )
    yield _row(
        "What is the Scope Change Impact formula?",
        "Change Impact = Additional Cost - Approved Budget. May be compensable (client pays) or non-compensable.",
        "6",
    )

    # Now the formula table with explanations: produce a Q&A for every formula
    # entry so the model sees both 'what is the formula' and 'plain English'.
    for key, formula, plain, section in _FORMULAS:
        yield _row(
            f"State the {key} formula exactly as written in construction_evm.md.",
            f"{formula}.",
            section,
        )
        yield _row(
            f"Explain what {key} means in plain English using the source's description.",
            f"{plain}. Formula: {formula}.",
            section,
        )

    # Plug a few additional values into each EVM formula using Section 20 numbers
    # so the model sees multiple worked examples for the same formula.
    s20 = {
        "BAC": _S20_BAC, "AC": _S20_AC, "EV": _S20_EV, "PV": _S20_PV,
        "CPI": _S20_CPI, "SPI": _S20_SPI, "EAC": _S20_EAC,
    }
    yield _row(
        f"Given BAC={s20['BAC']:,}, AC={s20['AC']:,}, EV={s20['EV']:,}, what is CPI?",
        f"CPI = EV/AC = {s20['EV']:,}/{s20['AC']:,} = {round(s20['EV']/s20['AC'], 2)} (RED - below 0.90).",
        "20",
    )
    yield _row(
        f"Given EV={s20['EV']:,} and PV={s20['PV']:,}, what is SPI?",
        f"SPI = EV/PV = {s20['EV']:,}/{s20['PV']:,} = {round(s20['EV']/s20['PV'], 2)} (AMBER).",
        "20",
    )
    yield _row(
        f"Given BAC={s20['BAC']:,} and CPI={s20['CPI']}, what is EAC?",
        f"EAC = BAC/CPI = {s20['BAC']:,}/{s20['CPI']} = $56,500,000.",
        "20",
    )
    yield _row(
        f"Given BAC={s20['BAC']:,} and EAC={s20['EAC']:,}, what is VAC?",
        f"VAC = BAC - EAC = {s20['BAC']:,} - {s20['EAC']:,} = -$6,500,000 (over budget at completion).",
        "20",
    )
    yield _row(
        f"Given BAC={s20['BAC']:,}, EV={s20['EV']:,}, AC={s20['AC']:,}, what is TCPI?",
        f"TCPI = (BAC - EV)/(BAC - AC) = (50,000,000 - 28,500,000)/(50,000,000 - 32,000,000) = 21,500,000/18,000,000 = 1.19.",
        "20",
    )
    # Section 7 plug-ins
    yield _row(
        f"Given BAC=$60,000,000, AC=$36,500,000, EV=$33,200,000, what is CPI?",
        f"CPI = EV/AC = $33,200,000/$36,500,000 = 0.91.",
        "7",
    )
    yield _row(
        f"Given BAC=$60,000,000 and CPI=0.91, what is EAC?",
        f"EAC = BAC/CPI = $60,000,000/0.91 = $65,934,066.",
        "7",
    )
    yield _row(
        f"Given EAC=$65,934,066 and AC=$36,500,000, what is ETC?",
        f"ETC = EAC - AC = $65,934,066 - $36,500,000 = $29,434,066.",
        "7",
    )
    yield _row(
        f"Given BAC=$60,000,000 and EAC=$65,934,066, what is VAC?",
        f"VAC = BAC - EAC = $60,000,000 - $65,934,066 = -$5,934,066 (OVER BUDGET).",
        "7",
    )
    # Section 11 plug-ins
    yield _row(
        f"Given BAC=$100M, EV=$40M, AC=$50M, what is TCPI and what does it mean?",
        f"TCPI = ($100M - $40M)/($100M - $50M) = $60M/$50M = 1.20. Must work 20% more efficiently for the rest of the project.",
        "11",
    )
    # CV/SV verbose worked examples in different terms
    yield _row(
        "If a project has EV = $33,200,000 and AC = $36,500,000, compute the Cost Variance.",
        "CV = EV - AC = $33,200,000 - $36,500,000 = -$3,300,000 (Over Budget).",
        "4/7",
    )
    yield _row(
        "If a project has EV = $28,500,000 and PV = $30,500,000, compute the Schedule Variance.",
        "SV = EV - PV = $28,500,000 - $30,500,000 = -$2,000,000 (Behind Schedule).",
        "4/20",
    )
    # CV/SV decision rules
    yield _row(
        "What does CV > 0 mean per the source?",
        "CV > 0 means Under Budget. CV = EV - AC.",
        "4",
    )
    yield _row(
        "What does CV < 0 mean per the source?",
        "CV < 0 means Over Budget. CV = EV - AC.",
        "4",
    )
    yield _row(
        "What does SV > 0 mean per the source?",
        "SV > 0 means Ahead of Schedule. SV = EV - PV.",
        "4",
    )
    yield _row(
        "What does SV < 0 mean per the source?",
        "SV < 0 means Behind Schedule. SV = EV - PV.",
        "4",
    )
    yield _row(
        "What does CPI = 1.0 mean per the source?",
        "CPI = 1.0 means On Budget. CPI = EV/AC.",
        "4",
    )
    yield _row(
        "What does SPI = 1.0 mean per the source?",
        "SPI = 1.0 means On Schedule. SPI = EV/PV.",
        "4",
    )
    yield _row(
        "What does VAC > 0 mean per the source?",
        "VAC > 0 means the project will finish under budget. VAC = BAC - EAC.",
        "4",
    )
    yield _row(
        "What does VAC < 0 mean per the source?",
        "VAC < 0 means the project will finish over budget. VAC = BAC - EAC.",
        "4",
    )
    # CPI interpretation phrase
    yield _row(
        "What is the CPI interpretation phrase given in Section 4?",
        "For every $1 spent, you get $[CPI] worth of work. Formula: CPI = EV / AC.",
        "4",
    )
    # alternate EAC formula
    yield _row(
        "What is the alternate EAC formula and its assumption?",
        "EAC (alternate) = BAC / CPI. Assumes future performance mirrors past.",
        "4",
    )
    # Additional plug-ins to round out formula coverage with worked numbers.
    yield _row(
        "Given Section 14 numbers (BAC $48,000,000, AC $32,000,000, Commitments $12,500,000), compute Exposure.",
        "Exposure = AC + Commitments = $32,000,000 + $12,500,000 = $44,500,000.",
        "14",
    )
    yield _row(
        "Given Section 14 numbers (BAC $48,000,000, AC $32,000,000, Commitments $12,500,000), compute True Remaining Budget.",
        "True Remaining Budget = BAC - AC - Commitments = $48,000,000 - $32,000,000 - $12,500,000 = $3,500,000.",
        "14",
    )
    yield _row(
        "If a project has BAC $50,000,000 and EAC $56,500,000, what is the VAC?",
        "VAC = BAC - EAC = $50,000,000 - $56,500,000 = -$6,500,000.",
        "20",
    )
    yield _row(
        "What is the formula relationship between EAC and ETC?",
        "ETC = EAC - AC. Section 7 worked example: $65,934,066 - $36,500,000 = $29,434,066.",
        "4",
    )
    yield _row(
        "What is the formula relationship between BAC, EAC and VAC?",
        "VAC = BAC - EAC. Section 20 worked example: $50,000,000 - $56,500,000 = -$6,500,000.",
        "4",
    )
    yield _row(
        "Why does EAC = BAC/CPI assume future performance mirrors past?",
        "Because dividing total budget by the current cost performance index extrapolates the current burn rate to project completion. Formula: EAC = BAC / CPI.",
        "4",
    )
    yield _row(
        "If CPI = 0.89, what does it mean per the source's interpretation phrase?",
        "For every $1 spent, you get $0.89 worth of work (Section 4 interpretation: 'For every $1 spent, you get $[CPI] worth of work').",
        "4",
    )
    yield _row(
        "Compute SV when EV = $33,200,000 and PV is unknown but SPI = 0.95.",
        "SV = EV - PV. With SPI = EV/PV = 0.95, PV = EV/0.95. Source instructs: always compute SV from the raw EV and PV values when both are available. Source values for the Section 7 example: BAC $60,000,000, AC $36,500,000, EV $33,200,000.",
        "4/7",
    )
    yield _row(
        "Compute the variance between Section 7 ETC and AC.",
        "ETC ($29,434,066) is the remaining cost. AC ($36,500,000) is the cost already incurred. EAC = ETC + AC = $29,434,066 + $36,500,000 = $65,934,066 (matches EAC = BAC/CPI).",
        "7",
    )


def gen_traffic_light_thresholds() -> Iterator[Dict[str, str]]:
    """Every CPI/SPI/health threshold from Section 5 + Section 8."""
    # Direct band Q&A
    for band, condition, meaning, action in _CPI_BANDS:
        yield _row(
            f"What CPI condition triggers {band} on the CPI traffic light?",
            f"{band}: {condition} - {meaning} - {action}.",
            "5",
        )
        yield _row(
            f"On the CPI traffic light, what is the {band} band's recommended action?",
            f"{action} (because {condition} indicates {meaning}).",
            "5",
        )
    for band, condition, meaning, action in _SPI_BANDS:
        yield _row(
            f"What SPI condition triggers {band} on the SPI traffic light?",
            f"{band}: {condition} - {meaning} - {action}.",
            "5",
        )
        yield _row(
            f"On the SPI traffic light, what is the {band} band's recommended action?",
            f"{action} (because {condition} indicates {meaning}).",
            "5",
        )

    # Direct numeric-threshold drills
    yield _row(
        "What CPI value triggers RED?",
        "RED is triggered when CPI < 0.90 (Over Budget, Cost Inefficient - Take Corrective Action NOW).",
        "5",
    )
    yield _row(
        "What CPI value triggers GREEN?",
        "GREEN is triggered when CPI > 1.00 (Under Budget, Cost Efficient - Keep Doing What Works).",
        "5",
    )
    yield _row(
        "What CPI band is the range 0.90 to 1.00?",
        "AMBER (0.90 <= CPI <= 1.00) - Slightly Over Budget - Investigate & Take Action.",
        "5",
    )
    yield _row(
        "What SPI value triggers RED?",
        "RED is triggered when SPI < 0.90 (Behind Schedule, At Risk - Take Corrective Action NOW).",
        "5",
    )
    yield _row(
        "What SPI value triggers GREEN?",
        "GREEN is triggered when SPI > 1.00 (Ahead of Schedule, Very Good - Maintain Momentum).",
        "5",
    )
    yield _row(
        "What SPI band is the range 0.90 to 1.00?",
        "AMBER (0.90 <= SPI <= 1.00) - Slightly Behind - Review Plan & Recover.",
        "5",
    )
    # Specific boundary probes
    for cpi_val, expected in [
        (0.85, "RED (below 0.90)"),
        (0.89, "RED (below 0.90)"),
        (0.90, "AMBER (within 0.90-1.00)"),
        (0.95, "AMBER (within 0.90-1.00)"),
        (1.00, "AMBER (within 0.90-1.00)"),
        (1.05, "GREEN (above 1.00)"),
        (1.20, "GREEN (above 1.00)"),
    ]:
        yield _row(
            f"What traffic-light band does CPI = {cpi_val} fall into?",
            f"CPI = {cpi_val} is {expected}.",
            "5",
        )
    for spi_val, expected in [
        (0.80, "RED (below 0.90)"),
        (0.89, "RED (below 0.90)"),
        (0.90, "AMBER (within 0.90-1.00)"),
        (0.93, "AMBER (within 0.90-1.00)"),
        (1.00, "AMBER (within 0.90-1.00)"),
        (1.02, "GREEN (above 1.00)"),
        (1.15, "GREEN (above 1.00)"),
    ]:
        yield _row(
            f"What traffic-light band does SPI = {spi_val} fall into?",
            f"SPI = {spi_val} is {expected}.",
            "5",
        )

    # Combined status interpretation
    for condition, meaning in _COMBINED_STATUS:
        yield _row(
            f"What is the combined-status interpretation when {condition}?",
            f"{meaning}.",
            "5",
        )

    # Section 8 traffic-light reporting
    for condition, meaning in _TRAFFIC_LIGHT_REPORTING:
        yield _row(
            f"On the executive dashboard, what does {condition} indicate?",
            f"{meaning}.",
            "8",
        )

    # Critical Rule thresholds
    yield _row(
        "When does the source say CPI requires immediate escalation as a crisis?",
        "When CPI < 0.90 - escalate immediately, this is a crisis.",
        "CRITICAL",
    )
    yield _row(
        "When does the source say SPI puts the critical path at risk and requires a recovery plan?",
        "When SPI < 0.85 - critical path is at risk, recovery plan required.",
        "CRITICAL",
    )
    # Extra threshold drills
    yield _row(
        "What is the exact upper bound of the AMBER CPI band?",
        "1.00 - the AMBER band is 0.90 <= CPI <= 1.00.",
        "5",
    )
    yield _row(
        "What is the exact lower bound of the AMBER CPI band?",
        "0.90 - the AMBER band is 0.90 <= CPI <= 1.00.",
        "5",
    )
    yield _row(
        "Does CPI = 0.90 fall in RED or AMBER?",
        "AMBER (the AMBER band is 0.90 <= CPI <= 1.00; RED is CPI < 0.90).",
        "5",
    )
    yield _row(
        "Does SPI = 0.90 fall in RED or AMBER?",
        "AMBER (the AMBER band is 0.90 <= SPI <= 1.00; RED is SPI < 0.90).",
        "5",
    )
    yield _row(
        "What's the key message of the CPI/SPI traffic light system per Section 5?",
        '"You cannot improve what you do not measure."',
        "5",
    )
    yield _row(
        "If CPI = 1.0 and SPI = 1.0, what does combined status indicate?",
        "Both metrics on track. Per Section 5, CPI > 1.0 AND SPI > 1.0 is 'Ahead & Under Budget - Great Performance'; at the 1.0 boundary the project is on budget and on schedule.",
        "5",
    )


def gen_tcpi_interpretation() -> Iterator[Dict[str, str]]:
    """Every TCPI boundary value from Section 11."""
    for boundary, meaning, verdict in _TCPI_BANDS:
        yield _row(
            f"What does {boundary} mean per Section 11?",
            f"{meaning} - {verdict}.",
            "11",
        )
        yield _row(
            f"How does Section 11 classify {boundary}?",
            f"{verdict}: {meaning}.",
            "11",
        )
    # Specific numeric probes
    for tcpi_val, expected in [
        (0.85, "Achievable - remaining work needs LESS efficiency than planned"),
        (0.95, "Achievable - remaining work needs LESS efficiency than planned"),
        (1.00, "On Track - remaining work needs EXACTLY planned efficiency"),
        (1.05, "Difficult - remaining work needs MORE efficiency than planned"),
        (1.10, "Difficult - remaining work needs MORE efficiency than planned (at the 1.10 boundary)"),
        (1.15, "Very Unlikely - remaining work needs 10%+ more efficiency"),
        (1.19, "Very Unlikely - remaining work needs 10%+ more efficiency (Section 20 case)"),
        (1.20, "RED - virtually impossible without scope reduction or budget increase"),
        (1.25, "RED - virtually impossible without scope reduction or budget increase"),
    ]:
        yield _row(
            f"How should TCPI = {tcpi_val} be interpreted per Section 11?",
            f"TCPI = {tcpi_val}: {expected}.",
            "11",
        )

    # Action triggers
    for idx, action in enumerate(_TCPI_RECOMMENDATIONS, start=1):
        yield _row(
            f"When TCPI > 1.10, what is recommendation #{idx}?",
            f"{action}.",
            "11",
        )
    yield _row(
        "List all four recommendations when TCPI > 1.10.",
        "1. Request budget increase (EAC revision to management). 2. Scope reduction options. 3. Productivity improvement plan. 4. Re-baseline schedule and cost.",
        "11",
    )
    yield _row(
        "Why is TCPI considered the most forward-looking EVM metric?",
        "Because TCPI = (BAC - EV) / (BAC - AC) measures the efficiency rate required for the remaining work to finish within the original budget.",
        "11",
    )
    # Additional TCPI threshold rows
    yield _row(
        "At what TCPI threshold does Section 11 say remaining work needs 10%+ more efficiency?",
        "TCPI > 1.10 - remaining work needs 10%+ more efficiency - Very Unlikely.",
        "11",
    )
    yield _row(
        "At what TCPI threshold does Section 11 classify the project as virtually impossible without scope reduction or budget increase?",
        "TCPI > 1.20 - virtually impossible without scope reduction or budget increase - RED.",
        "11",
    )
    yield _row(
        "Is TCPI = 1.0 achievable per Section 11?",
        "TCPI = 1.0 means remaining work needs EXACTLY planned efficiency - On Track (achievable).",
        "11",
    )
    yield _row(
        "Is TCPI = 0.95 achievable per Section 11?",
        "Yes - TCPI < 1.0 means remaining work needs LESS efficiency than planned - Achievable.",
        "11",
    )
    yield _row(
        "Is TCPI = 1.20 achievable per Section 11?",
        "TCPI > 1.20 is classified RED - virtually impossible without scope reduction or budget increase.",
        "11",
    )
    yield _row(
        "What action does Section 11 recommend when TCPI = 1.19?",
        "TCPI = 1.19 is in the Very Unlikely band (TCPI > 1.10). Recommend: 1. Request budget increase, 2. Scope reduction options, 3. Productivity improvement plan, 4. Re-baseline schedule and cost.",
        "11/20",
    )


def gen_forecasting_scenarios() -> Iterator[Dict[str, str]]:
    """Section 18 + Section 7 forecasting examples."""
    # Section 18 scenarios
    for name, cpi, eac, vac, probability in _FORECAST_SCENARIOS:
        yield _row(
            f"In the Section 18 BAC=$100M example, what CPI does the {name} scenario assume?",
            f"{name} scenario assumes CPI = {cpi}.",
            "18",
        )
        yield _row(
            f"In the Section 18 BAC=$100M example, what is the EAC under the {name} scenario?",
            f"{name} EAC = {eac}.",
            "18",
        )
        yield _row(
            f"In the Section 18 BAC=$100M example, what is the VAC under the {name} scenario?",
            f"{name} VAC = {vac}.",
            "18",
        )
        yield _row(
            f"In the Section 18 BAC=$100M example, what probability is assigned to the {name} scenario?",
            f"{name} scenario probability = {probability}.",
            "18",
        )

    # Weighted EAC
    yield _row(
        "What is the weighted EAC computation in the Section 18 BAC=$100M example?",
        f"Weighted EAC = {_WEIGHTED_EAC_CALC} = {_WEIGHTED_EAC}.",
        "18",
    )
    yield _row(
        "What is the final weighted EAC value in the Section 18 BAC=$100M example?",
        f"Weighted EAC = {_WEIGHTED_EAC}.",
        "18",
    )

    # Use-when / risk for each scenario
    yield _row(
        "When should the Optimistic forecast scenario be used per Section 18?",
        "When clear corrective actions are implemented and management is committed. Risk: overconfident, may delay necessary escalation.",
        "18",
    )
    yield _row(
        "What does the Expected scenario assume per Section 18?",
        "Current performance trend continues unchanged. This is the BASE CASE - always show this prominently.",
        "18",
    )
    yield _row(
        "What does the Pessimistic scenario assume per Section 18?",
        "Conditions worsen, new risks materialize. Always present to management - they must know the worst case.",
        "18",
    )

    # Section 7 scenarios
    yield _row(
        "In Section 7, what CPI does the Optimistic forecast scenario assume?",
        "CPI = 1.05 - Best case, past inefficiencies won't recur.",
        "7",
    )
    yield _row(
        "In Section 7, what CPI does the Pessimistic forecast scenario assume?",
        "CPI = 0.80 - Worst case, conditions deteriorate.",
        "7",
    )
    yield _row(
        "What does the Expected forecast scenario assume per Section 7?",
        "CPI = current. Most likely - trends continue.",
        "7",
    )
    # Section 7 worked example full
    yield _row(
        "Walk through the full Section 7 example calculation.",
        "BAC = $60,000,000. AC = $36,500,000. EV = $33,200,000. CPI = EV/AC = 0.91. EAC = BAC/CPI = $65,934,066. ETC = EAC - AC = $29,434,066. VAC = BAC - EAC = -$5,934,066 (OVER BUDGET).",
        "7",
    )
    # Forecasting best practices and cost drivers
    forecasting_best = [
        "Update forecasts regularly",
        "Use actual data, not assumptions",
        "Analyze trends, not just numbers",
        "Communicate early and clearly",
        "Take action based on forecasts",
    ]
    for bp in forecasting_best:
        yield _row(
            f"What forecasting best practice does Section 7 list: '{bp}'?",
            f"{bp}.",
            "7",
        )
    cost_drivers = [
        "Productivity changes",
        "Material price fluctuations",
        "Scope changes",
        "Rework/quality issues",
        "Claims & delays impact",
        "Market & inflation",
    ]
    for cd in cost_drivers:
        yield _row(
            f"What is one cost driver behind forecast changes per Section 7: '{cd}'?",
            f"{cd}.",
            "7",
        )
    yield _row(
        "List all forecasting best practices per Section 7.",
        "; ".join(forecasting_best) + ".",
        "7",
    )
    yield _row(
        "List all cost drivers behind forecast changes per Section 7.",
        "; ".join(cost_drivers) + ".",
        "7",
    )


def gen_commitment_tracking() -> Iterator[Dict[str, str]]:
    """Section 14 cost tracking example - every line item."""
    for label, value in _COMMITMENT_LINES:
        yield _row(
            f"In the Section 14 cost tracking example, what is the {label}?",
            f"{label}: {value}.",
            "14",
        )
        yield _row(
            f"In the Section 14 example, state the value of {label}.",
            f"{value}.",
            "14",
        )
    # Computation rows
    yield _row(
        "In the Section 14 cost tracking example, how is True Exposure computed?",
        "True Exposure = AC + Commitments = $32,000,000 + $12,500,000 = $44,500,000.",
        "14",
    )
    yield _row(
        "In the Section 14 cost tracking example, how is True Remaining Unencumbered Budget computed?",
        "True Remaining Unencumbered Budget = BAC - AC - Commitments = $48,000,000 - $32,000,000 - $12,500,000 = $3,500,000.",
        "14",
    )
    yield _row(
        "In the Section 14 cost tracking example, what percentage of budget is truly unencumbered?",
        "Only 7.3% of budget is truly unencumbered ($3,500,000 / $48,000,000).",
        "14",
    )
    yield _row(
        "Why does Section 14 illustrate a $5M subcontract commitment with $0 invoiced?",
        "To show that a subcontract signed for $5M is a $5M commitment even if $0 invoiced - ignoring commitments gives a false sense of budget availability.",
        "14",
    )
    yield _row(
        "What is the consequence of ignoring commitments per Section 14?",
        "Projects 'run out of money' when commitments exceed remaining budget.",
        "14",
    )
    yield _row(
        "Per Section 14, what should always be reported alongside budget spent?",
        "Always report Budget Spent + Commitments as 'Total Exposure'.",
        "14",
    )
    # Each cost state
    yield _row(
        "Per Section 14, what is the definition of ACTUALS (AC)?",
        "Invoices received and approved, money paid.",
        "14",
    )
    yield _row(
        "Per Section 14, what is the definition of COMMITMENTS?",
        "Purchase orders and contracts signed, not yet invoiced.",
        "14",
    )
    yield _row(
        "Per Section 14, what is the definition of BUDGET REMAINING?",
        "BAC - AC - Commitments.",
        "14",
    )
    # Extra commitment rows
    yield _row(
        "If BAC is $48M, AC is $32M, and Commitments are $12.5M, what is the True Remaining Budget?",
        "True Remaining Budget = $48M - $32M - $12.5M = $3.5M.",
        "14",
    )
    yield _row(
        "If BAC is $48M, AC is $32M, and Commitments are $12.5M, what is the Total Exposure?",
        "Total Exposure = AC + Commitments = $32M + $12.5M = $44.5M.",
        "14",
    )
    yield _row(
        "If a project has $44.5M total exposure against $48M BAC, what percent of budget is truly unencumbered?",
        "($48M - $44.5M) / $48M = 7.3% truly unencumbered.",
        "14",
    )
    yield _row(
        "Per Section 14, why is reporting only spent costs misleading?",
        "Because commitments (purchase orders and contracts signed but not yet invoiced) are real obligations. Ignoring commitments gives a false sense of budget availability. Projects 'run out of money' when commitments exceed remaining budget.",
        "14",
    )
    yield _row(
        "What is the most commonly missed cost element per Section 14?",
        "Commitments - contractual obligations not yet invoiced.",
        "14",
    )
    yield _row(
        "If a subcontract is signed for $5M but $0 has been invoiced, what is its commitment value?",
        "$5M - per Section 14: 'A subcontract signed for $5M is a $5M commitment even if $0 invoiced.'",
        "14",
    )
    yield _row(
        "Per Section 14, what three things must every cost manager track?",
        "1. ACTUALS (AC) - Invoices received and approved, money paid. 2. COMMITMENTS - Purchase orders and contracts signed, not yet invoiced. 3. BUDGET REMAINING - BAC - AC - Commitments.",
        "14",
    )
    yield _row(
        "What is the headline risk in the Section 14 cost tracking example?",
        "Only 7.3% of budget truly unencumbered ($3.5M remaining out of $48M BAC after $32M AC + $12.5M commitments).",
        "14",
    )
    yield _row(
        "In the Section 14 example, sum AC and Commitments and compare to BAC.",
        "AC + Commitments = $32M + $12.5M = $44.5M. BAC = $48M. Difference = $3.5M (only 7.3% truly unencumbered).",
        "14",
    )


def gen_common_mistakes() -> Iterator[Dict[str, str]]:
    """Section 16 common reporting mistakes."""
    for i, (title, wrong, right) in enumerate(_MISTAKES, start=1):
        yield _row(
            f"What is Cost Reporting Mistake {i} per Section 16?",
            f"Mistake {i}: {title}. Wrong: {wrong}. Right: {right}",
            "16",
        )
        yield _row(
            f"How should you correctly handle the '{title}' mistake?",
            f"{right}",
            "16",
        )
        yield _row(
            f"What is the wrong way to handle the '{title}' issue per Section 16?",
            f"Wrong: {wrong}",
            "16",
        )
    # Three required questions
    for i, q in enumerate(_REPORT_THREE_QUESTIONS, start=1):
        yield _row(
            f"What is question #{i} that every cost report must answer per Section 16?",
            f"{q}.",
            "16",
        )
    yield _row(
        "List the three questions every cost report must answer per Section 16.",
        f"1. {_REPORT_THREE_QUESTIONS[0]}. 2. {_REPORT_THREE_QUESTIONS[1]}. 3. {_REPORT_THREE_QUESTIONS[2]}.",
        "16",
    )
    yield _row(
        "How many dangerous cost-reporting mistakes are listed in Section 16?",
        "5 dangerous mistakes.",
        "16",
    )
    # Specific anchored examples
    yield _row(
        "Per Section 16, how should you correctly report 'CPI is 0.87'?",
        "CPI is 0.87 - we are getting only $0.87 of value for every dollar spent. Primary cause is MEP rework. Immediate action: re-inspect all Level 3 MEP before proceeding to Level 4.",
        "16",
    )
    yield _row(
        "Per Section 16, what makes 'Costs are over budget' a poor report?",
        "It has no clear actions or ownership. Right: 'Costs are over budget. [NAME] will re-baseline the mechanical subcontract by [DATE]. Target: recover 0.05 CPI points by end of next month.'",
        "16",
    )
    yield _row(
        "Per Section 16, what is the rule about declining CPI trends?",
        "Three consecutive months of declining CPI is an emergency regardless of current CPI value.",
        "16",
    )
    # Additional rows: explicit per-mistake "what's wrong / what's right" framing
    yield _row(
        "Per Section 16, what is the right way to report a 50-page cost report situation?",
        "Exception-based reporting. Show only RED items at executive level. Detail available on request.",
        "16",
    )
    yield _row(
        "Per Section 16, what is the right way to pair past performance with forecasting?",
        "Always pair actuals with forecast - 'We spent $X last month AND we forecast $Y at completion.'",
        "16",
    )
    yield _row(
        "Per Section 16, what is wrong with only doing monthly snapshot reporting?",
        "It ignores forecasts and trends. Right answer: show the trend line. Three consecutive months of declining CPI is an emergency regardless of current CPI value.",
        "16",
    )
    yield _row(
        "What is Mistake 2 in cost reporting per Section 16?",
        "Mistake 2: Focusing only on past performance. Wrong: reporting only what happened last month. Right: pair actuals with forecast.",
        "16",
    )
    yield _row(
        "What is Mistake 3 in cost reporting per Section 16?",
        "Mistake 3: Ignoring forecasts and trends. Wrong: monthly snapshot reporting only. Right: show the trend line - three consecutive months of declining CPI is an emergency regardless of current CPI value.",
        "16",
    )
    yield _row(
        "What is Mistake 5 in cost reporting per Section 16?",
        "Mistake 5: No clear actions or ownership. Wrong: 'Costs are over budget.' Right: name owner, date, target recovery.",
        "16",
    )
    yield _row(
        "Per Section 16, why must every report list actions with owners and dates?",
        "Because answering 'What are we doing about it?' is one of the three questions every cost report must answer.",
        "16",
    )


def gen_root_causes() -> Iterator[Dict[str, str]]:
    """Section 17 root causes and diagnostic patterns."""
    for rank, name, detail in _ROOT_CAUSES:
        yield _row(
            f"What is root cause #{rank} of cost overrun per Section 17?",
            f"{name} - {detail}.",
            "17",
        )
        yield _row(
            f"Per Section 17, describe the '{name}' root cause.",
            f"{detail}.",
            "17",
        )
    yield _row(
        "List all 8 top root causes of cost overruns per Section 17, in order.",
        ", ".join(f"{r}. {n}" for r, n, _ in _ROOT_CAUSES) + ".",
        "17",
    )
    # Fishbone categories from Section 6
    for cat, desc in _FISHBONE:
        yield _row(
            f"Per Section 6 (fishbone analysis), what does the {cat} category include?",
            f"{desc}.",
            "6",
        )
    yield _row(
        "List the 6 fishbone categories used in root cause analysis per Section 6.",
        ", ".join(c for c, _ in _FISHBONE) + ".",
        "6",
    )
    # Diagnostic patterns
    for condition, checks in _DECLINING_DIAGNOSIS:
        for check in checks:
            yield _row(
                f"When {condition}, what is one diagnostic check per Section 17?",
                f"{check}.",
                "17",
            )
        yield _row(
            f"List all diagnostic checks for the condition: {condition}.",
            "; ".join(checks) + ".",
            "17",
        )
    # The rule
    yield _row(
        "What is the rule for variance analysis per Section 6?",
        '"Treat Causes, Not Symptoms."',
        "6",
    )
    # Variance best practices
    for bp in _VARIANCE_BEST_PRACTICES:
        yield _row(
            f"Per Section 6 best practices, what is the practice: '{bp}'?",
            f"{bp}.",
            "6",
        )


def gen_executive_dashboard() -> Iterator[Dict[str, str]]:
    """Section 8 dashboard required fields."""
    for idx, kpi, full in _EXEC_KPIS:
        yield _row(
            f"What is KPI #{idx} on the Executive KPI Dashboard?",
            f"{kpi} - {full}.",
            "8",
        )
        yield _row(
            f"On the Executive KPI Dashboard, what does the abbreviation {kpi} stand for?",
            f"{full}.",
            "8",
        )
    yield _row(
        "How many KPIs are on the Executive KPI Dashboard per Section 8?",
        "6 KPIs: CPI, SPI, CV, SV, EAC, VAC.",
        "8",
    )
    # Reporting best practices
    for idx, practice in _REPORTING_BEST_PRACTICES:
        yield _row(
            f"Per Section 8, what is reporting best practice #{idx}?",
            f"{practice}.",
            "8",
        )
    # Management summary items
    for idx, item in _MGMT_SUMMARY:
        yield _row(
            f"Per Section 8, what is Management Summary item #{idx}?",
            f"{item}.",
            "8",
        )
    yield _row(
        "Per Section 8, what is the monthly cost summary table format?",
        "Month | PV | EV | AC | CV | SV | CPI | SPI.",
        "8",
    )
    # S-curve lines
    for line, desc in _S_CURVE_LINES:
        yield _row(
            f"In S-Curve Analysis per Section 8, what does the {line} represent?",
            f"{desc}.",
            "8",
        )
    yield _row(
        "What is the closing quote of Section 8 on reporting?",
        '"Good Reporting Drives Good Decisions."',
        "8",
    )


def gen_section_20_worked_overrun() -> Iterator[Dict[str, str]]:
    """Section 20 full overrun example."""
    yield _row(
        "What kind of project is the Section 20 worked example?",
        "Commercial Building Project - At Risk Status.",
        "20",
    )
    # Every input line
    inputs = [
        ("Contract Value (BAC)", "$50,000,000"),
        ("Actual Cost to Date (AC)", "$32,000,000"),
        ("Earned Value (EV)", "$28,500,000"),
        ("Cost Variance (CV)", "-$3,500,000 (OVER BUDGET)"),
        ("Schedule Variance (SV)", "-$2,000,000 (BEHIND SCHEDULE)"),
        ("VAC (Variance at Completion)", "-$6,500,000"),
        ("EAC (BAC/CPI)", "$56,500,000"),
    ]
    for label, value in inputs:
        yield _row(
            f"In the Section 20 worked example, what is the {label}?",
            f"{label}: {value}.",
            "20",
        )
    # Every formula step
    yield _row(
        "In the Section 20 example, how is CPI computed and what is its band?",
        "CPI = $28.5M / $32M = 0.89 - RED - Take Corrective Action NOW.",
        "20",
    )
    yield _row(
        "In the Section 20 example, how is SPI computed and what is its band?",
        "SPI = $28.5M / $30.5M = 0.93 - AMBER - Monitor Closely.",
        "20",
    )
    yield _row(
        "In the Section 20 example, what is the value-per-dollar interpretation of CPI?",
        "For every $1 spent, only $0.89 of work is being done.",
        "20",
    )
    yield _row(
        "In the Section 20 example, what overrun does the project forecast?",
        "Project forecasts to finish $6.5M over budget.",
        "20",
    )
    yield _row(
        "In the Section 20 example, how is TCPI computed?",
        "TCPI = ($50M - $28.5M) / ($50M - $32M) = $21.5M / $18M = 1.19.",
        "20",
    )
    yield _row(
        "In the Section 20 example, what does TCPI = 1.19 imply?",
        "Need 19% efficiency improvement - very challenging.",
        "20",
    )
    # Primary corrective actions
    actions = [
        (1, "Re-inspect MEP systems for rework drivers", "this week"),
        (2, "Re-sequence critical path to recover 2-week delay", "this month"),
        (3, "Request change order review with client for scope additions", "this month"),
        (4, "Revise EAC upward and present to executive team", "immediately"),
    ]
    for idx, action, timeline in actions:
        yield _row(
            f"In the Section 20 example, what is Primary Corrective Action #{idx}?",
            f"{action} ({timeline}).",
            "20",
        )
    yield _row(
        "List all primary corrective actions from the Section 20 example.",
        "; ".join(f"{i}. {a} ({t})" for i, a, t in actions) + ".",
        "20",
    )
    # Aggregate summary
    yield _row(
        "Summarise the Section 20 Commercial Building project status.",
        "BAC $50M, AC $32M, EV $28.5M, CV -$3.5M, SV -$2.0M, VAC -$6.5M, EAC $56.5M, CPI 0.89 (RED), SPI 0.93 (AMBER), TCPI 1.19.",
        "20",
    )
    yield _row(
        "What is the headline risk status of the Section 20 project?",
        "At Risk - CPI 0.89 (RED), forecasts $6.5M over budget, needs 19% efficiency improvement.",
        "20",
    )
    yield _row(
        "What is the relationship between CV (-$3,500,000) and EV/AC in Section 20?",
        "CV = EV - AC = $28,500,000 - $32,000,000 = -$3,500,000.",
        "20",
    )
    yield _row(
        "What is the relationship between VAC (-$6,500,000) and BAC/EAC in Section 20?",
        "VAC = BAC - EAC = $50,000,000 - $56,500,000 = -$6,500,000.",
        "20",
    )
    # Extra Section 20 rows
    yield _row(
        "What is the SV value in the Section 20 example?",
        "SV = -$2,000,000 (BEHIND SCHEDULE).",
        "20",
    )
    yield _row(
        "What is the CV emoji marker shown in Section 20?",
        "CV = -$3,500,000 marked RED (OVER BUDGET).",
        "20",
    )
    yield _row(
        "What is the SV emoji marker shown in Section 20?",
        "SV = -$2,000,000 marked RED (BEHIND SCHEDULE).",
        "20",
    )
    yield _row(
        "In Section 20, by how much is the project forecast over budget?",
        "$6.5M over budget at completion (VAC = -$6,500,000).",
        "20",
    )
    yield _row(
        "In Section 20, how does CPI 0.89 compare to the AMBER lower bound 0.90?",
        "CPI 0.89 falls below the AMBER lower bound of 0.90, placing it firmly in the RED band.",
        "20",
    )
    yield _row(
        "In Section 20, why is SPI 0.93 AMBER not RED?",
        "Because 0.93 is between 0.90 and 1.00 - the AMBER band (RED begins below 0.90).",
        "20",
    )
    yield _row(
        "In Section 20, what is the implied PV value used to compute SPI 0.93?",
        "EV/SPI = $28.5M / 0.93 = approximately $30.5M (used in SV = EV - PV = -$2.0M).",
        "20",
    )


def gen_cost_categories_and_pillars() -> Iterator[Dict[str, str]]:
    """5 pillars, 5 cost categories, direct vs indirect."""
    # Pillars
    for i, (pillar, desc) in enumerate(_PILLARS, start=1):
        yield _row(
            f"What is Pillar #{i} of cost management per Section 1?",
            f"{pillar} - {desc}.",
            "1",
        )
        yield _row(
            f"Describe the '{pillar}' pillar per Section 1.",
            f"{desc}.",
            "1",
        )
    yield _row(
        "How many pillars of cost management does Section 1 list?",
        "5 pillars: Budget Planning, Cost Tracking, Forecasting, Cost Reporting, Variance Analysis.",
        "1",
    )
    # Cost categories
    for cat, examples in _COST_CATEGORIES:
        yield _row(
            f"What is the {cat} cost category per Section 1?",
            f"{cat}: {examples}.",
            "1",
        )
        yield _row(
            f"Give examples of the {cat} cost category.",
            f"{examples}.",
            "1",
        )
    yield _row(
        "List the 5 cost categories per Section 1.",
        ", ".join(c for c, _ in _COST_CATEGORIES) + ".",
        "1",
    )
    # Direct vs indirect
    for kind, examples in _DIRECT_INDIRECT:
        yield _row(
            f"Per Section 1, what examples does the source give for {kind} costs?",
            f"{examples}.",
            "1",
        )
    yield _row(
        "What is the distinction between Direct and Indirect costs per Section 1?",
        "Direct (Traceable): concrete for foundation, rebar for structure, equipment for excavation. Indirect (Non-Traceable): site office & utilities, PM team, security & safety, insurance & bonds.",
        "1",
    )
    yield _row(
        "What is the Key Principle from Section 1?",
        '"Projects fail more often from poor cost control than poor planning."',
        "1",
    )
    # Budget split percentages (Section 9)
    for label, pct in _BUDGET_SPLIT:
        yield _row(
            f"Per Section 9, what is the typical budget percentage for {label}?",
            f"{label}: {pct}.",
            "9",
        )
    yield _row(
        "List the typical budget split by cost type per Section 9.",
        ", ".join(f"{l} {p}" for l, p in _BUDGET_SPLIT) + ".",
        "9",
    )
    # S-curve phases
    for phase, pct in _S_CURVE_PHASES:
        yield _row(
            f"Per Section 9 S-curve, what cumulative % is spent during the '{phase}' phase?",
            f"{phase}: {pct}.",
            "9",
        )
    yield _row(
        "How many S-curve phases does Section 9 describe?",
        "4 phases: Slow start (mobilization), Acceleration phase, Peak expenditure, Slowdown (commissioning).",
        "9",
    )
    yield _row(
        "Per Section 9, during which phase does Peak expenditure occur on the S-curve?",
        "Peak expenditure: 50-85% cumulative budget.",
        "9",
    )
    yield _row(
        "Per Section 9, what is the cumulative budget % range for the Slowdown (commissioning) phase?",
        "85-100%.",
        "9",
    )
    yield _row(
        "Sum the Section 9 budget percentages for Labour and Material.",
        "Labour (35%) + Material (40%) = 75%. Equipment 15% + Subcontract 7% + Other Costs 3% = 25%.",
        "9",
    )


def gen_lifecycle_estimate_ranges() -> Iterator[Dict[str, str]]:
    """Section 1 lifecycle phases & estimate accuracy."""
    for phase, accuracy in _LIFECYCLE_PHASES:
        yield _row(
            f"What is the estimate accuracy at the {phase} phase per Section 1?",
            f"{phase}: {accuracy}.",
            "1",
        )
        yield _row(
            f"At which lifecycle phase is the estimate accuracy {accuracy}?",
            f"{phase} phase.",
            "1",
        )
    yield _row(
        "How many project lifecycle phases does Section 1 list?",
        "5 phases: Concept, Planning, Execution, Close-Out, Operations.",
        "1",
    )
    yield _row(
        "List all 5 lifecycle phases with their estimate accuracy per Section 1.",
        ", ".join(f"{p} {a}" for p, a in _LIFECYCLE_PHASES) + ".",
        "1",
    )


def gen_cbs_hierarchy() -> Iterator[Dict[str, str]]:
    """Section 2 CBS levels and code example."""
    for level, name, detail in _CBS_LEVELS:
        yield _row(
            f"What is CBS Level {level} per Section 2?",
            f"Level {level}: {name} - {detail}.",
            "2",
        )
        yield _row(
            f"In the CBS hierarchy, what does Level {level} represent?",
            f"{name}.",
            "2",
        )
    yield _row(
        "How many levels does the CBS hierarchy have per Section 2?",
        "5 levels: Total Project Cost, Area, Discipline, Work Package, Cost Account.",
        "2",
    )
    # Code parts
    for code, meaning in _CBS_CODE_PARTS:
        yield _row(
            f"In the cost coding example 02.20.ME.P.101A, what does '{code}' represent?",
            f"{code} = {meaning}.",
            "2",
        )
    yield _row(
        "Decode the cost coding example 02.20.ME.P.101A per Section 2.",
        "02 = Area, 20 = Discipline, ME = System (Mechanical), P = Work Package, 101A = Cost Account Identifier.",
        "2",
    )
    yield _row(
        "What is the CBS rule per Section 2?",
        '"If the cost structure is wrong, the reporting is wrong."',
        "2",
    )
    yield _row(
        "What is the difference between WBS and CBS per Section 2?",
        "WBS defines WHAT will be done (scope & deliverables, owned by PM). CBS defines HOW MUCH it will cost (costs & budget, owned by Cost Manager).",
        "2",
    )
    yield _row(
        "Who owns the CBS per Section 2?",
        "The Cost Manager owns the CBS (costs & budget).",
        "2",
    )
    yield _row(
        "Who owns the WBS per Section 2?",
        "The PM owns the WBS (scope & deliverables).",
        "2",
    )


def gen_principles_and_rules() -> Iterator[Dict[str, str]]:
    """Every 'Rule:' or 'Key Principle:' / closing quote in the file."""
    rules: List[Tuple[str, str]] = [
        ("1", '"Projects fail more often from poor cost control than poor planning." (Key Principle, Section 1)'),
        ("2", '"If the cost structure is wrong, the reporting is wrong." (Rule, Section 2)'),
        ("5", '"You cannot improve what you do not measure." (Key Message, Section 5)'),
        ("6", '"Treat Causes, Not Symptoms." (Rule, Section 6)'),
        ("7", '"Forecasts provide time to act before problems become crises." (Key Message, Section 7)'),
        ("7", '"The sooner you see it, the easier it is to fix it." (Key Message, Section 7)'),
        ("8", '"Good Reporting Drives Good Decisions." (Closing quote, Section 8)'),
        ("13", "Never skip a milestone forecast. Each gate is a decision point. (Rule, Section 13)"),
        ("15", '"A decision without monitoring is just a hope." (Key Principle, Section 15)'),
        ("end", '"A project without cost control is simply a project waiting to surprise you." (closing)'),
        ("end", '"Earned Value tells you where the project really stands." (closing)'),
        ("end", '"What gets measured gets managed. What gets managed gets improved." (closing)'),
    ]
    for section, quote in rules:
        yield _row(
            f"What is the Section {section} principle/quote from construction_evm.md?",
            quote,
            section,
        )
    # Each Critical Rule from the final block
    for idx, rule_text in _CRITICAL_RULES_FINAL:
        yield _row(
            f"What is Critical Rule #{idx} at the end of construction_evm.md?",
            f"{rule_text}.",
            "CRITICAL",
        )
        if idx == 6:
            yield _row(
                "Per the Critical Rules, when does the source say to escalate immediately?",
                "When CPI < 0.90 - escalate immediately, this is a crisis.",
                "CRITICAL",
            )
    # Critical Rules - list them all in one row too
    yield _row(
        "List all 9 Critical Rules from the end of construction_evm.md.",
        "; ".join(f"{i}. {t}" for i, t in _CRITICAL_RULES_FINAL) + ".",
        "CRITICAL",
    )
    # Decision framework key principle
    yield _row(
        "What is the Key Principle of Section 15 (Decision Framework)?",
        '"A decision without monitoring is just a hope."',
        "15",
    )
    # Reporting rule
    yield _row(
        "What is the Section 16 Rule about cost reports?",
        "Every cost report must answer three questions: 1. Where are we? (Current status). 2. Where are we going? (Forecast). 3. What are we doing about it? (Actions with owners and dates).",
        "16",
    )
    # Power-of-EVM principle
    yield _row(
        "What is the Section 12 'Power of EVM' principle?",
        "EVM integrates all three elements (scope, schedule, cost) simultaneously. You cannot game the system - if scope, schedule, AND cost all look good, performance is genuinely good. One weak element exposes the others.",
        "12",
    )
    # Section 13 milestone rule re-emphasised
    yield _row(
        "What is the Section 13 milestone rule?",
        "Never skip a milestone forecast. Each gate is a decision point.",
        "13",
    )
    # Section 4 EAC assumption
    yield _row(
        "What is the assumption baked into EAC = BAC/CPI per Section 4?",
        "Assumes future performance mirrors past.",
        "4",
    )
    # Critical Rule emphasis when CPI < 0.90
    yield _row(
        "Per the closing Critical Rules, what does CPI < 0.90 require?",
        "When CPI < 0.90 - escalate immediately, this is a crisis.",
        "CRITICAL",
    )
    # Critical Rule emphasis when SPI < 0.85
    yield _row(
        "Per the closing Critical Rules, what does SPI < 0.85 require?",
        "When SPI < 0.85 - critical path is at risk, recovery plan required.",
        "CRITICAL",
    )
    yield _row(
        "Per the closing Critical Rules, why is EV called 'the bridge'?",
        "Because EV is the bridge between cost and schedule - it measures both.",
        "CRITICAL",
    )
    yield _row(
        "Per Critical Rule #1, what must always be calculated when given project data?",
        "Always calculate EAC and VAC when given project data - never skip this.",
        "CRITICAL",
    )


# ── orchestrator ──────────────────────────────────────────────────────────

_GENERATORS: List[Tuple[str, Callable[[], Iterator[Dict[str, str]]]]] = [
    ("section_concepts", gen_section_concepts),
    ("formulas_with_worked_examples", gen_formulas_with_worked_examples),
    ("traffic_light_thresholds", gen_traffic_light_thresholds),
    ("tcpi_interpretation", gen_tcpi_interpretation),
    ("forecasting_scenarios", gen_forecasting_scenarios),
    ("commitment_tracking", gen_commitment_tracking),
    ("common_mistakes", gen_common_mistakes),
    ("root_causes", gen_root_causes),
    ("executive_dashboard", gen_executive_dashboard),
    ("section_20_worked_overrun", gen_section_20_worked_overrun),
    ("cost_categories_and_pillars", gen_cost_categories_and_pillars),
    ("lifecycle_estimate_ranges", gen_lifecycle_estimate_ranges),
    ("cbs_hierarchy", gen_cbs_hierarchy),
    ("principles_and_rules", gen_principles_and_rules),
]


def generate_all() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _, gen in _GENERATORS:
        rows.extend(gen())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/learning/evm_scenarios.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output file instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts per generator without writing.")
    args = parser.parse_args()

    counts: Dict[str, int] = {}
    rows: List[Dict[str, str]] = []
    for name, gen in _GENERATORS:
        produced = list(gen())
        counts[name] = len(produced)
        rows.extend(produced)

    print("== generator counts ==", file=sys.stderr)
    for name, n in counts.items():
        print(f"  {name:<34} {n}", file=sys.stderr)
    print(f"  {'TOTAL':<34} {len(rows)}", file=sys.stderr)

    if args.dry_run:
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (mode={mode})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
