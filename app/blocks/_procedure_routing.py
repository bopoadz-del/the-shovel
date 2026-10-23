"""
SMART ORCHESTRATOR - CONSTRUCTION PROCEDURE ROUTING ADDITIONS
==============================================================
Procedure-specific routing patterns that prepend to ACTION_PATTERNS in
app/blocks/smart_orchestrator.py so procedure-specific queries are caught
first, before generic construction keywords.

These map real user language to the correct platform action, one entry per
procedure KIND in the shipped catalogue (app/data/procedures). No document
codes live here: a code the user types is resolved at run time against the
project's own documents (app.core.procedure_catalogue.expand_codes) and
routed by its kind's label.
"""

PROCEDURE_ROUTING_ADDITIONS = [

    # -- DESIGN MANAGEMENT ---------------------------------------------------
    # Design review workflows, acceptance forms, design directives
    (
        "design_review_workflow",
        [
            "design review", "review package", "review workshop", "design acceptance", "design acceptance form",
            "for comment", "buy-off", "design buy off", "design comments schedule",
            "design review schedule", "review status", "design package acceptance",
            "project decision note", "PDN", "design coordination meeting",
            "design progress evaluation", "design workshop decision",
            "distribution period", "review window",
        ]
    ),
    (
        "design_directive",
        [
            "design directive", "DD-",
            "design instruction", "design change instruction",
            "employer instruction", "design order",
            "site instruction", "site instructions",
            "si vs vo", "si versus vo",
        ]
    ),

    # -- PROJECT CONTROLS ----------------------------------------------------
    # RFI, risk register, work package
    (
        "rfi_management",
        [
            "request for information",
            "RFI-", "technical query", "technical enquiry",
            "contractor query", "design query", "clarification request",
            "rfi log", "rfi register", "open rfis", "overdue rfi",
            "overdue rfis", "rfi status", "rfi tracker",
            "rfis are open", "how many rfi",
        ]
    ),
    (
        "risk_register_auto_populate",
        [
            "risk register", "risk log", "risk matrix",
            "risk score", "probability impact", "risk mitigation",
            "risk identification", "risk assessment", "risk appetite",
            "amber risk", "red risk", "green risk",
            "if then risk", "risk statement",
        ]
    ),
    (
        "work_package_control",
        [
            "work package", "WP-", "package control",
            "work package register",
            "package status", "package milestone", "package overdue",
        ]
    ),

    # -- QUALITY & CONSTRUCTION ----------------------------------------------
    # QA audit, NCR, T&C, handover, inspection, HSE
    (
        "qa_audit",
        [
            "qa audit", "quality audit", "qc audit",
            "audit program", "audit programme", "audit finding",
            "critical finding", "major finding", "minor finding",
            "audit observation", "2nd party audit",
        ]
    ),
    (
        "ncr_management",
        [
            "NCR", "non-conformance", "non conformance",
            "nonconformance", "NCR-", "disposition",
            "issue ncr", "issue an ncr", "raise an ncr", "raise ncr",
            "issue a ncr",
            "use as is", "reject and replace", "concession request",
            "corrective action", "ncr closure", "close ncr",
            "ncr register", "open ncr", "physical ncr", "documentary ncr",
        ]
    ),
    (
        "commissioning_checklist",
        [
            "testing commissioning", "test and commission",
            "T&C", "ITP", "inspection test plan",
            "commissioning result", "punch list", "pre-commissioning",
            "rides scope", "mep commissioning", "building commissioning",
        ]
    ),
    (
        "handover_management",
        [
            "handover", "practical completion", "CPC",
            "certificate of practical completion", "DLP",
            "defects liability", "snag list", "as-built",
            "o&m manual", "handover register", "handover checklist",
            "handover prerequisites",
        ]
    ),
    (
        "inspection_request",
        [
            "inspection request", "IR-",
            "contractor inspection", "material inspection",
            "witness inspection", "hold point release",
            "inspection result", "inspection rejection",
            "WIR", "work inspection request", "work inspection",
            "WIR form", "WIR template",
            "hold point", "hold points", "witness point", "review point",
            "prepare a wir", "raise a wir", "issue a wir",
        ]
    ),
    (
        "safety_compliance_audit",
        [
            "hse audit", "hse inspection",
            "near miss",
            "fatality risk", "serious injury", "environmental incident",
            "toolbox talk", "ppe compliance", "hse finding",
            "work resumption", "hse register",
            "hse compliance audit", "working at height", "work at height", "fall protection",
            "site safety audit", "compliance audit checklist",
        ]
    ),

    # -- TENDERING & PROCUREMENT ---------------------------------------------
    # Job requisition, RFP, tender analysis, award
    (
        "job_requisition",
        [
            "job requisition", "JR-", "JR number",
            "prequalification", "prequalify", "procurement strategy",
            "packaging strategy", "rfp preparation",
        ]
    ),
    (
        "rfp_management",
        [
            "request for proposal", "RFP",
            "tender package", "instructions to tenderers",
            "form of tender", "tender documents",
            "scope of work document",
            # TASK 1 vocabulary patch (additive): "RFP" alone scores 0.2,
            # below the 0.3 non-generative gate; the sweep's verbatim miss
            # was "prepare an RFP for the landscaping subcontract package".
            "prepare an rfp", "rfp for", "issue an rfp", "draft an rfp",
            "rfp document",
        ]
    ),
    (
        "tender_bid_analysis",
        [
            "tender analysis", "tender evaluation",
            "TER", "tender evaluation report",
            "bid scoring", "bid comparison", "tender recommendation",
            "RAP", "rapid approval",
            "technical score", "commercial score",
            "contractor bids", "compare bids", "score bids", "tender scoring", "evaluate bids",
        ]
    ),
    (
        "contract_award",
        [
            "contract award", "letter of award", "LOA",
            "performance bond", "advance payment bond",
            "award approval", "award threshold",
        ]
    ),

    # -- COMMERCIAL ----------------------------------------------------------
    # Payments, change management, variation orders
    (
        "payment_certificate",
        [
            "interim payment", "payment request", "PR-",
            "payment certificate", "payment certification",
            "retention release", "retention calculation",
            "payment workflow", "certified amount", "disputed amount",
            "cumulative billed", "payment status",
        ]
    ),
    (
        "change_order_impact",
        [
            "change management", "request for modification",
            "RFM-", "variation order", "VO-",
            "provisional sum directive", "PSD",
            "type a change", "type b change", "type c change",
            "owner initiated change", "contractor claim variation",
            "regulatory change", "scope change authority",
            "variation settlement", "contract account",
        ]
    ),
]
