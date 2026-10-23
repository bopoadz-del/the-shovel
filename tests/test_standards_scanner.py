"""
Test the broadened standards-advisory scanner (W8).

Original: single `no_approved_on_design` rule.
Broadened: 9 rules (1 core + 8 broadened), advisory only.
"""

from __future__ import annotations

import pytest

from app.core.standards_scanner import (
    BROADENED,
    StandardsFlag,
    StandardsScanner,
    scan_text,
    scan_to_md,
)


# -- Test fixtures --------------------------------------------------------

DESIGN_DOC_WITHOUT_APPROVAL = """
Structural Drawing Set -- Revision C
Drawn by: Eng. A. Smith
Checked by: Eng. B. Jones
Date: 2026-03-15

This drawing shows the foundation layout for Building A.
"""

INSPECTION_WITHOUT_ITP = """
Inspection Report -- Concrete Pour, Level 3
Inspector: M. Khan
Date: 2026-04-20

The concrete pour was inspected per the project quality plan.
No ITP reference is available for this pour.
Slump test passed. No defects observed.
"""

MATERIAL_SUBSTITUTION_UNVERIFIED = """
Material Substitution Request
Original: C40 concrete specified
Proposed: Use equivalent C35 concrete with additives
Justification: Supplier recommendation
"""

WELDING_WITHOUT_WPS = """
Welding Inspection Report
Welder: Ali H.
Joint: Column base plate to foundation
Visual inspection: acceptable
No WPS or PQR reference provided.
"""

COMMISSIONING_NO_SIGNOFF = """
Commissioning Report -- HVAC System
Functional testing completed.
Performance testing: within specification.
No taking-over certificate referenced.
"""

METHOD_STATEMENT_MENTION = """
Work Plan -- Piling Operations
Method statement to be prepared.
High-risk activity identified: deep excavation (>6m).
"""

AS_BUILT_WITHOUT_SURVEY = """
As-Built Drawing -- Drainage Layout
Recorded as constructed.
No survey verification data included.
"""

CLEAN_DOC = """
Monthly Progress Report -- May 2026
Activities completed: formwork Level 5, rebar Level 4.
Milestones on track.
"""


# -- Tests ----------------------------------------------------------------


class TestCoreRule:
    """The original no_approved_on_design rule always runs."""

    def test_design_doc_without_approval(self):
        scanner = StandardsScanner(broadened=False)
        flags = scanner.scan(
            DESIGN_DOC_WITHOUT_APPROVAL,
            context={"document_type": "design_drawing", "has_approved_stamp": False},
        )
        assert len(flags) == 1
        assert flags[0].rule_id == "no_approved_on_design"

    def test_design_doc_with_approval(self):
        scanner = StandardsScanner(broadened=False)
        flags = scanner.scan(
            DESIGN_DOC_WITHOUT_APPROVAL,
            context={"document_type": "design_drawing", "has_approved_stamp": True},
        )
        assert len(flags) == 0

    def test_no_context_advisory(self):
        """When approval status unknown, flag advisory."""
        scanner = StandardsScanner(broadened=False)
        flags = scanner.scan(
            DESIGN_DOC_WITHOUT_APPROVAL,
            context={"document_type": "design_drawing"},
        )
        assert len(flags) == 1
        assert "check manually" in flags[0].message.lower()


class TestBroadenedRules:
    """The 8 broadened rules fire on appropriate content."""

    @pytest.fixture
    def scanner(self):
        return StandardsScanner(broadened=True)

    def test_inspection_without_itp(self, scanner):
        flags = scanner.scan(
            INSPECTION_WITHOUT_ITP,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "missing_itp_reference" in ids

    def test_material_substitution(self, scanner):
        flags = scanner.scan(
            MATERIAL_SUBSTITUTION_UNVERIFIED,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "unverified_material_substitution" in ids

    def test_welding_without_wps(self, scanner):
        flags = scanner.scan(
            WELDING_WITHOUT_WPS,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "undocumented_welding_procedure" in ids

    def test_commissioning_no_signoff(self, scanner):
        flags = scanner.scan(
            COMMISSIONING_NO_SIGNOFF,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "missing_commissioning_signoff" in ids

    def test_method_statement_missing(self, scanner):
        flags = scanner.scan(
            METHOD_STATEMENT_MENTION,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "missing_method_statement" in ids

    def test_as_built_without_survey(self, scanner):
        flags = scanner.scan(
            AS_BUILT_WITHOUT_SURVEY,
            context={"document_type": "test_report"},
        )
        ids = [f.rule_id for f in flags]
        assert "unverified_as_built_status" in ids

    def test_regulatory_requirement(self, scanner):
        text = "The civil defense authority must approve the fire safety design."
        flags = scanner.scan(text, context={"document_type": "test_report"})
        ids = [f.rule_id for f in flags]
        assert "uncited_regulatory_requirement" in ids

    def test_multiple_flags(self, scanner):
        """A document can trigger multiple flags."""
        combined = (
            INSPECTION_WITHOUT_ITP + "\n" +
            WELDING_WITHOUT_WPS + "\n" +
            COMMISSIONING_NO_SIGNOFF + "\n" +
            MATERIAL_SUBSTITUTION_UNVERIFIED
        )
        flags = scanner.scan(combined, context={"document_type": "test_report"})
        ids = {f.rule_id for f in flags}
        assert "missing_itp_reference" in ids
        assert "undocumented_welding_procedure" in ids
        assert "missing_commissioning_signoff" in ids
        assert "unverified_material_substitution" in ids


class TestCleanDocument:
    """Clean documents produce no flags."""

    def test_clean_doc_no_flags(self):
        scanner = StandardsScanner(broadened=True)
        flags = scanner.scan(CLEAN_DOC, context={"document_type": "progress_report"})
        assert len(flags) == 0


class TestFlagStructure:
    """Every flag has the required structure."""

    def test_flag_fields(self):
        scanner = StandardsScanner(broadened=True)
        flags = scanner.scan(
            WELDING_WITHOUT_WPS,
            context={"document_type": "test_report"},
        )
        assert len(flags) >= 1
        for f in flags:
            assert f.rule_id
            assert f.severity in ("advisory", "warning", "critical")
            assert f.message
            assert f.triggered is True


class TestMarkdownOutput:
    """scan_to_markdown produces valid markdown advisory block."""

    def test_md_contains_advisory_header(self):
        scanner = StandardsScanner(broadened=True)
        md = scanner.scan_to_markdown(
            WELDING_WITHOUT_WPS + "\n" + INSPECTION_WITHOUT_ITP,
            context={"document_type": "test_report"},
        )
        assert "Standards Advisory" in md
        assert "advisory" in md.lower() or "ADVISORY" in md

    def test_md_empty_when_no_flags(self):
        scanner = StandardsScanner(broadened=True)
        md = scanner.scan_to_markdown(CLEAN_DOC, context={"document_type": "progress_report"})
        assert md == ""


class TestRuleCounts:
    """Rule counts match expected."""

    def test_core_only_has_one_rule(self):
        scanner = StandardsScanner(broadened=False)
        assert scanner.rule_count() == 1
        assert scanner.rule_ids() == ["no_approved_on_design"]

    def test_broadened_has_nine_rules(self):
        scanner = StandardsScanner(broadened=True)
        assert scanner.rule_count() == 9  # 1 core + 8 broadened


class TestEnvFlag:
    """BROADENED env flag controls behavior."""

    def test_env_flag_type(self):
        assert isinstance(BROADENED, bool)

    def test_scanner_ignores_env_when_explicit(self):
        """Scanner respects explicit broadened parameter over env."""
        scanner_on = StandardsScanner(broadened=True)
        scanner_off = StandardsScanner(broadened=False)
        assert scanner_on.rule_count() > scanner_off.rule_count()


class TestAdvisoryOnly:
    """All flags are advisory -- never blocking."""

    def test_all_flags_advisory(self):
        scanner = StandardsScanner(broadened=True)
        combined = (
            INSPECTION_WITHOUT_ITP + "\n" +
            WELDING_WITHOUT_WPS + "\n" +
            MATERIAL_SUBSTITUTION_UNVERIFIED + "\n" +
            COMMISSIONING_NO_SIGNOFF + "\n" +
            METHOD_STATEMENT_MENTION
        )
        flags = scanner.scan(combined, context={"document_type": "test_report"})
        assert len(flags) >= 4, f"Expected >= 4 flags, got {len(flags)}: {[f.rule_id for f in flags]}"
        for f in flags:
            assert f.severity == "advisory", f"Flag {f.rule_id} is {f.severity}, expected advisory"
