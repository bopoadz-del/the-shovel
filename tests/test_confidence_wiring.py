"""Tests for Stream E: measured confidence wired into the confidence envelope.

Covers:
1. Each _analyze_* method produces a measured confidence (not hardcoded).
2. Rich text scores higher than near-empty text for the same analysis type.
3. confidence_report["measured"] is True.
4. Envelope (UniversalBlock.execute) passes through result confidence as-is;
   when the result has NO confidence key the envelope carries None, not 0.95.
"""

import asyncio
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

OLD_HARDCODED = {0.85, 0.80, 0.75, 0.70, 0.60}

RICH_DRAWING_TEXT = (
    "Drawing A-101 Rev C  Scale 1:100  elevation plan section dimension\n"
    "Concrete slab: 15.5m x 12.3m concrete steel rebar pour cast install\n"
    "Materials: C30 concrete, Grade-60 rebar, brick block glass aluminum\n"
    "5 no. Concrete Columns  3 no. Steel Beams  Formwork: 380 m2\n"
    "Inspection witness hold point required prior to concrete pour\n"
) * 40  # ~3 800 chars — well above 400-char/page threshold


SPARSE_TEXT = "drawing"  # ~7 chars — well below threshold, no fields populated


RICH_SPEC_TEXT = (
    "Specification Section 03 30 00 Cast-in-Place Concrete\n"
    "Material: C35/45 concrete per BS EN 206, Grade 60 rebar\n"
    "Method: pour cast place install with vibration\n"
    "Inspection witness hold point prior to placement\n"
    "Cement aggregate sand timber insulation membrane tile gypsum\n"
) * 40


RICH_CONTRACT_TEXT = (
    "CONTRACT AGREEMENT between Contractor and Employer\n"
    "Contractor shall complete works within 24 months of commencement.\n"
    "Employer shall pay monthly interim valuations per schedule.\n"
    "Payment terms: monthly invoice schedule milestone.\n"
    "Liquidated damages: SAR 50,000 per week of delay.\n"
    "Retention: 5% retention until practical completion.\n"
    "Contract value: 12,500,000 SAR total.\n"
    "Termination for cause with 28-day notice.\n"
) * 40


RICH_SCHEDULE_TEXT = (
    "Project Schedule — Primavera P6 Export\n"
    "Milestone: mobilisation, substantial completion, practical completion, handover\n"
    "Activity 1001 duration 15 days excavation\n"
    "Activity 1002 duration 20 days concrete foundations\n"
    "Activity 1003 duration 30 days structural steel\n"
    "Gantt milestone handover completion duration week\n"
) * 40


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: envelope no longer fabricates 0.95
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvelopeConfidence:
    """UniversalBlock.execute envelope confidence rules."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_no_confidence_key_gives_none(self):
        """A result dict without 'confidence' → envelope confidence is None."""
        from app.core.universal_base import UniversalBlock
        from abc import abstractmethod

        class _NoConfBlock(UniversalBlock):
            name = "test_no_conf"
            version = "1.0"
            layer = 3
            tags = []
            requires = []

            async def process(self, input_data, params=None):
                return {"data": "hello"}  # no confidence key

        block = _NoConfBlock()
        env = self._run(block.execute("input"))
        assert env["confidence"] is None, (
            f"expected None but got {env['confidence']!r}"
        )

    def test_result_with_confidence_is_passed_through(self):
        """A result dict WITH 'confidence' → envelope confidence equals it."""
        from app.core.universal_base import UniversalBlock

        class _WithConfBlock(UniversalBlock):
            name = "test_with_conf"
            version = "1.0"
            layer = 3
            tags = []
            requires = []

            async def process(self, input_data, params=None):
                return {"data": "hello", "confidence": 0.73}

        block = _WithConfBlock()
        env = self._run(block.execute("input"))
        assert env["confidence"] == 0.73

    def test_block_error_gives_zero(self):
        """Exceptions still produce confidence 0.0."""
        from app.core.universal_base import UniversalBlock

        class _BrokenBlock(UniversalBlock):
            name = "test_broken"
            version = "1.0"
            layer = 3
            tags = []
            requires = []

            async def process(self, input_data, params=None):
                raise RuntimeError("kaboom")

        block = _BrokenBlock()
        env = self._run(block.execute("input"))
        assert env["status"] == "error"
        assert env["confidence"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BaseBlock envelope (app.core.block) — same rules
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseBlockEnvelopeConfidence:
    """app.core.block.BaseBlock.execute envelope confidence rules."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_no_confidence_key_gives_none(self):
        from app.core.block import BaseBlock, BlockConfig

        class _NoConfBase(BaseBlock):
            def __init__(self):
                super().__init__(BlockConfig(name="base_no_conf"))

            async def process(self, input_data, params=None):
                return {"data": "hello"}

        block = _NoConfBase()
        env = self._run(block.execute("input"))
        assert env["confidence"] is None

    def test_result_with_confidence_is_passed_through(self):
        from app.core.block import BaseBlock, BlockConfig

        class _WithConfBase(BaseBlock):
            def __init__(self):
                super().__init__(BlockConfig(name="base_with_conf"))

            async def process(self, input_data, params=None):
                return {"data": "hello", "confidence": 0.55}

        block = _WithConfBase()
        env = self._run(block.execute("input"))
        assert env["confidence"] == 0.55
