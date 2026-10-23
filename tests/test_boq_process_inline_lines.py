"""Live tip 4ab55613 FIXTURE-d: boq_process ask1 with SYNTHETIC CSV lines in the
message must process those lines — not refuse for missing file_path and not
steal to construction_calc.

Exit evidence: tools=['construction_calc']; answer said boq_processor only
reads an uploaded file and needs a real file_path.
"""
from __future__ import annotations

import pytest


INLINE_CSV = (
    "Use boq_process on these SYNTHETIC CSV/BOQ lines in the message:\n"
    "Item,Desc,Qty,Unit,Rate,Amount\n"
    "1.1,Excavation,500,m3,45,22500\n"
    "1.2,Concrete footings,120,m3,480,57600\n"
    "1.3,Rebar,15,t,3200,48000\n"
    "1.4,Blockwork,600,m2,85,51000\n"
    "Process and give section/total values."
)


def test_inline_boq_lines_detected():
    from app.core.site_vocab import message_has_inline_boq_lines

    assert message_has_inline_boq_lines(INLINE_CSV)
    assert not message_has_inline_boq_lines(
        "Please process the uploaded priced_boq.xlsx with boq_process"
    )


def test_named_calculator_does_not_steal_inline_boq():
    from app.agents.runtime import _message_wants_named_calculator

    assert _message_wants_named_calculator(INLINE_CSV) is False


@pytest.mark.asyncio
async def test_boq_processor_accepts_inline_csv_text():
    from app.blocks.boq_processor import BOQProcessorBlock

    block = BOQProcessorBlock()
    result = await block.process(
        {"text": INLINE_CSV},
        {"action": "process"},
    )
    assert result.get("status") in {"success", "ok", "completed"}, result
    blob = str(result).lower()
    assert "excavation" in blob or "1.1" in blob
    # must not be the file_path refusal
    assert "no file_path" not in blob
    assert "requires an .xlsx" not in blob
