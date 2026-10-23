"""Construction container — qto submodule."""

from typing import Any, Dict


class ConstructionQtoMixin:
    async def drawing_qto(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to DrawingQTOBlock: DXF quantity take-off."""
        block = self._resolve_block("drawing_qto")
        if block is None:
            return {"status": "error", "error": "drawing_qto block unavailable"}
        return await block.process(input_data, params)
