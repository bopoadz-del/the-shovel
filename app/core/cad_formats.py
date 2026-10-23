"""CAD formats the platform does not read.

DWG take-off is removed (owner ruling): no block converts or reads a DWG.
Take-off and BIM answer a DWG with one message and one client-error status,
so the caller knows what to do instead of seeing a conversion failure.
"""
from __future__ import annotations

import os
from typing import Any

DWG_NOT_SUPPORTED = "DWG is not supported. Export DXF or PDF from your CAD software."

#: 415 Unsupported Media Type: the request is well formed, the format is not
#: one the platform reads.
DWG_NOT_SUPPORTED_STATUS = 415


def is_dwg(path: Any) -> bool:
    """True when ``path`` names a ``.dwg`` file (by extension, any case)."""
    if not isinstance(path, str) or not path:
        return False
    return os.path.splitext(path)[1].lower() == ".dwg"


def dwg_not_supported() -> dict[str, Any]:
    """The error envelope a block returns for a DWG input.

    ``client_error_status`` tells the HTTP layer (``app/routers/execute.py``)
    to answer with that 4xx instead of a 200 carrying an error.
    """
    return {
        "status": "error",
        "error": DWG_NOT_SUPPORTED,
        "format": "dwg",
        "client_error_status": DWG_NOT_SUPPORTED_STATUS,
    }
