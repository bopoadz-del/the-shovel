"""One-shot validation of DrawingQTOBlock against 5 representative drawings.

Run: .venv/Scripts/python.exe scripts/_validate_drawing_reader.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.blocks.drawing_qto import DrawingQTOBlock

ROOT = os.environ.get("FORK_DRIVE_ROOT", "/data/example")
DRAWINGS = [
    ("TM", os.path.join(ROOT, r"Contract Docs\Contractor\ITT\02-Drawings\TM-Traffic Management\QZ-ENG-310-0000-EXC-DWG-TM-200-0000001-A.pdf")),
    ("SG", os.path.join(ROOT, r"Contract Docs\Contractor\ITT\02-Drawings\SG-Sewage\QZ-ENG-310-0000-EXC-DWG-SG-200-1001000-A.pdf")),
    ("WS", os.path.join(ROOT, r"Contract Docs\Contractor\ITT\02-Drawings\WS-Water Supply\QZ-ENG-310-0000-EXC-DWG-WS-600-0000001-C.pdf")),
    ("EL", os.path.join(ROOT, r"Contract Docs\Contractor\ITT\02-Drawings\EL-Electrical LV\QZ-ENG-310-0000-EXC-DWG-EL-600-0200068-B.pdf")),
    ("TL", os.path.join(ROOT, r"Contract Docs\Contractor\ITT\02-Drawings\TL-Telecom\QZ-ENG-310-0000-EXC-DWG-TL-600-0000002-D.pdf")),
]


async def main() -> None:
    block = DrawingQTOBlock()
    rows = []
    for disc, path in DRAWINGS:
        try:
            result = await block.process({"file_path": path}, {})
            drw = result.get("drawing") or {}
            rows.append({
                "disc": disc,
                "fname": os.path.basename(path),
                "status": result.get("status"),
                "errors": result.get("errors") or [],
                "drawing_number": drw.get("drawing_number"),
                "drawing_title": drw.get("drawing_title"),
                "discipline": drw.get("discipline"),
                "discipline_full": drw.get("discipline_full"),
                "revision": drw.get("revision"),
                "n_notes": len(drw.get("notes") or []),
                "n_dims": len(drw.get("dimensions") or []),
                "n_cross_refs": len(drw.get("cross_refs") or []),
                "cad_tags_filtered": drw.get("cad_tags_filtered_count", 0),
                "raw_chunk_preview": (result.get("text") or "")[:240].replace("\n", " | "),
                "notes_preview": (drw.get("notes") or [])[:3],
            })
        except Exception as e:
            rows.append({"disc": disc, "fname": os.path.basename(path), "exception": repr(e)[:200]})

    print()
    print("=" * 100)
    print("VALIDATION TABLE  -  5 pilot drawings, new DrawingQTOBlock")
    print("=" * 100)
    for r in rows:
        print()
        print(f"--- {r['disc']}  |  {r['fname']}  ---")
        if "exception" in r:
            print(f"  EXCEPTION: {r['exception']}")
            continue
        print(f"  status: {r['status']}   errors: {r['errors']}")
        print(f"  drawing_number: {r['drawing_number']}")
        print(f"  drawing_title:  {r['drawing_title']}")
        print(f"  discipline:     {r['discipline']} ({r['discipline_full']})    revision: {r['revision']}")
        print(f"  n_notes: {r['n_notes']}   n_dimensions: {r['n_dims']}   n_cross_refs: {r['n_cross_refs']}   cad_tags_filtered: {r['cad_tags_filtered']}")
        print(f"  notes_preview: {r['notes_preview']}")
        print(f"  raw_chunk_preview: {r['raw_chunk_preview']}")


if __name__ == "__main__":
    asyncio.run(main())
