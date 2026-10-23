"""Synthetic stand-ins for fixtures that used to be real client files.

Every file here is generated at test time from invented content: the project
is "Sample Works", the drawing numbers use the invented "QZ-SWK" prefix, and
no name, number or path comes from a real project. Each builder documents the
properties of the file that the code under test depends on, and returns the
values a test should expect, so the assertions are computed from the same
source as the file instead of being copied from a document nobody can see.

Builders are deterministic: the same call writes the same content.
"""
from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Drawing sheets (PyMuPDF; reportlab is not a dependency) ─────────────────

# Full long-form drawing numbers in the shape `_DWG_NUMBER_FULL` accepts:
#   <proj>-<pkg>-<nnn>-<zone>-<orig>-DWG-<disc>-<nnn>-<seq>-<rev>
DETAIL_DRAWING_NUMBER = "QZ-SWK-100-0000-ABC-DWG-TM-200-0000101-A"
KEYPLAN_DRAWING_NUMBER = "QZ-SWK-100-0000-ABC-DWG-TM-200-0000100-B"
DETAIL_TITLE = "TEMPORARY TRAFFIC DIVERSION DETAIL"
KEYPLAN_TITLE = "TRAFFIC MANAGEMENT KEY PLAN"

# Sheet size: A3 landscape in points. The title-block band is y0 > 85% of the
# height (fitz y grows downward), i.e. y0 > 715.7 here.
_PAGE_W, _PAGE_H = 1190.0, 842.0

_DETAIL_NOTES = [
    "1. ALL DIMENSIONS ARE IN METRES UNLESS OTHERWISE STATED.",
    "2. TEMPORARY BARRIERS SHALL BE INSPECTED DAILY BY THE SITE ENGINEER.",
    "3. DIVERSION SIGNAGE TO BE INSTALLED BEFORE THE LANE CLOSURE STARTS.",
    "4. PEDESTRIAN ROUTES SHALL REMAIN OPEN AND LIT AT ALL TIMES.",
    # A near-duplicate of note 4 (edit distance 1) the dedup must collapse.
    "4. PEDESTRIAN ROUTES SHALL REMAIN OPEN AND LIT AT ALL TIMEZ.",
]
# Two distinct sheet targets; the 03 callout appears twice so the cross-ref
# dedup has something to collapse.
_DETAIL_MATCH_LINES = [
    "MATCH LINE : FOR REFERENCE REFER TO SHEET NO : 03",
    "MATCH LINE : FOR REFERENCE REFER TO SHEET NO : 03",
    "MATCH LINE : FOR REFERENCE REFER TO SHEET NO : 05",
]


def _cad_tags(n: int) -> list[str]:
    """Pure CAD-tag strings (all caps + digits + hyphens, no spaces)."""
    return [f"DE{10 + i // 30}-TS-{i % 30:03d}" for i in range(n)]


def _place_tag_grid(page, tags: list[str], *, x0: float, y0: float,
                    cols: int, dx: float, dy: float, size: float = 5.0) -> None:
    # Spacing keeps neighbouring tags > 30pt apart so each one is its own
    # cluster (a merged "TAG TAG" line has a space and is not a CAD tag).
    import fitz

    for i, tag in enumerate(tags):
        r, c = divmod(i, cols)
        page.insert_text(fitz.Point(x0 + c * dx, y0 + r * dy), tag, fontsize=size)


def _title_block(page, *, number: str, title: str, sheet: str,
                 extra_callout: str | None = None) -> None:
    """Bottom-band title block with >= 5 lines (the richness gate)."""
    import fitz

    x = 700.0
    page.insert_text(fitz.Point(x, 735), "PROJECT: SAMPLE WORKS ROAD UPGRADE", fontsize=7)
    if extra_callout:
        # Larger than the title on purpose: the title picker must reject a
        # match-line callout even when it is the biggest text in the block.
        page.insert_text(fitz.Point(60, 760), extra_callout, fontsize=16)
    page.insert_text(fitz.Point(x, 752), title, fontsize=13)
    page.insert_text(fitz.Point(x, 770), "SCALE 1:500", fontsize=7)
    page.insert_text(fitz.Point(x, 784), "DATE 12/03/24", fontsize=7)
    page.insert_text(fitz.Point(x, 798), f"SHEET {sheet}", fontsize=7)
    page.insert_text(fitz.Point(x, 812), f"DRAWING NO: {number}", fontsize=7)


def build_drawing_detail_pdf(path: Path) -> dict[str, Any]:
    """A traffic-management *detail* sheet.

    Properties the drawing reader relies on:
    - a bottom-band title block carrying a full long-form drawing number
      (revision A; no DISCIPLINE label), a scale, a date and a sheet number;
    - general notes in the drawing zone (> 10 words, one near-duplicate);
    - repeated MATCH LINE callouts (two distinct sheet targets);
    - a block of CAD tags that must be filtered out of the chunk;
    - well over 100 text spans on the page.
    """
    import fitz

    tags = _cad_tags(90)
    doc = fitz.open()
    page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
    page.insert_text(fitz.Point(60, 60), "GENERAL NOTES:", fontsize=9)
    for i, note in enumerate(_DETAIL_NOTES):
        page.insert_text(fitz.Point(60, 80 + i * 14), note, fontsize=7)
    for i, callout in enumerate(_DETAIL_MATCH_LINES):
        page.insert_text(fitz.Point(700, 80 + i * 40), callout + " ", fontsize=6)
    _place_tag_grid(page, tags, x0=60, y0=220, cols=10, dx=80, dy=18)
    # A handful of chainage labels: numeric text that is not a note.
    for i in range(12):
        page.insert_text(fitz.Point(60 + i * 80, 420), f"0+{100 + i * 25}.000", fontsize=5)
    # Road edge geometry so the sheet is not text-only.
    page.draw_line(fitz.Point(60, 450), fitz.Point(1100, 450))
    page.draw_line(fitz.Point(60, 520), fitz.Point(1100, 520))
    page.draw_rect(fitz.Rect(400, 470, 520, 500))
    _title_block(page, number=DETAIL_DRAWING_NUMBER, title=DETAIL_TITLE,
                 sheet="2 OF 6", extra_callout=_DETAIL_MATCH_LINES[-1])
    doc.save(str(path))
    doc.close()
    return {
        "drawing_number": DETAIL_DRAWING_NUMBER,
        "drawing_title": DETAIL_TITLE,
        "revision": "A",
        "scale": "1:500",
        "match_line_targets": {"03", "05"},
        "cad_tags": len(tags),
    }


# Key-plan legend: count-style quantities ("<n> NO <ITEM>") the measurement
# extractor reads, and three material grades for the specification pass.
_KEYPLAN_LEGEND = [
    "12 NO SIGN POSTS",
    "4 NO WATER FILLED BARRIERS",
    "6 Traffic Cones",
    "3 NO VARIABLE MESSAGE SIGNS",
    "8 NO LIGHTING COLUMNS",
    "2 NO PEDESTRIAN CROSSINGS",
    "5 NO TEMPORARY KERB RAMPS",
    "KERBS IN CONCRETE C40",
    "BARRIER POSTS STEEL S275",
    "BEDDING MORTAR M20",
]
KEYPLAN_COUNT_ITEMS = 7
KEYPLAN_GRADES = ("C40", "S275", "M20")


def build_drawing_keyplan_pdf(path: Path, *, pages: int = 2) -> dict[str, Any]:
    """A traffic-management *key plan*: the CAD-tag-dense sheet.

    Properties the readers rely on:
    - a legend with count-style quantities and material grades near the start
      of the text layer (the measurement pass reads the first 2000 chars);
    - no area/volume/WxH dimension strings, so every measurement is a counted
      item and carries an ``item`` field;
    - hundreds of CAD tags and coordinate pairs to be filtered;
    - a title block with a full drawing number; ``pages`` sheets.
    """
    import fitz

    doc = fitz.open()
    tags_total = 0
    for p in range(pages):
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        if p == 0:
            page.insert_text(fitz.Point(60, 50), "LEGEND", fontsize=9)
            for i, line in enumerate(_KEYPLAN_LEGEND):
                page.insert_text(fitz.Point(60, 66 + i * 12), line, fontsize=7)
        tags = _cad_tags(140)
        tags_total += len(tags)
        _place_tag_grid(page, tags, x0=320, y0=60, cols=10, dx=78, dy=16)
        for i in range(10):
            page.insert_text(fitz.Point(60 + i * 100, 300),
                             f"{2100.5 + i:.1f},{1450.25 + i:.2f}", fontsize=4)
        for k in range(6):
            page.draw_line(fitz.Point(60, 320 + k * 40), fitz.Point(1100, 330 + k * 40))
        _title_block(page, number=KEYPLAN_DRAWING_NUMBER, title=KEYPLAN_TITLE,
                     sheet=f"{p + 1} OF {pages}")
    doc.save(str(path))
    doc.close()
    return {
        "drawing_number": KEYPLAN_DRAWING_NUMBER,
        "pages": pages,
        "cad_tags": tags_total,
        "count_items": KEYPLAN_COUNT_ITEMS,
        "grades": KEYPLAN_GRADES,
    }


# ── Primavera P6 baseline programme (.xer text) ─────────────────────────────

_XER_PROJECT = "SAMPLEWORKS"
_XER_START = datetime(2025, 3, 3, 8, 0, tzinfo=timezone.utc)
_XER_DATA_DATE = datetime(2025, 2, 24, 8, 0, tzinfo=timezone.utc)
_XER_CRITICAL_DURATIONS = [5, 10, 15, 10, 20, 15, 10, 25, 20, 15, 10, 5]  # days
_XER_SIDE_BRANCHES = 4          # branches hanging off the critical chain
_XER_SIDE_PER_BRANCH = 9        # activities in each branch


def _xer_dt(d: datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M")


def build_baseline_xer(path: Path) -> dict[str, Any]:
    """A Primavera P6 baseline programme in XER (TABLE/FIELDS/ROW) format.

    One critical chain of 12 zero-float activities plus four non-critical
    branches with positive float, linked FS in TASKPRED, with a PROJECT data
    date (last_recalc_date), a calendar, WBS, resources and assignments --
    the tables a real P6 export carries for the schedule family. Dates are
    calendar-day arithmetic, so the expected duration is simply last critical
    finish minus first critical start.
    """
    tasks: list[dict[str, Any]] = []
    preds: list[tuple] = []
    tid = 5000
    cursor = _XER_START
    critical_ids: list[int] = []
    for n, days in enumerate(_XER_CRITICAL_DURATIONS, 1):
        tid += 1
        start, finish = cursor, cursor + timedelta(days=days)
        tasks.append({
            "task_id": tid, "code": f"SW-C{n * 10:04d}",
            "name": f"Main works stage {n}", "wbs": 101, "days": days,
            "start": start, "finish": finish, "float_h": 0,
            "type": "TT_Task",
        })
        if critical_ids:
            preds.append((tid, critical_ids[-1]))
        critical_ids.append(tid)
        cursor = finish
    critical_finish = cursor
    # Milestones bracket the chain: zero duration, zero float.
    for code, name, when, ttype in (
        ("SW-M0000", "Start of works milestone", _XER_START, "TT_StartMile"),
        ("SW-M9999", "Practical completion milestone", critical_finish, "TT_FinMile"),
    ):
        tid += 1
        tasks.append({"task_id": tid, "code": code, "name": name, "wbs": 100,
                      "days": 0, "start": when, "finish": when, "float_h": 0,
                      "type": ttype})
    milestone_ids = [tid - 1, tid]
    preds.append((critical_ids[0], milestone_ids[0]))
    preds.append((milestone_ids[1], critical_ids[-1]))

    for b in range(_XER_SIDE_BRANCHES):
        anchor = tasks[2 + b * 2]
        branch_cursor = anchor["start"]
        prev = anchor["task_id"]
        for k in range(_XER_SIDE_PER_BRANCH):
            tid += 1
            days = 2 + (k % 3)
            start, finish = branch_cursor, branch_cursor + timedelta(days=days)
            tasks.append({
                "task_id": tid, "code": f"SW-B{b + 1}{k + 1:03d}",
                "name": f"Utility diversion {b + 1} part {k + 1}",
                "wbs": 102 + b, "days": days, "start": start, "finish": finish,
                "float_h": 8 * (10 + 5 * b + k), "type": "TT_Task",
            })
            preds.append((tid, prev))
            prev = tid
            branch_cursor = finish

    lines: list[str] = [
        "ERMHDR\t19.12\t2025-02-24\tProject\tadmin\tSample Planner\tdbxDatabaseNoName\tProject Management\tGBP",
        "%T\tCURRTYPE",
        "%F\tcurr_id\tdecimal_digit_cnt\tcurr_symbol\tcurr_short_name\tcurr_type",
        "%R\t1\t2\t£\tGBP\tPound Sterling",
        "%T\tPROJECT",
        "%F\tproj_id\tproj_short_name\tclndr_id\tlast_recalc_date\tplan_start_date\tplan_end_date\tscd_end_date",
        (
            f"%R\t900\t{_XER_PROJECT}\t1\t{_xer_dt(_XER_DATA_DATE)}\t{_xer_dt(_XER_START)}"
            f"\t{_xer_dt(critical_finish)}\t{_xer_dt(critical_finish)}"
        ),
        "%T\tCALENDAR",
        "%F\tclndr_id\tclndr_name\tday_hr_cnt\twk_hr_cnt",
        "%R\t1\tSample 5-day\t8\t40",
        "%T\tPROJWBS",
        "%F\twbs_id\tproj_id\tparent_wbs_id\twbs_short_name\twbs_name\tseq_num",
        f"%R\t100\t900\t\t{_XER_PROJECT}\tSample Works\t0",
        "%R\t101\t900\t100\tMAIN\tMain works\t1",
    ]
    for b in range(_XER_SIDE_BRANCHES):
        lines.append(f"%R\t{102 + b}\t900\t100\tUD{b + 1}\tUtility diversion {b + 1}\t{2 + b}")
    lines += [
        "%T\tRSRC",
        "%F\trsrc_id\trsrc_name\trsrc_short_name\trsrc_type\tunit_id",
        "%R\t1\tGeneral labour\tLAB\tRT_Labor\t",
        "%R\t2\tExcavator 20t\tEXC\tRT_Equip\t",
        "%T\tTASK",
        (
            "%F\ttask_id\tproj_id\twbs_id\tclndr_id\ttask_type\tstatus_code\ttask_code\ttask_name"
            "\ttotal_float_hr_cnt\tremain_drtn_hr_cnt\ttarget_drtn_hr_cnt\tphys_complete_pct"
            "\tearly_start_date\tearly_end_date\ttarget_start_date\ttarget_end_date"
        ),
    ]
    for t in tasks:
        hrs = t["days"] * 8
        lines.append(
            f"%R\t{t['task_id']}\t900\t{t['wbs']}\t1\t{t['type']}\tTK_NotStart\t{t['code']}"
            f"\t{t['name']}\t{t['float_h']}\t{hrs}\t{hrs}\t0"
            f"\t{_xer_dt(t['start'])}\t{_xer_dt(t['finish'])}"
            f"\t{_xer_dt(t['start'])}\t{_xer_dt(t['finish'])}"
        )
    lines += [
        "%T\tTASKPRED",
        "%F\ttask_pred_id\ttask_id\tpred_task_id\tproj_id\tpred_proj_id\tpred_type\tlag_hr_cnt",
    ]
    for i, (succ, pred) in enumerate(preds, 1):
        lines.append(f"%R\t{7000 + i}\t{succ}\t{pred}\t900\t900\tPR_FS\t0")
    lines += [
        "%T\tTASKRSRC",
        "%F\ttaskrsrc_id\ttask_id\tproj_id\trsrc_id\ttarget_qty\tremain_qty\ttarget_start_date\ttarget_end_date",
    ]
    for i, t in enumerate(tasks[:len(_XER_CRITICAL_DURATIONS)], 1):
        lines.append(
            f"%R\t{8000 + i}\t{t['task_id']}\t900\t1\t{t['days'] * 16}\t{t['days'] * 16}"
            f"\t{_xer_dt(t['start'])}\t{_xer_dt(t['finish'])}"
        )
    lines.append("%E")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="cp1252")

    n_critical = len(_XER_CRITICAL_DURATIONS) + len(milestone_ids)
    return {
        "total_activities": len(tasks),
        "critical_activities": n_critical,
        "project_duration": (critical_finish - _XER_START).days,
        "data_date": _XER_DATA_DATE.strftime("%Y-%m-%d"),
    }


# ── Ingest-shard sample folder ──────────────────────────────────────────────

INGEST_SAMPLE_FOLDER = "sample-works-001"
# 40 docs cycling through 8 extensions + 20 drawings; .gdoc is unsupported.
_DOC_EXTS = [".docx", ".xlsx", ".txt", ".png", ".kmz", ".gdoc", ".csv", ".pdf"]


def build_ingest_shard_folder(root: Path, *, docs: int = 40, drawings: int = 20) -> dict[str, Any]:
    """A small Drive-shaped folder tree for the sharded ingest dry run.

    ``root/sample-works-001/docs/file_<n><ext>`` and ``.../drawings/dwg_<n>.pdf``,
    each a few bytes of placeholder text.
    """
    base = root / INGEST_SAMPLE_FOLDER
    (base / "docs").mkdir(parents=True, exist_ok=True)
    (base / "drawings").mkdir(parents=True, exist_ok=True)
    unsupported = 0
    for n in range(1, docs + 1):
        ext = _DOC_EXTS[(n - 1) % len(_DOC_EXTS)]
        unsupported += ext == ".gdoc"
        (base / "docs" / f"file_{n}{ext}").write_text(f"sample {n}\n", encoding="utf-8")
    for n in range(1, drawings + 1):
        (base / "drawings" / f"dwg_{n}.pdf").write_text("sample\n", encoding="utf-8")
    total = docs + drawings
    return {"folder": INGEST_SAMPLE_FOLDER, "total": total,
            "unsupported": unsupported, "supported": total - unsupported}


# ── Legacy binary PowerPoint 97-2003 (.ppt) ─────────────────────────────────

_CFB_FREE, _CFB_END, _CFB_FAT, _CFB_NOSTREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF
_SECTOR = 512
_MINI_CUTOFF = 4096


def _ppt_record(rec_type: int, body: bytes, *, container: bool = False, instance: int = 0) -> bytes:
    ver_inst = (instance << 4) | (0x000F if container else 0x0000)
    return struct.pack("<HHI", ver_inst, rec_type, len(body)) + body


def _ppt_document_stream(title: str, body_text: str) -> bytes:
    """A minimal PowerPoint Document record tree.

    DocumentContainer
      MainMasterContainer -> "Click to edit ..." prompt (must be skipped)
      SlideListWithTextContainer
        TextHeaderAtom(title) + TextCharsAtom (UTF-16LE)
        TextHeaderAtom(body)  + TextBytesAtom (8-bit)
    """
    master = _ppt_record(0x03F8, _ppt_record(0x0FA0, "Click to edit Master title style".encode("utf-16-le")),
                         container=True)
    slide_text = (
        _ppt_record(0x0F9F, struct.pack("<I", 0))
        + _ppt_record(0x0FA0, title.encode("utf-16-le"))
        + _ppt_record(0x0F9F, struct.pack("<I", 1))
        + _ppt_record(0x0FA8, body_text.encode("latin-1"))
    )
    slwt = _ppt_record(0x0FF0, slide_text, container=True, instance=0)
    return _ppt_record(0x03E8, master + slwt, container=True, instance=1)


def _cfb_dir_entry(name: str, etype: int, *, left=_CFB_NOSTREAM, right=_CFB_NOSTREAM,
                   child=_CFB_NOSTREAM, start=_CFB_END, size=0) -> bytes:
    raw = (name + "\0").encode("utf-16-le") if name else b""
    return (
        raw.ljust(64, b"\0")
        + struct.pack("<HBB", len(raw), etype, 1)          # length, type, black
        + struct.pack("<III", left, right, child)
        + b"\0" * 16 + struct.pack("<I", 0)                 # clsid, state bits
        + b"\0" * 16                                        # create / modify times
        + struct.pack("<IQ", start, size)
    )


def _write_cfb(path: Path, streams: dict[str, bytes]) -> None:
    """Write a version-3 compound file with every stream in the regular FAT.

    Streams are padded to the mini-stream cutoff so no mini FAT is needed;
    readers use the declared size, and the padding is zero bytes.
    """
    names = list(streams)  # exactly two: sibling tree is child + left
    assert len(names) == 2
    fat: list[int] = []
    data = b""
    starts, sizes = {}, {}
    for name in names:
        blob = streams[name].ljust(_MINI_CUTOFF, b"\0")
        n = -(-len(blob) // _SECTOR)
        starts[name], sizes[name] = len(fat), len(blob)
        fat += [len(fat) + i + 1 for i in range(n - 1)] + [_CFB_END]
        data += blob.ljust(n * _SECTOR, b"\0")
    dir_sector = len(fat)
    fat.append(_CFB_END)
    fat_sector = len(fat)
    fat.append(_CFB_FAT)
    fat += [_CFB_FREE] * (_SECTOR // 4 - len(fat))

    # Sibling order: shorter name first, so "Current User" < "PowerPoint Document".
    small, big = sorted(names, key=lambda s: (len(s), s.upper()))
    directory = (
        _cfb_dir_entry("Root Entry", 5, child=2)
        + _cfb_dir_entry(small, 2, start=starts[small], size=sizes[small])
        + _cfb_dir_entry(big, 2, left=1, start=starts[big], size=sizes[big])
        + _cfb_dir_entry("", 0)
    )
    header = (
        bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 16
        + struct.pack("<HHHHH", 0x003E, 0x0003, 0xFFFE, 9, 6) + b"\0" * 6
        + struct.pack("<IIIIIIIII", 0, 1, dir_sector, 0, _MINI_CUTOFF, _CFB_END, 0, _CFB_END, 0)
        + struct.pack("<I", fat_sector) + struct.pack("<I", _CFB_FREE) * 108
    )
    assert len(header) == _SECTOR
    path.write_bytes(header + data + directory + struct.pack(f"<{len(fat)}I", *fat))


def build_legacy_ppt(path: Path, *, title: str, body: str) -> Path:
    """A PowerPoint 97-2003 file: OLE compound document with the
    "PowerPoint Document" record stream and a "Current User" stream."""
    current_user = struct.pack("<IIIHHBBH", 20, 0xE391C05F, 0, 8, 0x03F4, 3, 0, 0) + b"sample\0\0"
    _write_cfb(path, {
        "Current User": _ppt_record(0x0FF6, current_user),
        "PowerPoint Document": _ppt_document_stream(title, body),
    })
    return path
