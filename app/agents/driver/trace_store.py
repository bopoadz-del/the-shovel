"""Where trace records (``trace/1``) are kept, and the last working sheet of
each conversation.

One JSON line per answered turn in ``$DATA_DIR/trace_records/<UTC date>.jsonl``
(on live, DATA_DIR is the shared file system), appended under a lock. The
owner and the testers download them through the admin route; nothing here
reads a record's contents to decide anything.

The last sheet of a conversation is what the next turn's sheet starts from
(its givens and the inputs already asked for carry over). It is the last
record's sheet for that conversation, held in memory and read back from the
day's file after a restart.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
#: Conversations whose last sheet is held in memory (oldest dropped first).
_MAX_HELD = 512
_LAST_SHEET: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def directory() -> Path:
    return Path(os.getenv("DATA_DIR") or "./data") / "trace_records"


def _day_file(day: str) -> Path:
    _dt.date.fromisoformat(day)  # a day that is not a date never becomes a path
    return directory() / f"{day}.jsonl"


def save(record: Dict[str, Any], conversation_id: Optional[str] = None,
         now: Optional[_dt.datetime] = None) -> Optional[str]:
    """Append ``record`` to today's file; returns its id (``<day>/<uuid>``)
    or None when it could not be written (the turn's answer is never lost
    for a record that was not kept; the failure is logged)."""
    stamp = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(_dt.timezone.utc)
    day = stamp.date().isoformat()
    record_id = f"{day}/{uuid.uuid4().hex}"
    row = dict(record, record_id=record_id, conversation=str(conversation_id or ""),
               saved_at=stamp.isoformat())
    line = json.dumps(row, ensure_ascii=False, default=str, sort_keys=True)
    try:
        with _LOCK:
            path = _day_file(day)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if conversation_id and isinstance(record.get("sheet"), dict):
                _LAST_SHEET[str(conversation_id)] = record["sheet"]
                _LAST_SHEET.move_to_end(str(conversation_id))
                while len(_LAST_SHEET) > _MAX_HELD:
                    _LAST_SHEET.popitem(last=False)
    except OSError:
        _LOG.warning("trace record not written", exc_info=True)
        return None
    return record_id


def _rows(day: str) -> Iterator[Dict[str, Any]]:
    path = _day_file(day)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                _LOG.warning("trace record line in %s is not JSON; skipped", path.name)
                continue
            if isinstance(row, dict):
                yield row


def records(day: str, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """The records of one UTC day (``YYYY-MM-DD``), in the order written;
    only one conversation's when ``conversation_id`` is given."""
    return [r for r in _rows(day) if not conversation_id or r.get("conversation") == conversation_id]


def last_sheet(conversation_id: Optional[str], now: Optional[_dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """The last sheet this conversation's turns wrote (today or yesterday,
    UTC), or None."""
    if not conversation_id:
        return None
    held = _LAST_SHEET.get(str(conversation_id))
    if held is not None:
        return held
    today = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(_dt.timezone.utc).date()
    for day in (today, today - _dt.timedelta(days=1)):
        found = None
        for row in records(day.isoformat(), str(conversation_id)):
            if isinstance(row.get("sheet"), dict):
                found = row["sheet"]
        if found is not None:
            return found
    return None


def forget_held() -> None:
    """Drop the in-memory sheets (a restart; used by tests)."""
    with _LOCK:
        _LAST_SHEET.clear()
