"""The Q-1 scorer's real output read by the trace page's score reader:
one invented run through both scripts, no live service."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


scorer = _module("answer_source_score")
tp = _module("trace_page")

T0 = 1_800_000_000.0


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize("keep_key,joined", [(True, True), (False, False)])
def test_scorer_output_joins_onto_its_row_on_the_trace_page(tmp_path, capsys, keep_key, joined):
    # one invented tester row: repeat 2 of row Z9, an answer citing a
    # document that was not handed and stating a number no source holds
    row = {"id": "Z9", "rep": 2, "time_utc": _iso(T0 + 30), "timing": {"t_end": 30},
           "answer": "The depth is 417 mm [source: Invented Note.pdf].", "handed": []}
    rows_file = tmp_path / "Z9_v2_x.json"
    rows_file.write_text(json.dumps(row))
    assert scorer.main([str(rows_file)]) == 0
    out = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert (out[0]["row"], out[0]["rep"]) == ("Z9", 2)
    if not keep_key:  # the first output's shape: 'id' only, no 'row' / 'rep'
        out = [{k: v for k, v in rec.items() if k not in ("row", "rep")} for rec in out]
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps(r) for r in out) + "\n")

    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"events": []}))
    page_file = tmp_path / "page.html"
    assert tp.main(["--events", str(ev), "--rows", str(tmp_path / "*_v2_*.json"),
                    "--out", str(page_file), "--scores", str(scores)]) == 0
    page = page_file.read_text(encoding="utf-8")
    assert ("<th>citations_not_handed</th><td>1</td>" in page) == joined
    assert "Invented Note" not in page and "417" not in page
