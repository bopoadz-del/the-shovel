"""scan_hardwiring --gaps: the forms the six-form gate cannot see.

Each test plants one form in a temp product tree and expects ``gap_findings``
to name it; each also plants a generic look-alike that must stay clean.
The gaps pass is opt-in (``--gaps``): the default gate, its baseline (0 entries)
and its allowlist are unchanged and are checked here too.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "scan_hardwiring", ROOT / "scripts" / "scan_hardwiring.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    pkg = tmp_path / "app" / "core"
    pkg.mkdir(parents=True)
    (pkg / name).write_text(body, encoding="utf-8")
    return tmp_path


def _kinds(sh, root: Path) -> set[str]:
    return {f.split("::")[1] for f in sh.gap_findings(root)}


def test_probe_id_near_live_wording_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "# Live X6: the hat stopped early\nx = 1\n")
    assert "app/core/mod.py::gap_probe::X6" in sh.gap_findings(root)



def test_document_code_prefix_collection_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(
        tmp_path,
        "ROUTES = [{'words': ['notice', 'AB-', 'directive', 'CD-', 'EF-']}]\n",
    )
    assert "gap_ref" in _kinds(sh, root)


def test_tuple_of_short_reference_literals_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "REFS = ('AB-1201', 'AB-1202', 'CD-77', 'CD-78')\n")
    assert "gap_ref" in _kinds(sh, root)


def test_plain_word_tuple_is_clean(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "WORDS = ('alpha', 'beta', 'gamma', 'delta')\n")
    assert sh.gap_findings(root) == []


def test_non_rag_kill_switch_knob_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(
        tmp_path,
        "import os\nON = os.getenv('SOME_FEATURE_FETCH', '1') != '0'\n",
    )
    assert "app/core/mod.py::gap_knob::SOME_FEATURE_FETCH" in sh.gap_findings(root)


def test_infrastructure_env_var_is_clean(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "import os\nURL = os.getenv('DATABASE_URL', '')\n")
    assert sh.gap_findings(root) == []



def test_module_level_regex_with_filename_literals_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "import re\n_NOISE = r'^(~\\$|alpha-menu|BetaCorp_Application)'\n")
    assert "gap_regex" in _kinds(sh, root)



def test_plain_regex_is_clean(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "import re\n_WORD = re.compile(r'[a-z0-9]+')\n")
    assert sh.gap_findings(root) == []


def test_gaps_flag_exits_nonzero_on_a_planted_tree_and_default_gate_is_unchanged():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_hardwiring.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert out.returncode == 0, out.stdout  # default gate: still 0 new
    base = json.loads((ROOT / "scripts" / "hardwiring_baseline.json").read_text())
    assert base["symbols"] == {} and base["probe_counts"] == {}


def test_gaps_pass_adds_no_allowlist_entry():
    sh = _load()
    assert len(sh.PROBE_LINE_ALLOWLIST) == 7  # the count on main; may only shrink


def test_knob_name_held_in_a_constant_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "FLAG_ENV = 'SOME_FEATURE_FETCH'\n")
    assert "app/core/mod.py::gap_knob::SOME_FEATURE_FETCH" in sh.gap_findings(root)


def test_single_document_code_prefix_among_words_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "KEYS = ('request', 'RFX-', 'notice')\n")
    assert "gap_ref" in _kinds(sh, root)


# ── structural refinements: each false-positive class stays silent ───────────

import pytest


@pytest.mark.parametrize("line", [
    "# set C30 concrete for the slab",             # 'set' as a verb, id is a grade
    "# numbers live in the field; use M25 mix",     # 'live' far from the id
    "# measured live on an A1 sheet",               # id is a paper size
    "# K2 rejects forced tool_choice across three live rounds",  # id before the word
    "# Primavera P6 exports XER",                   # product name
    "# L2B live project record",                    # layer name, id before 'live'
    "# the A4 sheet, R2 bucket, P95 latency",
])
def test_probe_shape_without_a_citing_word_before_it_is_silent(tmp_path: Path, line: str):
    sh = _load()
    assert sh.gap_findings(_tree(tmp_path, line + "\nx = 1\n")) == []


@pytest.mark.parametrize("line,ids", [
    ("# Live M15: the hat ended early", {"M15"}),
    ("# Live M1/M2/M6: remaining already drafted", {"M1", "M2", "M6"}),
    ("# Leftover-hat L4: validation", {"L4"}),
    ("# WAVE 2 B4: priced row", {"B4"}),
    ("# Live UI pack E4: aliased", {"E4"}),
    ("# Live find 2026-08-15 (F22): a request", {"F22"}),
    ("# (unseen Set 3 B3: page)", {"B3"}),
])
def test_probe_citations_still_fire(tmp_path: Path, line: str, ids: set):
    sh = _load()
    got = {f.split("::")[2] for f in sh.gap_findings(_tree(tmp_path, line + "\nx = 1\n"))}
    assert got == ids


@pytest.mark.parametrize("body", [
    "tags = ['infrastructure', 'cache', 'hospital']\n",                       # metadata tags
    "SCHEMA = {'enum': ['data_center', 'hospital', 'tower', 'school']}\n",    # declared enum
    "CATALOG = ['tower crane', 'warehouse forklift', 'shoring tower']\n",      # equipment
    "PHRASES = ('quantity takeoff', 'infrastructure drawings', 'floor area')\n",  # one noun, phrase
])
def test_project_noun_false_positive_classes_are_silent(tmp_path: Path, body: str):
    sh = _load()
    assert sh.gap_findings(_tree(tmp_path, body)) == []


def test_two_whole_noun_elements_still_fire(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "KINDS = ('hospital', 'school', 'tower', 'misc')\n")
    assert "gap_noun" in _kinds(sh, root)


@pytest.mark.parametrize("body", [
    "GRADES = ['C50', 'C60', 'S460', 'S500']\n",                      # material classes
])
def test_single_letter_grade_lists_are_silent(tmp_path: Path, body: str):
    sh = _load()
    assert sh.gap_findings(_tree(tmp_path, body)) == []


@pytest.mark.parametrize("body", [
    "import re\n_CASH = re.compile(r'cash[_ -]?flow|s-curve|spend curve')\n",
    "import re\n_LEAK = re.compile(r'<(?:function_calls|invoke)>')\n",
    "import re\n_CAUSE = re.compile(r'contractor-caused|our delay')\n",
    "MODEL = 'vendor/model-base-8M'\n",
    "import re\n_GAP = r'[^0-9%]{0,30}?'\n",
    "NOTE = ('Rates are typically 30-50% higher than standard rates for this kind of work '\n        'in remote areas')\n",
    "RATES = {'concrete': {'unit': 'm3', 'base': 150.0, 'note': 'about 30 MPa'}}\n",
    "LIMITS = ('limit 70 mm', 'cover 25 mm', 'class')\n",                 # never used for matching
])
def test_regex_and_term_false_positive_classes_are_silent(tmp_path: Path, body: str):
    sh = _load()
    assert sh.gap_findings(_tree(tmp_path, body)) == []


def test_filter_regex_named_for_file_names_still_fires(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "_NOISE_DEFAULT = r'^(~\\$|alpha-menu|BetaCorp_Application)'\n")
    assert "app/core/mod.py::gap_regex::_NOISE_DEFAULT" in sh.gap_findings(root)


def test_number_and_unit_term_table_used_for_matching_still_fires(tmp_path: Path):
    sh = _load()
    root = _tree(
        tmp_path,
        "LIMITS = ('limit 70 mm', 'cover 25 mm', 'class')\n"
        "def hit(t):\n    return any(x in t for x in LIMITS) or 'a' in LIMITS\n",
    )
    assert "app/core/mod.py::gap_regex::LIMITS" in sh.gap_findings(root)


def test_number_and_unit_compiled_regex_used_for_matching_still_fires(tmp_path: Path):
    sh = _load()
    root = _tree(
        tmp_path,
        "import re\n_LIM = re.compile(r'limit 70 mm|cover 25 mm')\n"
        "def hit(t):\n    return _LIM.search(t)\n",
    )
    assert "app/core/mod.py::gap_regex::_LIM" in sh.gap_findings(root)


# ── the command line: main() is really invoked ────────────────────────────────

def _run(args, root: Path):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_hardwiring.py"), *args, "--root", str(root)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )


def test_gaps_subprocess_exits_1_with_findings_and_0_with_report(tmp_path: Path):
    root = _tree(tmp_path, "# Live M15: planted\nx = 1\n")
    blocking = _run(["--gaps"], root)
    assert blocking.returncode == 1, blocking.stdout
    assert "app/core/mod.py::gap_probe::M15" in blocking.stdout
    report = _run(["--gaps", "--report"], root)
    assert report.returncode == 0, report.stdout
    assert "app/core/mod.py::gap_probe::M15" in report.stdout
    assert "GAPS: 1 finding(s)" in report.stdout


def test_gaps_subprocess_on_a_clean_tree_exits_0(tmp_path: Path):
    root = _tree(tmp_path, "x = 1\n")
    out = _run(["--gaps"], root)
    assert out.returncode == 0 and "GAPS: 0 finding(s)" in out.stdout


def test_main_is_invoked_in_process_with_gaps_and_report(tmp_path: Path, monkeypatch, capsys):
    sh = _load()
    root = _tree(tmp_path, "# Live M15: planted\nx = 1\n")
    monkeypatch.setattr(sys, "argv", ["scan_hardwiring.py", "--gaps", "--report", "--root", str(root)])
    assert sh.main() == 0
    assert "gap_probe::M15" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["scan_hardwiring.py", "--gaps", "--root", str(root)])
    assert sh.main() == 1


# ── repair 2: restored fire tests (same names as the first commit, true inputs) ──
#
# REVERSED-WITH-EVIDENCE NOTE. Two inputs of the first commit's tests were reversed
# by the refinement commit without a note. They are restored as fire tests below on
# inputs that are true positives; the old inputs are pinned as SILENT, with the
# false positive and the ruling that made them silent:
#   * `HINTS = ('floor plan', 'hospital drawings', 'sheet')` is the shape of
#     app/agents/runtime.py:882 (`_DRAWING_QTO_PHRASES`: phrases that merely contain a
#     noun). G2 ruled that finding a false positive (PR #7 re-grade at 3b25fff).
#   * `LIMITS = ('limit 70 °C', 'cover 25 mm', 'class C40')` that nothing matches
#     against fired on any number+unit (G2 ruling: "gap_regex firing on any number+unit").


def test_project_noun_keyword_tuple_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "HINTS = ('hospital', 'school', 'warehouse', 'sheet')\n")
    assert "app/core/mod.py::gap_noun::line1" in sh.gap_findings(root)


def test_project_noun_phrase_tuple_is_silent_g2_ruling(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "HINTS = ('floor plan', 'hospital drawings', 'sheet')\n")
    assert sh.gap_findings(root) == []


def test_module_level_term_table_with_number_and_unit_is_flagged(tmp_path: Path):
    sh = _load()
    root = _tree(
        tmp_path,
        "LIMITS = ('limit 70 mm', 'cover 25 mm', 'class')\n"
        "def hit(text):\n    return any(t in text for t in LIMITS)\n",
    )
    assert "app/core/mod.py::gap_regex::LIMITS" in sh.gap_findings(root)


def test_number_and_unit_table_never_used_for_matching_is_silent_g2_ruling(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "LIMITS = ('limit 70 \u00b0C', 'cover 25 mm', 'class C40')\n")
    assert sh.gap_findings(root) == []


def test_probe_id_without_probe_wording_is_clean(tmp_path: Path):
    sh = _load()
    root = _tree(tmp_path, "# the A4 sheet, R2 bucket, P95 latency\nx = 1\n")
    assert sh.gap_findings(root) == []


# ── repair 2: M1/M3/M4 reached (trailing id/probe-word window, 'question <id>') ──

@pytest.mark.parametrize("line,ids", [
    ("# THE INCIDENT (gate battery 2026-08-31, question F2). The WBS", {"F2"}),   # M1
    ("# question F2: a WBS was composed", {"F2"}),
    ("# F26 (live OOM 2026-08-15): dpi alone does not bound memory", {"F26"}),    # M3
    ("# W4 planted-RFP probe with a minimal config", {"W4"}),                    # M4
    ("# F7 probe: one question", {"F7"}),
])
def test_trailing_window_and_question_cue_reach_the_citation(tmp_path: Path, line: str, ids: set):
    sh = _load()
    got = {f.split("::")[2] for f in sh.gap_findings(_tree(tmp_path, line + "\nx = 1\n"))}
    assert got == ids


@pytest.mark.parametrize("line", [
    "# Live 2026-10-07: one question issued BM25 eight times.",   # BM25 is an algorithm
    '"""W6 \u2014 live activation of the construction learning loop.',  # W6 is a feature id
    "# K2 rejects forced tool_choice, and across three live rounds the",
    "# Primavera P6 exports XER in a way that is not a probe",     # product name; 'probe' too far
])
def test_bm25_w6_k2_class_stays_silent_with_the_new_cues(tmp_path: Path, line: str):
    sh = _load()
    assert sh.gap_findings(_tree(tmp_path, line + "\nx = 1\n")) == []


@pytest.mark.xfail(
    strict=True,
    reason="M2 (F19) stays out of reach: '(F19, phase-3 campaign). Two live gates' has no "
    "probe word adjacent to the id; widening the window re-fires the K2/W6 class",
)
def test_m2_f19_is_not_reached_known_loss(tmp_path: Path):
    sh = _load()
    line = "# Empty-router dispatch enforcement (F19, phase-3 campaign). Two live gates"
    got = {f.split("::")[2] for f in sh.gap_findings(_tree(tmp_path, line + "\nx = 1\n"))}
    assert got == {"F19"}
