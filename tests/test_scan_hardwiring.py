"""scan_hardwiring: the NO HARDWIRING gate has teeth, and its baseline is honest.

Two things, mirroring how scan_exception_pass is tested:
  * a planted fixture in a temp tree is FLAGGED -- all four forms;
  * the real baseline is a ceiling that may only shrink and refuses stale keys,
    so the gate can never be "passed" by growing the baseline.
"""
from __future__ import annotations

import importlib.util
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


def test_planted_hardwiring_is_flagged(tmp_path: Path):
    """Each of the four forms, planted in product code with no baseline, is NEW."""
    sh = _load()
    pkg = tmp_path / "app" / "core"
    pkg.mkdir(parents=True)
    (pkg / "retr.py").write_text(
        "import os\n"
        "_LUX_NEEDLES = ['concrete placement lux']\n"
        "FLAG = os.getenv('RAG_LUX_TABLE_RESCUE')\n"
        "def _rescue_lux_table(chunks):\n"
        "    # tuned for R18\n"
        "    return chunks\n",
        encoding="utf-8",
    )
    findings = sh.scan(tmp_path)
    joined = "\n".join(findings)
    assert "::rescue::_rescue_lux_table" in joined
    assert "::needles::_LUX_NEEDLES" in joined
    assert "::knob::RAG_LUX_TABLE_RESCUE" in joined
    assert "::probe::R18" in joined
    assert all(f.startswith("NEW ") for f in findings)


def test_clean_product_code_passes(tmp_path: Path):
    """A mechanism fix with no per-case form produces no findings."""
    sh = _load()
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "ok.py").write_text(
        "def rank_table_chunks(chunks, k):\n"
        "    return sorted(chunks, key=lambda c: c.score)[:k]\n",
        encoding="utf-8",
    )
    assert sh.scan(tmp_path) == []


def test_code_tables_and_name_lists_in_blocks_are_flagged(tmp_path: Path):
    """Forms 5 and 6: a code -> name table and a place-name list in app/blocks/."""
    sh = _load()
    pkg = tmp_path / "app" / "blocks"
    pkg.mkdir(parents=True)
    (pkg / "reader.py").write_text(
        "DISCIPLINE_NAMES = {'QX': 'Quarry Works', 'VB': 'Vent Ducts',\n"
        "                    'ZL': 'Zinc Lining', 'KR4': 'Kerb Runs'}\n"
        "EXCLUDED_PLACE_NAMES = {'FENWICK REACH', 'KESTREL QUAY'}\n"
        "def lookup(code):\n"
        "    local = {'AA': 'One', 'BB': 'Two', 'CC': 'Three', 'DD': 'Four'}\n"
        "    return local.get(code)\n",
        encoding="utf-8",
    )
    findings = sh.scan(tmp_path)
    joined = "\n".join(findings)
    assert "app/blocks/reader.py::code_table::DISCIPLINE_NAMES" in joined
    assert "app/blocks/reader.py::code_table::local" in joined
    assert "app/blocks/reader.py::name_list::EXCLUDED_PLACE_NAMES" in joined


def test_block_table_forms_ignore_small_dicts_and_non_code_keys(tmp_path: Path):
    """A switch of three entries, a units map keyed by numbers, and a map
    keyed by words are not code tables; outside app/blocks/ forms 5-6 do not
    apply."""
    sh = _load()
    blocks = tmp_path / "app" / "blocks"
    blocks.mkdir(parents=True)
    (blocks / "ok.py").write_text(
        "MODES = {'ON': 'enabled', 'OFF': 'disabled', 'AUTO': 'automatic'}\n"
        "UNITS = {1: 'inch', 2: 'foot', 4: 'millimetre', 6: 'metre'}\n"
        "LABELS = {'Drawing': 'a', 'Title': 'b', 'Scale': 'c', 'Date': 'd'}\n",
        encoding="utf-8",
    )
    core = tmp_path / "app" / "core"
    core.mkdir(parents=True)
    (core / "elsewhere.py").write_text(
        "T = {'AA': 'One', 'BB': 'Two', 'CC': 'Three', 'DD': 'Four'}\n",
        encoding="utf-8",
    )
    assert sh.scan(tmp_path) == []


def test_probe_count_growth_is_flagged(tmp_path: Path):
    """A probe ID already baselined may not gain occurrences."""
    sh = _load()
    pkg = tmp_path / "app"
    pkg.mkdir()
    (tmp_path / "scripts").mkdir()
    (pkg / "m.py").write_text("# E1\n# E1\n", encoding="utf-8")
    sh.write_baseline(tmp_path)  # baseline: E1 x2
    (pkg / "m.py").write_text("# E1\n# E1\n# E1\n", encoding="utf-8")
    findings = sh.scan(tmp_path)
    assert any(f.startswith("GREW ") and "::probe::E1" in f for f in findings)


def test_line_allowlist_skips_only_the_named_fragment(tmp_path: Path):
    """An allowlisted (file, ID, fragment) line is skipped; the same ID on any
    other line of that file, or in another file, still counts."""
    sh = _load()
    pkg = tmp_path / "app" / "lib"
    pkg.mkdir(parents=True)
    (pkg / "pm_excel.py").write_text(
        'ws.add_chart(chart, "G3")\n'
        "# tuned for G3\n",
        encoding="utf-8",
    )
    (pkg / "other.py").write_text('ws.add_chart(chart, "G3")\n', encoding="utf-8")
    _syms, probes = sh.inventory(tmp_path)
    assert probes.get("app/lib/pm_excel.py::probe::G3") == 1
    assert probes.get("app/lib/other.py::probe::G3") == 1


def test_real_baseline_is_a_ceiling_with_no_stale_keys():
    """The committed baseline matches reality: nothing new, nothing grown, and
    every baselined key still exists (a deleted form must be removed by
    regenerating the baseline, never left behind as a stale allowance)."""
    sh = _load()
    # The file must exist (an absent file would read as "nothing allowed" and
    # hide a broken path); an EMPTY baseline is the goal state, not an error.
    assert (sh.repo_root() / sh.BASELINE_PATH).is_file(), "baseline missing: run --write-baseline"
    base_syms, base_probes = sh.load_baseline()
    # Ceiling: the gate itself is green against main.
    assert sh.scan() == []
    # No stale keys: every grandfathered form is still present.
    symbols, probes = sh.inventory()
    stale_syms = sorted(k for k in base_syms if k not in symbols)
    stale_probes = sorted(k for k in base_probes if k not in probes)
    assert not stale_syms, f"stale baseline symbols (regenerate): {stale_syms}"
    assert not stale_probes, f"stale baseline probe keys (regenerate): {stale_probes}"


def test_run_time_inputs_catch_battery_text_names_and_project_ids(tmp_path):
    """Battery question text, case ids, distinctive expected figures, live
    document names / reference codes and project ids are caught in code,
    prompts and config -- read from inputs given at run time, never baselined."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("scan_hw", "scripts/scan_hardwiring.py")
    scan_hw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan_hw)

    (tmp_path / "app" / "agents" / "configs").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "app" / "clean.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "app" / "leaky.py").write_text(
        "# fixes ZQ7: how many pallets of blue widgets fit inside the north hangar today\n"
        "TOTAL = 918273.64\nPID = 'acme_tower_2'\n", encoding="utf-8")
    (tmp_path / "app" / "agents" / "configs" / "agent.md").write_text(
        "Always cite Quarterly Pallet Audit Northern Hangar Report first.", encoding="utf-8")
    (tmp_path / "config" / "x.yaml").write_text("ref: QPA-HNG-0042-REV\n", encoding="utf-8")

    cases = {"ZQ7": {"questions": ["How many pallets of blue widgets fit inside the north hangar today?"],
                     "expected": ["918,273.64 pallets"]}}
    live = {"documents": ["Quarterly Pallet Audit Northern Hangar Report.pdf", "QPA-HNG-0042-REV.pdf"],
            "projects": ["acme_tower_2"]}

    found = scan_hw.leakage_findings(cases=cases, live=live, root=tmp_path)
    kinds = {f.split()[0] for f in found}
    assert kinds == {"CASE-ID", "QUESTION-TEXT", "EXPECTED-FIGURE", "DOCUMENT-NAME", "DOCUMENT-REF", "PROJECT-ID"}
    assert not [f for f in found if "clean.py" in f]
    assert scan_hw.leakage_findings(cases={"QQ1": {"questions": [], "expected": ["2500 kg"]}},
                                    live=None, root=tmp_path) == []  # 2500: not distinctive


def test_leakage_case_id_honours_the_line_allowlist(tmp_path: Path):
    """A battery case id that is ALSO a domain token (an Excel cell, a paper
    size) is skipped on the allowlisted (file, id, fragment) line only -- the
    same id on another line or in another file is still a CASE-ID finding."""
    sh = _load()
    pkg = tmp_path / "app" / "lib"
    pkg.mkdir(parents=True)
    (pkg / "pm_excel.py").write_text('ws.add_chart(chart, "G3")\n', encoding="utf-8")
    (pkg / "other.py").write_text('ws.add_chart(chart, "G3")\n', encoding="utf-8")
    cases = {"G3": {"questions": [], "expected": []}}
    found = sh.leakage_findings(cases=cases, live=None, root=tmp_path)
    assert found == ["CASE-ID app/lib/other.py: G3"]
    (pkg / "pm_excel.py").write_text('ws.add_chart(chart, "G3")\n# tuned for G3\n', encoding="utf-8")
    found = sh.leakage_findings(cases=cases, live=None, root=tmp_path)
    assert "CASE-ID app/lib/pm_excel.py: G3" in found


def _registry_tree(tmp_path: Path) -> Path:
    core = tmp_path / "app" / "core"
    core.mkdir(parents=True)
    (core / "system_projects.py").write_text(
        'SHARED_REFERENCE_PROJECT = "zz_shared_reference"\n'
        "SYSTEM_PROJECTS = {\n"
        '    SHARED_REFERENCE_PROJECT: "product-seeded reference layer",\n'
        "}\n",
        encoding="utf-8",
    )
    return core


def test_system_project_registry_is_read_at_run_time(tmp_path: Path):
    """System namespaces are declared once in the registry; customer projects
    are not. The registry ids come from the registry file itself (AST), not
    from the scanner."""
    sh = _load()
    _registry_tree(tmp_path)
    assert sh.system_project_ids(tmp_path) == {"zz_shared_reference"}
    assert sh.system_project_ids(tmp_path / "nowhere") == set()


def test_project_id_check_exempts_only_registry_declared_system_ids(tmp_path: Path):
    """A customer project id in product text is flagged anywhere. A system id
    declared in the registry and used via its constant is not -- but the same
    id written as a literal outside the registry is (define it once)."""
    sh = _load()
    core = _registry_tree(tmp_path)
    (core / "uses_constant.py").write_text(
        "from app.core.system_projects import SHARED_REFERENCE_PROJECT\n"
        "PID = SHARED_REFERENCE_PROJECT\n",
        encoding="utf-8",
    )
    (core / "customer.py").write_text("PID = 'acme_tower_2'\n", encoding="utf-8")
    live = {"documents": [], "projects": ["acme_tower_2", "zz_shared_reference"]}
    found = sh.leakage_findings(cases=None, live=live, root=tmp_path)
    assert found == ["PROJECT-ID app/core/customer.py: acme_tower_2"]

    (core / "scattered.py").write_text("PID = 'zz_shared_reference'\n", encoding="utf-8")
    found = sh.leakage_findings(cases=None, live=live, root=tmp_path)
    assert any(f.startswith("PROJECT-ID app/core/scattered.py: zz_shared_reference") for f in found)
    # A customer id cannot hide in the registry file either.
    (core / "system_projects.py").write_text(
        (core / "system_projects.py").read_text(encoding="utf-8") + "# acme_tower_2\n",
        encoding="utf-8",
    )
    found = sh.leakage_findings(cases=None, live=live, root=tmp_path)
    assert "PROJECT-ID app/core/system_projects.py: acme_tower_2" in found


def test_generic_document_name_is_flagged_only_in_its_file_name_form(tmp_path: Path):
    """A live name of three or fewer plain words ("terms of engagement") is
    ordinary vocabulary: prose using the phrase is not a document name. Its
    file-name form (with the extension, or with its own ``_``/``-`` joins) still
    is. A distinctive name (four+ words, or carrying a digit) is still caught
    as a phrase."""
    sh = _load()
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "prose.py").write_text("# notice under the Terms of Engagement clause\n", encoding="utf-8")
    (pkg / "fname.py").write_text("# see terms_of_engagement for the clause\n", encoding="utf-8")
    (pkg / "fext.py").write_text("# open Terms of Engagement.PDF first\n", encoding="utf-8")
    (pkg / "digit.py").write_text("# the zeta 2031 handbook says\n", encoding="utf-8")
    live = {"documents": ["terms_of_engagement.pdf", "Terms of Engagement.pdf", "Zeta 2031 Handbook.pdf"],
            "projects": []}
    found = sh.leakage_findings(cases=None, live=live, root=tmp_path)
    files = {f.split()[1].rstrip(":") for f in found if f.startswith("DOCUMENT-NAME")}
    assert files == {"app/fname.py", "app/fext.py", "app/digit.py"}


def test_live_document_ids_in_product_text_are_flagged(tmp_path: Path):
    """A live document id (or its 8-char prefix), word-bounded, in product text
    is a DOCUMENT-ID finding; the same hex run inside a longer token is not."""
    sh = _load()
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "seed.py").write_text('STALE = "c0ffee12"\n', encoding="utf-8")
    (pkg / "prefix.py").write_text("# see doc 7e57ab1e for the copy\n", encoding="utf-8")
    (pkg / "inside.py").write_text('SHA = "00c0ffee1299"\n', encoding="utf-8")
    live = {"documents": [], "projects": [], "document_ids": ["c0ffee12", "7e57ab1e-9f00-4c1d-8e2a-1234567890ab", "abc"]}
    found = sh.leakage_findings(cases=None, live=live, root=tmp_path)
    assert found == [
        "DOCUMENT-ID app/prefix.py: 7e57ab1e",
        "DOCUMENT-ID app/seed.py: c0ffee12",
    ]
