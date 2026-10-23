"""The repo-hygiene gate (scripts/scan_repo_hygiene.py) on synthetic trees.

Every input here is generated in tmp_path: fake binaries, an oversized file,
and a synthetic "live" name hashed into a synthetic hash file. Nothing reads
the real database; only the last checks read the committed hash file and
model manifest, to prove they parse and carry digests only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_model
import gen_live_name_hashes as gen
import scan_repo_hygiene as hyg

SYNTHETIC_NAME = "Zephyr Quarry Annex 7741 Method Statement.pdf"
SYNTHETIC_CODE = "ZQX-4417-SYN-09"
SYNTHETIC_PROJECT = "synthetic_project_alpha"


def _write(root: Path, rel: str, data: bytes | str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return rel


def _hash_file(tmp_path: Path, live: dict) -> hyg.LiveHashes:
    tokens, seqs = gen.records(live, system_ids=set())
    out = tmp_path / "hashes.txt"
    out.write_text(gen.render("00ff" * 8, tokens, seqs), encoding="utf-8")
    loaded = hyg.LiveHashes.load(out)
    assert loaded is not None
    return loaded


def _scan(root, files, hashes=None, **kw):
    kw.setdefault("allowed_binaries", {})
    kw.setdefault("allowed_large", {})
    return hyg.scan(root, files, hashes, **kw)


def test_clean_text_tree_passes(tmp_path):
    files = [_write(tmp_path, "app/mod.py", "def f():\n    return 1\n")]
    findings, warnings = _scan(tmp_path, files)
    assert findings == [] and warnings == []


@pytest.mark.parametrize("rel,data", [
    ("tests/fixtures/fake.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
    ("docs/contract.pdf", b"%PDF-1.4\n%synthetic\n"),
    ("docs/notes.txt", b"%PDF-1.7 renamed pdf with a text extension"),
    ("data/file_0123456789abcdef", b"SQLite format 3\x00" + b"\x00" * 64),
    ("models/weights.onnx", b"\x08\x07\x12\x04synthetic"),
    ("reports/pack.zip", b"PK\x03\x04synthetic"),
    ("tests/fixtures/report.docx", b"PK\x03\x04[Content_Types].xml"),
])
def test_binary_outside_the_allow_list_fails(tmp_path, rel, data):
    files = [_write(tmp_path, rel, data)]
    findings, _ = _scan(tmp_path, files)
    assert len(findings) == 1 and findings[0].startswith(f"BINARY {rel}:")


def test_allow_listed_binary_passes_and_stale_entry_warns(tmp_path):
    files = [_write(tmp_path, "tests/fixtures/tiny.png", b"\x89PNG\r\n\x1a\n")]
    allowed = {"tests/fixtures/tiny.png": "generated", "tests/fixtures/gone.pdf": "generated"}
    findings, warnings = _scan(tmp_path, files, allowed_binaries=allowed)
    assert findings == []
    assert warnings == ["stale ALLOWED_BINARIES entry (no longer tracked): tests/fixtures/gone.pdf"]


def test_file_above_500_kb_fails(tmp_path):
    files = [_write(tmp_path, "docs/big.md", "x" * 600_000)]
    findings, _ = _scan(tmp_path, files)
    assert findings == [f"SIZE docs/big.md: 600000 bytes > {hyg.MAX_BYTES}"]


def test_size_allow_list_is_a_ceiling_not_an_exemption(tmp_path):
    files = [_write(tmp_path, "app/huge.py", "#" * 600_000)]
    assert _scan(tmp_path, files, allowed_large={"app/huge.py": 700_000})[0] == []
    findings, _ = _scan(tmp_path, files, allowed_large={"app/huge.py": 550_000})
    assert findings == ["SIZE app/huge.py: 600000 bytes > 550000"]


def test_the_real_allow_lists_are_exact_paths():
    for rel in list(hyg.ALLOWED_BINARIES) + list(hyg.ALLOWED_LARGE):
        assert not any(ch in rel for ch in "*?["), f"glob in allow-list: {rel}"
        assert not rel.endswith("/"), f"directory in allow-list: {rel}"


def test_synthetic_live_name_in_the_hash_file_fails(tmp_path):
    hashes = _hash_file(tmp_path, {"documents": [SYNTHETIC_NAME], "document_ids": [],
                                   "projects": []})
    files = [
        _write(tmp_path, "tests/test_x.py",
               "# regression from zephyr-quarry annex 7741 method_statement upload\n"),
        _write(tmp_path, "app/ok.py", "QUARRY = 'annex'\n"),
    ]
    findings, _ = _scan(tmp_path, files, hashes)
    assert len(findings) == 1
    assert findings[0].startswith("LIVE-NAME tests/test_x.py:1: live document name")
    assert "zephyr" not in findings[0]  # CI output names the place, not the name
    # The hash file holds digests only, never the name.
    text = (tmp_path / "hashes.txt").read_text(encoding="utf-8").lower()
    assert "zephyr" not in text and "quarry" not in text


def test_synthetic_code_and_project_id_and_path_fail(tmp_path):
    hashes = _hash_file(tmp_path, {"documents": [f"{SYNTHETIC_CODE} Rev B.pdf"],
                                   "document_ids": ["5eed5eed-0000"],
                                   "projects": [SYNTHETIC_PROJECT]})
    files = [
        _write(tmp_path, "docs/a.md", f"see {SYNTHETIC_CODE} for details\n"),
        _write(tmp_path, "docs/b.md", f"project_id = '{SYNTHETIC_PROJECT}'\n"),
        _write(tmp_path, "docs/c.md", "doc 5eed5eed was re-ingested\n"),
        _write(tmp_path, f"tests/fixtures/{SYNTHETIC_PROJECT}/n.txt", "plain\n"),
    ]
    findings, _ = _scan(tmp_path, files, hashes)
    where = sorted(f.split(": live")[0] for f in findings)
    assert where == ["LIVE-NAME docs/a.md:1", "LIVE-NAME docs/b.md:1", "LIVE-NAME docs/c.md:1",
                     f"LIVE-NAME tests/fixtures/{SYNTHETIC_PROJECT}/n.txt (the path itself)"]


def test_short_generic_names_and_system_projects_are_not_hashed(tmp_path):
    live = {"documents": ["Conditions of Contract.pdf", "BOQ.xlsx"],
            "document_ids": ["abc"], "projects": ["training_material", "kb"]}
    tokens, seqs = gen.records(live, system_ids={"training_material"})
    # A three-plain-word name is domain vocabulary: only its file-name form counts.
    assert seqs == {(4, "conditions of contract pdf")}
    assert tokens == set()
    out = tmp_path / "h.txt"
    out.write_text(gen.render("ab" * 16, tokens, seqs), encoding="utf-8")
    hashes = hyg.LiveHashes.load(out)
    files = [_write(tmp_path, "docs/a.md", "The Conditions of Contract govern; see training_material.\n"),
             _write(tmp_path, "docs/b.md", "attached: Conditions of Contract.pdf\n")]
    findings, _ = _scan(tmp_path, files, hashes)
    assert [f.split(": live")[0] for f in findings] == ["LIVE-NAME docs/b.md:1"]


def test_repo_authored_names_are_not_hashed():
    authored = " " + " ".join(hyg.normalize_words("zephyr quarry annex 7741 method statement")) + " "
    tokens, seqs = gen.records({"documents": [SYNTHETIC_NAME], "document_ids": [], "projects": []},
                               system_ids=set(), authored=authored)
    assert seqs == set() and tokens == set()


def test_baseline_grandfathers_but_only_shrinks(tmp_path):
    hashes = _hash_file(tmp_path, {"documents": [], "document_ids": [],
                                   "projects": [SYNTHETIC_PROJECT]})
    files = [_write(tmp_path, "docs/a.md", f"{SYNTHETIC_PROJECT}\n")]
    baseline = hyg.hit_counts(hyg.live_hits(tmp_path, files, hashes))
    assert _scan(tmp_path, files, hashes, baseline=baseline)[0] == []
    # A second occurrence in the same file is a NEW leak.
    _write(tmp_path, "docs/a.md", f"{SYNTHETIC_PROJECT}\n{SYNTHETIC_PROJECT}\n")
    findings, _ = _scan(tmp_path, files, hashes, baseline=baseline)
    assert len(findings) == 2  # both lines reported for the over-count digest
    # Removing it leaves a baseline that can shrink (warning, not failure).
    _write(tmp_path, "docs/a.md", "clean\n")
    findings, warnings = _scan(tmp_path, files, hashes, baseline=baseline)
    assert findings == [] and len(warnings) == 1 and "can shrink" in warnings[0]


def test_committed_hash_file_parses_and_holds_digests_only():
    hashes = hyg.LiveHashes.load(ROOT / hyg.HASH_FILE)
    assert hashes is not None and hashes.salt and (hashes.tokens or hashes.seqs)
    for line in (ROOT / hyg.HASH_FILE).read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "salt ", "width ")):
            continue
        kind, value = line.split(" ")
        assert kind == "A" or kind == "T" or kind[1:].isdigit()
        assert len(value) == hashes.width and int(value, 16) >= 0


def test_fetch_model_check_fails_on_checksum_mismatch(tmp_path):
    good = b"synthetic weights"
    (tmp_path / "w.onnx").write_bytes(good)
    manifest = tmp_path / "manifest.json"
    entry = {"name": "w.onnx", "sha256": hashlib.sha256(good).hexdigest(), "size": len(good),
             "required_for_image": True}
    manifest.write_text(json.dumps({"repo": "x/y", "release_tag": "t", "files": [entry]}))
    assert fetch_model.main(["--check", "--manifest", str(manifest)]) == 0
    (tmp_path / "w.onnx").write_bytes(b"synthetic weightz")  # same size, different bytes
    assert fetch_model.main(["--check", "--manifest", str(manifest)]) == 1
    (tmp_path / "w.onnx").unlink()
    assert fetch_model.main(["--check", "--manifest", str(manifest)]) == 1


def test_model_manifest_pins_the_image_weights():
    m = json.loads((ROOT / "data/models/manifest.json").read_text(encoding="utf-8"))
    names = {f["name"]: f for f in m["files"]}
    assert names["safety_world_v2.onnx"]["required_for_image"] is True
    assert len(names["safety_world_v2.onnx"]["sha256"]) == 64
    assert not m["release_tag"].startswith("v")  # docker-publish fires on v* tags
