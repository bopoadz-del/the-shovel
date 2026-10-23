"""Every GitHub workflow file must be YAML GitHub can load.

A workflow run that finishes in 0s with 0 jobs means the file did not parse.
The deploy-aws failure mode was a multi-line python3 -c whose continuation
lines sat at column 0 inside a run: | block. Loading each file with a YAML
parser catches that class of break, and the deploy-aws assertions keep the
two triggers and a non-empty job map.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_READER = _ROOT / "scripts" / "read_livez_build_sha.py"


def _workflow_paths() -> list[Path]:
    paths = sorted(_WORKFLOWS.glob("*.yml")) + sorted(_WORKFLOWS.glob("*.yaml"))
    assert paths, f"no workflow files under {_WORKFLOWS}"
    return paths


def _load(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} did not parse to a mapping"
    return loaded


def _trigger(doc: dict) -> dict:
    # PyYAML's YAML 1.1 resolver turns the bare key `on` into boolean True.
    # GitHub keeps it as the string key. Accept either.
    raw = doc["on"] if "on" in doc else doc[True]
    assert isinstance(raw, dict), "workflow trigger is not a mapping"
    return raw


def test_every_workflow_file_parses():
    for path in _workflow_paths():
        _load(path)


def test_deploy_aws_has_jobs_workflow_run_and_dispatch():
    doc = _load(_WORKFLOWS / "deploy-aws.yml")
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs
    trigger = _trigger(doc)
    assert "workflow_run" in trigger
    assert "workflow_dispatch" in trigger
    assert trigger["workflow_run"]["workflows"] == ["tests"]
    assert trigger["workflow_run"]["branches"] == ["main"]


def _reader():
    spec = importlib.util.spec_from_file_location("read_livez_build_sha", _READER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_livez_reader_prints_build_sha_or_blank(tmp_path, capsys):
    module = _reader()
    assert module.build_sha_from_livez('{"build_sha": "abc123"}') == "abc123"
    assert module.build_sha_from_livez('{"build_sha": "  abc123  "}') == "abc123"
    assert module.build_sha_from_livez("{}") == ""
    assert module.build_sha_from_livez('{"build_sha": null}') == ""
    assert module.build_sha_from_livez("not-json") == ""
    assert module.build_sha_from_livez("[1, 2]") == ""
    assert module.build_sha_from_livez("") == ""

    present = tmp_path / "livez.json"
    present.write_text('{"status": "ok", "build_sha": "deadbeef"}\n', encoding="utf-8")
    assert module.main([str(present)]) == 0
    assert capsys.readouterr().out == "deadbeef\n"

    assert module.main([str(tmp_path / "missing.json")]) == 0
    assert capsys.readouterr().out == "\n"
