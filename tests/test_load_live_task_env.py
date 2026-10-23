"""Live-task env loader: mask values, never print them."""
from __future__ import annotations

import importlib

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _mod():
    return importlib.import_module("load_live_task_env")


def test_export_masks_and_does_not_print_values(tmp_path, capsys, monkeypatch):
    mod = _mod()
    github_env = tmp_path / "github.env"
    n = mod.export_env(
        {"RAG_SCRUB_RULES": "line-one\nline-two", "PORT": "8000"},
        str(github_env),
    )
    assert n == 1  # PORT is skipped
    out = capsys.readouterr().out
    assert "::add-mask::line-one" in out
    assert "::add-mask::line-two" in out
    # the mask directive contains the value (Actions needs it) but we never
    # print a bare unmasked assignment.
    assert "RAG_SCRUB_RULES=line-one" not in out
    written = github_env.read_text()
    assert "RAG_SCRUB_RULES" in written
    assert "PORT" not in written


def test_workflow_declares_expected_inputs():
    text = (ROOT / ".github/workflows/scrub-verify.yml").read_text()
    assert "workflow_dispatch" in text
    assert "measure" in text and "repair" in text and "dryrun" in text
    assert "aws-actions/configure-aws-credentials@v4" in text
    assert "NEON_API" not in text  # the Neon key never comes to GitHub
    assert "set +x" in text
    assert "echo $" not in text
    assert "secrets.RAG_SCRUB_RULES" not in text


@pytest.mark.parametrize("ref,expected", [
    ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123:SOME_KEY::",
     ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123", "SOME_KEY")),
    ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123",
     ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123", "")),
    ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123:SOME_KEY:AWSCURRENT:",
     ("arn:aws:secretsmanager:us-west-2:111122223333:secret:app/env-AbC123", "SOME_KEY")),
    ("plain-secret-name", ("plain-secret-name", "")),
])
def test_an_ecs_secret_reference_splits_into_id_and_json_key(ref, expected):
    from load_live_task_env import split_secret_ref
    assert split_secret_ref(ref) == expected
