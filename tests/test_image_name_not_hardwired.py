"""#8: the published image name is derived from ${{ github.repository }}, never
typed as a literal owner/repo, in the workflow and in the docs that quote it."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/docker-publish.yml"
DOCS = [ROOT / "deploy/README.md", ROOT / "docs/EDGE_PORT.md"]
# ghcr.io/<owner>/<repo> with literal segments (not <placeholder> or ${{ ... }})
LITERAL_GHCR = re.compile(r"ghcr\.io/(?!<|\$\{\{)[A-Za-z0-9_.-]+/(?!<|\$\{\{)[A-Za-z0-9_.-]+")


def literal_image_refs(text: str) -> list:
    return LITERAL_GHCR.findall(text)


def image_name(workflow_text: str) -> str:
    return str(yaml.safe_load(workflow_text)["env"]["IMAGE_NAME"])


def test_workflow_image_name_is_derived_from_the_repository():
    assert image_name(WORKFLOW.read_text(encoding="utf-8")) == "${{ github.repository }}"


def test_workflow_tags_use_the_derived_image_name():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}" in text


@pytest.mark.parametrize("path", [WORKFLOW, *DOCS], ids=lambda p: p.name)
def test_no_literal_ghcr_owner_repo(path):
    assert literal_image_refs(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "text, found",
    [
        ("docker pull ghcr.io/<owner>/<repo>:latest", []),
        ("images: ghcr.io/${{ github.repository }}", []),
        ("docker pull ghcr.io/someorg/some-image:latest", ["ghcr.io/someorg/some-image"]),
        ("tag `ghcr.io/other/thing:edge`", ["ghcr.io/other/thing"]),
    ],
)
def test_literal_detector_two_inputs(text, found):
    assert literal_image_refs(text) == found


@pytest.mark.parametrize(
    "value, derived",
    [("${{ github.repository }}", True), ("someorg/the-shovel", False)],
)
def test_image_name_reader_two_inputs(value, derived):
    text = f"env:\n  IMAGE_NAME: {value}\n"
    assert (image_name(text) == "${{ github.repository }}") is derived
