import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from scripts.survey_photo_corpus import survey_folder


@pytest.fixture
def fixture_folder(tmp_path: Path) -> Path:
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(tmp_path / name)
    return tmp_path


def _fake_dino(image_path, class_names):
    if image_path.name == "a.jpg":
        return [
            {"class": "hard hat", "confidence": 0.7, "bbox": [0, 0, 32, 32]},
            {"class": "crack in concrete wall", "confidence": 0.6, "bbox": [16, 16, 48, 48]},
        ]
    return [{"class": "hard hat", "confidence": 0.5, "bbox": [0, 0, 32, 32]}]


def test_survey_counts_per_class_and_per_photo(fixture_folder, tmp_path):
    out_json = tmp_path / "out.json"
    with patch("scripts.survey_photo_corpus.detect_with_dino", side_effect=_fake_dino):
        report = survey_folder(fixture_folder, out_json)

    assert report["total_images"] == 2
    assert report["per_class"]["hard hat"]["detections"] == 2
    assert report["per_class"]["hard hat"]["photos_with_at_least_one"] == 2
    assert report["per_class"]["crack in concrete wall"]["detections"] == 1
    assert report["per_class"]["crack in concrete wall"]["photos_with_at_least_one"] == 1

    persisted = json.loads(out_json.read_text())
    assert persisted == report
    assert len(persisted["per_photo"]) == 2
