"""A chat-attached image has its text read and handed to the model (DEFECT E1).

The image block's OCR pass already read the photo, but analyze-photo returned
only detector observations, so a screenshot of a table reached the model as
``observations: none detected`` and nothing else. The endpoint now returns the
text and the ready-made ``[attached: ...]`` marker that carries it.

Every test builds its own input: images are rendered by the test with PIL
from invented lines; detector output is a stub. Each rule is pinned by two
inputs that must come out differently.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.blocks import image as image_block
from app.core.rag.retriever import ATTACHED_MARKER_RE, attachment_markers_carry_content
from app.main import app
from app.routers import chat_photos
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("WARM_MODELS_BLOCKING", "1")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mp.undo()


def _route_path(endpoint):
    for route in app.routes:
        routed = getattr(route, "endpoint", None)
        if routed is not None and (routed.__module__, routed.__qualname__) == (
            endpoint.__module__, endpoint.__qualname__,
        ):
            return route.path
    raise AssertionError(f"{endpoint.__qualname__} is not routed")


def _png(lines):
    """A white PNG with ``lines`` drawn in large black type."""
    font = ImageFont.load_default(size=48)
    img = Image.new("RGB", (1400, 120 + 90 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 90 * i), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _OcrOnlyBlock:
    """The image block's OCR pass for real; the detector ran and saw nothing."""

    async def execute(self, inputs, params):
        text, conf = image_block._tesseract_ocr(inputs["file_path"])
        return {"result": {"extracted_text": text, "ocr_confidence": conf,
                           "safety_qaqc": [], "safety_qaqc_ran": True}}


class _FixedBlock:
    def __init__(self, result):
        self.result = result

    async def execute(self, inputs, params):
        return {"result": dict(self.result)}


def _post(client, monkeypatch, block, name, data):
    monkeypatch.setattr("app.dependencies.get_block_instance", lambda _n: block)
    r = client.post(
        _route_path(chat_photos.analyze_chat_photo),
        files={"file": (name, data, "image/png")},
        headers=H,
    )
    assert r.status_code == 200, r.text
    return r.json()


def _norm(s):
    return " ".join(s.lower().split())


@pytest.mark.parametrize(
    "lines,absent",
    [
        (["Pipe bedding 34 m", "Manhole cover 7 nr"], "plasterboard"),
        (["Plasterboard lining 210 m2", "Skirting 96 m"], "manhole"),
    ],
)
def test_text_rendered_into_an_image_reaches_the_marker(client, monkeypatch, lines, absent):
    body = _post(client, monkeypatch, _OcrOnlyBlock(), "shot.png", _png(lines))
    got = _norm(body["image_text"])
    for line in lines:
        assert _norm(line) in got, (line, body["image_text"])
    assert absent not in got
    m = ATTACHED_MARKER_RE.search(body["attachment_marker"])
    assert m is not None and m.group(1) == "shot.png"
    assert _norm(lines[0]) in _norm(m.group(2))
    assert attachment_markers_carry_content(body["attachment_marker"])


@pytest.mark.parametrize(
    "ocr,name",
    [
        ("Item [A] 12 nr\nTotal ] 40", "plan [rev] | b.png"),
        ("Row one\n\n   \nRow two [x]   ", "table.png"),
    ],
)
def test_marker_survives_delimiters_inside_text_and_name(client, monkeypatch, ocr, name):
    body = _post(client, monkeypatch, _FixedBlock({"extracted_text": ocr,
                                                   "safety_qaqc_ran": True}), name, b"\x89PNG\r\n\x1a\n")
    marker = body["attachment_marker"]
    matches = list(ATTACHED_MARKER_RE.finditer(marker))
    assert len(matches) == 1 and matches[0].group(0) == marker
    payload = matches[0].group(2)
    for word in ("12 nr", "40", "Row two"):
        if word in ocr:
            assert word in payload
    assert "\n\n" not in body["image_text"]
    assert not any(ln != ln.rstrip() for ln in body["image_text"].splitlines())


@pytest.mark.parametrize("size,cut", [(chat_photos._IMAGE_TEXT_MAX_CHARS - 10, False),
                                      (chat_photos._IMAGE_TEXT_MAX_CHARS + 500, True)])
def test_cap_is_said_never_silent(size, cut):
    text, truncated = chat_photos._image_text({"extracted_text": "q" * size})
    assert truncated is cut
    assert len(text) <= chat_photos._IMAGE_TEXT_MAX_CHARS
    marker = chat_photos._attachment_marker("a.png", [], text, truncated)
    assert ("cut at" in marker) is cut


@pytest.mark.parametrize(
    "body,observations,expected",
    [
        ({"extracted_text": "   \n "}, ["ladder detected"],
         "[attached: a.png | observations: ladder detected]"),
        ({"extracted_text": None}, [], "[attached: a.png]"),
    ],
)
def test_no_text_means_no_text_segment(body, observations, expected):
    text, truncated = chat_photos._image_text(body)
    assert text == "" and truncated is False
    assert chat_photos._attachment_marker("a.png", observations, text, truncated) == expected


@pytest.mark.parametrize("lang", ["eng", "ara+eng"])
def test_image_ocr_uses_the_configured_language(monkeypatch, tmp_path, lang):
    seen = []

    def _to_string(img, lang=None, **kw):
        seen.append(("string", lang))
        return "x"

    def _to_data(img, lang=None, **kw):
        seen.append(("data", lang))
        return {"conf": ["90"]}

    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_string", _to_string)
    monkeypatch.setattr(pytesseract, "image_to_data", _to_data)
    monkeypatch.setenv("RAG_OCR_LANG", lang)
    p = tmp_path / "t.png"
    p.write_bytes(_png(["any"]))
    image_block._tesseract_ocr(str(p))
    assert seen == [("string", lang), ("data", lang)]
