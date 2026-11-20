"""Decode tracked-file bytes to text, whatever the encoding.

Every scanner here reads tracked files as text. Each one used to decode UTF-8
only, so the ENCODING of a file decided whether the gate saw it at all:

- ``utf-8`` with ``errors="ignore"`` turns a UTF-16 file into text whose
  characters are separated by NULs, and no rule matches that;
- ``utf-8`` with ``errors="strict"`` raises ``UnicodeDecodeError``, and the
  caller skipped the file outright;
- one NUL byte in the first 4 KB made the repo-hygiene gate call the file a
  binary, and an allow-listed binary is never checked for live names.

A UTF-16 file is what Windows PowerShell 5.1 writes for ``>`` and ``Out-File``,
so this is the ordinary save path on a Windows host, not a corner case. One
decoder, used by every reader, means the encoding a leak is written in cannot
decide whether the leak is seen.

The ladder is fixed and general: a byte-order mark names the encoding and wins;
otherwise valid UTF-8 is UTF-8; otherwise an even-length head that is mostly
NUL-interleaved is UTF-16 without a mark; otherwise latin-1, which is lossless
byte for byte, so no content is dropped. Nothing here looks at what the text
says -- the rules do that.
"""
from __future__ import annotations

import codecs
from pathlib import Path

#: Byte-order marks, checked in order. Kept as marks rather than as a codec
#: list because ``utf-8-sig`` is the only one that also strips on its own.
BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

#: A UTF-16 file written without a BOM still carries a NUL byte on every
#: line break, space and digit. Its absence is what says "not UTF-16".
SNIFF_BYTES = 4096


def _strip_mark(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def _ascii_share(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch in "\n\r\t" or " " <= ch < "\x7f")
    return good / len(text)


def _utf16_without_bom(data: bytes) -> str | None:
    """Decode ``data`` as UTF-16 when it really is, else None.

    Two facts make this decidable without a BOM. Pure-ASCII UTF-16 is also
    VALID UTF-8 -- as NUL-separated characters -- so a UTF-8 attempt cannot
    come first: it would succeed and the NULs would hide the text from every
    rule. And the WRONG endianness does not fail to decode, it decodes into
    block-drawing code points, so "it decoded" is not a test either. Source
    text is overwhelmingly ASCII, and that is the discriminator: the reading
    with the high ASCII share is the right endian, the wrong one has almost
    none. A file with no ASCII in it falls through to latin-1, which keeps
    every byte rather than guessing.
    """
    head = data[:SNIFF_BYTES]
    if len(head) < 4 or head.count(b"\x00") < 2:
        return None
    readings: list[tuple[float, str]] = []
    for endian in ("utf-16-le", "utf-16-be"):
        try:
            text = data.decode(endian)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            readings.append((_ascii_share(text), text))
    if not readings:
        return None
    best_share, best = max(readings)
    other = min(share for share, _ in readings) if len(readings) > 1 else 0.0
    if best_share >= 0.5 and best_share >= 2 * other:
        return best
    return None


def decode_bytes(data: bytes) -> str:
    """``data`` as text. Never raises and never drops bytes."""
    for mark, codec in BOMS:
        if data.startswith(mark):
            return _strip_mark(data.decode(codec, errors="replace"))
    guessed = _utf16_without_bom(data)
    if guessed is not None:
        return _strip_mark(guessed)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    return data.decode("latin-1")


def read(path: Path) -> str:
    """The file at ``path`` as text; ``OSError`` only if it cannot be read."""
    return decode_bytes(Path(path).read_bytes())


def looks_like_text(head: bytes) -> bool:
    """False when ``head`` decodes to something that still carries NULs.

    A UTF-16 head used to look binary because of the NUL in every other byte;
    once decoded, its text carries none. A PNG, archive or database still does.
    """
    return "\x00" not in decode_bytes(head)
