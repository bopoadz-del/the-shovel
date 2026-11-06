"""Compressed files are never ingested, opened or unpacked (owner ruling).

A compressed folder (zip, rar, 7z, tar, gzip, ...) is never a project
document and no extractor is registered for one (app/core/text_extractors.py),
so ingest excludes it at discovery like any other non-ingestible file. This
module only names it for the uploader: an upload that IS a compressed file
gets one clear message instead of a generic "type not ingestible", whatever
its extension. Recognised by the container's own signature -- an Office
package is a zip on disk but is a document, so it is recognised as one first.
"""
from __future__ import annotations

from typing import Optional

from app.core.ingest_status import is_ingestible

COMPRESSED_UPLOAD_DETAIL = (
    "Compressed files are not supported. Unzip it and upload the documents."
)

# Leading bytes of compressed-folder containers (offset, signature).
_SIGNATURES = (
    (0, b"PK\x03\x04"), (0, b"PK\x05\x06"), (0, b"PK\x07\x08"),  # zip family
    (0, b"Rar!\x1a\x07"),                                       # rar 4 / 5
    (0, b"7z\xbc\xaf\x27\x1c"),                                 # 7-Zip
    (0, b"\x1f\x8b"),                                           # gzip
    (0, b"BZh"),                                                # bzip2
    (0, b"\xfd7zXZ\x00"),                                       # xz
    (0, b"\x28\xb5\x2f\xfd"),                                   # zstd
    (0, b"MSCF"),                                               # cabinet
    (257, b"ustar"),                                            # tar
)
HEAD_BYTES = 262


def is_compressed(filename: Optional[str], head: bytes) -> bool:
    """True when ``head`` (the file's first bytes) is a compressed folder.

    A file whose name is an ingestible document (``.docx`` is a zip inside)
    is a document, not a compressed folder.
    """
    if filename and is_ingestible(filename):
        return False
    return any(head[at:at + len(sig)] == sig for at, sig in _SIGNATURES)


def head_of(fileobj) -> bytes:
    """The first ``HEAD_BYTES`` of an upload stream, leaving it rewound."""
    fileobj.seek(0)
    head = fileobj.read(HEAD_BYTES)
    fileobj.seek(0)
    return head
