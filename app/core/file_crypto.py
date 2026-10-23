"""Encryption at rest for uploaded documents — Roadmap V2 · Epic 6.

The Fork stores client documents in ``DATA_DIR`` as files. This module adds
optional symmetric encryption (Fernet / AES-128-CBC + HMAC) so those files are
ciphertext on disk.

Design — OPT-IN, transparent, backward-compatible
-------------------------------------------------
* The feature is driven entirely by the ``DATA_ENCRYPTION_KEY`` env var. It
  must hold a valid Fernet key (generate one with
  ``Fernet.generate_key()``). If the var is unset, encryption is OFF and every
  function behaves exactly as before — plaintext in, plaintext out — so the
  default test/dev experience is unchanged.
* Reading is backward-compatible: ``read_document`` / ``open_plaintext``
  inspect the bytes on disk and only decrypt files that actually look like a
  Fernet token. Pre-existing plaintext files in ``DATA_DIR`` keep working even
  after a key is configured.

Public API
----------
* ``encryption_enabled() -> bool``
* ``encrypt_bytes(data) -> bytes`` / ``decrypt_bytes(token) -> bytes``
* ``looks_encrypted(blob) -> bool``
* ``write_document(path, data)``        — encrypts iff enabled
* ``read_document(path) -> bytes``      — returns plaintext, decrypting iff needed
* ``open_plaintext(path)`` (context manager) — yields a real filesystem path
  containing plaintext (the original file when unencrypted, a secure temp copy
  when encrypted; the temp copy is removed on exit).
"""

import contextlib
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

from cryptography.fernet import Fernet, InvalidToken

# Fernet tokens are URL-safe base64 and the decoded payload starts with the
# version byte 0x80 (per the Fernet spec). We use that to tell a real token
# apart from legacy plaintext files.
_FERNET_VERSION = 0x80

_ENV_KEY = "DATA_ENCRYPTION_KEY"


class DecryptionError(Exception):
    """A blob that is a real Fernet token could not be decrypted — almost
    always because DATA_ENCRYPTION_KEY does not match the key it was written
    with (e.g. the key was rotated). Raised instead of silently returning
    ciphertext, which would corrupt the document."""


#: Prefix of every short-lived plaintext copy in the temp dir (decrypt-to-read
#: and the ingest's transient extraction copy). ``sweep_stale_plaintext`` reaps
#: anything left behind under it.
DECRYPTED_TEMP_PREFIX = "fork_dec_"


def _load_fernet() -> Optional[Fernet]:
    """Build a Fernet instance from the env var, or None if unset/invalid."""
    raw = os.getenv(_ENV_KEY)
    if not raw:
        return None
    try:
        return Fernet(raw.encode() if isinstance(raw, str) else raw)
    except (ValueError, TypeError) as exc:  # malformed key
        raise ValueError(
            f"{_ENV_KEY} is set but is not a valid Fernet key. Generate one "
            f"with cryptography.fernet.Fernet.generate_key(). ({exc})"
        ) from exc


def encryption_enabled() -> bool:
    """True when a valid DATA_ENCRYPTION_KEY is configured."""
    return _load_fernet() is not None


def looks_encrypted(blob: bytes) -> bool:
    """Heuristically detect whether ``blob`` is a Fernet token.

    A Fernet token is URL-safe base64; decoded it begins with the 0x80 version
    byte and is at least 57 bytes (version + 8B timestamp + 16B IV + 32B HMAC).
    Legacy plaintext documents (PDF, images, text, ...) will not decode cleanly
    to that shape, so this lets the reader transparently pass legacy files
    through. Any uncertainty errs on the side of "not encrypted".
    """
    if not blob:
        return False
    if blob.startswith(_STREAM_MAGIC):
        return True
    try:
        import base64

        decoded = base64.urlsafe_b64decode(blob)
    except Exception:
        logger.debug(
            "blob is not a Fernet token; treating as plaintext",
            exc_info=True,
        )
        return False
    return len(decoded) >= 57 and decoded[0] == _FERNET_VERSION


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt ``data``. Returns ``data`` unchanged when encryption is off."""
    fernet = _load_fernet()
    if fernet is None:
        return data
    return fernet.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    """Decrypt a Fernet token. Returns the input unchanged when encryption is
    off, or when the input is not actually a Fernet token (legacy plaintext)."""
    fernet = _load_fernet()
    if fernet is None:
        return token
    if not looks_encrypted(token):
        return token
    if token.startswith(_STREAM_MAGIC):
        import io

        stream = io.BytesIO(token)
        _is_stream_file(stream)
        return b"".join(_iter_stream_plaintext(stream))
    try:
        return fernet.decrypt(token)
    except InvalidToken as exc:
        # The blob is base64 with a Fernet version byte and the right length —
        # the odds of legacy plaintext matching that by chance are negligible,
        # so a decrypt failure here means the key is wrong/rotated. Fail loud
        # rather than silently handing back ciphertext as if it were plaintext.
        raise DecryptionError(
            "A stored document is encrypted but could not be decrypted — "
            "DATA_ENCRYPTION_KEY does not match the key it was written with."
        ) from exc


#: Header of the framed (streamed) encrypted format: after it, repeated
#: ``[8-byte big-endian length][Fernet token of one plaintext block]``. Plain
#: ASCII, so it can never be mistaken for a whole-file Fernet token or for a
#: legacy plaintext document's own first bytes in practice.
_STREAM_MAGIC = b"FORK-ENC-STREAM-1\n"
_FRAME_LEN_BYTES = 8


def _is_stream_file(fh) -> bool:
    """True when the open file starts with the framed header (fh is left just
    past the header when True, at the start otherwise)."""
    head = fh.read(len(_STREAM_MAGIC))
    if head == _STREAM_MAGIC:
        return True
    fh.seek(0)
    return False


def _iter_stream_plaintext(fh):
    """Decrypt a framed file frame by frame (fh positioned after the header)."""
    fernet = _load_fernet()
    if fernet is None:
        raise DecryptionError(
            "A stored document is encrypted but DATA_ENCRYPTION_KEY is not set.")
    while True:
        prefix = fh.read(_FRAME_LEN_BYTES)
        if not prefix:
            return
        size = int.from_bytes(prefix, "big")
        token = fh.read(size)
        if len(prefix) < _FRAME_LEN_BYTES or len(token) < size:
            raise DecryptionError("A stored encrypted document is truncated.")
        try:
            yield fernet.decrypt(token)
        except InvalidToken as exc:
            raise DecryptionError(
                "A stored document is encrypted but could not be decrypted — "
                "DATA_ENCRYPTION_KEY does not match the key it was written with."
            ) from exc


def write_document(path: str, data: bytes) -> None:
    """Write ``data`` to ``path``, encrypting it iff encryption is enabled."""
    payload = encrypt_bytes(data)
    with open(path, "wb") as fh:
        fh.write(payload)


def read_document(path: str) -> bytes:
    """Read ``path`` and return plaintext bytes.

    Transparently decrypts encrypted files and passes legacy plaintext files
    through untouched.
    """
    with open(path, "rb") as fh:
        if _is_stream_file(fh):
            return b"".join(_iter_stream_plaintext(fh))
        raw = fh.read()
    return decrypt_bytes(raw)


class UploadTooLarge(Exception):
    """A streamed write exceeded its ``max_bytes`` budget. Carries the limit so
    the caller can build the 413 without re-reading config."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"stream exceeded {limit} bytes")
        self.limit = limit


def plaintext_size(path: str) -> int:
    """Byte length of the document's PLAINTEXT content at ``path``.

    ``os.path.getsize`` reports the CIPHERTEXT length for an encrypted file
    (Fernet base64 inflates by ~33%), so it is NOT the number to record as a
    document's size. Encryption off / legacy plaintext takes the cheap stat
    path; a real Fernet token is decrypted to be measured.

    Propagates OSError when the file is unreadable — callers decide whether a
    missing file is fatal or merely unknown.
    """
    with open(path, "rb") as fh:
        if _is_stream_file(fh):
            return sum(len(block) for block in _iter_stream_plaintext(fh))
    if not encryption_enabled():
        return os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    if not looks_encrypted(raw):
        return len(raw)
    return len(decrypt_bytes(raw))


def write_document_stream(
    path: str,
    fileobj,
    *,
    max_bytes: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
    hasher=None,
) -> int:
    """Copy ``fileobj`` to ``path``, returning the PLAINTEXT byte count written.

    ``hasher`` (e.g. ``hashlib.sha256()``) is fed every plaintext block as it
    passes, so a caller gets the content hash without reading the file back.

    Why this exists: the upload routes used ``file.file.read()``, which holds
    the ENTIRE document in memory (and again while encrypting) — a 345 MB
    drawing set on a shared worker is an OOM that drops every concurrent user,
    not just the uploader. Copying in ``chunk_size`` blocks keeps peak memory
    flat regardless of file size.

    Fernet has no streaming mode, so when encryption is ON each block is
    encrypted on its own and written as a length-prefixed frame after the
    ``_STREAM_MAGIC`` header (the framed format every reader here understands).
    Memory stays one block whatever the file size.

    ``max_bytes`` (when set) aborts with :class:`UploadTooLarge` the moment the
    stream exceeds the limit, removing the partial file first — so a rejected
    upload leaves nothing behind on disk.
    """
    try:
        fileobj.seek(0)
    except (AttributeError, OSError):
        pass  # non-seekable stream — read from wherever it is

    fernet = _load_fernet()
    written = 0
    try:
        with open(path, "wb") as out:
            if fernet is not None:
                out.write(_STREAM_MAGIC)
            while True:
                block = fileobj.read(chunk_size)
                if not block:
                    break
                written += len(block)
                if max_bytes is not None and written > max_bytes:
                    raise UploadTooLarge(max_bytes)
                if hasher is not None:
                    hasher.update(block)
                if fernet is not None:
                    token = fernet.encrypt(block)
                    out.write(len(token).to_bytes(_FRAME_LEN_BYTES, "big"))
                    out.write(token)
                else:
                    out.write(block)
    except UploadTooLarge:
        # Never leave a truncated document behind for a rejected upload — a
        # partial file on disk is indistinguishable from a complete one later.
        try:
            os.remove(path)
        except OSError:
            logger.warning("could not remove partial upload %s", path, exc_info=True)
        raise
    return written


@contextlib.contextmanager
def open_plaintext(path: str):
    """Yield a filesystem path that contains the document's plaintext.

    Libraries like PIL, Tesseract and PyMuPDF need a real file. When the file
    on disk is encrypted this decrypts it to a secure temp file and yields that
    path, removing the temp file on exit. When the file is plaintext (legacy
    files, or encryption disabled) it simply yields the original path — no copy
    is made.
    """
    with open(path, "rb") as fh:
        streamed = _is_stream_file(fh)
        raw = b"" if streamed else fh.read()

    if not streamed and not (encryption_enabled() and looks_encrypted(raw)):
        # Plaintext on disk — hand back the original path, nothing to clean up.
        yield path
        return

    # Preserve the suffix so downstream code that sniffs by extension still works.
    suffix = os.path.splitext(path)[1] or ""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=DECRYPTED_TEMP_PREFIX)
    try:
        with os.fdopen(fd, "wb") as tmp:
            if streamed:
                with open(path, "rb") as fh:
                    _is_stream_file(fh)
                    for block in _iter_stream_plaintext(fh):
                        tmp.write(block)
            else:
                tmp.write(decrypt_bytes(raw))
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            # NOT harmless enough to swallow silently: tmp_path holds the
            # DECRYPTED document. If the unlink fails the plaintext stays on
            # disk, which is exactly the state encryption exists to prevent.
            # Defence in depth: shred (zero-overwrite) the contents so even a
            # file whose NAME survives holds no plaintext, then log. Cleanup
            # still must not raise (that would mask whatever the caller was
            # doing).
            shredded = shred_file(tmp_path)
            logger.warning(
                "could not remove decrypted temp file %s — contents %s",
                tmp_path,
                "zero-overwritten (no plaintext remains)" if shredded
                else "COULD NOT be overwritten; plaintext may remain on disk",
                exc_info=True,
            )


def shred_file(path: str) -> bool:
    """Zero-overwrite a file's contents in place, then retry the unlink.

    For decrypted temp files whose removal failed: a name that lingers is
    cosmetic, plaintext that lingers is an incident. Returns True when the
    contents were overwritten (whether or not the unlink then succeeded).
    """
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.write(b"\0" * size)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        logger.warning("could not shred decrypted temp file %s", path, exc_info=True)
        return False
    try:
        os.remove(path)
    except OSError:
        # Zeroed but not removed — the sweeper will reap the empty husk.
        pass
    return True


def sweep_stale_plaintext(
    max_age_seconds: int = 3600, tmp_dir: str | None = None,
) -> dict:
    """Reap orphaned decrypted temp files (``fork_dec_*``).

    Every code path that materialises plaintext cleans up after itself, but
    a crash between decrypt and cleanup — or a failed unlink — can orphan a
    file holding CLIENT PLAINTEXT. This bounds that exposure: anything with
    the decrypted-temp prefix older than ``max_age_seconds`` is removed
    (shredded first if removal fails). Runs at boot, hourly, and on demand
    via /v1/admin/debug/sweep-plaintext.
    """
    import glob
    import time as _time

    tmp_dir = tmp_dir or tempfile.gettempdir()
    cutoff = _time.time() - max_age_seconds
    scanned = reaped = shredded = 0
    failed: list[str] = []
    for path in glob.glob(os.path.join(tmp_dir, DECRYPTED_TEMP_PREFIX + "*")):
        scanned += 1
        try:
            if os.path.getmtime(path) > cutoff:
                continue  # fresh — some request is still using it
            os.remove(path)
            reaped += 1
        except OSError:
            if shred_file(path):
                shredded += 1
            else:
                failed.append(path)
    result = {"scanned": scanned, "reaped": reaped,
              "shredded": shredded, "failed": failed}
    if reaped or shredded or failed:
        logger.warning("plaintext sweep: %s", result)
    else:
        logger.info("plaintext sweep: clean (%d scanned)", scanned)
    return result
