"""Tests for encryption at rest — app/core/file_crypto.py.

Roadmap V2 · Epic 6 — encryption-at-rest follow-up.

Encryption is OPT-IN: it is driven entirely by the DATA_ENCRYPTION_KEY env
var. With no key set, everything stays plaintext and behaves exactly as before
— these tests assert that backward-compatible default explicitly.
"""

import importlib
import os

import pytest
from cryptography.fernet import Fernet

from app.core import file_crypto


@pytest.fixture
def fresh_crypto():
    """Reload file_crypto so module-level env reads pick up monkeypatched vars."""
    importlib.reload(file_crypto)
    return file_crypto


# ── encryption_enabled() reflects the env var ───────────────────────────────

def test_encryption_disabled_when_key_unset(monkeypatch, fresh_crypto):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    fc = importlib.reload(fresh_crypto)
    assert fc.encryption_enabled() is False


def test_encryption_enabled_when_key_set(monkeypatch, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    assert fc.encryption_enabled() is True


# ── round-trip encrypt / decrypt ────────────────────────────────────────────

def test_encrypt_decrypt_round_trip(monkeypatch, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    data = b"%PDF-1.4 confidential client data \x00\x01\x02"
    token = fc.encrypt_bytes(data)
    assert token != data
    assert fc.decrypt_bytes(token) == data


def test_encrypt_is_noop_when_disabled(monkeypatch, fresh_crypto):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    fc = importlib.reload(fresh_crypto)
    data = b"plaintext stays plaintext"
    assert fc.encrypt_bytes(data) == data
    assert fc.decrypt_bytes(data) == data


# ── write_document encrypts iff enabled ─────────────────────────────────────

def test_write_document_encrypts_on_disk_when_enabled(monkeypatch, tmp_path, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    path = str(tmp_path / "doc.pdf")
    data = b"%PDF-1.4 secret"
    fc.write_document(path, data)
    on_disk = open(path, "rb").read()
    assert on_disk != data  # ciphertext, not plaintext
    assert fc.read_document(path) == data  # transparently decrypts


def test_write_document_stays_plaintext_when_disabled(monkeypatch, tmp_path, fresh_crypto):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    fc = importlib.reload(fresh_crypto)
    path = str(tmp_path / "doc.pdf")
    data = b"%PDF-1.4 plain"
    fc.write_document(path, data)
    assert open(path, "rb").read() == data  # plaintext on disk


# ── backward compatibility — legacy plaintext files keep working ────────────

def test_read_legacy_plaintext_with_key_set(monkeypatch, tmp_path, fresh_crypto):
    """A pre-existing unencrypted file must still read back as plaintext even
    after a key is configured (legacy files predate encryption)."""
    path = str(tmp_path / "legacy.pdf")
    data = b"%PDF-1.4 pre-existing unencrypted file"
    with open(path, "wb") as f:
        f.write(data)
    # Now enable encryption — the legacy file was written before the key existed.
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    assert fc.read_document(path) == data


def test_read_encrypted_file_with_key(monkeypatch, tmp_path, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    path = str(tmp_path / "enc.pdf")
    data = b"encrypted body"
    fc.write_document(path, data)
    assert fc.read_document(path) == data


def test_looks_encrypted_detection(monkeypatch, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    token = fc.encrypt_bytes(b"some data")
    assert fc.looks_encrypted(token) is True
    assert fc.looks_encrypted(b"%PDF-1.4 plain text") is False
    assert fc.looks_encrypted(b"") is False


# ── open_plaintext context manager ──────────────────────────────────────────

def test_open_plaintext_yields_original_path_when_plaintext(monkeypatch, tmp_path, fresh_crypto):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    fc = importlib.reload(fresh_crypto)
    path = str(tmp_path / "plain.txt")
    data = b"hello plaintext"
    fc.write_document(path, data)
    with fc.open_plaintext(path) as p:
        assert p == path  # no temp copy needed
        assert open(p, "rb").read() == data


def test_open_plaintext_decrypts_to_temp_and_cleans_up(monkeypatch, tmp_path, fresh_crypto):
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    path = str(tmp_path / "enc.txt")
    data = b"hello encrypted body"
    fc.write_document(path, data)
    with fc.open_plaintext(path) as p:
        assert p != path  # decrypted temp copy
        assert os.path.exists(p)
        assert open(p, "rb").read() == data
        temp_path = p
    # temp file is removed on exit
    assert not os.path.exists(temp_path)


def test_open_plaintext_handles_legacy_file_with_key(monkeypatch, tmp_path, fresh_crypto):
    """A legacy plaintext file opened with a key set yields its own path."""
    path = str(tmp_path / "legacy.txt")
    data = b"legacy plaintext under a key"
    with open(path, "wb") as f:
        f.write(data)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    fc = importlib.reload(fresh_crypto)
    with fc.open_plaintext(path) as p:
        assert p == path
        assert open(p, "rb").read() == data
