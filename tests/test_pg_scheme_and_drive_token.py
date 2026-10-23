"""Regression tests for the 2026-07-11 prod hotfix:

1. Raw ``create_engine(DATABASE_URL)`` calls crashed with
   ``ModuleNotFoundError: No module named 'psycopg2'`` once the app pointed at
   Postgres, because a bare ``postgresql://`` URL routes to psycopg2 (not
   installed; psycopg v3 only). ``to_psycopg_url`` normalizes the scheme.
2. A Google Drive token encrypted with a rotated ``DATA_ENCRYPTION_KEY`` made
   every ``/v1/admin/drive/*`` call 500. ``load_token`` now degrades to
   not-connected and clears the dead file.
"""
from cryptography.fernet import Fernet

from app.core.db import to_psycopg_url


def test_to_psycopg_url_rewrites_bare_postgres_schemes():
    assert to_psycopg_url("postgresql://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert to_psycopg_url("postgres://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    # Already-correct and non-postgres URLs pass through unchanged (idempotent).
    assert to_psycopg_url("postgresql+psycopg://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert to_psycopg_url("sqlite:///x.db") == "sqlite:///x.db"


def test_load_token_returns_none_and_clears_undecryptable_token(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())

    from app.core import drive_auth

    drive_auth.save_token("user-1", {"refresh_token": "secret", "access_token": "a"})
    path = drive_auth._token_path("user-1")
    assert path.exists()

    # Rotate the key: the stored token can no longer be decrypted.
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())

    # Must not raise; treats the user as not-connected and removes the dead file.
    assert drive_auth.load_token("user-1") is None
    assert not path.exists()
