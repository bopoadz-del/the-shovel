"""The secret scanner has to catch the shapes that actually got committed.

Two live credentials were sitting in this repository when the scanner was
written — a Fork API key in a recovery runbook and a live deploy API key hardcoded in
a scheduled-task script. A third pair survived the FIRST version of the
scanner, because it was written as

    TOK=rnd_<32 letters and digits>

with no quotes and a variable name (`TOK`) that no secret-name pattern
matches. That near-miss is why the provider-prefix rule exists and why
quotes are optional, and it is pinned here so neither can regress.

These tests exercise the rules directly rather than the repo walk, so they
stay true regardless of what the tree contains.

The `# pragma: allowlist secret` markers below are not a workaround — they
are the scanner's own escape hatch, used on the only lines in the repository
where secret-shaped strings are the point. The moment this file was committed
it became tracked, and the scanner correctly flagged its own fixtures. That
is the mechanism working. Marking each line keeps them visible and greppable
rather than carving out a path exclusion that would silently widen later.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import random
import string

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets", Path(__file__).resolve().parents[1] / "scripts" / "scan_secrets.py"
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules["scan_secrets"] = scan_secrets
_SPEC.loader.exec_module(scan_secrets)


def _hits(line: str, *, test_path: bool = False):
    """Which rules fire on one line, mirroring scan()'s rule selection."""
    rules = list(scan_secrets.RULES)
    if not test_path:
        rules += scan_secrets.KDF_RULES
    names = [name for name, pattern, _ in rules if pattern.search(line)]
    for name, pattern, _, accepts in scan_secrets.URL_CREDENTIAL_RULES:
        match = pattern.search(line)
        if match is not None and accepts(match):
            names.append(name)
    return names


# ── the ones that were really in the tree ─────────────────────────────────

# Every vector is generated here, in the committed shape, so no credential --
# not even a revoked one -- is stored in the tree.
_RNG = random.Random(20261016)


def _fake(prefix: str, n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return prefix + "".join(_RNG.choices(alphabet, k=n))


_PROVIDER = _fake("rnd" + "_", 32)
_KEY_BLOCK = "-----BEGIN " + "{}PRIVATE KEY-----"

MUST_CATCH = [
    # The exact shape that slipped past the first version: bare, unquoted,
    # and assigned to a name no secret-name pattern matches.
    ("TOK=" + _PROVIDER, "provider-credential"),
    ("$RenderApiKey   = '" + _PROVIDER + "'", "provider-credential"),
    ('export FORK_API_KEY="' + _fake("", 43) + '"', "assigned-long-literal"),
    ("aws_access_key_id = " + _fake("AK" + "IA", 16, string.ascii_uppercase + string.digits), "aws-access-key-id"),
    (_KEY_BLOCK.format("RSA "), "private-key-block"),
    (_KEY_BLOCK.format(""), "private-key-block"),
    ("OPENAI_KEY=" + _fake("sk" + "-proj-", 32, string.ascii_lowercase + string.digits), "provider-credential"),
    ("GROQ=" + _fake("gsk" + "_", 30, string.ascii_lowercase + string.digits), "provider-credential"),
    ("hf_token: " + _fake("hf" + "_", 36, string.ascii_lowercase + string.digits), "provider-credential"),
    ("token = " + _fake("gh" + "p_", 36, string.ascii_lowercase + string.digits), "provider-credential"),
]


@pytest.mark.parametrize("line,expected_rule", MUST_CATCH)
def test_the_shapes_that_were_committed_are_caught(line, expected_rule):
    fired = _hits(line)
    assert expected_rule in fired, f"{line!r} matched {fired} — expected {expected_rule}"


def test_an_unquoted_assignment_is_not_a_blind_spot():
    """Dotenv and shell lines are bare, and those are the ones pasted into
    runbooks. The first version required quotes and missed them."""
    value = _fake("", 32, "0123456789abcdef")
    assert _hits("SECRET_KEY=" + value)
    assert _hits('SECRET_KEY="' + value + '"')


# ── things that must NOT fire, or nobody will run it ──────────────────────

MUST_NOT_CATCH = [
    'SECRET_KEY = os.getenv("SECRET_KEY")',
    'api_key = config["api_key"]',
    'FORK_API_KEY="<FORK_API_KEY -- take it from your .env>"',
    "$RenderApiKey   = $env:RENDER_API_KEY",
    "# Set RENDER_API_KEY in your environment before running this",
    'password = ""',
    "token: str | None = None",
    "--hash=sha256:9f0b1e6d4a2c8f3e5b7d9a1c3e5f7a9b1d3f5a7c9e1b3d5f7a9c1e3b5d7f9a1c",
    '"integrity": "sha512-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH=="',
    "commit 5f09f88a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e",
]


@pytest.mark.parametrize("line", MUST_NOT_CATCH)
def test_ordinary_lines_do_not_fire(line):
    """A scanner with false positives gets ignored, which is worse than
    no scanner at all."""
    assert not _hits(line), f"false positive on {line!r}: {_hits(line)}"


# ── the KDF rules, and why they are non-test-paths only ───────────────────

def test_a_hardcoded_salt_is_caught_in_production_code():
    """The quiet one: it does not look like a password, but it makes every
    derived key reproducible by anyone with the source."""
    assert "hardcoded-salt" in _hits('salt = b"fork-static-salt-v1"')


def test_the_same_salt_is_allowed_in_a_test():
    """A fixed salt is how you make a fixture deterministic."""
    assert "hardcoded-salt" not in _hits('salt = b"fork-static-salt-v1"', test_path=True)


def test_a_fernet_key_built_from_a_literal_is_caught():
    assert "hardcoded-fernet-key" in _hits(
        "f = Fernet('bXlzdXBlcnNlY3JldGZlcm5ldGtleXZhbHVlMTIzNDU2Nzg5MA==')"
    )


# ── rule 5: a connection string that carries a literal password ─────────────
#
# Rules 1-4 key on a name or a provider prefix. A URL credential has neither,
# which is why a bot scan reporting 29 such lines across 16 files produced zero
# hits here until this rule existed. The password is what decides, and it is
# generated here so no credential-shaped literal is stored in the tree.

_URL_PW = _fake("", 24)


def test_a_connection_string_with_a_literal_password_is_caught():
    assert "url-credential" in _hits(
        "DATABASE_URL=postgresql://svc:" + _URL_PW + "@db.internal:5432/app"
    )


def test_the_length_floor_is_eight_characters():
    """Exactly eight fires, seven does not: the floor is what separates a
    redaction stand-in from a password, so it is pinned on both sides."""
    eight = _fake("", scan_secrets.MIN_LITERAL_PASSWORD)
    seven = eight[:-1]
    assert "url-credential" in _hits("postgresql://svc:" + eight + "@db.internal/app")
    assert "url-credential" not in _hits("postgresql://svc:" + seven + "@db.internal/app")


@pytest.mark.parametrize(
    "password",
    [
        "${DB_PASSWORD}",
        "$DB_PASSWORD",
        "%(DB_PASSWORD)s",
        "<password>",
        "{{ db_password }}",
        "[password]",
        "********",
        "xxxxxxxx",
        "...",
        "password",
        "changeme",
        "postgres",
        "x",
        "pw",
    ],
)
def test_a_connection_string_without_credential_material_does_not_fire(password):
    """Interpolations, masks, redaction stand-ins and placeholder words are
    not credential material; firing on them trains people to ignore the gate."""
    assert "url-credential" not in _hits(
        "DATABASE_URL=postgresql://svc:" + password + "@db.internal:5432/app"
    )


@pytest.mark.parametrize("password", [_fake("", 24), _fake("pw-", 12)])
def test_the_walk_reports_a_connection_string_credential(tmp_path, monkeypatch, password):
    """The rule is wired into scan(), not merely defined beside it."""
    (tmp_path / "compose.yml").write_text(
        "DATABASE_URL=postgresql://svc:" + password + "@db.internal:5432/app\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_secrets, "tracked_files", lambda: ["compose.yml"])

    findings = scan_secrets.scan()
    assert len(findings) == 1
    assert "[url-credential]" in findings[0]
    assert password not in findings[0]


# ── the allowlist has to work, or people will delete the check ────────────

def test_an_allowlisted_line_is_ignored_by_the_walk(tmp_path, monkeypatch):
    marked = f"TOK={_PROVIDER}  # {scan_secrets.ALLOW_MARKER}"  # pragma: allowlist secret
    (tmp_path / "runbook.md").write_text(marked, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_secrets, "tracked_files", lambda: ["runbook.md"])

    assert scan_secrets.scan() == []


def test_the_walk_reports_an_unmarked_secret(tmp_path, monkeypatch):
    (tmp_path / "runbook.md").write_text(
        "TOK=" + _PROVIDER, encoding="utf-8"  # pragma: allowlist secret
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_secrets, "tracked_files", lambda: ["runbook.md"])

    findings = scan_secrets.scan()
    assert len(findings) == 1
    assert "runbook.md:1" in findings[0]


def test_the_tree_is_clean_right_now(monkeypatch):
    """The gate itself. If this fails, a secret was committed.

    The env-fed denylist must be present or the script refuses (fail-closed).
    The pattern here is a structural dummy that does not match this tree;
    live client patterns live in the repository secret, not in git.
    """
    import subprocess
    monkeypatch.setenv(scan_secrets.ENV_VAR, r"FORK_SCAN_DUMMY_\d{16}")
    result = subprocess.run(
        [sys.executable, "scripts/scan_secrets.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"secret material in the tree:\n{result.stdout}\n{result.stderr}"


# ── CI twin 2a: env-fed client-pattern denylist, fail-closed ───────────────
#
# Ported from The_Level's scan_client_patterns contract. Patterns arrive in
# SECRET_SCAN_PATTERNS. A missing denylist is a red gate, not a skip. Dummy
# tokens below are invented for this file; they are not battery strings,
# canaries, or live client figures.


@pytest.mark.parametrize("value", [None, "", "   ", "\n\n", "# only a comment"])
def test_unset_or_empty_denylist_refuses_to_report_clean(value):
    """(1) unset / empty env → non-zero. Never conclude 'clean'."""
    env = {} if value is None else {scan_secrets.ENV_VAR: value}
    with pytest.raises(ValueError) as excinfo:
        scan_secrets.load_client_patterns(env)
    message = str(excinfo.value).lower()
    assert "refuses" in message or "no usable patterns" in message
    assert "clean" not in message or "refuses to report a clean" in str(excinfo.value)


def test_unset_env_exits_nonzero_and_does_not_print_clean(monkeypatch, capsys):
    """(1) CLI: unset SECRET_SCAN_PATTERNS → exit 2, no 'clean' on stdout."""
    monkeypatch.delenv(scan_secrets.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.argv", ["scan_secrets.py", "--paths"])

    assert scan_secrets.main() == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    combined = captured.out + captured.err
    assert "NO SECRET MATERIAL" not in combined
    # Refusal may mention 'clean' as the conclusion it refuses to draw.
    # The earned-clean banner must not appear.
    assert "client-pattern scan: clean" not in combined


def test_dummy_token_in_fixture_exits_nonzero(tmp_path, capsys, monkeypatch):
    """(2) set env with a dummy token present in a fixture file → non-zero."""
    # Built at runtime so this tracked test file never contains a matching
    # literal (the tree-clean test uses the same structural pattern).
    dummy = "FORK_SCAN_DUMMY_" + ("0" * 15) + "1"
    fixture = tmp_path / "fixture.txt"
    fixture.write_text(f"neutral prose containing {dummy}\n", encoding="utf-8")
    monkeypatch.setenv(scan_secrets.ENV_VAR, r"FORK_SCAN_DUMMY_\d{16}")
    monkeypatch.setattr(
        "sys.argv", ["scan_secrets.py", "--paths", str(fixture)]
    )

    assert scan_secrets.main() == 1
    err = capsys.readouterr().err
    assert "fixture.txt:1" in err
    assert "pattern #0" in err
    assert dummy not in err, "the dummy token was echoed into the log"
    assert r"FORK_SCAN_DUMMY_\d{16}" not in err, "the secret pattern was echoed"


def test_set_env_with_no_match_exits_zero(tmp_path, capsys, monkeypatch):
    """(3) set env with no match → zero."""
    fixture = tmp_path / "clean.txt"
    fixture.write_text("bicycle wheels\n", encoding="utf-8")
    monkeypatch.setenv(scan_secrets.ENV_VAR, r"FORK_SCAN_DUMMY_\d{16}")
    monkeypatch.setattr(
        "sys.argv", ["scan_secrets.py", "--paths", str(fixture)]
    )

    assert scan_secrets.main() == 0
    out = capsys.readouterr().out
    assert "clean" in out
    assert "1 pattern(s)" in out
    assert "1 file(s)" in out


def test_escaped_newlines_are_accepted_as_separators():
    """A GitHub secret is often pasted as one line with \\n between entries."""
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: r"alpha\nbeta"}
    )
    assert len(patterns) == 2


def test_a_broken_pattern_is_reported_without_printing_the_pattern():
    with pytest.raises(ValueError) as excinfo:
        scan_secrets.load_client_patterns(
            {scan_secrets.ENV_VAR: "unclosed(group"}
        )
    message = str(excinfo.value)
    assert "not a valid regex" in message
    assert "unclosed(group" not in message, "the secret pattern was echoed"


@pytest.mark.parametrize(
    "text,expected",
    [
        (r'name = "Document from \u0645\u0631\u0633\u0644"' + "\nplain line\n", [(1, 0)]),
        (r'x = "\U0001F1F8 \u007a\u0065bra"' + "\n", [(1, 1)]),
        ("plain line\nanother plain line\n", []),
    ],
)
def test_a_denylisted_word_written_as_unicode_escapes_is_caught(text, expected):
    r"""A non-Latin word in a string literal is often written as \u escapes.

    The raw line holds only backslashes and hex, so a pattern in the word's own
    script never matched it. The scan matches the decoded form as well.
    """
    patterns = scan_secrets.load_client_patterns(
        {scan_secrets.ENV_VAR: "\u0645\u0631\u0633\u0644\nzebra"}
    )
    assert scan_secrets.scan_client_text(text, patterns) == expected


@pytest.mark.parametrize(
    "raw,decoded",
    [
        (r"a \UFFFFFFFF b", r"a \UFFFFFFFF b"),
        (r"\u0041\U00000042", "AB"),
        ("no escapes", "no escapes"),
    ],
)
def test_an_escape_that_names_no_character_is_left_as_written(raw, decoded):
    assert scan_secrets.decode_escapes(raw) == decoded


# -- rule 6: a personal mailbox address, never a product contact -------------
#
# A committed address cannot be rotated the way a credential can: the commit
# is permanent. The rule is keyed on the PROVIDER CLASS -- consumer webmail
# exists to serve individuals -- so a person's address is a finding while a
# product or support contact at the project's own domain is not. Every address
# below is invented for this file; the allowlist markers keep the tree-clean
# test green, exactly as the file's docstring describes.


@pytest.mark.parametrize(
    "address",
    [
        "slm.lane.probe@gmail.com",        # pragma: allowlist secret
        "slm.probe.two@outlook.com",       # pragma: allowlist secret
        "slm.probe.three@proton.me",       # pragma: allowlist secret
        "slm.probe.four@yahoo.co.uk",      # pragma: allowlist secret
    ],
)
def test_a_consumer_mailbox_address_is_caught(address):
    assert "consumer-mailbox-address" in _hits(f"contact: {address}")


@pytest.mark.parametrize(
    "line",
    [
        "support@example.com",
        "ops@theshovel.ai",
        "noreply@example.test",
        "service@example-project.iam.gserviceaccount.com",
        "a@notgmail.com",
        "x@sub.example.com",
        "a@gmail.com.evil",
    ],
)
def test_documentation_and_product_addresses_do_not_fire(line):
    """The provider class, not the word 'email', is what fires. A product
    contact at its own domain is public by design and must stay clean."""
    assert "consumer-mailbox-address" not in _hits(line)


def test_the_walk_reports_a_personal_address(tmp_path, monkeypatch):
    """The rule is wired into scan(), not merely defined beside it."""
    (tmp_path / "notes.md").write_text(
        "reach the team at slm.lane.probe@gmail.com\n",  # pragma: allowlist secret
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_secrets, "tracked_files", lambda: ["notes.md"])

    findings = scan_secrets.scan()
    assert len(findings) == 1
    assert "notes.md:1" in findings[0]
    assert "[consumer-mailbox-address]" in findings[0]


def test_a_deliberate_fixture_address_can_be_allowlisted(tmp_path, monkeypatch):
    (tmp_path / "fixture.md").write_text(
        "example fixture: slm.probe.three@proton.me  # pragma: allowlist secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_secrets, "tracked_files", lambda: ["fixture.md"])

    assert scan_secrets.scan() == []
