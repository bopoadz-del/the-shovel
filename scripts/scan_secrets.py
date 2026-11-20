#!/usr/bin/env python3
"""Fail the build if secret material is in the working tree.

Seven hardcoded credential-shape rules, in descending order of certainty:

  1. AWS access key ids   -- `AKIA` + 16 uppercase alphanumerics. Unambiguous.
  2. Private key blocks   -- `-----BEGIN ... PRIVATE KEY`. Unambiguous.
  3. Long literals assigned to a secret-ish name -- 32+ chars of hex or
     base64 assigned to something called secret/token/key/password/etc.
  4. Hardcoded key-derivation constants on a non-test path -- a literal
     salt, IV or Fernet key. These are the quiet ones: they do not look
     like a password, and they make every derived key reproducible by
     anyone holding the source.
  5. Connection strings that carry a literal password --
     `scheme://user:password@host`. Rules 1-4 all key on a NAME or on a
     provider prefix, so a bot scan that reported 29 such lines across
     16 files in this tree produced zero hits here: a URL credential has
     neither. The password decides -- an interpolation, a redaction mask,
     a short stand-in or a well-known placeholder word is not credential
     material; a literal of eight or more characters is. See
     `PREDICATE_RULES`.
  6. Personal mailbox addresses -- an address at a consumer webmail
     provider (`@gmail.com`, `@outlook.com`, ...). Every rule above keys on
     a credential, and a credential can be rotated; an address cannot. It is
     identity, it is permanent in history, and a committed one is a privacy
     defect even when it is not a secret. The rule is keyed on the PROVIDER
     CLASS, never on a person: consumer webmail exists to serve individuals,
     so an address there is personal by construction, while a product or
     support contact uses the product's own domain and does not fire.
  7. Session identifiers committed as literals -- a session-ish field
     (`session_id` and its spellings) assigned an opaque value: a UUID, a
     32+ hex run, or a long token whose digit density only randomness
     produces. A session id is a bearer credential -- whoever holds it
     resumes the conversation -- and rule 3 walks past it, because the field
     is named `session`, not `token`. A human-readable scope label is a
     name, not a credential, and does not fire. See `PREDICATE_RULES`.

Plus an env-fed client-pattern denylist (``SECRET_SCAN_PATTERNS``): one
regex per line, from a repository secret. Those patterns are themselves
secret — a committed list of the client names you must not commit *is a
list of client names*. A match is reported as ``file:line (pattern #N)``
only; the matched text and the regex are never echoed.

FAIL-CLOSED: if ``SECRET_SCAN_PATTERNS`` is unset, empty, or comments-only,
this exits non-zero and never prints or concludes "clean". A scan that
inspected nothing because it was given nothing to look for has not found
nothing. A missing denylist is a red gate, not a skip.

Exit 0 if clean (config present AND no hits). Exit 2 if the denylist is
missing. Exit 1 if anything is found.

WHY IT ONLY SCANS TRACKED FILES: an untracked `.env` on a developer's disk
is where secrets are SUPPOSED to live. Flagging it would train everyone to
ignore this tool. What must never happen is a secret being committed, so
the scan follows `git ls-files` — exactly the set that gets pushed.

To allow a specific line (a documented placeholder, a test fixture key),
put `pragma: allowlist secret` in a comment on that line. Deliberate,
visible, and greppable — unlike a path exclusion that silently widens.
The allowlist applies to credential-shape rules only, not to the env-fed
client-pattern denylist.

Usage:
    python scripts/scan_secrets.py            # scan tracked files
    python scripts/scan_secrets.py --paths a b
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ENV_VAR = "SECRET_SCAN_PATTERNS"

ALLOW_MARKER = "pragma: allowlist secret"

# Binary and vendored paths that would produce noise, not findings.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".xz",
    ".woff", ".woff2", ".ttf", ".eot", ".onnx", ".bin", ".pyc", ".so",
    ".dll", ".dylib", ".xlsx", ".xls", ".docx", ".ifc", ".mp4", ".wav",
}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
SKIP_DIRS = ("node_modules/", ".venv/", "dist/", "build/", ".worktrees/",
             ".claude/")

SECRET_NAME = r"(?i)\b\w*(secret|token|api[_-]?key|password|passwd|pwd|credential|private[_-]?key)\w*\b"
# 32+ chars of hex, or of base64 alphabet. Quoted literal only — an
# os.getenv() call has no literal to leak.
LONG_HEX = r"[0-9a-fA-F]{32,}"
LONG_B64 = r"[A-Za-z0-9+/_-]{32,}={0,2}"

RULES = [
    (
        "aws-access-key-id",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "AWS access key id",
    ),
    (
        "private-key-block",
        re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
        "private key block",
    ),
    (
        # Provider-issued credentials carry their own prefix. These need NO
        # variable name and NO quotes to be certain, which is what makes them
        # worth their own rule: the first version of this scanner keyed only
        # on the assigned-name rule below and walked straight past a live
        # Render key written as `TOK=rnd_...` in a runbook.
        "provider-credential",
        re.compile(
            r"\b("
            r"rnd_[A-Za-z0-9]{20,}"              # Render
            r"|sk-[A-Za-z0-9_-]{20,}"            # OpenAI and lookalikes
            r"|sk_live_[A-Za-z0-9]{20,}"         # Stripe live
            r"|gh[pousr]_[A-Za-z0-9]{30,}"       # GitHub token
            r"|github_pat_[A-Za-z0-9_]{30,}"     # GitHub fine-grained PAT
            r"|glpat-[A-Za-z0-9_-]{20,}"         # GitLab
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"     # Slack
            r"|AIza[0-9A-Za-z_-]{35}"            # Google API key
            r"|gsk_[A-Za-z0-9]{20,}"             # Groq
            r"|hf_[A-Za-z0-9]{30,}"              # Hugging Face
            r")\b"
        ),
        "provider-issued credential",
    ),
    (
        # Quotes are OPTIONAL here for the same reason: shell and dotenv
        # assignments are bare, and those are exactly the lines that end up
        # pasted into a runbook.
        "assigned-long-literal",
        re.compile(
            SECRET_NAME + r"\s*[:=]\s*[\"']?(" + LONG_HEX + r"|" + LONG_B64 + r")[\"']?"
        ),
        "long literal assigned to a secret-named field",
    ),
    (
        # A committed personal address is the one privacy defect in this
        # scanner that no rotation can undo: the commit is forever. This rule
        # was added because a bot scan of this tree reported addresses as a
        # pattern class and the reader had to adjudicate them BY EYE -- no
        # rule here fired on an address at all, so the next one would have
        # landed unread. It fires on the provider class, not on any person, so
        # a product, support or documentation address at the project's own
        # domain is not a finding. The trailing lookahead is what makes it a
        # provider and not a prefix: `@gmail.com.evil` is a different domain
        # and does not fire. No test-path gate: a fixture address in a test
        # is still published, so deliberate ones carry the allowlist marker
        # like every other fixture in tests/test_scan_secrets.py.
        "consumer-mailbox-address",
        re.compile(
            r"[\w.+-]+@(?:"
            r"gmail\.com|googlemail\.com"
            r"|outlook\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|hotmail\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|live\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|yahoo\.[a-z]{2,3}(?:\.[a-z]{2})?|ymail\.com|rocketmail\.com"
            r"|icloud\.com|me\.com|mac\.com"
            r"|proton\.me|protonmail\.com"
            r"|aol\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|gmx\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|mail\.ru|yandex\.[a-z]{2,3}(?:\.[a-z]{2})?"
            r"|qq\.com|163\.com|126\.com"
            r")(?![A-Za-z0-9.-])",
            re.I,
        ),
        "personal address at a consumer mailbox provider",
    ),
]

# Rule 4 — key-derivation material. Only meaningful outside tests, where a
# fixed salt is a legitimate way to make a fixture deterministic.
KDF_RULES = [
    (
        "hardcoded-salt",
        re.compile(r"(?i)\b(salt|iv|nonce)\s*[:=]\s*b?[\"'][^\"']{8,}[\"']"),
        "hardcoded salt/IV/nonce",
    ),
    (
        "hardcoded-fernet-key",
        re.compile(r"Fernet\s*\(\s*[\"'][A-Za-z0-9+/_-]{40,}={0,2}[\"']"),
        "Fernet key built from a literal",
    ),
    (
        "hardcoded-kdf-password",
        re.compile(
            r"(?i)\b(PBKDF2HMAC|pbkdf2_hmac|scrypt|bcrypt\.hashpw)\b[^\n]*"
            r"[\"'][^\"']{8,}[\"']"
        ),
        "key derivation seeded from a literal",
    ),
]

# Rule 5 — a connection string carrying a literal password. The regex finds the
# shape; `_accepts_password` decides whether the password is credential
# material, because regex alone cannot tell `:${DB_PASSWORD}@` from a real one,
# and a rule that flagged every fixture URL would be switched off within a
# week. Eight characters is the floor: shorter literals in this tree are
# redaction stand-ins (`:x@`, `:pw@`), not passwords. This rule has NO
# non-test gate — a connection string in a test is still published, so the
# deliberate fixtures carry the allowlist marker instead of a path exclusion.
URL_CREDENTIAL = re.compile(
    r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:(?P<password>[^/\s@]+)@[^\s/]+", re.I
)
#: A password that IS one of these is a stand-in, not a secret. Whole-value
#: match only: a real password merely containing "test" still fails the gate.
PLACEHOLDER_PASSWORDS = frozenset({
    "password", "passwd", "pass", "pwd", "secret", "changeme", "change_me",
    "changeit", "example", "test", "testpass", "postgres", "postgresql",
    "mysql", "redis", "mongo", "mongodb", "user", "username", "admin",
    "root", "demo", "placeholder", "yourpassword", "your_password",
    "password123", "development", "local", "guest", "dummy", "sample",
    "default", "password1",
})
#: `$VAR`, `${...}`, `%(var)s`, `<password>`, `{{ var }}`, `[password]` — any
#: of these means the value is supplied at deploy time, not committed.
PASSWORD_REFERENCE = re.compile(r"[$%]|<|\{\{|\}|\[|\]")
#: `***`, `xxx`, `...` — a redaction, and its length proves nothing.
PASSWORD_MASK = re.compile(r"^(?:[*xX._\-]){3,}$")
MIN_LITERAL_PASSWORD = 8


def _accepts_password(match: re.Match[str]) -> bool:
    """True when the captured password looks like committed credential material."""
    password = match.group("password")
    if PASSWORD_REFERENCE.search(password):
        return False
    if PASSWORD_MASK.match(password):
        return False
    if len(password) < MIN_LITERAL_PASSWORD:
        return False
    return password.lower() not in PLACEHOLDER_PASSWORDS


# Rule 6 -- a session identifier committed as a literal. A session id is a
# bearer credential: whoever holds it resumes the conversation. Nothing above
# fires on one -- the assigned-name rule keys on names like token/secret and
# the field here is called session, and a session id carries no provider
# prefix -- so a tree could carry a live one and still scan clean. The value
# test is what keeps the rule honest. A UUID or a 32+ hex run is an identifier
# by construction. Anything else has to look like randomness rather than a
# label: 24+ characters, letters and digits, at a digit density the usual
# encodings produce (~1 in 3 to 1 in 6) and a readable scope name does not
# reach. A label long enough to look like a token can still be marked on its
# own line with the allowlist marker, which is visible and greppable.
SESSION_FIELD = r"(?i)session[_-]?id\b"
SESSION_ID = re.compile(
    SESSION_FIELD + r"\s*[\"']?\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9_\-+/=]{16,})"
)
_UUID_LITERAL = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_HEX_LITERAL = re.compile(r"[0-9a-fA-F]{32,}")
MIN_OPAQUE_SESSION = 24
MIN_OPAQUE_DISTINCT = 12
#: A random alphanumeric token carries a digit roughly one character in six to
#: one in three. A readable scope label that happens to end in a number carries
#: one in twelve or fewer. The ratio, not the presence of a digit, is what
#: separates an opaque identifier from a name.
MAX_LABEL_DIGIT_DENSITY = 1 / 12


def _accepts_session_id(match: re.Match[str]) -> bool:
    """True when the captured session value is an opaque identifier."""
    value = match.group("value")
    if _UUID_LITERAL.fullmatch(value) or _HEX_LITERAL.fullmatch(value):
        return True
    if len(value) < MIN_OPAQUE_SESSION:
        return False
    if not any(character.isalpha() for character in value):
        return False
    digits = sum(character.isdigit() for character in value)
    if digits / len(value) <= MAX_LABEL_DIGIT_DENSITY:
        return False
    return len(set(value)) >= MIN_OPAQUE_DISTINCT


#: Rules whose regex alone cannot decide an answer: the match must also pass
#: the rule's predicate on the captured value.
PREDICATE_RULES = [
    (
        "url-credential",
        URL_CREDENTIAL,
        "literal password inside a connection string",
        _accepts_password,
    ),
    (
        "session-identifier",
        SESSION_ID,
        "session id committed as a literal",
        _accepts_session_id,
    ),
]


def is_test_path(rel: str) -> bool:
    return rel.startswith("tests/") or "/tests/" in rel or "/test_" in rel \
        or Path(rel).name.startswith("test_") or rel.startswith("scripts/audit_")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def should_scan(rel: str) -> bool:
    if any(rel.startswith(d) or f"/{d}" in rel for d in SKIP_DIRS):
        return False
    p = Path(rel)
    return p.suffix.lower() not in SKIP_SUFFIXES and p.name not in SKIP_NAMES


def load_client_patterns(env: dict[str, str] | None = None) -> list[re.Pattern[str]]:
    """Compile the env-fed client-pattern denylist.

    Raises ``ValueError`` when none are configured -- fail-closed. A scan
    that finds nothing because it was given nothing to look for is not a
    clean scan, and reporting it as one is the silent-failure class this
    gate exists to catch.
    """
    source = os.environ if env is None else env
    raw = (source.get(ENV_VAR) or "").strip()
    if not raw:
        raise ValueError(
            f"{ENV_VAR} is empty or unset. This scan refuses to report a clean "
            "result it did not earn: with no patterns configured it inspected "
            "nothing, and calling that 'no secret material found' is the "
            "silent failure this gate exists to catch. Set the repository "
            "secret."
        )
    patterns = []
    for index, line in enumerate(raw.replace("\\n", "\n").splitlines()):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            patterns.append(re.compile(text, re.IGNORECASE))
        except re.error as exc:
            # The bad pattern is NOT echoed: it is a client pattern.
            raise ValueError(
                f"pattern {index} in {ENV_VAR} is not a valid regex "
                f"({exc.msg}). The pattern itself is not printed -- it is "
                "the secret."
            ) from exc
    if not patterns:
        raise ValueError(
            f"{ENV_VAR} contained no usable patterns (only blanks or comments)."
        )
    return patterns


def scan_client_text(
    text: str, patterns: list[re.Pattern[str]]
) -> list[tuple[int, int]]:
    """Return ``(line_number, pattern_index)`` for every client-pattern hit.

    Deliberately returns no matched text. Echoing a hit would write the
    client's content into a CI log, which is the leak this scan exists to
    prevent.
    """
    hits: list[tuple[int, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        forms = (line, decode_escapes(line))
        for index, pattern in enumerate(patterns):
            if any(pattern.search(form) for form in forms):
                hits.append((line_number, index))
    return hits


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")


def decode_escapes(line: str) -> str:
    r"""Spell out ``\uXXXX`` / ``\UXXXXXXXX`` escapes as the characters they name.

    A denylisted word written in a non-Latin script is just as present in a
    source file when it is written as escapes (a Python or JS string literal
    with ``\u....``) as when it is written literally, but a regex over the raw
    text only ever sees the backslashes. Matching the decoded form too closes
    that path for every pattern, whatever script it is in. An escape that
    names no character (beyond U+10FFFF) is left as written.
    """

    def _char(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1) or match.group(2), 16))
        except (ValueError, OverflowError):
            return match.group(0)

    return _ESCAPE.sub(_char, line)


def scan_client_paths(
    paths: list[Path], patterns: list[re.Pattern[str]]
) -> list[tuple[str, int, int]]:
    findings: list[tuple[str, int, int]] = []
    for path in paths:
        rel = str(path).replace("\\", "/")
        if not should_scan(rel):
            continue
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or binary; nothing a text regex can judge
        findings.extend(
            (rel, line_number, index)
            for line_number, index in scan_client_text(text, patterns)
        )
    return findings


def _resolve_scan_paths(rel_paths: list[str] | None) -> list[str]:
    if rel_paths is not None:
        return rel_paths
    return tracked_files()


def scan(rel_paths: list[str] | None = None) -> list[str]:
    findings: list[str] = []
    for rel in _resolve_scan_paths(rel_paths):
        if not should_scan(rel):
            continue
        path = Path(rel)
        if not path.exists():          # staged deletion
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"WARN  could not read {rel}: {exc}", file=sys.stderr)
            continue

        rules = list(RULES)
        if not is_test_path(rel):
            rules += KDF_RULES

        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            reported = False
            for name, pattern, description in rules:
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}  [{name}] {description}")
                    reported = True
                    break
            if reported:
                continue
            for name, pattern, description, accepts in PREDICATE_RULES:
                match = pattern.search(line)
                if match is not None and accepts(match):
                    findings.append(f"{rel}:{lineno}  [{name}] {description}")
                    break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Credential-shape + env-fed client-pattern scan (fail-closed)."
    )
    parser.add_argument("--paths", nargs="*", type=Path)
    args = parser.parse_args()

    try:
        patterns = load_client_patterns()
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if args.paths is not None:
        path_list = list(args.paths)
        rel_paths = [str(p) for p in path_list]
    else:
        rel_paths = tracked_files()
        path_list = [Path(rel) for rel in rel_paths]

    findings = scan(rel_paths)
    client_findings = scan_client_paths(path_list, patterns)

    if findings or client_findings:
        if findings:
            print("SECRET MATERIAL FOUND:")
            for f in findings:
                print("  " + f)
            print(f"TOTAL: {len(findings)}")
            print(
                "\nIf one of these is a documented placeholder or a test fixture, "
                f"put `{ALLOW_MARKER}` in a comment on that line."
            )
        if client_findings:
            print(
                f"CLIENT CONTENT DETECTED -- {len(client_findings)} hit(s). "
                "The matched text is not printed here on purpose: echoing it "
                "would write the client's content into a CI log, which is the "
                "leak this scan exists to prevent. Open the file and line below.",
                file=sys.stderr,
            )
            for rel, line_number, index in client_findings:
                print(f"  {rel}:{line_number}  (pattern #{index})", file=sys.stderr)
        return 1

    print(
        f"NO SECRET MATERIAL IN TRACKED FILES. "
        f"client-pattern scan: clean ({len(patterns)} pattern(s), "
        f"{len(path_list)} file(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
