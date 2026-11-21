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

A denylisted pattern is compared against three views of every line, because
the pattern is a written form and a person has many: the literal line, the
escape-decoded line, and a folded line in which case, accents, invisibles,
and the choice of separator are normalised away. A phrase wrapped across a
line break is the same phrase. A denylist entry that is a written proper
name -- every word capitalised -- is additionally matched as a name is
written: surname first, in mixed case or in capitals, in any script that
marks case, with up to two words -- a middle name and an initial -- between
the name's own. See ``fold_for_match``, ``name_pattern_words`` and
``scan_client_document``.

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
import unicodedata
from bisect import bisect_right
from pathlib import Path

import scan_text

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

    Two views of each line are matched, because a denylisted word can be
    written in two ways and each defeats the other:

    * the raw line -- the literal spelling;
    * the escape-decoded line -- ``\\u0627`` spells the word without its
      letters ever appearing (see ``decode_escapes``).

    Folding lives in ``scan_client_document``, which is what the file walk
    uses: a folded match has to be checked against the text as written before
    it can be called a hit, and that check belongs beside the folding.
    """
    hits: list[tuple[int, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        forms = (line, decode_escapes(line))
        for index, pattern in enumerate(patterns):
            if any(pattern.search(form) for form in forms):
                hits.append((line_number, index))
    return hits


# Characters that carry no sound and no space of their own: a zero-width
# space, joiner, non-joiner, word joiner, byte-order mark, or soft hyphen.
# They are invisible in an editor, so a word split by one reads normally to a
# human and as a different string to a regex.
_INVISIBLE = dict.fromkeys(
    map(ord, "\u00ad\u180e\u200b\u200c\u200d\u2060\ufeff"), None
)

# Every kind of whitespace that is not the ASCII space, folded to one space:
# tab, newline, vertical tab, form feed, carriage return, NEL, NBSP, the whole
# Unicode space-separator run (en/em/thin/hair space, ideographic space), and
# the dash a writer reaches for when joining two words.
#
# NOT folded: underscore, period, middle dot, slash and backslash. Those are
# not the punctuation of prose, they are the joiners of code: ``scan_secrets``
# and ``docs/notes`` are one identifier each, and folding them into separate
# words is what made a client-pattern scan report ordinary file names. A
# name is never spelled with an underscore between its words, and a gate that
# says it is manufactures hits in every tree that has underscores in it.
_ASCII_SEPARATORS = dict.fromkeys(
    map(ord, "\t\n\x0b\x0c\r-"), " "
)
_UNICODE_SEPARATORS = dict.fromkeys(
    map(
        ord,
        "\t\n\x0b\x0c\r\x85\u00a0\u1680\u2000\u2001\u2002\u2003\u2004"
        "\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
        "-\u2010\u2011\u2012\u2013\u2014",
    ),
    " ",
)

# Letters in another script that are drawn the same as a Latin letter in
# ordinary type. Writing one of them inside a denylisted name is the cheapest
# way to keep the name while defeating a literal comparison -- and it does
# defeat it, in every word order, exact spelling included. The property this
# encodes is shape, not vocabulary: a Latin name written in any of these
# letters is the same name to a reader, so the gate reads it the same way.
# The table is the high-frequency confusables of common UI fonts, not the
# exhaustive Unicode set; case needs no entries because the fold case-folds
# before this table is applied, and Cyrillic and Greek capitals fold onto
# these lower-case forms.
_LOOKALIKE_LETTERS = {
    # Cyrillic
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u0455": "s",
    "\u0501": "d",
    "\u051b": "q",
    "\u051d": "w",
    "\u04bb": "h",
    "\u04cf": "l",
    "\u043a": "k",
    "\u043c": "m",
    "\u043d": "h",
    "\u043f": "n",
    "\u0438": "u",
    "\u0448": "w",
    "\u044c": "b",
    "\u0442": "t",
    # Greek
    "\u03b1": "a",
    "\u03bf": "o",
    "\u03bd": "v",
    "\u03c1": "p",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u03b5": "e",
    "\u03b7": "n",
    "\u03c9": "w",
    "\u03b3": "y",
    "\u03c3": "o",
    "\u03c4": "t",
    "\u03bc": "u",
    # A Latin letter that is not the one it looks like
    "\u0261": "g",
}
_CONFUSABLE = {ord(source): latin for source, latin in _LOOKALIKE_LETTERS.items()}

# Any of the invisible characters, so the second folding is only built for a
# line that actually has one. Most lines do not, and most of the tree is
# ASCII: the fast paths keep the added cost of the folded comparison small
# enough that the scan stays a lint step rather than a coffee break.
_INVISIBLE_RE = re.compile("[\u00ad\u180e\u200b\u200c\u200d\u2060\ufeff]")


def fold_for_match(line: str, *, join_invisible: bool = False) -> str:
    """Fold one line to the form a denylist pattern is compared against.

    Compatibility-normalise (so full-width letters and digits become their
    ASCII forms), remove invisible characters, decompose and strip combining
    marks (so ``\u00e9`` and ``e`` + accent compare equal), turn every
    non-ASCII-space whitespace and every joining punctuation mark into a
    single space, and case-fold. What survives is the word sequence, which is
    what makes a name a name; what is dropped is presentation.

    Finally, each letter from another script that is drawn as a Latin letter
    is replaced by that letter: a name protected by a Cyrillic ``a`` or a
    Greek ``o`` is the same name, and a comparison that lets one substituted
    letter hide it protects nothing.

    ``join_invisible`` decides what an invisible character becomes. Removed
    (the default) it splits two words a hidden character was standing
    between; joined it fuses the halves of one word a hidden character was
    pushed into. Both are real spellings of the same name, so
    ``fold_views`` produces both and a match on either is a match.
    """
    if line.isascii():
        # No compatibility variants, no combining marks and no invisible
        # characters exist below U+0080: the fold is separators and case.
        return " ".join(line.translate(_ASCII_SEPARATORS).split()).casefold()
    folded = unicodedata.normalize("NFKC", line).translate(
        dict.fromkeys(_INVISIBLE, "" if join_invisible else " ")
    )
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.translate(_UNICODE_SEPARATORS)
    folded = " ".join(folded.split()).casefold()
    return folded.translate(_CONFUSABLE)


def fold_views(line: str) -> tuple[str, ...]:
    """Every folding of one line worth comparing a pattern against."""
    split_view = fold_for_match(line)
    if not _INVISIBLE_RE.search(line):
        return (split_view,)      # the two foldings would be identical
    joined_view = fold_for_match(line, join_invisible=True)
    if joined_view == split_view:
        return (split_view,)
    return (split_view, joined_view)


# A character that makes a run of text an identifier rather than prose: a
# letter, a digit, or one of the two joining characters code uses to build one
# name out of several words. A denylisted phrase sitting inside such a run --
# the middle of ``build_rate_card``, or one segment of ``some/path/name`` -- is
# code naming a thing, not prose naming a person. Matching those was how a
# folded comparison turned ordinary files of this repository into client
# content hits, and a gate that reports code as a leak is a gate that gets
# switched off.
_IDENTIFIER_CHARS = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
)


def fold_with_map(
    raw: str, *, join_invisible: bool = False
) -> tuple[str, list[int]]:
    """``fold_for_match`` character by character, remembering each origin.

    The folded text alone cannot say whether a match sat in the middle of an
    identifier, because folding is precisely what erased the identifier's own
    separators. So every folded character keeps the position in ``raw`` it
    came from, and a caller can look at the text as written on either side of
    a match.

    ``join_invisible`` mirrors ``fold_for_match``: an invisible character
    either breaks two words apart or is deleted from inside one, and which of
    those the author meant cannot be known, so both readings are offered.
    """
    folded: list[str] = []
    index: list[int] = []
    spaced = False
    for position, char in enumerate(raw):
        code = ord(char)
        if code in _INVISIBLE:
            if join_invisible:
                continue
            spaced = bool(folded)
            continue
        if char.isspace() or code in _ASCII_SEPARATORS or code in _UNICODE_SEPARATORS:
            spaced = bool(folded)      # a leading space is not a word break
            continue
        piece = fold_for_match(char)
        if not piece:
            continue
        if spaced:
            folded.append(" ")
            index.append(position)
            spaced = False
        for out in piece:
            folded.append(out)
            index.append(position)
    return "".join(folded), index


def sits_in_identifier(raw: str, index: list[int], start: int, end: int) -> bool:
    """True when a folded match is a piece of a longer identifier.

    Only the two edges of the folded span are examined, so a name joined by a
    hyphen *between* its words (``First-Last``, surrounded by spaces) is
    untouched, while the same words inside ``rate_card`` or ``a/b`` are not.
    A dot, slash or backslash counts only when it is welded to a word
    character -- ``card.py`` is a file, ``Vantrey.`` is the end of a sentence.
    An underscore anywhere inside the span is code's own joiner: no name is
    spelled with one, so a match that spans it is an identifier.
    """
    first = index[start]
    last = index[end - 1]
    if "_" in raw[first:last + 1]:
        return True      # an underscore between two words is code, not a name
    left = raw[first - 1] if first > 0 else ""
    right = raw[last + 1] if last + 1 < len(raw) else ""
    if left in _IDENTIFIER_CHARS or right in _IDENTIFIER_CHARS:
        return True
    if left in "./\\":
        before = raw[first - 2] if first - 2 >= 0 else ""
        if before.isalnum():
            return True
    if right in "./\\":
        after = raw[last + 2] if last + 2 < len(raw) else ""
        if after.isalnum():
            return True
    return False


def _blocks(text: str) -> list[tuple[str, list[tuple[int, int]]]]:
    """Group consecutive non-blank lines, with each line's place in the raw text.

    A blank line ends a block, so two paragraphs are never joined: the line
    break recovered here is the one a word processor inserts inside a single
    sentence, not the structural break between sections. Each entry carries
    the block's own raw text (lines re-joined with a newline, so offsets map
    back to the document) and, per line, its (offset in that text, number).
    """
    blocks: list[tuple[str, list[tuple[int, int]]]] = []
    current: list[str] = []
    numbers: list[tuple[int, int]] = []
    offset = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            numbers.append((offset, line_number))
            current.append(line)
            offset += len(line) + 1
            continue
        if current:
            blocks.append(("\n".join(current), numbers))
            current, numbers, offset = [], [], 0
    if current:
        blocks.append(("\n".join(current), numbers))
    return blocks


# A markdown table row prints one record across cells, and a cell boundary is
# layout, not a word break: ``| Surname | Given |`` is the same name a byline
# prints on one line, split by the column rule. The bar cannot join
# ``_NAME_SEPARATORS`` because in code and prose it separates a pipeline or an
# alternative, where no one writes a name. So a row is read a second time with
# its bars read as the spaces between cells -- and that reading is one line,
# never a block, so two rows are never joined across the line break between
# them.
_TABLE_BAR = "|"


def _table_row_phrase(line: str) -> str | None:
    """The line read across its cells as one phrase, or ``None`` for no row.

    A row is a line whose first and last non-space characters are bars and
    which carries at least two cells. That is the shape a markdown table is
    written in, and it is not the shape of a shell pipeline or a boolean
    expression -- those are ordinary code lines, where a written name is not a
    thing. The phrase returned is built from this line alone: the caller never
    joins two rows, so a name cannot be assembled out of two records.
    """
    stripped = line.strip()
    if len(stripped) < 3:
        return None
    if not (stripped.startswith(_TABLE_BAR) and stripped.endswith(_TABLE_BAR)):
        return None
    if stripped.count(_TABLE_BAR) < 3:
        return None
    return line.replace(_TABLE_BAR, " ")


# Regex syntax in a denylist entry means the author wrote a machine pattern,
# not a fixed phrase: there is no literal text to fold, and escaping a regex
# would destroy it. Such an entry is compared to folded text exactly as
# written, which is what it was written for.
_REGEX_SYNTAX = re.compile(r"[\\\[\]()|?*+{}^$]")


def folded_matcher(pattern: re.Pattern[str]) -> re.Pattern[str]:
    """The matcher to use against folded text, for one denylist entry.

    Both sides of that comparison have to be folded, or a pattern written with
    an underscore could never match a line where the same words are separated
    by a space (which is the whole case this is for). A plain phrase is
    therefore folded here, word-bounded, and escaped; anything else is left
    exactly as its author wrote it.
    """
    if _REGEX_SYNTAX.search(pattern.pattern):
        return pattern
    folded = fold_for_match(pattern.pattern)
    if not folded:
        return pattern
    return re.compile(r"(?<!\w)" + re.escape(folded) + r"(?!\w)", re.IGNORECASE)


# A written personal name is a run of capitalised words: ``Given Surname``,
# ``Surname Given``, ``Given A. Surname``, and the same name set in capitals
# for a title block. That is a property of names, not a guess: prose
# mid-sentence is lower case, code is an identifier, and only a proper noun is
# spelled with a capital at every word. So the rule below applies to a denylist
# entry only when EVERY word of it is Titlecase -- an entry that spells a name.
# The document side is judged by the same shape the entry has, capitals
# included: whether a page sets a name in mixed case or in capitals is a
# typographic choice, not a different name. An order-independent, gap-tolerant
# match over ordinary words was tried and withdrawn because it matched word
# sets across a sentence; this one never forgets which word is which: it reads
# the exact sequence, forwards or backwards, with at most one interior
# insertion.
_NAME_SEPARATORS = frozenset(
    " \t\n\x0b\x0c\r\x85\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000-\u2010\u2011\u2012"
    "\u2013\u2014"
)
# Punctuation a word may be wrapped in by a sentence and still be that word.
# An interior apostrophe is NOT here: it makes the token not a name word at
# all, which breaks the run instead of guessing at the apostrophe's meaning.
_EDGE_PUNCTUATION = "\"'`.,;:!?()[]{}<>*_/\\|~#@$%^&+=…“”‘’\u060c\u3001\uff0c"
# The comma, as each script that has one spells it: the inverted listing form
# ``Surname, Given`` is how a byline, an index entry and a table cell write a
# name. A comma separates entries; it does not end one, so it does not end a
# run of name words. Every other closing mark does. The Arabic, ideographic
# and fullwidth commas are the same mark in another script and are stripped as
# edge punctuation for that reason.
_ENTRY_COMMA = frozenset(",\u060c\u3001\uff0c")
# The shape of a name word is a property of the alphabet, not of ASCII: a
# name written in Cyrillic, Greek or Armenian is spelled with a capital at
# every word exactly as a Latin one is, and a rule that only recognises the
# Latin range gives such an entry no order-independent match at all -- only
# the spelling the list happened to be typed in. So the shape is asked of the
# characters themselves. A script without case (Han, Kana, Hangul syllables)
# has no capital to read as a name word, which is stated as a limit of the
# rule rather than papered over; the literal comparison still covers it.


def _is_titlecase_word(word: str) -> bool:
    """True for a leading capital and lower-case letters, in any script.

    Letters only: a word carrying a digit is an identifier, and an identifier
    is never a written name however it is capitalised.
    """
    return (
        len(word) >= 2
        and word[0].isupper()
        and word[1:].isalpha()
        and word[1:].islower()
    )


def _is_capitalised_word(word: str) -> bool:
    """True for a word spelled entirely in capitals, in any script.

    A single capital letter is the one-letter case of it, so this predicate
    answers for an initial too.
    """
    return bool(word) and word.isalpha() and word.isupper()


def _is_initial(word: str) -> bool:
    """True for one capital letter, in any script: the ``A.`` of a middle name."""
    return len(word) == 1 and word.isalpha() and word.isupper()


def _is_name_word(word: str) -> bool:
    """True for a capitalised word: Titlecase, all capitals, or one initial.

    A name is spelled with a capital at every word. The same name is also
    printed in capitals -- on a drawing title block, a table header, a
    signature line -- and there every letter carries the capital instead, so
    the run a reader sees is still a name. A lower-case word is prose, and an
    identifier is never split into words, so neither joins a run: the case a
    document happens to use cannot decide whether a name is hidden.
    """
    return _is_titlecase_word(word) or _is_capitalised_word(word)


def _closes_with_initial(shape: str, core: str) -> bool:
    """True when ``core`` is one capital letter and its own dot follows it.

    ``Q.`` is a word to a reader and a letter to a tokeniser: the dot is not
    the sentence's full stop but the mark that says the letter stands for a
    name. The same test decides that the run stays open across the dot, and
    that the word is read as the initial it is written as.
    """
    if not _is_initial(core):
        return False
    start = shape.find(core)
    return shape[start + len(core):] == "."


def _keeps_run(shape: str, core: str) -> bool:
    """True when what wraps ``core`` leaves the run of name words open.

    A word with nothing written after it keeps the run open, and so does the
    dot of an initial -- the dot belongs to the letter. A word closed by a
    comma keeps it open too: in the inverted listing form the comma stands
    between the surname and the given name, and reading it as a sentence
    break loses the name exactly where a document prints it most formally.
    A word closed by a possessive mark keeps it open for the same reason: the
    clitic exists to attach what follows, so ``the Vantreys' crew`` is the same
    sentence as ``the Vantrey's crew`` and the mark is not a full stop. Every
    other closing mark ends the entry.
    """
    start = shape.find(core) if core else 0
    trailing = shape[start + len(core):]
    if not trailing:
        return True
    if trailing == "." and _is_initial(core):
        return True
    return all(char in _ENTRY_COMMA or char in _POSSESSIVE_MARKS for char in trailing)


# The clitic a sentence attaches to a word to attribute what follows to it.
# ``Q. Vantrey's crew`` names the same person as ``Q. Vantrey``: the mark and
# its ``s`` are the sentence's grammar, not letters of the name. The mark is
# each apostrophe a document is written with, and the ``s`` is the same letter
# in capitals on a title block as in a sentence.
_POSSESSIVE_MARKS = frozenset("'\u2019")


def _without_possessive(shape: str, core: str) -> tuple[str, str]:
    """The word a possessive clitic is attached to, and its shape without it.

    A document writes the possessive to attribute what follows to the person
    it has just named -- a project's crew, a company's sheet -- and that is the
    place a name is most often written in prose. The clitic is not part of the
    word, so a reader takes it off before asking what the word is; leaving it
    on made the word name-shaped in no case at all, which ended the run at the
    name and lost every form the run reader exists to read.

    Edge punctuation is stripped before this point, so the clitic sits inside
    the token here. A token that does not end with one is returned unchanged,
    and so is one whose remainder is neither letters nor a written initial
    (``2010's``): a possessive of a number or an identifier is not a written
    name. The shape keeps whatever marks follow the clitic, so the run still
    ends where the sentence ends it.

    A written initial takes the clitic too. ``Q.'s crew`` attributes the crew
    to the person the initial stands for, exactly as ``Vantrey's crew`` does
    for the surname, and an entry copied out of an index prints the given name
    that way more often than in full. The dot is the reader's evidence that
    the letter is an initial, so it stays with the letter: the word read off
    the token is the letter alone and the shape keeps the dot, which is the
    same pair the tokeniser produces for ``Q.`` written with no clitic at all.
    """
    if len(core) < 3 or core[-2] not in _POSSESSIVE_MARKS or core[-1] not in "sS":
        return core, shape
    word = core[:-2]
    kept = word
    if not word.isalpha():
        if not (word.endswith(".") and _is_initial(word[:-1])):
            return core, shape
        word = word[:-1]
    at = shape.rfind(core)
    if at < 0:
        return core, shape
    return word, shape[:at] + kept + shape[at + len(core):]


def _entry_words(text: str) -> list[str] | None:
    """The words of a denylist entry, read as a reader reads them written.

    An entry is typed by a person, usually copied from the place the name was
    found: a byline, an index entry, a licence line, a table cell. Those places
    wrap a name in marks that belong to the layout, not to the name -- the
    comma that stands between a surname and a given name above all, and the
    closing dot or quotation mark of the sentence it was copied out of. So a
    word is read with the same edge punctuation the document reader strips,
    and the entry comma separates words exactly as the space beside it does.
    Without this the entry ``Surname, Given`` is not a set of two words at all,
    and the order-independent rule is silently off for the entry -- the one
    form in which a document most often prints a name.

    ``None`` when a token carries no word once its marks are stripped: an
    entry like that is a phrase with a joiner in it, not a sequence of words.
    """
    spread = text
    for comma in _ENTRY_COMMA:
        spread = spread.replace(comma, " ")
    words: list[str] = []
    for token in re.split(r"[\s\-\u2010-\u2014]+", spread):
        if not token:
            continue
        core = token.strip(_EDGE_PUNCTUATION)
        if not core:
            return None
        words.append(core)
    return words


def name_pattern_words(pattern: re.Pattern[str]) -> tuple[str, ...] | None:
    """The folded words of a denylist entry that spells a written name.

    ``None`` for every other entry, and deliberately: a regex entry has no
    literal words to sequence, and an entry with a lower-case word is a phrase
    -- the folded phrase match already covers it, and reordering its words
    would be inventing names out of vocabulary. Two to four words, every one
    Titlecase, is the shape of a person's name on a denylist; the entry is
    stored folded, and the written forms compared against it are read from the
    document, caps included. The words are read past the marks a written word
    may be wrapped in (``_entry_words``), so the rule does not depend on how
    the secret was typed.
    """
    if _REGEX_SYNTAX.search(pattern.pattern):
        return None
    words = _entry_words(pattern.pattern)
    if words is None:
        return None
    if not 2 <= len(words) <= 4:
        return None
    if not all(_is_titlecase_word(word) for word in words):
        return None
    return tuple(fold_for_match(word) for word in words)


# A markdown link writes its label in brackets and the place it points at in
# the parentheses that follow. The destination is layout -- a path, an anchor,
# a URL -- and never a word the label spells. When the label closes a token,
# the two are welded together in the raw text (``[Q. Surname](x)``), so the
# token a reader sees as the surname arrives carrying a destination that no
# name word is spelled with. That is the same defect the table bar had: layout
# read as part of a word. The destination is taken off before the word is
# judged, and its label's closing bracket with it -- together they are the
# link's closing mark, and neither is part of the word. The opening paren is
# found by balancing, so a URL that itself contains parentheses is still one
# destination. What follows the destination stays, so the sentence's own comma
# or full stop still ends the entry.
_LINK_DESTINATION_OPEN = "]("


def _without_link_destination(token: str) -> str:
    """The token with a markdown link destination taken off, if one is on it.

    The closing bracket goes with the destination: a link ends with ``](...)``
    as one mark, and leaving the bracket behind would hide an initial's own
    dot behind it. What follows the destination's closing parenthesis is kept
    (the comma or full stop the sentence put after the link), so the run still
    ends where the line ends it. A token that carries no ``](`` is returned
    unchanged, and so is one whose parentheses never balance -- an unbalanced
    mark is punctuation, not a link.
    """
    at = token.find(_LINK_DESTINATION_OPEN)
    if at < 0:
        return token
    depth = 1
    index = at + len(_LINK_DESTINATION_OPEN)
    while index < len(token):
        char = token[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return token[:at] + token[index + 1:]
        index += 1
    return token


# A reference link writes its label in brackets and names the reference to
# follow in a second bracket pair: ``[Q. Surname][ref]``. The second pair is
# layout -- an identifier the document resolves elsewhere, or nothing at all in
# the collapsed form ``[Q. Surname][]`` -- and never a word the label spells.
# No space falls between the label's closing bracket and the reference, so the
# surname and the reference arrive as one token, and a word welded to
# ``][...]`` is name-shaped in no case at all. The reference is taken off
# before the word is judged, the label's closing bracket with it, exactly as
# the parenthesised destination is; brackets inside the reference are found by
# balancing, and what follows the reference stays, so a sentence's comma or
# full stop still ends the entry.
_LINK_REFERENCE_OPEN = "]["


def _without_link_reference(token: str) -> str:
    """The token with a markdown reference-link mark taken off, if on it.

    The closing bracket goes with the reference: a full reference link ends
    with ``][ref]`` as one mark, and leaving the label's bracket behind would
    hide an initial's own dot behind it. What follows the reference's closing
    bracket is kept (the comma or full stop the sentence put after the link),
    so the run still ends where the line ends it. A token that carries no
    ``][`` is returned unchanged, and so is one whose brackets never balance --
    an unbalanced mark is punctuation, not a reference link.
    """
    at = token.find(_LINK_REFERENCE_OPEN)
    if at < 0:
        return token
    depth = 1
    index = at + len(_LINK_REFERENCE_OPEN)
    while index < len(token):
        char = token[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return token[:at] + token[index + 1:]
        index += 1
    return token


def name_runs(text: str) -> list[tuple[int, list[str]]]:
    """Maximal runs of name-shaped words, each with the offset it starts at.

    A token whose edges carry punctuation closes the run it is in: the dot of
    ``Vantrey.`` ends a sentence, so the capitalised word after it is not part
    of the same written name. Tokens are split only on separators a name is
    written with -- space, tab, dash, Unicode space -- never on the joiners of
    code, so an identifier is one token and is never a name word.

    A markdown link destination welded to a label's last word is taken off
    first (``_without_link_destination``): it is where the link points, not
    how the name is spelled.

    Invisible characters are deleted, not treated as separators: a soft
    hyphen pushed into the middle of a surname leaves one word to a reader,
    and a rule that cut it in two would miss the name it was written as.
    Every offset maps back through ``origin`` so the caller still reports the
    line the run starts on.
    """
    if _INVISIBLE_RE.search(text):
        origin: list[int] = []
        kept: list[str] = []
        for position, char in enumerate(text):
            if ord(char) in _INVISIBLE:
                continue
            kept.append(char)
            origin.append(position)
        clean = "".join(kept)
    else:
        clean = text
        origin = list(range(len(text)))
    words: list[tuple[int, str, bool]] = []
    index = 0
    length = len(clean)
    while index < length:
        char = clean[index]
        if char in _NAME_SEPARATORS:
            index += 1
            continue
        end = index + 1
        while end < length and clean[end] not in _NAME_SEPARATORS:
            end += 1
        token = _without_link_reference(_without_link_destination(clean[index:end]))
        # Compatibility forms and accents are presentation: ``Vantre\u0301y``,
        # ``Vantr\u00e9y`` and ``Vantrey`` are one name, and a full-width
        # letter is the same letter. Judge the shape of the word, not its
        # encoding.
        shape = token
        if not token.isascii():
            shape = "".join(
                ch
                for ch in unicodedata.normalize(
                    "NFKD", unicodedata.normalize("NFKC", token)
                )
                if not unicodedata.combining(ch)
            )
            # A letter borrowed from another script is the letter it is drawn
            # as, so the word is judged -- and later compared -- in Latin.
            # Without this a name protected by one lookalike letter would be
            # name-shaped in no script at all, and would be missed in every
            # word order but the literal one.
            shape = shape.translate(_CONFUSABLE)
        core = shape.strip(_EDGE_PUNCTUATION)
        after = shape[shape.find(core) + len(core) :] if core else ""
        # A possessive clitic belongs to the sentence, not to the word.
        bare = core
        core, shape = _without_possessive(shape, core)
        # A family's possessive puts the mark after the plural's own ``s``
        # (``the Vantreys' crew``) and nothing is left to remove; the mark is
        # still the sentence attributing what follows, so the word carries one.
        carries = core != bare or (
            after[:1] in _POSSESSIVE_MARKS and bare[-1:] in ("s", "S")
        )
        # A written initial keeps its dot in the run: without it the run
        # cannot tell ``Q.`` (a letter standing for a name) from ``Q`` (a
        # one-letter word), and the dot is the reader's only evidence.
        written = fold_for_match(core)
        if _closes_with_initial(shape, core):
            written += "."
        words.append((origin[index], core, written, _keeps_run(shape, core), carries))
        index = end
    runs: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    # A possessive clitic attributes what follows to the name just written --
    # ``Vantrey's crew``, and in the surname-first listing ``Vantrey's crew,
    # Quillon`` the given name follows the attributed noun. The noun is not
    # part of the name, so one lower-case word written after the clitic is
    # passed over and the run resumes at the next name word, exactly as it
    # would have continued had the noun not been written. Only one word is
    # passed over, and only while the noun keeps the entry open (a comma or no
    # mark at all): a sentence break is the document's own end of the entry.
    clitic = False
    attributed = False
    for offset, core, written, keeps, possessive in words:
        if core and _is_name_word(core):
            if not current:
                start = offset
            current.append(written)
            clitic = possessive and keeps
            attributed = False
            if keeps:
                continue
        elif not core and keeps:
            # A token made only of separating marks -- a comma standing
            # between two spaces -- carries no word at all, so it neither
            # joins the run nor ends it.
            continue
        elif clitic and not attributed and keeps and core.isalpha() and core.islower():
            # The noun the clitic attributes: the entry continues past it.
            attributed = True
            continue
        if len(current) >= 2:
            runs.append((start, current))
        current = []
        clitic = False
        attributed = False
    if len(current) >= 2:
        runs.append((start, current))
    return runs


# How many capitalised words may sit between the denylisted words of a name
# and still be that name. One carries a middle name or an initial; two carry
# both -- ``Given Alaric B. Surname`` and ``Given Alaric Bramwell Surname``
# are the same person written out in full. Beyond that the run is a list of
# people, a column header or a heading, not one name, and matching it would be
# matching vocabulary rather than an identity.
_MAX_INTERIOR_INSERTIONS = 2


def _word_reads_as(written: str, word: str) -> bool:
    """True when a run word is the denylisted word, its plural, or its initial.

    A byline, a citation and an index entry abbreviate a given name to its
    first letter and close it with a dot: ``G. Surname`` for the denylisted
    ``Given Surname``. The dot is the document's own evidence that the
    letter stands for a word, so only a word written with it is read as an
    initial; a bare capital letter is an ordinary one-letter word and goes on
    matching nothing but itself. What the initial abbreviates is decided by
    the denylist entry -- the first letter of a word the list already names --
    so no vocabulary is added anywhere.

    A family is named in the plural: ``the Surnames`` and ``the Surnames'
    crew`` name the household the entry names in the singular, and the plural
    is English morphology on the word, not a different word. Only the two
    regular spellings are folded -- ``-s``, and ``-es`` after a word that
    already ends in ``s`` -- so an irregular or a longer word is still a
    different word, and the rest of the name still has to be read beside it.
    """
    if written == word:
        return True
    if word.isalpha() and written == f"{word}s":
        return True
    if word.isalpha() and word.endswith("s") and written == f"{word}es":
        return True
    return (
        len(written) == 2
        and written.endswith(".")
        and written[0] == word[0]
    )


def _word_or_initial_in(word: str, *views: str) -> bool:
    """True when a block carries the word, or the initial the word begins with.

    The candidate test in front of the run reader asks whether a denylisted
    word is written anywhere in the block at all. A byline that abbreviates a
    given name -- ``G. Surname`` for ``Given Surname`` -- writes the word
    nowhere, so a test that only looks for the word drops the block before the
    run reader can read the initial. The initial is the word's first letter
    and its dot, which is the same evidence the run reader uses; whether the
    run really reads as the name is still decided there.
    """
    for view in views:
        if word in view or f"{word[0]}." in view:
            return True
    return False


def _reads_as(run: list[str], start: int, form: tuple[str, ...]) -> bool:
    """True when the run reads as ``form`` from ``start``, with a bounded gap.

    The first word is anchored by the caller. Each further word is taken at
    its earliest place at or after the previous one, and the words skipped
    over on the way are counted: taking the earliest place is what minimises
    that count, so the total is the smallest number of words a reader of this
    run has to pass over to read the name, and one comparison against the
    bound decides. Skipped words are already name words -- the caller only
    passes runs of them -- so a gap is made of capitalised text, not prose.
    """
    position = start + 1
    for needle in form[1:]:
        while position < len(run) and not _word_reads_as(run[position], needle):
            position += 1
        if position >= len(run):
            return False
        position += 1
    return position - start - len(form) <= _MAX_INTERIOR_INSERTIONS


def name_run_matches(run: list[str], words: tuple[str, ...]) -> bool:
    """True when the run reads as the denylisted name, however it is written.

    The exact sequence and the sequence reversed (a surname-first
    attribution), and either of those with up to two words inserted between
    the name's words -- a middle name, a middle initial, or both at once.
    What is written after the surname is not part of the match, and neither
    is what is written before the first word: the match is anchored at the
    first word and ends where the last one is read, exactly as a reader reads
    a name.
    """
    size = len(words)
    for form in (words, words[::-1]):
        for start in range(len(run) - size + 1):
            if _word_reads_as(run[start], form[0]) and _reads_as(run, start, form):
                return True
    return False


def scan_client_document(
    text: str, patterns: list[re.Pattern[str]], *, allow_line_join: bool = True
) -> list[tuple[int, int]]:
    """Every client-pattern hit in one document, each location reported once.

    Adds to ``scan_client_text`` the spellings a literal comparison cannot see,
    and only those it can defend:

    * case, accents, invisible characters and the choice of separator between
      two words are normalised away (see ``fold_for_match``), so a name is
      caught whether it was typed hyphenated, in full-width letters, with a
      zero-width space inside it, or with an accent spelled two ways;
    * with ``allow_line_join``, the same phrase is matched across the line
      break of a wrapped line, reported at the line it starts on. The caller
      sets this for prose only: in a source file a newline separates
      statements, and joining them manufactures phrases nobody wrote;
    * a match that sits inside a longer identifier is NOT a hit. ``rate_card``
      and ``docs/notes`` are one word each to a reader, and without this rule
      the scan reports code naming things as client content, which is how a
      red gate teaches a team to bypass it.

    An order-independent, gap-tolerant word-set match was tried and withdrawn:
    on the real denylist it turned seventeen ordinary lines of this repository
    into hits. What a written name actually does -- surname first, a middle
    name between -- is matched instead by the exact sequence rule in
    ``name_run_matches``, which only ever applies to an entry whose every word
    is a capitalised name word.
    """
    found: set[tuple[int, int]] = set(scan_client_text(text, patterns))
    matchers = [folded_matcher(pattern) for pattern in patterns]
    name_patterns = [
        (index, words)
        for index, pattern in enumerate(patterns)
        if (words := name_pattern_words(pattern)) is not None
    ]
    blocks = (
        _blocks(text)
        if allow_line_join
        else [
            (line, [(0, number)])
            for number, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
    )
    for raw_block, numbers in blocks:
        split_fold = fold_for_match(raw_block)
        joined_fold = (
            fold_for_match(raw_block, join_invisible=True)
            if _INVISIBLE_RE.search(raw_block)
            else split_fold
        )
        starts = [offset for offset, _ in numbers]
        sparse = [
            (index, words)
            for index, words in name_patterns
            if all(_word_or_initial_in(word, split_fold, joined_fold) for word in words)
        ]
        if sparse:
            for run_start, run in name_runs(raw_block):
                for pattern_index, words in sparse:
                    if name_run_matches(run, words):
                        position = numbers[bisect_right(starts, run_start) - 1][1]
                        found.add((position, pattern_index))
            for line, (_, line_number) in zip(raw_block.split("\n"), numbers):
                phrase = _table_row_phrase(line)
                if phrase is None:
                    continue
                for _, run in name_runs(phrase):
                    for pattern_index, words in sparse:
                        if name_run_matches(run, words):
                            found.add((line_number, pattern_index))
        cheap = {split_fold, joined_fold}
        if not any(matcher.search(form) for matcher in matchers for form in cheap):
            continue                     # no candidate: no need for the map
        views = [fold_with_map(raw_block)]
        if joined_fold != split_fold:
            views.append(fold_with_map(raw_block, join_invisible=True))
        for mapped, index in views:
            for pattern_index, matcher in enumerate(matchers):
                for match in matcher.finditer(mapped):
                    if sits_in_identifier(raw_block, index, match.start(), match.end()):
                        continue
                    position = index[match.start()]
                    line_number = numbers[bisect_right(starts, position) - 1][1]
                    found.add((line_number, pattern_index))
    return sorted(found)


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


# Line break inside a phrase is a typesetting artifact in prose and a statement
# separator in code. Joining the lines of a source file invents adjacencies no
# reader ever sees -- one docstring line ending in a word, the next starting
# with another -- which is how a wrapped-line match turned ordinary fixtures
# into client content. So the cross-line comparison is limited to the formats
# where a human reads a paragraph, not a run of statements.
PROSE_SUFFIXES = frozenset({
    ".md", ".markdown", ".rst", ".txt", ".adoc", ".html", ".htm", ".tex",
})


def is_prose(path: str | Path) -> bool:
    """True when a file is read as a paragraph rather than as statements."""
    return Path(str(path).replace("\\", "/")).suffix.lower() in PROSE_SUFFIXES


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
            # Whatever the encoding, so a UTF-16 or latin-1 file is judged
            # by its text instead of being dropped unread. See scan_text.
            text = scan_text.read(path)
        except (OSError, UnicodeDecodeError):
            continue  # unreadable; no text a regex can judge
        findings.extend(
            (rel, line_number, index)
            for line_number, index in scan_client_document(
                text, patterns, allow_line_join=is_prose(rel)
            )
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
            text = scan_text.read(path)
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
