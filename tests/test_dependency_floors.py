"""Security floors for pinned dependencies.

Exists because the dependency gate (pip-audit) ran report-only for months —
`continue-on-error: true` plus `|| true` meant its green check could never
fail, and HIGH-severity pins rode straight through PR review (found in the
2026-08-04 PRR: cryptography CVE-2026-69247, aiohttp CVE-2026-69244).

pip-audit is now enforcing, but its database can lag a freshly published
advisory. These floors pin the *adjudicated* minimums so a merge that
resolves a requirements conflict by reverting to an older pin — exactly how
the Redis merge reintroduced two other defects — fails loudly with the CVE
it would be reintroducing.

When a new advisory raises a floor: bump the pin, then record the new floor
here with its advisory ID.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# package -> (minimum safe version, advisory that set the floor)
#
# click is deliberately NOT here: PYSEC-2026-2132 wants click>=8.3.3, but
# gtts (latest 2.5.4) hard-caps click<8.2, so the lockfile cannot carry the
# fixed version. That advisory is instead an adjudicated `--ignore-vuln` in
# .github/workflows/dependency-audit.yml with the rationale inline.
FLOORS = {
    "aiohttp": ((3, 14, 3), "CVE-2026-69244 (HIGH, OOB heap read) + CVE-2026-69243 (request smuggling)"),
    "cryptography": ((50, 0, 0), "CVE-2026-69247 (HIGH, Bleichenbacher oracle in PKCS#7 decrypt)"),
    "pytest": ((9, 0, 3), "PYSEC-2026-1845"),
    "h2": ((4, 4, 1), "GHSA h2 < 4.4.1 (2026-08-08, both requirements files)"),
    "restrictedpython": ((8, 3), "CVE-2026-55830 (sandbox escape via positional-only args; 8.3+ stable on PyPI)"),
    "anyio": ((4, 14, 2), "CVE-2026-63374 + CVE-2026-64847 (fixed in 4.14.2; surfaced on #603 the same day soupsieve was cleared)"),
    "soupsieve": ((2, 9), "CVE-2026-85999 + CVE-2026-86000 (fixed in 2.9; Dependabot #601 dropped the uvloop marker)"),
}

REQUIREMENT_FILES = ["requirements.txt", "requirements-ml.txt"]

_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?==(?P<ver>[0-9][0-9a-zA-Z.]*)")


def _parse_version(ver: str) -> tuple:
    parts = []
    for piece in ver.split("."):
        digits = re.match(r"\d+", piece)
        parts.append(int(digits.group()) if digits else 0)
    return tuple(parts)


def _pins(path: Path) -> dict:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _PIN.match(line.strip())
        if m:
            pins[m.group("name").lower().replace("_", "-")] = m.group("ver")
    return pins


def test_no_requirement_file_pins_below_a_security_floor():
    violations = []
    for fname in REQUIREMENT_FILES:
        path = REPO / fname
        if not path.exists():
            continue
        pins = _pins(path)
        for pkg, (floor, advisory) in FLOORS.items():
            ver = pins.get(pkg)
            if ver is not None and _parse_version(ver) < floor:
                violations.append(
                    f"{fname}: {pkg}=={ver} is below the security floor "
                    f"{'.'.join(map(str, floor))} — reintroduces {advisory}"
                )
    assert not violations, "\n".join(violations)


def test_restrictedpython_pin_is_stable_not_prerelease():
    """CVE-2026-55830 is fixed in RestrictedPython 8.3. Dependabot #437
    proposed 8.3a1.dev0 — an alpha must never satisfy the floor."""
    pins = _pins(REPO / "requirements.txt")
    ver = pins.get("restrictedpython")
    assert ver is not None, "restrictedpython must stay pinned in requirements.txt"
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", ver), (
        f"restrictedpython=={ver} is not a stable X.Y / X.Y.Z pin "
        "(refuse a/b/rc/dev — wait for a stable release)"
    )
    assert _parse_version(ver) >= (8, 3), (
        f"restrictedpython=={ver} is below the CVE-2026-55830 floor 8.3"
    )


def test_floors_cover_the_packages_they_claim():
    # The floor table is only useful if the packages are actually pinned
    # somewhere — a typo'd name here would silently guard nothing.
    all_pins = {}
    for fname in REQUIREMENT_FILES:
        path = REPO / fname
        if path.exists():
            all_pins.update(_pins(path))
    missing = [pkg for pkg in FLOORS if pkg not in all_pins]
    assert not missing, f"floor entries match no pin in any requirements file: {missing}"


# Environment markers that were hand-added to the LOCKFILE and which
# pip-compile does not regenerate — so every re-resolve silently drops them.
#
# uvloop: Dependabot PR #321 (h2 bump, 2026-08-08) rewrote the pin as a bare
# `uvloop==0.22.1`. uvloop publishes no Windows wheels, so that breaks
# `pip install -r requirements.txt` on the maintainer's own machine — and CI
# never catches it, because CI is Linux and installs fine either way. The
# marker is not in requirements.in; it exists only here, which is exactly why
# it needs a test rather than a comment.
REQUIRED_MARKERS = [
    ("requirements.txt", "uvloop==", 'platform_system != "Windows"',
     "uvloop has no Windows wheels; without the marker pip install fails on Windows"),
]


def test_handwritten_environment_markers_survive_a_re_resolve():
    violations = []
    for fname, pin_prefix, marker, why in REQUIRED_MARKERS:
        path = REPO / fname
        if not path.exists():
            continue
        lines = [
            ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith(pin_prefix)
        ]
        if not lines:
            violations.append(f"{fname}: no pin starting {pin_prefix!r} — did the package move?")
            continue
        for ln in lines:
            if marker not in ln:
                violations.append(f"{fname}: {ln!r} lost its {marker!r} marker — {why}")
    assert not violations, "\n".join(violations)
