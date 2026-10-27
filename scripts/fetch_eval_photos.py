"""Download the evaluation photographs named in the fixture manifest.

The images are NOT committed — only tests/fixtures/photo_eval_manifest.json is.
This resolves each Commons file title through the Wikimedia API, re-reads the
licence at fetch time (so the manifest cannot quietly drift from the source),
and writes the images plus a licences.json record into a gitignored directory.

    python scripts/fetch_eval_photos.py            # download everything
    python scripts/fetch_eval_photos.py --check    # verify only, no download

Exits non-zero if any file cannot be resolved or its licence is not one this
project accepts. Tests that use these fixtures skip (with a reason) when the
directory is absent, so CI never depends on the network.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "photo_eval_manifest.json"
DEST = REPO_ROOT / "tests" / "fixtures" / "photos"
API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's User-Agent policy (https://w.wiki/4wJS) requires automated
# clients to identify themselves with a contact URL; requests without one are
# refused with a 429 citing the robot policy. This is that identification.
UA = ("TheFork-photo-eval/1.0 "
      "(https://github.com/bopoadz-del/the-shovel; construction QA research) "
      "python-urllib/3.11")

# Permissive only. Anything else is not used — an unclear licence is a reason
# to drop a photo, never a reason to "probably it's fine".
ACCEPTED = ("cc0", "cc by", "cc-by", "public domain", "pd-")

MAX_WIDTH = 1280  # plenty for detection; keeps the fixture dir small


# Commons rate-limits anonymous clients; a tight loop over 20 files gets a 429
# partway through and leaves a half-populated fixture set that looks like a
# detector problem later. Be a well-behaved client instead.
THROTTLE_SECONDS = 1.0
MAX_ATTEMPTS = 4


def _open(req, timeout):
    """urlopen with backoff on 429/503 — returns the response body as bytes."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == MAX_ATTEMPTS:
                raise
            wait = THROTTLE_SECONDS * (2 ** attempt)
            print(f"      {exc.code} from Commons — waiting {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    body = _open(req, timeout=60)
    time.sleep(THROTTLE_SECONDS)
    return json.loads(body.decode("utf-8"))


def resolve(title: str = "", pageid: int = 0) -> dict:
    """Current direct URL + licence for one Commons file, named by its title
    or by its page id (an entry whose title would carry a person's name names
    the file by id instead)."""
    data = _api({
        "action": "query", "format": "json", **({"pageids": str(pageid)} if pageid else {"titles": title}),
        "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": str(MAX_WIDTH),
    })
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for page in pages.values():
        if "missing" in page:
            raise LookupError(f"not found on Commons: {title or pageid}")
        title = title or page.get("title") or ""
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {}) or {}
        return {
            "url": info.get("thumburl") or info.get("url"),
            "licence": (meta.get("LicenseShortName", {}).get("value") or "").strip(),
            "artist": (meta.get("Artist", {}).get("value") or "").strip(),
            "page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        }
    raise LookupError(f"no imageinfo for {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="resolve and verify licences without downloading")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    photos = manifest["photos"]
    DEST.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for entry in photos:
        pid, title = entry["id"], entry.get("title", "")
        try:
            resolved = resolve(title, int(entry.get("pageid") or 0))
        except Exception as exc:
            failures.append(f"{pid}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {pid}: {exc}")
            continue

        licence = resolved["licence"]
        if not any(tok in licence.lower() for tok in ACCEPTED):
            failures.append(f"{pid}: licence {licence!r} is not accepted")
            print(f"  REJECT {pid}: licence {licence!r}")
            continue

        target = DEST / f"{pid}.jpg"
        existed = target.exists()
        if not args.check and not existed:
            req = urllib.request.Request(resolved["url"], headers={"User-Agent": UA})
            target.write_bytes(_open(req, timeout=120))
            time.sleep(THROTTLE_SECONDS)
        status = "ok" if args.check else ("cached" if existed else "downloaded")
        print(f"  {status:<10} {pid:<32} {licence}")
        records.append({
            "id": pid, "file": target.name, "title": title or f"pageid {entry.get('pageid')}",
            "licence_at_fetch": licence, "licence_in_manifest": entry.get("licence"),
            "page": resolved["page"], "artist": resolved["artist"],
        })

    if not args.check:
        (DEST / "licences.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(records)}/{len(photos)} usable")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
