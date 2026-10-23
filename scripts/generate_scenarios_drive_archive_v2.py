#!/usr/bin/env python3
"""V2: balanced 5,000-10,000-row scenario generator over `drive_archive`.

Forked from generate_scenarios_drive_archive.py (the 498-row pilot). The pilot
file at training_scenarios_drive_archive.jsonl is FROZEN — this writes to a
new path. Hard guards on output paths at startup.

Differences from pilot:
- Per-doc cap raised to 4 (was 3).
- Per-discipline quota = min(900, pool_size) instead of TARGET // n_disc.
- Within a discipline: round-robin across source-folders (dirname of the
  file) before going back for seconds, so we don't binge one folder.
- Across disciplines: round-robin to build the processing order, so a
  truncated wall-clock run is still balanced (NOT contract-first).
- Resume seeds done_keys from both the state file AND the output file.
- Per-discipline written tally in the summary.

LLM routing: ONLY Ollama. Primary qwen3-coder:480b-cloud, fallback
qwen2.5:7b-instruct after 5 consecutive primary failures.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

# Force UTF-8 on stdout/stderr for Windows.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass

# Config
OLLAMA_URL = "http://127.0.0.1:11434"
PRIMARY_MODEL = "qwen3-coder:480b-cloud"
FALLBACK_MODEL = "qwen2.5:7b-instruct"
_REPO = os.environ.get("FORK_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTORS_DB = os.path.join(_REPO, "data", "rag", "vectors.db")
OUT_PATH = os.path.join(_REPO, "data", "learning", "training_scenarios_drive_archive_v2.jsonl")
STATE_PATH = os.path.join(_REPO, "data", "logs", "scenario_gen_state_drive_archive_v2.json")
PILOT_OUT = os.path.join(_REPO, "data", "learning", "training_scenarios_drive_archive.jsonl")

PER_DISCIPLINE_CAP = 900
PER_DOC_CAP = 4
MIN_CHUNK_CHARS = 250
WALL_CLOCK_BUDGET_SEC = 75 * 60   # leave 15 min headroom inside the 90-min hard wall
SAVE_EVERY = 100

random.seed(42)

# Guard: irreversible mistake protection.
assert OUT_PATH.endswith("_v2.jsonl"), "OUT_PATH must end with _v2.jsonl — pilot file is frozen"
assert STATE_PATH.endswith("_v2.json"), "STATE_PATH must end with _v2.json"
assert OUT_PATH != PILOT_OUT, "OUT_PATH must NOT equal pilot path"

# Discipline classification — exact same rules as the pilot.
DISCIPLINE_RULES: List[Tuple[str, str]] = [
    (r"structural|\\ST\\|\\ST-|\bSTR\b", "structural"),
    (r"street\s*light|lighting|\\LI-|\\LI\\", "lighting"),
    (r"\\MEP\\|mechanical|electrical|plumbing|HVAC", "mep"),
    (r"contract\s*docs?|PMC|PSA|signed\s*contract", "contract"),
    (r"BOQ|bill\s*of\s*quantit|pricing|tender|priced", "boq"),
    (r"specification|\bspec\b|\\SPEC\\", "spec"),
    (r"schedule|programme|baseline\s*p6", "schedule"),
    (r"geotech|borehole|BH-\d+|soil\s*invest", "geotech"),
    (r"survey|topograph|level\s*plan", "survey"),
    (r"drainage|storm\s*water|sewerage|\\DR\\", "drainage"),
    (r"road|highway|pavement|asphalt", "roads"),
    (r"landscape|hardscape|softscape|irrigation", "landscape"),
    (r"architectur|\\AR\\|\\AR-", "architecture"),
    (r"drawing|drwg|\\DWG-|design\s*drawings", "drawings"),
    (r"safety|HSE|risk\s*assessment|method\s*statement", "hse"),
    (r"report|study|analysis", "report"),
]


def classify_discipline(source_path: str) -> str:
    for pattern, bucket in DISCIPLINE_RULES:
        if re.search(pattern, source_path, re.IGNORECASE):
            return bucket
    return "other"


def parse_source(text: str) -> str:
    m = re.match(r"\[source:\s*([^\]]+)\]", text)
    return m.group(1).strip() if m else ""


def source_folder(source_path: str) -> str:
    """Bucket key for stratifying within a discipline — the file's immediate
    parent directory. Falls back to the doc_id if path is empty."""
    if not source_path:
        return "(unknown)"
    # Use backslash for Windows paths; fall back to forward slash.
    sep = "\\" if "\\" in source_path else "/"
    parts = source_path.rsplit(sep, 1)
    return parts[0] if len(parts) > 1 else "(root)"


def build_processing_order() -> List[Dict[str, object]]:
    """Read all eligible drive_archive chunks, build a per-discipline ordered
    list with per-doc cap + source-folder round-robin, then round-robin across
    disciplines so the head of the list is balanced even if truncated."""
    print(f"[sample] opening {VECTORS_DB}", flush=True)
    con = sqlite3.connect(VECTORS_DB)
    cur = con.cursor()
    cur.execute(
        "SELECT doc_id, chunk_index, text FROM chunks "
        "WHERE project_id='drive_archive' AND length(text) >= ?",
        (MIN_CHUNK_CHARS,),
    )

    # First pass: bucket by discipline -> source_folder -> doc_id -> [chunks]
    # Apply MIN_CHUNK_CHARS already in SQL.
    disc_folder_doc: Dict[str, Dict[str, Dict[str, List[Tuple[int, str, str]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    total_eligible = 0
    for doc_id, chunk_index, text in cur:
        src = parse_source(text)
        disc = classify_discipline(src)
        folder = source_folder(src)
        disc_folder_doc[disc][folder][doc_id].append((chunk_index, text, src))
        total_eligible += 1
    con.close()
    print(f"[sample] eligible chunks: {total_eligible}", flush=True)

    # Pool sizes per discipline (informational).
    pool_sizes: Dict[str, int] = {}
    for disc, folders in disc_folder_doc.items():
        n = sum(len(chunks) for docs in folders.values() for chunks in docs.values())
        pool_sizes[disc] = n
    print("[sample] discipline pool sizes:", flush=True)
    for d, n in sorted(pool_sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {d:14s} {n}", flush=True)

    # For each discipline: round-robin across source-folders, within a folder
    # round-robin across docs (cap PER_DOC_CAP per doc), pick chunks evenly
    # spaced through each doc.
    per_discipline_ordered: Dict[str, List[Dict[str, object]]] = {}
    quotas: Dict[str, int] = {}
    for disc, folders in disc_folder_doc.items():
        quota = min(PER_DISCIPLINE_CAP, pool_sizes[disc])
        quotas[disc] = quota

        # For each (folder, doc) pre-pick up to PER_DOC_CAP chunks, evenly
        # spread across the doc (first/middle/last/mid-late style).
        # Sort chunks within doc by chunk_index then sample.
        per_doc_picks: Dict[str, Dict[str, List[Tuple[int, str, str]]]] = {}
        for folder, docs in folders.items():
            per_doc_picks[folder] = {}
            for doc_id, chunks in docs.items():
                chunks_sorted = sorted(chunks, key=lambda r: r[0])
                n = len(chunks_sorted)
                if n <= PER_DOC_CAP:
                    pick = chunks_sorted
                else:
                    # Evenly-spaced indices [0, n//k, 2n//k, ...].
                    idxs = [int(round(j * (n - 1) / (PER_DOC_CAP - 1))) for j in range(PER_DOC_CAP)]
                    pick = [chunks_sorted[k] for k in sorted(set(idxs))]
                per_doc_picks[folder][doc_id] = pick

        # Build ordered list by round-robin: across folders, then across docs
        # within a folder, then take pick[0] from each (one-per-folder round),
        # then pick[1], etc.
        folder_keys = sorted(per_doc_picks.keys())
        # Per-folder per-doc cursors.
        folder_state: Dict[str, Dict[str, object]] = {}
        for folder in folder_keys:
            doc_keys = sorted(per_doc_picks[folder].keys())
            folder_state[folder] = {
                "doc_keys": doc_keys,
                "doc_cursor": 0,        # which doc to draw from next
                "doc_picks_idx": defaultdict(int),  # per-doc cursor into pick list
            }

        ordered: List[Dict[str, object]] = []
        # Round counter: in each round, visit each folder once, take ONE chunk
        # from the folder's current doc, then advance the folder's doc cursor.
        while len(ordered) < quota:
            progressed = False
            for folder in folder_keys:
                state = folder_state[folder]
                doc_keys = state["doc_keys"]
                if not doc_keys:
                    continue
                # Advance to a doc that still has picks left.
                start_cursor = state["doc_cursor"]
                tries = 0
                while tries < len(doc_keys):
                    doc_id = doc_keys[state["doc_cursor"]]
                    pick_list = per_doc_picks[folder][doc_id]
                    pi = state["doc_picks_idx"][doc_id]
                    if pi < len(pick_list):
                        chunk_idx, text, src = pick_list[pi]
                        ordered.append({
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "text": text,
                            "source_path": src,
                            "discipline": disc,
                        })
                        state["doc_picks_idx"][doc_id] = pi + 1
                        # Advance doc_cursor for next round of this folder.
                        state["doc_cursor"] = (state["doc_cursor"] + 1) % len(doc_keys)
                        progressed = True
                        break
                    state["doc_cursor"] = (state["doc_cursor"] + 1) % len(doc_keys)
                    tries += 1
                if len(ordered) >= quota:
                    break
            if not progressed:
                break

        per_discipline_ordered[disc] = ordered[:quota]
        print(f"  [order] {disc:14s} quota={quota:4d}  built={len(ordered):4d}", flush=True)

    # Cross-discipline round-robin to build final processing order. This is
    # the key balance-under-truncation move.
    queues: Dict[str, deque] = {d: deque(per_discipline_ordered[d]) for d in per_discipline_ordered}
    disc_keys = list(queues.keys())
    # Shuffle disc_keys order for fairness (but stable per seed).
    random.shuffle(disc_keys)
    final: List[Dict[str, object]] = []
    while any(queues[d] for d in disc_keys):
        for d in disc_keys:
            if queues[d]:
                final.append(queues[d].popleft())

    print(f"[order] final cross-discipline interleaved length: {len(final)}", flush=True)
    return final


# Ollama call — same prompt as pilot.
SYSTEM_PROMPT = (
    "You are generating training data for a construction-domain RAG LLM. "
    "Given a chunk from a real construction document, produce ONE Q&A pair: "
    "a realistic engineer question answerable from the chunk + a grounded "
    "answer. Do not invent facts. Return JSON only with keys instruction "
    "and response."
)


def call_ollama(model: str, chunk_text: str, source_path: str, timeout: int = 90) -> Optional[str]:
    user = (
        f"Context (chunk from {source_path}):\n{chunk_text}\n\n"
        'Return JSON only: {"instruction": "...", "response": "..."}'
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 600},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return (resp.get("message", {}) or {}).get("content")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"[ollama] ERROR {model}: {exc}", flush=True)
        return None


def parse_qa(raw: str) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*\}", "}", blob))
        except json.JSONDecodeError:
            return None
    instr = (obj.get("instruction") or "").strip()
    resp = (obj.get("response") or "").strip()
    if len(instr) < 10 or len(resp) < 30:
        return None
    lower_q = instr.lower()
    if any(p in lower_q for p in ("what does this say", "summarize this", "what is this about")):
        return None
    return {"instruction": instr, "response": resp}


def load_state_and_seed_done_keys() -> Tuple[Dict[str, object], set]:
    """Load checkpoint AND seed done_keys from any already-written rows in
    the output file. Closes the append-vs-checkpoint dup window."""
    state: Dict[str, object] = {"completed_keys": []}
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    done_keys = set(state.get("completed_keys", []))

    # Also scan output file for already-written rows.
    if os.path.exists(OUT_PATH):
        n_from_file = 0
        try:
            with open(OUT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        src = obj.get("source", "")
                        # source = "drive_archive:{doc_id}:{chunk_index}"
                        if src.startswith("drive_archive:"):
                            key = src[len("drive_archive:"):]
                            if key not in done_keys:
                                done_keys.add(key)
                                n_from_file += 1
                    except json.JSONDecodeError:
                        continue
            if n_from_file:
                print(f"[resume] +{n_from_file} keys seeded from {OUT_PATH}", flush=True)
        except OSError:
            pass

    return state, done_keys


def save_state(state: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def append_row(row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_URL"] = OLLAMA_URL
    os.environ["LOCAL_LLM_MODEL"] = PRIMARY_MODEL

    t0 = time.time()
    samples = build_processing_order()

    state, done_keys = load_state_and_seed_done_keys()
    if done_keys:
        print(f"[resume] {len(done_keys)} total keys already done", flush=True)

    model = PRIMARY_MODEL
    primary_failures = 0
    primary_calls = 0
    fallback_calls = 0
    switched_at: Optional[int] = None
    written = 0
    dropped_parse = 0
    dropped_short_or_generic = 0
    per_disc_written: Dict[str, int] = defaultdict(int)

    for i, c in enumerate(samples, start=1):
        elapsed = time.time() - t0
        if elapsed > WALL_CLOCK_BUDGET_SEC:
            print(f"[budget] hit {WALL_CLOCK_BUDGET_SEC}s wall — stopping at processed={i-1}/{len(samples)}", flush=True)
            break

        key = f"{c['doc_id']}:{c['chunk_index']}"
        if key in done_keys:
            continue

        raw = call_ollama(model, c["text"], c["source_path"] or c["doc_id"])
        if model == PRIMARY_MODEL:
            primary_calls += 1
            if raw is None:
                primary_failures += 1
                if primary_failures >= 5 and switched_at is None:
                    print(f"[switch] {primary_failures} primary failures → switching to {FALLBACK_MODEL} at row {i}", flush=True)
                    model = FALLBACK_MODEL
                    switched_at = i
                continue
            primary_failures = 0
        else:
            fallback_calls += 1
            if raw is None:
                dropped_parse += 1
                continue

        qa = parse_qa(raw)
        if qa is None:
            if raw and ("{" in raw or "instruction" in raw.lower()):
                dropped_short_or_generic += 1
            else:
                dropped_parse += 1
            continue

        row = {
            "instruction": qa["instruction"],
            "context": c["text"][:600],
            "response": qa["response"],
            "source": f"drive_archive:{c['doc_id']}:{c['chunk_index']}",
            "source_doc_path": c["source_path"],
            "discipline": c["discipline"],
        }
        append_row(row)
        done_keys.add(key)
        written += 1
        per_disc_written[c["discipline"]] += 1

        if written % SAVE_EVERY == 0:
            state["completed_keys"] = sorted(done_keys)
            save_state(state)
            rate = written / max(1.0, time.time() - t0)
            print(
                f"[progress] {written} written | {i}/{len(samples)} processed | "
                f"model={model} | {rate:.2f} rows/s | "
                f"elapsed={int(time.time()-t0)}s",
                flush=True,
            )

    state["completed_keys"] = sorted(done_keys)
    save_state(state)

    total = time.time() - t0
    print("\n── SUMMARY ──", flush=True)
    print(f"  written:          {written}", flush=True)
    print(f"  primary calls:    {primary_calls} ({PRIMARY_MODEL})", flush=True)
    print(f"  fallback calls:   {fallback_calls} ({FALLBACK_MODEL})", flush=True)
    print(f"  switched at:      {switched_at}", flush=True)
    print(f"  dropped (parse):  {dropped_parse}", flush=True)
    print(f"  dropped (qual):   {dropped_short_or_generic}", flush=True)
    print(f"  wall-clock:       {total:.1f}s ({total/60:.1f} min)", flush=True)
    if written:
        print(f"  avg per row:      {total/written:.2f}s", flush=True)
    print("  per-discipline written:", flush=True)
    for d, n in sorted(per_disc_written.items(), key=lambda kv: -kv[1]):
        print(f"    {d:14s} {n}", flush=True)
    print(f"  output:           {OUT_PATH}", flush=True)
    print(f"  state:            {STATE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
