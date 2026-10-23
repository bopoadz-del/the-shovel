"""Sanitise U+FFFD (mojibake) from the 498-row pilot.

Rules (deterministic, applied per row):
  - If any field is missing -> count as malformed, drop.
  - Count U+FFFD characters in the row's textual fields (instruction +
    context + response).
  - DROP if:
      * U+FFFD count > 10 in any single field, OR
      * U+FFFD ratio (fffd_chars / total_chars) > 5% on the row.
  - REPAIR otherwise: strip U+FFFD from each field, collapse the leftover
    whitespace. Keep the row.
  - Also DROP if after repair the response length <30 chars or
    instruction <10 chars (post-repair quality bar; same thresholds the
    generator used at write time).

Writes:
  - training_scenarios_drive_archive_clean.jsonl
  - sanitise_report.json (counts, sample IDs)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path("data/learning/training_scenarios_drive_archive.jsonl")
OUT = Path("data/learning/training_scenarios_drive_archive_clean.jsonl")
REPORT = Path("data/learning/sanitise_report.json")

REPLACEMENT = "�"


def count_fffd(s: str) -> int:
    return s.count(REPLACEMENT)


def repair(s: str) -> str:
    # Drop U+FFFD; collapse runs of >=2 whitespace.
    s = s.replace(REPLACEMENT, "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def main() -> None:
    n_total = 0
    n_dropped_malformed = 0
    n_dropped_heavy_fffd = 0
    n_dropped_post_repair_short = 0
    n_repaired = 0
    n_clean_passthrough = 0
    sample_drops = []
    sample_repairs = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with SRC.open("r", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as g:
        for line_no, line in enumerate(f, 1):
            n_total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_dropped_malformed += 1
                continue
            instr = row.get("instruction") or ""
            ctx = row.get("context") or ""
            resp = row.get("response") or ""
            if not (instr and resp):
                n_dropped_malformed += 1
                continue
            fffd_per_field = {
                "instruction": count_fffd(instr),
                "context": count_fffd(ctx),
                "response": count_fffd(resp),
            }
            total_chars = len(instr) + len(ctx) + len(resp)
            total_fffd = sum(fffd_per_field.values())
            ratio = (total_fffd / total_chars) if total_chars > 0 else 0
            heavy = any(v > 10 for v in fffd_per_field.values()) or ratio > 0.05
            if heavy:
                n_dropped_heavy_fffd += 1
                if len(sample_drops) < 5:
                    sample_drops.append({
                        "line": line_no,
                        "fffd_per_field": fffd_per_field,
                        "ratio": round(ratio, 4),
                        "instruction": instr[:120],
                    })
                continue
            if total_fffd == 0:
                n_clean_passthrough += 1
                g.write(line if line.endswith("\n") else line + "\n")
                continue
            # Repair path
            new_instr = repair(instr)
            new_ctx = repair(ctx) if ctx else ctx
            new_resp = repair(resp)
            if len(new_resp) < 30 or len(new_instr) < 10:
                n_dropped_post_repair_short += 1
                continue
            row["instruction"] = new_instr
            row["context"] = new_ctx
            row["response"] = new_resp
            row["sanitised"] = True
            row["sanitised_fffd_removed"] = total_fffd
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_repaired += 1
            if len(sample_repairs) < 5:
                sample_repairs.append({
                    "line": line_no,
                    "fffd_removed": total_fffd,
                    "ratio_before": round(ratio, 4),
                    "instruction": new_instr[:120],
                })

    report = {
        "src": str(SRC),
        "out": str(OUT),
        "total_rows_in": n_total,
        "rows_kept_clean_passthrough": n_clean_passthrough,
        "rows_repaired": n_repaired,
        "rows_dropped_malformed": n_dropped_malformed,
        "rows_dropped_heavy_fffd": n_dropped_heavy_fffd,
        "rows_dropped_post_repair_short": n_dropped_post_repair_short,
        "rows_out": n_clean_passthrough + n_repaired,
        "sample_drops": sample_drops,
        "sample_repairs": sample_repairs,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("sample")}, indent=2))


if __name__ == "__main__":
    main()
