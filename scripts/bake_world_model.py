"""Bake a YOLO-Worldv2 model with a frozen prompt vocabulary and
export to ONNX for shipment.

YOLO-World is open-vocabulary at training time but supports a
"prompt-then-detect" deployment: you call ``set_classes(prompts)`` once
with CLIP loaded, then export. We export to ONNX (not .save()) because:
  * .save() writes a Python pickle that REFERENCES the CLIP module
    class hierarchy at load time -- ultralytics will try to import the
    `clip` package on YOLO() init even though no CLIP forward pass
    happens. That defeats the "no CLIP at runtime" guarantee.
  * export(format="onnx") writes the IR with text vectors baked into
    the head and zero Python class dependencies. It loads with plain
    YOLO() backed by onnxruntime -- no CLIP, no torch model imports.

Run this ON THE OPERATOR PC (where 338 MB CLIP weights are tolerable).
Ship ONLY the resulting .onnx to Render -- the production container has
zero CLIP dependency.

Inputs:
  app/blocks/safety_world_prompts.json   (the prompt list, version-pinned)
  yolov8s-worldv2.pt                     (downloaded on first run)

Output:
  data/models/safety_world_v2.onnx       (self-contained, ship this)

Usage:
  python scripts/bake_world_model.py
  python scripts/bake_world_model.py --base yolov8m-worldv2.pt   # bigger / slower / more accurate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    prompts_path = repo / "app" / "blocks" / "safety_world_prompts.json"
    if not prompts_path.is_file():
        print(f"missing {prompts_path}", file=sys.stderr)
        return 1
    spec = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = list(spec.get("prompts") or [])
    if not prompts:
        print("empty prompts list", file=sys.stderr)
        return 1

    out_path = repo / spec.get("baked_model_file", "data/models/safety_world_v2.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolov8s-worldv2.pt",
                   help="Base YOLO-Worldv2 checkpoint to start from")
    p.add_argument("--out", default=None,
                   help="Override output path (defaults to spec.baked_model_file)")
    args = p.parse_args()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"prompts: {len(prompts)}")
    for i, p_text in enumerate(prompts):
        print(f"  [{i:>2}] {p_text}")

    print(f"\nloading {args.base} (may pull weights on first run)...")
    t0 = time.time()
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.base)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nset_classes() -- CLIP runs here, encodes {len(prompts)} prompts...")
    t1 = time.time()
    model.set_classes(prompts)
    print(f"  set_classes done in {time.time()-t1:.1f}s")

    # Intermediate pickle -- pickled object references the CLIP class
    # hierarchy and so cannot load on a CLIP-free Render. We need it
    # only as the source for the ONNX export below; deleted at the end.
    intermediate_pt = out_path.with_suffix(".intermediate.pt")
    print(f"\nsave intermediate -> {intermediate_pt}")
    model.save(str(intermediate_pt))

    print(f"\nexport ONNX (reparameterized, CLIP-free) ...")
    onnx_out = model.export(format="onnx", opset=12)
    onnx_path = Path(onnx_out)
    # Place the exported .onnx at the slot the spec advertises (with .onnx ext).
    final_path = out_path.with_suffix(".onnx")
    if onnx_path != final_path:
        onnx_path.replace(final_path)
    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {final_path}  ({size_mb:.1f} MB)")

    try:
        intermediate_pt.unlink()
    except OSError:
        pass

    # Provenance manifest sits next to the .onnx so we can see which
    # prompts this checkpoint was baked with without loading the model.
    manifest_path = final_path.with_suffix(".prompts.json")
    manifest_path.write_text(json.dumps({
        "baked_from": args.base,
        "prompt_count": len(prompts),
        "prompts": prompts,
        "spec_version": spec.get("version"),
    }, indent=2), encoding="utf-8")
    print(f"  manifest -> {manifest_path}")

    print("\nDONE. Ship this .onnx to Render -- no CLIP needed at inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
