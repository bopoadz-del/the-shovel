# Edge Port Plan — When the Orin Arrives

This is a one-pager naming the future commands. **None of this work is in any merged PR.** It exists so the future port is obvious — and so PR 3 code doesn't paint into a corner that makes the port hard.

## Context

- PR 3a (LoRA fine-tune) and PR 3b (YOLO object detection) ship CPU+CUDA-portable code. No hardcoded device strings, HF format only for adapters, ultralytics-native `.pt` for YOLO weights.
- The cloud GPU is the **training** environment, not the deployment ceiling.
- The Orin (Jetson edge device) is the **deployment** target when it arrives.

## What changes when the Orin shows up

### 1. Quantize the LoRA adapter for memory-constrained inference

The Orin has 8-16 GB unified memory. A 3B-param model in fp16 takes ~6 GB; in 4-bit it takes ~1.5 GB. We quantize **after** training, not during.

Two paths, pick the one that fits the runtime you want on the Orin:

**4-bit via bitsandbytes** (Transformers-compatible, easy):

```bash
pip install bitsandbytes
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "data/learning/adapters/construction_v1")
model.save_pretrained("data/learning/adapters/construction_v1_4bit", safe_serialization=True)
PY
```

**GGUF via llama.cpp** (smaller binary, faster on edge, no torch runtime):

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
# Merge adapter into base then convert
python convert_hf_to_gguf.py /path/to/merged_model --outfile construction_v1.gguf --outtype q4_k_m
# Move construction_v1.gguf to the Orin; serve via llama-server or directly via the Python binding
```

The PR 3a adapter format works with both paths because it's standard HF safetensors. No re-training needed.

### 2. Compile YOLO weights to TensorRT for the Orin's NVDLA

The Orin has hardware accelerators that need TensorRT engines, not raw torch weights. Ultralytics has a built-in exporter:

```bash
pip install ultralytics tensorrt
yolo export model=data/cv/best.pt format=engine device=0 half=True
# Produces best.engine — ship this to the Orin
```

The PR 3b code reads from `YOLO_MODEL` env. On the Orin, set `YOLO_MODEL=/path/to/best.engine` and the existing block code calls into the TensorRT runtime — no app code change.

### 3. Add an arm64 Docker build

Currently `Dockerfile` builds for `linux/amd64` only (PR #23 dropped arm64 because QEMU emulation in CI was hitting Azure cache token expiry). For the Orin:

- Add an arm64 build job that runs on a **native arm64 runner** (GitHub's arm64 large runners, or self-hosted on the Orin itself)
- Base image switches to `nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3` (JetPack-compatible)
- Tag separately: `ghcr.io/bopoadz-del/cerebrum-blocks:edge-arm64`

Not a parallel `platforms:` entry in the existing job — a separate workflow file or job. Keeps the amd64 build fast.

### 4. Wire the `jetson_gateway` block to a real edge service

`app/blocks/jetson_gateway.py` is currently a placeholder. When the Orin is on a network the platform can reach, this block becomes the dispatch shim — the cloud app calls `jetson_gateway.run(...)` which forwards the inference request over HTTP/gRPC to a tiny FastAPI service running on the Orin that wraps the quantized model + TensorRT YOLO.

Architecture:

```
[cloud Fork]  ──jetson_gateway──>  [Orin: FastAPI shim]  ──>  [llama.cpp or transformers + bnb 4bit]
                                                          └─>  [TensorRT YOLO engine]
```

The shim is small (~200 LOC) and re-uses The Fork's `ChatBlock` + `ImageBlock` response shapes verbatim, so callers can't tell whether the answer came from cloud or edge.

## What this document is NOT

- Not a project plan. Don't start work here until an Orin is in hand.
- Not a commitment to specific commands working on day 1 of having the device. Test each step.
- Not exhaustive — there will be Jetson-specific surprises (CUDA toolkit version mismatches, JetPack base image quirks). The point is to have the map, not the turn-by-turn.

## What this document IS

A receipt that **PR 3a's HF format and PR 3b's ultralytics-native format were chosen specifically because they convert cleanly to edge formats**. If the future ports get blocked by a format choice we made now, this doc should let us catch it before merging.
