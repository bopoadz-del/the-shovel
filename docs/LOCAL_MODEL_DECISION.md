# LOCAL_MODEL_DECISION — on-prem LLM selection (STEP 3)

STEP 3 of the on-prem sovereignty program. Which local model drives The Fork's
agent loop on our own hardware? This document is the **methodology + candidate
slate + the evidence executable on the available hardware**, honestly labelled.

> **Hardware honesty.** The bench that fully decides this needs the target box
> (24 GB VRAM x86, and later an AGX Orin). That hardware was **not available in
> this session** (a Windows dev PC: Ollama CPU/iGPU, local `qwen2.5:3b` +
> `qwen2.5:7b` only). So: the **8B-class** referee is executed here on
> `qwen2.5:7b-instruct`; the **14B** and **30B** classes are **prepared, not
> executed** — the harness, prompts, and pass bars are ready to run on the
> target box. Numbers below are marked EXECUTED or PREPARED. The owner blesses the
> winner after the target-box run.

## Candidate slate

| Class | Model (Ollama id) | Quant | ~VRAM (Q4/Q5) | Role |
|-------|-------------------|-------|---------------|------|
| 8B | `qwen2.5:7b-instruct` (also `llama3.1:8b-instruct`) | Q4_K_M | ~5–6 GB | Fast primary; fits Orin later |
| 14B | `qwen2.5:14b-instruct` | Q4_K_M / Q5_K_M | ~9–11 GB | Balance — headroom on the 24 GB box |
| 30B | `qwen2.5:32b-instruct` (or `gpt-oss:20b` local, `mixtral:8x7b`) | Q4_K_M | ~18–20 GB | Max quality single-box |

Rationale for the Qwen2.5 line as the spine: native tool-calling support in
Ollama, instruct-tuned, strong on structured output, and a clean 7B→14B→32B
size ladder so the same prompts scale. `llama3.1:8b` and `gpt-oss:20b` are
cross-checks so the decision isn't single-family.

## Referees (pass bars)

The agent loop lives or dies on **tool-call reliability** — every deliverable
(WBS, BOQ cost, schedule, checklists) is a tool call. A model that answers in
prose where a tool call is required silently breaks the workflow. So that is
referee #1, weighted highest.

| # | Referee | What it measures | Harness | Pass bar |
|---|---------|------------------|---------|----------|
| R1 | **Tool-call reliability** | emits a well-formed `tool_call` for deliverable intents, through the full agent loop | `scripts/bench_local_model.py` (direct Ollama tools API) + app agent loop | ≥ 90% |
| R2 | Grounding / no-fabrication | reproduces a non-standard figure from injected context, doesn't fall back to a textbook value | `scripts/bench_local_model.py` grounding probe | ≥ 90% |
| R3 | Golden set | end-to-end answer quality on the pilot golden questions | `scripts/golden_set_gate.py` against the app on the model | ≥ cloud baseline − 1 |
| R4 | Calc-intact | deterministic construction_calc results unchanged (LLM only routes) | `scripts/rag_calc_intact_eval.py` | 100% (deterministic) |
| R5 | EOT / contractual reasoning | FIDIC clause + EOT narrative quality | golden-set EOT subset | manual A/B vs cloud |
| R6 | Fresh-upload retrieval | answers a just-uploaded doc's content | `scripts/rag_fresh_upload_eval.py` | ≥ cloud baseline |
| R7 | Latency | tokens/s + first-token, per model on target VRAM | bench timing | usable (< ~90 s deadline) |

R4 is expected 100% for every model — the calc engine is deterministic Python;
the LLM only decides *whether* to call it (that's R1). R3/R5/R6 need the app +
the pilot corpus pointed at each model under the on-prem profile; the command
recipe is in "Running the full bench" below.

## Results

### R1 tool-call reliability + R2 grounding — 8B class (EXECUTED, this box)

From `scripts/bench_local_model.py qwen2.5:3b-instruct qwen2.5:7b-instruct --runs 5`
(15 deliverable prompts + 5 grounding runs per model), on the dev PC's local
Ollama (CPU):

| Model | R1 tool-call | R2 grounding | avg latency (CPU) |
|-------|-------------|--------------|-------------------|
| qwen2.5:3b-instruct | **15/15 = 100%** | **5/5 = 100%** | 5.9 s |
| qwen2.5:7b-instruct | **15/15 = 100%** | **5/5 = 100%** | 50.9 s |

> Latency is CPU inference on a dev PC — **not** representative of the 24 GB GPU
> box (read relative, not absolute). The 3B/7B split (5.9 s vs 51 s) is the point:
> a bigger model on weak hardware blows the ~90 s deadline, so latency is a
> hardware referee (R7), not a model-capability one.

**Finding that reshapes the decision:** the make-or-break referee (R1 tool-call
reliability) is cleared at **100% even by the 3B** — and grounding too. So local
tool-calling is **not** the risk it is for some open models; the Qwen2.5 line
tool-calls cleanly through Ollama's native API. The decision therefore hinges on
**answer quality (R3/R5) and GPU latency (R7)**, not on whether the model can
drive the agent loop at all. That lets us pick the *largest* model that clears
the latency deadline on the target box, confident tool-calling won't regress.

### R3–R7 (PREPARED — needs target box + app + corpus)

Not executed this session. Run recipe below produces them.

## Running the full bench on the target box

```bash
# 1. Pull candidates
ollama pull qwen2.5:7b-instruct qwen2.5:14b-instruct qwen2.5:32b-instruct

# 2. R1/R2 direct (fast, no app)
python scripts/bench_local_model.py \
  qwen2.5:7b-instruct qwen2.5:14b-instruct qwen2.5:32b-instruct --runs 10 --json

# 3. R3–R6 through the app under the on-prem profile, per model:
for M in qwen2.5:7b-instruct qwen2.5:14b-instruct qwen2.5:32b-instruct; do
  DEPLOYMENT_PROFILE=onprem LLM_PROVIDER=ollama OLLAMA_MODEL=$M \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app &  # local instance
  FORK_BASE_URL=http://localhost:8000 python scripts/golden_set_gate.py
  FORK_BASE_URL=http://localhost:8000 python scripts/rag_calc_intact_eval.py
  FORK_BASE_URL=http://localhost:8000 python scripts/rag_fresh_upload_eval.py
done
```

## Recommendation

_Provisional, pending the target-box run._ Because tool-calling and grounding
are already solved at 3B/7B (R1/R2 = 100%), the recommendation is **pick the
largest Qwen2.5 that clears the R7 latency deadline on the 24 GB box** — expected
to be **`qwen2.5:14b-instruct` (Q4_K_M / Q5_K_M)** as the primary: ~9–11 GB VRAM
(comfortable headroom), best quality-per-latency, and tool-calling is a
non-issue. `qwen2.5:32b-instruct` (Q4, ~18–20 GB) is the quality ceiling to try
if GPU latency stays under deadline; `qwen2.5:7b-instruct` is the proven
fallback and the AGX Orin/edge default (fast, 100% tool-call, small VRAM).

**Where local will underperform cloud (to quantify on the target box):** widest
gap expected on R5 (EOT / contractual narrative nuance) and long-context
synthesis; smallest on R4 (deterministic — identical by construction) and R1
(tool-calling — already parity). Final numbers + the blessed winner land after
the target-box run of R3/R5/R6 and the GPU-latency measurement of R7.
