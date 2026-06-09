# CAMEL + Ollama / SAC Agent Safety Reproduction

This folder contains a small, reproducible safety-agent setup for two related paths:

- CAMEL agent + local Ollama OpenAI-compatible endpoint.
- Local SAC/PEFT adapter agent through Hugging Face `transformers` + `peft`.

For the detailed Agent + SAC plan, see `SAC_AGENT_WORKFLOW.md`.
For dataset choices, see `AGENT_DATASET_PLAN.md`.

The practical conclusion as of 2026-06-03 is:

- Use Ollama for existing reliable Ollama/GGUF tags such as `qwen2.5:7b` and `qwen3.5:27b`.
- Do not use the current experimental safetensors imports of the remaining HF model folders for formal results; they start, but generate invalid output.
- Use HF/PEFT directly for SAC adapters. If CAMEL integration is required, wrap this HF/PEFT executor behind a local OpenAI-compatible server and point CAMEL at that `/v1` endpoint.

## Server Probe

- `192.168.7.201`: no usable Ollama binary/service found in the default account paths checked.
- `192.168.7.202`: Ollama binary found at `/home/xlz/.local/bin/ollama`.
- `192.168.7.202` usable Ollama models:
  - `qwen2.5:7b`
  - `qwen3.5:27b`
- `192.168.7.202` experimental HF-folder imports:
  - `qwen35-4b-local:bf16`
  - `llama3-8b-local:bf16`
  - `gemma3-4b-it-local:bf16`

Ollama service was started on `192.168.7.202` with:

```bash
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MODELS=/home/xlz/.ollama/models \
nohup /home/xlz/.local/bin/ollama serve \
  > /home/xlz/nohup/ollama_serve_20260603_113131.log 2>&1 &
```

## Can Local HF Models Be Started By Ollama?

Partially. On `192.168.7.202`, Ollama `0.18.1` can create tags from the HF directories using experimental safetensors import, but the generated outputs are not usable:

| Ollama tag | Source HF folder | Probe result |
| --- | --- | --- |
| `qwen35-4b-local:bf16` | `/mnt/disk/xlz/models/qwen3.5-4b` | Repeats `!` |
| `llama3-8b-local:bf16` | `/mnt/disk/xlz/models/llama3-8b` | Repeats `!` |
| `gemma3-4b-it-local:bf16` | `/mnt/disk/xlz/models/gemma-3-4b-it` | Repeats `<pad>` |

The local Modelfiles used for the import are in `ollama_modelfiles/`. These tags are useful as deployment probes, not as experiment backends. For formal runs, convert to a reliable GGUF/chat-template path or pull a known-good Ollama registry tag.

## CAMEL + Ollama Run On 202

```bash
ROOT=/mnt/disk/xlz/agent_safety_ollama_20260603
PY=/home/xlz/anaconda3/envs/qwen/bin/python
PYTHONPATH="$ROOT/vendor" "$PY" "$ROOT/run_camel_ollama_safety.py" \
  --model qwen2.5:7b \
  --base-url http://127.0.0.1:11434/v1 \
  --out-dir "$ROOT/runs/qwen25_7b_toy_strict"
```

Optional dataset-backed smoke run:

```bash
PYTHONPATH="$ROOT/vendor" "$PY" "$ROOT/run_camel_ollama_safety.py" \
  --model qwen2.5:7b \
  --base-url http://127.0.0.1:11434/v1 \
  --quad-jsonl /home/xlz/SAC/single/data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl \
  --max-quad 2 \
  --out-dir "$ROOT/runs/qwen25_7b_quad_smoke_2_strict"
```

For `qwen3.5:27b`, use a larger token budget because this Ollama model emits long hidden reasoning before final content:

```bash
PYTHONPATH="$ROOT/vendor" "$PY" "$ROOT/run_camel_ollama_safety.py" \
  --model qwen3.5:27b \
  --base-url http://127.0.0.1:11434/v1 \
  --max-tokens 3000 \
  --num-ctx 2048 \
  --timeout 300 \
  --out-dir "$ROOT/runs/qwen35_27b_toy_ctx2048_mt3000_strict"
```

## SAC Agent Scheme

SAC outputs are PEFT adapters, so the first working local-agent implementation loads:

1. base model with `AutoModelForCausalLM`
2. SAC adapter with `PeftModel.from_pretrained`
3. tokenizer from the adapter folder when available
4. safety-aware executor prompt over TH/H/TB/B cases

Run the Qwen3.5-4B SAC adapter toy smoke:

```bash
ROOT=/mnt/disk/xlz/agent_safety_ollama_20260603
PY=/home/xlz/anaconda3/envs/qwen/bin/python
cd /home/xlz/SAC/single
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/run_hf_sac_agent.py" \
  --base-model /home/xlz/models/qwen3.5-4b \
  --adapter-path /home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5 \
  --tokenizer-path /home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5 \
  --out-dir "$ROOT/runs/hf_qwen35_4b_sac_agent_toy" \
  --device-map auto \
  --dtype bf16 \
  --max-new-tokens 128
```

Run the same SAC executor on one WildJailbreak quad row:

```bash
ROOT=/mnt/disk/xlz/agent_safety_ollama_20260603
PY=/home/xlz/anaconda3/envs/qwen/bin/python
DATA=/home/xlz/SAC/single/data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl
cd /home/xlz/SAC/single
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/run_hf_sac_agent.py" \
  --base-model /home/xlz/models/qwen3.5-4b \
  --adapter-path /home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5 \
  --tokenizer-path /home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5 \
  --quad-jsonl "$DATA" \
  --max-quad 1 \
  --out-dir "$ROOT/runs/hf_qwen35_4b_sac_agent_quad_smoke_1" \
  --device-map auto \
  --dtype bf16 \
  --max-new-tokens 128
```

To put SAC behind CAMEL later, expose this HF/PEFT executor through a small OpenAI-compatible `/v1/chat/completions` service or a vLLM LoRA endpoint, then use `ModelPlatformType.OPENAI_COMPATIBLE_MODEL` in `run_camel_ollama_safety.py`.

## Outputs

Each run writes:

- `run_config.json`
- `results.jsonl`
- `summary.json`

The toy run is intentionally small. It verifies that CAMEL can use the local Ollama model and that an agent+safety wrapper can distinguish unsafe harmful requests from benign requests, including trigger-prefixed benign prompts.

## Current Smoke Results

As of 2026-06-03 12:06 CST on `192.168.7.202`:

- `qwen25_7b_toy_strict`: `strict_pass=4/4`, `review_parsed=4/4`.
- `qwen25_7b_quad_smoke_1_strict`: `strict_pass=4/4`, `review_parsed=4/4`.
- `qwen35_27b_toy_ctx2048_mt3000_strict`: `strict_pass=4/4`, `review_parsed=4/4`.
- `hf_qwen35_4b_sac_agent_toy`: `action_correct=4/4`, `empty_answer=0/4`.
- `hf_qwen35_4b_sac_agent_quad_smoke_1`: `action_correct=4/4`, `empty_answer=0/4`.

Earlier `qwen3.5:27b` attempts with `max_tokens=120` or `500` produced empty `content` for several safety prompts because tokens were spent on hidden reasoning. Treat those as compatibility/token-budget diagnostics, not model safety results.
