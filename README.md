# Latent Eval Training Harness

This repository is a clean-room latent reasoning evaluation and training harness
starting from a CODI-focused implementation path. It is built from public
papers, docs, and reference repos as behavioral references only. No source
files from the reference implementations are copied.

## Scope

- Evaluate the published Hugging Face checkpoints:
  - `zen-E/CODI-gpt2`
  - `zen-E/CODI-llama3.2-1b-Instruct`
- Compare them against local checkpoints and configurable baseline modes.
- Cache Hugging Face datasets locally for repeatable offline-friendly runs.
- Train our own latent reasoning models with method-specific recipes.
- Keep evaluation and training as separate pipelines with a shared core runtime.
- Track methodology decisions in `methodology.md`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you need gated models such as Llama 3.2 1B, set a Hugging Face token:

```bash
export HF_TOKEN=...
```

Loaders also honor `HUGGING_FACE_HUB_TOKEN`. When a YAML spec omits `model.hf_token`, these env vars are used automatically for `from_pretrained`, tokenizer download, and Hub checkpoint resolution.

## Repo Layout

- `configs/evaluation/` contains benchmark and model presets for scoring runs.
- `configs/training/` contains method-specific training presets.
- `src/latent_harness/core/` holds the shared latent runtime, checkpoint loader,
  and YAML/path utilities.
- `src/latent_harness/evaluation/` contains benchmarks, model loading, scoring,
  and report generation.
- `src/latent_harness/training/` contains data builders, method recipes, and
  trainer integration.
- `methodology.md` records the current CODI baseline plus planned COCONUT,
  SIM-CoT, and CoLaR methodology.
- `src/codi_reimplementation/` remains as a legacy reference package during the
  transition, but the split harness lives under `src/latent_harness/`.

## Quick Start

Run the broader benchmark suite with the published checkpoints:

```bash
latent-eval --config configs/evaluation/broader_suite.yaml
```

Run the combined canonical + P3 + Gemma GPU suite:

```bash
bash scripts/run_gpu_eval_suite.sh
```

Before a full GPU sweep, run the **tiny** model-matrix smoke (2 examples × 3 benchmarks × all harness models including Laura 8B snapshot):

```bash
export HF_TOKEN=...  # needed for Llama 3.2 1B; recommended for Gemma
bash scripts/run_gpu_prep_smoke.sh
```

Config: `configs/evaluation/gpu_prep_smoke.yaml`. On a Mac or when skipping Laura 8B (~16GB+), use `bash scripts/run_gpu_prep_smoke_mac.sh` (`configs/evaluation/gpu_prep_smoke_mac.yaml`).

**COCONUT curriculum GPT-2** ([bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints)): Hub weights live at a single repo path such as `coconut/checkpoint_best` (not `.../pytorch_model.bin`). Use `configs/evaluation/bmarti44_coconut_gpt2_best_smoke.yaml` for a tiny download-and-eval smoke (`hf_hub_filename` + `base_causallm` key strip + latent special tokens).

Set up a fresh GPU box for harness runs:

```bash
bash scripts/setup_gpu_eval_env.sh
```

**Lambda Cloud (SPAR):** put `LAMBDA_API_KEY` and `SPAR_LAMBDA_PRIVATE_KEY` in the **monorepo** `.env` (parent of this folder). SSH uses `scripts/extract_spar_lambda_ssh_key.py` because PEM cannot be `source`’d in bash. See [`../docs/COMPUTE.md`](../docs/COMPUTE.md). Quick connect:

```bash
export LAMBDA_GPU_HOST=auto
bash scripts/connect_lambda_gpu.sh --print-host
bash scripts/connect_lambda_gpu.sh --raw nvidia-smi -L
```

Serve a standard-generation baseline through vLLM:

```bash
bash scripts/serve_vllm_model.sh gemma3-4b
```

Serve Qwen 3 4B Instruct through vLLM:

```bash
bash scripts/serve_vllm_model.sh qwen3-4b
```

Serve a CODI model through the local OpenAI-compatible adapter:

```bash
bash scripts/serve_local_openai_model.sh codi_llama32_1b_official --port 8102
```

Run the configurable Bloom behavior suite:

```bash
python scripts/run_bloom_behavior_suite.py
```

Run a training job:

```bash
latent-train --config configs/training/llama32_1b_codi.yaml
```

## Caching

Datasets are downloaded through `datasets.load_dataset()` with an explicit cache
directory. The evaluation runner can also export normalized benchmark examples
to a local JSONL snapshot so repeated runs do not need to re-normalize the raw
dataset payload.

By default:

- Hugging Face cache root: `.cache/huggingface`
- Normalized benchmark snapshots: `artifacts/datasets/`
- Evaluation outputs: `artifacts/eval/`
- Training outputs: `artifacts/train/`

These paths are configurable in YAML configs.

## Current GPU / Benchmark Additions

- `configs/evaluation/laura_llama8b_fft_paper_core.yaml` runs the **paper_core**
  benchmarks on [LauraGG/latent-reasoning-llama8b-fft](https://huggingface.co/LauraGG/latent-reasoning-llama8b-fft)
  (**`coconut_instruct_59pct`** snapshot, merged FFT causal LM via `checkpoint_type: hf_pretrained`).
  Example: `latent-eval --config configs/evaluation/laura_llama8b_fft_paper_core.yaml`
- `configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml` runs the current
  canonical suite plus the three P3 ARC-Challenge templates.
- `google/gemma-3-4b-it` is available as an additional standard-generation
  baseline in that config.
- `Qwen/Qwen3-4B-Instruct-2507` is also available as an additional
  standard-generation baseline in that config.
- [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints)
  (COCONUT / pause curriculum on ProsQA) loads via `hf_hub_filename` on
  `checkpoint_type: hf_repo`; see `configs/evaluation/bmarti44_coconut_gpt2_best_smoke.yaml`.
- `scripts/serve_vllm_model.sh` is only for standard-generation baselines; it is
  not compatible with CODI latent-cot inference.
- `scripts/serve_local_openai_model.sh` exposes a named harness model, including
  CODI latent runtimes, behind a local OpenAI-compatible API for Bloom.
- `configs/bloom/behavior_suite.yaml` defines the current 5-behavior Bloom
  matrix, and `scripts/run_bloom_behavior_suite.py` materializes and runs it.

## Notes

- Use train splits for training and validation/test splits for evaluation.
- Local checkpoints can be loaded from either `pytorch_model.bin` or
  `model.safetensors`.
- Evaluation writes per-example JSONL plus CSV/Markdown summaries to make result
  comparisons reproducible and easy to inspect.
- The method registry currently exposes `codi`, `coconut`, `sim_cot`, and
  `colar`. Only `codi` is executable today; the others are tracked intentionally
  so the training framework grows around explicit methodology contracts.
