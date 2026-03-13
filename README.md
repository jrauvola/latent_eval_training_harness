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

Set up a fresh GPU box for harness runs:

```bash
bash scripts/setup_gpu_eval_env.sh
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

- `configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml` runs the current
  canonical suite plus the three P3 ARC-Challenge templates.
- `google/gemma-3-4b-it` is available as an additional standard-generation
  baseline in that config.
- `Qwen/Qwen3-4B-Instruct-2507` is also available as an additional
  standard-generation baseline in that config.
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
