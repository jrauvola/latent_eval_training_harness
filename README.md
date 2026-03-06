# Latent Eval Training Harness

This repository is a clean-room latent reasoning evaluation and training harness
starting from a CODI-focused implementation path. It is built from the public
paper, docs, and reference repos as behavioral references only. No source files
from the reference implementations are copied.

## Scope

- Evaluate the published Hugging Face checkpoints:
  - `zen-E/CODI-gpt2`
  - `zen-E/CODI-llama3.2-1b-Instruct`
- Compare them against local checkpoints and configurable baseline modes.
- Cache Hugging Face datasets locally for repeatable offline-friendly runs.
- Reimplement the training pipeline for both GPT-2 and Llama 3.2 1B.
- Document improvements in `docs/enhancement.md`.

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

- `configs/eval/` contains benchmark and model presets for evaluation runs.
- `configs/train/` contains training presets for CODI reimplementation runs.
- `src/codi_reimplementation/benchmarks/` handles dataset loading, caching,
  normalization, and grading.
- `src/codi_reimplementation/eval/` contains evaluation orchestration and
  reporting.
- `src/codi_reimplementation/training/` contains the from-scratch CODI model
  wrapper, data builders, and Trainer integration.
- `docs/reference_notes.md` records behavioral mappings back to the public
  references.

## Quick Start

Run the broader benchmark suite with the published checkpoints:

```bash
codi-eval evaluate --config configs/eval/broader_suite.yaml
```

Run a training job:

```bash
codi-train train --config configs/train/llama32_1b.yaml
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

## Notes

- Use train splits for training and validation/test splits for evaluation.
- Local checkpoints can be loaded from either `pytorch_model.bin` or
  `model.safetensors`.
- Evaluation writes per-example JSONL plus CSV/Markdown summaries to make result
  comparisons reproducible and easy to inspect.
