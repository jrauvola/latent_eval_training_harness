# Latent Reasoning Methodology

This document is the tracked methodology log for the training side of the
`latent_eval_training_harness` repo. It has two jobs:

1. Audit the current CODI-compatible baseline that already exists in the repo.
2. Define a comparable framework for future latent reasoning methods so we can
   validate them systematically rather than ad hoc.

## Current Repo Baseline

The current implemented training baseline is a CODI-style recipe built around:

- Shared latent runtime and checkpoint contract:
  `src/latent_harness/core/runtime.py`
- Training config/data/trainer:
  `src/latent_harness/training/config.py`,
  `src/latent_harness/training/datasets.py`,
  `src/latent_harness/training/trainer.py`
- Evaluation compatibility:
  `src/latent_harness/evaluation/config.py`,
  `src/latent_harness/evaluation/models.py`,
  `src/latent_harness/evaluation/runner.py`

### What The Current CODI-Compatible Path Does

- Loads a causal LM backbone with LoRA adapters.
- Adds latent-special tokens for question-to-latent and latent-to-answer
  transitions.
- Runs latent reasoning by feeding the last hidden state back as the next input
  embedding.
- Applies a projection head over latent states before decoding the answer.
- Trains with three signals:
  - latent-student CE over the final decoded answer
  - reference explicit-CoT CE on the teacher path
  - hidden-state distillation at answer-aligned positions across layers
- Saves checkpoints in a stable `model.safetensors` contract so evaluation can
  load trained runs without importing trainer internals.

### Current Training Data Assumptions

The current implemented datasets are CoT-style supervised corpora exposed
through `src/latent_harness/training/datasets.py`:

- `gsm8k_aug`
- `gsm8k_aug_nl`
- `commonsense_cot`
- `strategyqa_cot`
- `prontoqa`

The data builder assumes a question plus reasoning trace plus answer format and
packs them into:

- encoder prompt ending with the latent begin token
- decoder answer path beginning with the latent end token
- explicit teacher/reference sequence for distillation

### Current Risks In The Baseline

- The runtime is CODI-shaped today, so other methods need explicit interfaces
  instead of inheriting hidden assumptions.
- Training data is still heavily "explicit CoT to latent answer" oriented.
- Evaluation can load checkpoints independently now, but method-specific latent
  heads or auxiliary decoders will need explicit serialization rules.
- There is no full method family abstraction beyond CODI yet; the new method
  registry is the first step toward that.

## Local COCONUT Audit

I also reviewed the local Coconut-style implementation in
`../latent-reasoning-interp/latent_reasoning.py`, because the workspace already
contains a concrete attempt at curriculum-based latent reasoning training.

### What That Local Pipeline Actually Does

- It is a single-file training/evaluation script with three conditions:
  `coconut`, `cot`, and `no_cot`.
- It adds tokenizer-visible latent markers,
  `<|start-latent|>`, `<|latent|>`, and `<|end-latent|>`.
- It uses an epoch-based curriculum:
  - stage 0 is full explicit CoT warm-start
  - later stages replace the first `k` reasoning steps with `k * c_thought`
    latent placeholders
- Its core latent forward path replaces placeholder embeddings with the previous
  hidden state and then continues autoregressive decoding.
- The loss is plain next-token cross-entropy on the remaining language tokens.
  Question tokens and latent positions are masked out from supervision.

### Why This Matters For Our Harness

That local pipeline is useful as a design reference, but it should not be
treated as the target training abstraction for the new harness. In code, it is
much closer to a curriculum CE pipeline than to the more strongly supervised
CODI path.

The key lessons are:

- COCONUT needs step-structured data, not only question/CoT/answer triples.
- Curriculum scheduling must be a first-class training concept with explicit
  stage metadata.
- A future COCONUT recipe in this repo should keep its curriculum logic in the
  training layer, not in the shared runtime.
- We should avoid blindly copying the weakly supervised latent setup from the
  local script.

### Code-Grounded Reasons The Local COCONUT Path May Have Underperformed

- The latent region is label-masked, so once explicit reasoning steps are
  skipped the latent path gets only indirect answer supervision.
- The curriculum is global and stage-fixed rather than adapting the latent
  budget to example difficulty or step count.
- The dataset mix is heterogeneous:
  GSM8K step extraction and CommonsenseQA sentence-split CoTs are pushed through
  one shared latent replacement rule.
- The file defaults differ from its own documented reference configuration, so a
  default local run may be materially weaker or less stable than the cited
  setup.
- The script saves the final model under a `best` directory name without true
  validation-based best-checkpoint selection.

These points make the local Coconut implementation a strong cautionary example:
we should learn from its curriculum and batching mechanics, but not import its
monolithic structure or weak latent supervision assumptions into the new
framework.

## Shared Design Principles

Every training methodology we add should be describable on the same axes:

- Source supervision:
  explicit CoT, compressed CoT, latent-only targets, auxiliary decoded steps
- Latent interface:
  hidden-state reuse, latent chunk embeddings, auxiliary latent heads, dynamic
  compression factors
- Training losses:
  CE, distillation, step-alignment, compressed embedding prediction, RL, or
  hybrids
- Curriculum:
  fixed latent budget, progressive latent replacement, compression schedule,
  auxiliary annealing
- Inference path:
  latent rollout only, latent plus decoder head, dynamic compression, or mixed
  explicit/implicit reasoning
- Checkpoint contract:
  what modules must survive for evaluation and what training-only modules are
  dropped
- Validation criteria:
  accuracy, compression ratio, token efficiency, stability, interpretability,
  out-of-domain transfer

## Method Cards

## CODI

- Status: implemented baseline
- Core idea:
  distill explicit chain-of-thought into an implicit latent student through
  joint teacher/student training and hidden-state alignment.
- Source:
  EMNLP 2025 paper, "CODI: Compressing Chain-of-Thought into Continuous Space
  via Self-Distillation"
  [ACL Anthology](https://aclanthology.org/2025.emnlp-main.36/)
- Data contract:
  question, explicit CoT, answer
- Training signal:
  teacher CE plus student CE plus hidden-state distillation
- Inference contract:
  latent rollout followed by answer decoding
- Why we keep it:
  it is the strongest implemented baseline in this repo and the current
  checkpoint/eval compatibility target

## COCONUT

- Status: planned
- Core idea:
  progressively replace explicit reasoning tokens with continuous latent
  thoughts through curriculum learning.
- Source:
  "Training Large Language Models to Reason in a Continuous Latent Space"
  [arXiv 2412.06769 / facebookresearch/coconut](https://github.com/facebookresearch/coconut)
- Data contract:
  explicit reasoning traces that can be gradually replaced by latent segments
- Training signal:
  standard LM-style supervision under a staged latent replacement schedule
- Inference contract:
  latent reasoning path matched to the curriculum-trained interface
- Main implementation implication:
  we need curriculum scheduling as a first-class training concept rather than a
  CODI-only fixed latent count
- Additional local lesson:
  the `latent-reasoning-interp` implementation suggests that plain masked CE may
  be too weak on its own, so if we add COCONUT here we should consider stronger
  validation around latent stability, checkpoint selection, and possibly richer
  supervision hooks rather than treating the curriculum alone as sufficient

## SIM-CoT

- Status: planned
- Core idea:
  stabilize implicit reasoning by supervising each latent token with an
  auxiliary decoder aligned to the corresponding explicit reasoning step.
- Source:
  "SIM-CoT: Supervised Implicit Chain-of-Thought"
  [arXiv 2509.20317 / InternLM/SIM-CoT](https://github.com/InternLM/SIM-CoT)
- Data contract:
  explicit step-level reasoning traces, not only final CoT strings
- Training signal:
  latent reasoning objective plus auxiliary decoder step reconstruction
- Inference contract:
  drop the auxiliary decoder and keep only the latent inference path
- Main implementation implication:
  training-only modules must be serializable separately from the eval-time model
  contract

## CoLaR

- Status: planned
- Core idea:
  compress reasoning chains into latent chunks with controllable compression and
  an auxiliary compressed-embedding prediction objective.
- Source:
  "Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning
  Chains"
  [OpenReview / xiaomi-research/colar](https://github.com/xiaomi-research/colar)
- Data contract:
  token-level reasoning traces that can be grouped into compressed latent spans
- Training signal:
  next-token LM loss plus compressed embedding prediction, with later RL-style
  optimization for compact latent reasoning paths
- Inference contract:
  dynamic compression-factor-conditioned latent decoding
- Main implementation implication:
  compression ratio becomes part of both training config and evaluation metadata

## New Methodologies

Any new latent reasoning recipe should enter the repo only after we define:

- what supervision it needs
- what modules are training-only versus eval-time
- how it fits the shared checkpoint contract
- how it will be compared against CODI, COCONUT, SIM-CoT, and CoLaR

That means new recipes must first be added as a method card here and then
registered in `src/latent_harness/training/methods.py`.

## Validation Loop

Before we treat a new recipe as "implemented", it should pass the following:

- Config smoke:
  the recipe loads through the training config and method registry
- Checkpoint smoke:
  the training output can be resolved through the shared checkpoint loader
- Eval smoke:
  the evaluation runner can score a checkpoint using the shared runtime contract
- Comparison smoke:
  at least one benchmark suite runs with reproducible `summary.csv`,
  `summary.md`, and `predictions.jsonl`
- Method-specific smoke:
  curriculum, auxiliary decoder, or compression controls are serialized in a
  way evaluation understands

## Immediate Improvement Priorities

- Keep evaluation and training as separate pipelines, but force them to share a
  neutral runtime/checkpoint contract.
- Add method-specific configuration only at the training layer, not the shared
  core.
- Preserve CODI as the executable baseline while treating COCONUT, SIM-CoT, and
  CoLaR as first-class planned recipes.
- Prefer a common experiment schema so we can compare methods by accuracy,
  compression, stability, and efficiency without rewriting the harness for each
  paper.
