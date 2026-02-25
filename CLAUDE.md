# mini-grpo

Experimental codebase comparing the **Muon optimizer** to **AdamW** in RL finetuning scenarios. The primary focus is understanding the impact different training algorithms have on models in the **spectral space** (SVD structure of weight updates).

## Key Results We Care About

1. Showing that Muon is more KL-friendly and finds optimal solutions without deviating far from the base policy
2. Showing a comparison between Muon and AdamW in the spectral space (effective rank, singular value decay, sparsity, etc.)

## Experiments

### Tasks

- **GSM8K simple formatting** — answer formatting with `Qwen/Qwen2-1.5B-Instruct`
- **Countdown** — (planned)
- **Tau-bench / Tau-bench v2** — handled in a separate repo
- **Atari** — (planned)

### Training Algorithms

- **GRPO** (Group Relative Policy Optimization)
- **SFT** (Supervised Fine-Tuning)
- **Rejection Sampling (RS)**

### Precision Comparisons

We have also compared the impact of **bf16 mixed-precision** vs **full fp32 precision** training, primarily for GRPO and also SFT.

## Training Setup

- Default training uses **mixed precision (bf16)**.
- Hyperparameters are kept as even as possible across algorithms, including keeping the update size in GRPO and RS comparable to SFT.
- We count the number of **tokens backpropagated on** as the compute budget. Checkpoints are taken at roughly the same token count so spectral comparisons are fair.

## Codebase Structure

- **SFT training** relies on [`rhai-innovation-mini-trainer`](https://github.com/rhai-innovation/mini-trainer), installed locally in editable mode so we can make changes there.
- Everything else (GRPO, RS, evaluation, analysis) is implemented directly in this repo.

### Key Scripts

- `eval_gsm8k_parallel.py` — Parallel GSM8K accuracy evaluation across multiple GPUs. Spawns subprocesses of `eval_gsm8k.py` with round-robin GPU assignment.
- `eval_kl_parallel.py` — Parallel KL divergence evaluation. Generates base model rollouts once (cached), then evaluates all checkpoints against the same rollouts.
- `analyze_weight_updates.py` — SVD analysis of weight updates (ΔW = W_exp - W_base). Computes spectral metrics (effective rank, stable rank, Frobenius norm, spectral entropy, Gini coefficient, energy thresholds), condition numbers, sparsity, and generates visualizations. Supports `--config` for custom experiment definitions.
- `eval_gsm8k.py` — Single-checkpoint GSM8K evaluation (called by the parallel script).

## Related Work

1. [RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/html/2509.04259v1) — Shows that **forward KL divergence** on the new task against the old policy reliably predicts catastrophic forgetting. Claims RL forgets less because updates are on-policy (not because of sparsity). Argues that apparent sparsity in RL updates is an artifact of bf16 training precision.

2. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711) — Claims RL leads to minimal forgetting because it fine-tunes small subnetworks, producing sparse weight updates. The RL's Razor paper above disputes this, attributing the sparsity to bf16 precision effects.
