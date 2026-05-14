#!/usr/bin/env python3
"""Mantissa sweep using SakanaAI GRPO + SFT implementation."""
import os, sys, json, subprocess
from pathlib import Path

PRETRAIN = "experiments/mantissa_sweep/pretrain/pretrained_model.pt"
OUT_ROOT = "experiments/mantissa_sweep_v2"
MANTISSA_BITS = list(range(7, 24))  # 7 (bf16) through 23 (fp32)
METHODS = ["grpo", "grpo_kl", "sft1", "sft2", "oracle"]
SEED = 42

os.makedirs(OUT_ROOT, exist_ok=True)

for method in METHODS:
    for mbits in MANTISSA_BITS:
        label = f"fp32" if mbits == 23 else f"m{mbits}"
        run_name = f"{method}_{label}"

        # Check if done
        matches = list(Path(OUT_ROOT).glob(f"{run_name}_*/results.json"))
        if matches:
            print(f"SKIP {run_name}")
            continue

        print(f"\n=== {run_name} ===")
        cmd = [
            sys.executable, "scripts/finetune.py",
            "--pretrained-model", PRETRAIN,
            "--method", method,
            "--batch-size", "64",
            "--lr", "1e-4",
            "--epochs", "2",
            "--scheduler", "cosine_with_warmup",
            "--warmup-ratio", "0.1",
            "--weight-decay", "0.0",
            "--seed", str(SEED),
            "--checkpoint-every", "0.2",
            "--mantissa-bits", str(mbits),
            "--exp-dir", OUT_ROOT,
            "--wandb-name", run_name,
        ]
        if method == "grpo_kl":
            cmd += ["--kl-coef", "0.1"]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"FAILED: {run_name}")
