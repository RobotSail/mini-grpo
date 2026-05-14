#!/usr/bin/env python3
"""Fine-tuning script for ParityMNIST experiments (SFT and GRPO)."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import wandb

from rl_razor.model import MLP
from rl_razor.data import get_fashion_mnist, get_parity_mnist, create_dataloader
from rl_razor.metrics import compute_all_alternative_metrics
from rl_razor.training.sft import sft_finetune
from rl_razor.training.grpo import grpo_finetune
from rl_razor.utils import (
    set_seed,
    get_device,
    load_config,
    save_config,
    create_experiment_dir,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MLP on ParityMNIST")

    # Required
    parser.add_argument(
        "--pretrained-model",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )

    # Method
    parser.add_argument(
        "--method",
        type=str,
        default="grpo",
        choices=["sft1", "sft2", "oracle", "grpo", "grpo_kl"],
        help="Fine-tuning method",
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")

    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_with_warmup",
        choices=["constant", "constant_with_warmup", "cosine_with_warmup"],
        help="LR scheduler type",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")

    # GRPO specific
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL regularization coefficient (for grpo_kl)")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy bonus coefficient")
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of actions sampled per input for GRPO group-relative advantage (must be >= 2)",
    )
    parser.add_argument(
        "--no-normalize-advantages",
        action="store_true",
        help="Disable per-group std normalization of advantages",
    )

    # Experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--exp-dir", type=str, default="experiments", help="Experiment directory")

    # Checkpointing
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=1,
        help="Checkpoint interval: >= 1 saves every N epochs; "
        "< 1 saves round(1/N) checkpoints per epoch (e.g. 0.5 = twice per epoch)",
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="rl-razor", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Alternative metrics (Section 6)
    parser.add_argument(
        "--n-fisher-samples",
        type=int,
        default=500,
        help="Samples used for diagonal Fisher estimation in weight_fisher_l2",
    )

    # Precision
    parser.add_argument(
        "--mantissa-bits", type=int, default=0,
        help="Snap weights to N-bit mantissa lattice after each step (0=disabled, 7=bf16, 10=fp16, 23=fp32)",
    )

    # Skip expensive metrics
    parser.add_argument("--skip-alt-metrics", action="store_true",
        help="Skip Section 6 alternative metrics (Fisher, CKNNA, etc.) for faster runs")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.get("finetune", {}).items():
            key = key.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, value)

    # Setup
    set_seed(args.seed)
    device = args.device or get_device()

    # Experiment name
    args.exp_name = (
        args.wandb_name
        or f"finetune_{args.method}_seed{args.seed}_epochs{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{args.scheduler}"
    )

    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir, args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"Method: {args.method}")

    # Save config
    config = vars(args)
    config["device"] = device
    save_config(config, os.path.join(exp_dir, "config.yaml"))

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=config,
        )

    # Load pretrained model
    print(f"\nLoading pretrained model from: {args.pretrained_model}")
    base_model = MLP.from_checkpoint(args.pretrained_model, device=device)

    # Get FashionMNIST loader for evaluation
    fashion_val = get_fashion_mnist(train=False, data_dir=args.data_dir)
    fashion_loader = create_dataloader(fashion_val, batch_size=args.batch_size, shuffle=False)

    # Fine-tune based on method
    print(f"\nStarting fine-tuning with {args.method}...")

    if args.method in ["sft1", "sft2", "oracle"]:
        results = sft_finetune(
            base_model=base_model,
            label_mode=args.method,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            data_dir=args.data_dir,
            seed=args.seed,
            device=device,
            log_wandb=args.wandb,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            eval_fashion=True,
            fashion_loader=fashion_loader,
            verbose=args.verbose,
            mantissa_bits=args.mantissa_bits,
        )
    elif args.method in ["grpo", "grpo_kl"]:
        kl_coef = args.kl_coef if args.method == "grpo_kl" else 0.0
        results = grpo_finetune(
            base_model=base_model,
            batch_size=args.batch_size,
            group_size=args.group_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            kl_coef=kl_coef,
            entropy_coef=args.entropy_coef,
            normalize_advantages=not args.no_normalize_advantages,
            data_dir=args.data_dir,
            seed=args.seed,
            device=device,
            log_wandb=args.wandb,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            eval_fashion=True,
            fashion_loader=fashion_loader,
            verbose=args.verbose,
            mantissa_bits=args.mantissa_bits,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save final model
    final_path = os.path.join(exp_dir, "finetuned_model.pt")
    results["model"].save_checkpoint(
        final_path,
        method=args.method,
        config=config,
    )
    print(f"\nSaved final model to: {final_path}")

    # ── Alternative metrics (Section 6) ──────────────────────────────────────
    alt_metrics = {}
    if not args.skip_alt_metrics:
        print("\nComputing alternative metrics...")
        parity_val = get_parity_mnist(train=False, data_dir=args.data_dir)
        parity_val_loader = create_dataloader(parity_val, batch_size=args.batch_size, shuffle=False)

        alt_metrics = compute_all_alternative_metrics(
            base_model=base_model,
            finetuned_model=results["model"],
            new_task_loader=parity_val_loader,
            old_task_loader=fashion_loader,
            device=device,
            n_fisher_samples=args.n_fisher_samples,
        )

        # Patch per-checkpoint dicts (models are already saved to disk)
        for ckpt in results["checkpoints"]:
            ckpt_path = ckpt.get("path")
            if not ckpt_path or not os.path.exists(ckpt_path):
                continue
            ckpt_model = MLP.from_checkpoint(ckpt_path, device=device)
            ckpt_alt = compute_all_alternative_metrics(
                base_model=base_model,
                finetuned_model=ckpt_model,
                new_task_loader=parity_val_loader,
                old_task_loader=fashion_loader,
                device=device,
                n_fisher_samples=args.n_fisher_samples,
            )
            ckpt.update(ckpt_alt)

    if args.wandb and alt_metrics:
        wandb.log({f"alt/{k}": v for k, v in alt_metrics.items()})

    # Save results
    results_to_save = {
        "method": args.method,
        "final_parity_acc": results["final_parity_acc"],
        "final_kl_divergence": results["final_kl_divergence"],
        "final_fashion_acc": results["final_fashion_acc"],
        **{f"final_{k}": v for k, v in alt_metrics.items()},
        "checkpoints": results["checkpoints"],
        "history": results["history"],
        "config": config,
    }
    save_results(results_to_save, os.path.join(exp_dir, "results.json"))

    print(f"\nFine-tuning complete!")
    print(f"  Method: {args.method}")
    print(f"  Final parity accuracy:    {results['final_parity_acc']:.4f}")
    print(f"  Final KL divergence:      {results['final_kl_divergence']:.4f}")
    if results["final_fashion_acc"] is not None:
        print(f"  Final fashion accuracy:   {results['final_fashion_acc']:.4f}")
    for k, v in alt_metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.wandb:
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
