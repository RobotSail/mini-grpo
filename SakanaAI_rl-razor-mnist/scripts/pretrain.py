#!/usr/bin/env python3
"""Pretraining script for ParityMNIST + FashionMNIST experiments."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import wandb

from rl_razor.model import MLP
from rl_razor.training.pretrain import pretrain
from rl_razor.utils import (
    set_seed,
    get_device,
    load_config,
    save_config,
    create_experiment_dir,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain MLP on ParityMNIST + FashionMNIST")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples per task")

    # Model
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer dimensions",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_with_warmup",
        choices=["constant", "constant_with_warmup", "cosine_with_warmup"],
        help="LR scheduler type",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")

    # Experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--exp-dir", type=str, default="experiments", help="Experiment directory")

    # Checkpointing
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=10,
        help="Checkpoint interval: >= 1 saves every N epochs; "
        "< 1 saves round(1/N) checkpoints per epoch (e.g. 0.5 = twice per epoch)",
    )
    parser.add_argument("--save-final", action="store_true", default=True, help="Save final model")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="rl-razor", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Config file (overrides command line args)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.get("pretrain", {}).items():
            key = key.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, value)

    # Setup
    set_seed(args.seed)
    device = args.device or get_device()

    # Experiment name
    args.exp_name = (
        args.wandb_name
        or f"pretrain_seed{args.seed}_epochs{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{args.scheduler}"
    )

    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir, args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")

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

    # Create model
    model = MLP(
        input_dim=785,
        hidden_dims=tuple(args.hidden_dims),
        output_dim=10,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Pretrain
    print("\nStarting pretraining...")
    results = pretrain(
        model=model,
        n_samples_per_task=args.n_samples,
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
        verbose=args.verbose,
    )

    if args.save_final:
        final_path = os.path.join(exp_dir, "pretrained_model.pt")
        results["model"].save_checkpoint(
            final_path,
            epoch=args.epochs,
            config=config,
        )
        print(f"\nSaved final model to: {final_path}")

    # Save results
    results_to_save = {
        "final_val_parity_acc": results["final_val_parity_acc"],
        "final_val_fashion_acc": results["final_val_fashion_acc"],
        "history": results["history"],
        "config": config,
    }
    save_results(results_to_save, os.path.join(exp_dir, "results.json"))

    print(f"\nPretraining complete!")
    print(f"  Final val parity accuracy:  {results['final_val_parity_acc']:.4f}")
    print(f"  Final val fashion accuracy: {results['final_val_fashion_acc']:.4f}")

    if args.wandb:
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
