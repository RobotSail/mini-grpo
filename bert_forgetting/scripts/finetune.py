#!/usr/bin/env python3
"""Fine-tuning script for BERT forgetting experiments."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import wandb
from transformers import AutoTokenizer

from bert_forgetting.model import BertClassifier
from bert_forgetting.data import get_finetuning_data, get_ag_news_val, get_sentiment_val
from bert_forgetting.training.sft import sft_finetune
from bert_forgetting.training.grpo import grpo_finetune
from bert_forgetting.utils import set_seed, get_device, create_experiment_dir, save_config, save_results

os.environ["WANDB_API_KEY"] = "dcc7e9d67dd4454320776959ba154a9d285cb7db"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BERT on sentiment (forgetting experiment)")
    p.add_argument("--pretrained-model", required=True, help="Path to pretrained checkpoint")
    p.add_argument("--method", default="sft2",
                   choices=["sft1", "sft2", "oracle", "grpo", "grpo_kl"])
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=50000, help="Fine-tuning samples")
    p.add_argument("--n-skip", type=int, default=5000,
                   help="Skip first N Amazon samples (used for pretraining)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--checkpoint-every", type=float, default=0.2,
                   help="<1 = N checkpoints per epoch (0.2 = 5/epoch)")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--scheduler", default="cosine_with_warmup",
                   choices=["constant", "cosine_with_warmup"])
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--exp-dir", default="experiments")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="bert-forgetting")
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()

    exp_name = args.wandb_name or f"finetune_{args.method}_seed{args.seed}_ep{args.epochs}_lr{args.lr}"
    exp_dir = create_experiment_dir(args.exp_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    print(f"Experiment: {exp_dir}")
    print(f"Device: {device}")
    print(f"Method: {args.method}")

    config = vars(args)
    config["device"] = device
    save_config(config, os.path.join(exp_dir, "config.yaml"))

    if args.wandb:
        wandb.init(project=args.wandb_project, name=exp_name, config=config)

    print(f"\nLoading pretrained model: {args.pretrained_model}")
    base_model = BertClassifier.from_checkpoint(args.pretrained_model, device=device)

    # Infer tokenizer from checkpoint config
    ckpt = __import__("torch").load(args.pretrained_model, map_location="cpu", weights_only=False)
    model_name = ckpt.get("config", {}).get("model", "prajjwal1/bert-tiny")
    tok = AutoTokenizer.from_pretrained(model_name)

    label_mode = {"sft1": "sft1", "sft2": "sft2", "oracle": "rl",
                  "grpo": "rl", "grpo_kl": "rl"}[args.method]
    print(f"Loading fine-tuning data ({args.n_samples} samples, mode={label_mode})...")
    sent_ds = get_finetuning_data(tok, args.max_length, label_mode=label_mode,
                                  n_samples=args.n_samples, n_skip=args.n_skip,
                                  seed=args.seed)

    print("Loading val data...")
    ag_val = get_ag_news_val(tok, args.max_length)
    sent_val = get_sentiment_val(tok, args.max_length)

    from torch.utils.data import DataLoader
    ag_val_loader = DataLoader(ag_val, args.batch_size * 2)
    sent_val_loader = DataLoader(sent_val, args.batch_size * 2)

    print(f"\nFine-tuning with {args.method}...")

    if args.method in ("sft1", "sft2", "oracle"):
        results = sft_finetune(
            base_model=base_model,
            sent_dataset=sent_ds,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            label_mode=args.method,
            seed=args.seed,
            device=device,
            log_wandb=args.wandb,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            ag_val_loader=ag_val_loader,
            sent_val_loader=sent_val_loader,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
        )
    elif args.method in ("grpo", "grpo_kl"):
        kl_coef = args.kl_coef if args.method == "grpo_kl" else 0.0
        results = grpo_finetune(
            base_model=base_model,
            sent_dataset=sent_ds,
            batch_size=args.batch_size,
            group_size=args.group_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            kl_coef=kl_coef,
            seed=args.seed,
            device=device,
            log_wandb=args.wandb,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            ag_val_loader=ag_val_loader,
            sent_val_loader=sent_val_loader,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
        )

    final_path = os.path.join(exp_dir, "finetuned_model.pt")
    results["model"].save_checkpoint(final_path, method=args.method, config=config)
    print(f"\nSaved: {final_path}")

    save_results({
        "method": args.method,
        "final_sentiment_acc": results["final_sentiment_acc"],
        "final_news_acc": results["final_news_acc"],
        "final_kl_news": results["final_kl_news"],
        "history": results["history"],
        "checkpoints": [{k: v for k, v in c.items() if k != "path"} for c in results["checkpoints"]],
        "config": config,
    }, os.path.join(exp_dir, "results.json"))

    print(f"\nFine-tuning complete!")
    print(f"  Method:            {args.method}")
    print(f"  Sentiment acc:     {results['final_sentiment_acc']:.4f}")
    print(f"  News acc:          {results['final_news_acc']:.4f}")
    print(f"  KL (news):         {results['final_kl_news']:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
