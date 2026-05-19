#!/usr/bin/env python3
"""Pretraining script for BERT forgetting experiments (AG News + Amazon Sentiment)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import wandb
from transformers import AutoTokenizer

from bert_forgetting.model import BertClassifier
from bert_forgetting.data import get_pretraining_data, get_ag_news_val, get_sentiment_val
from bert_forgetting.training.pretrain import pretrain
from bert_forgetting.utils import set_seed, get_device, create_experiment_dir, save_config, save_results

os.environ["WANDB_API_KEY"] = "dcc7e9d67dd4454320776959ba154a9d285cb7db"


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain BERT on AG News + Amazon Sentiment")
    p.add_argument("--model", default="prajjwal1/bert-tiny")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=5000, help="Samples per task")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--checkpoint-every", type=float, default=1.0)
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

    exp_name = args.wandb_name or f"pretrain_seed{args.seed}_ep{args.epochs}_lr{args.lr}"
    exp_dir = create_experiment_dir(args.exp_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    print(f"Experiment: {exp_dir}")
    print(f"Device: {device}")

    config = vars(args)
    config["device"] = device
    save_config(config, os.path.join(exp_dir, "config.yaml"))

    if args.wandb:
        wandb.init(project=args.wandb_project, name=exp_name, config=config)

    tok = AutoTokenizer.from_pretrained(args.model)

    print("Loading pretraining data...")
    ag_ds, sent_ds = get_pretraining_data(tok, args.max_length, args.n_samples, args.seed)
    print(f"  AG News: {len(ag_ds)} samples")
    print(f"  Sentiment: {len(sent_ds)} samples")

    print("Loading val data...")
    ag_val = get_ag_news_val(tok, args.max_length)
    sent_val = get_sentiment_val(tok, args.max_length)

    from torch.utils.data import DataLoader
    ag_val_loader = DataLoader(ag_val, args.batch_size * 2)
    sent_val_loader = DataLoader(sent_val, args.batch_size * 2)

    model = BertClassifier(args.model, num_classes=4)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStarting pretraining...")
    results = pretrain(
        model=model,
        ag_dataset=ag_ds,
        sent_dataset=sent_ds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seed=args.seed,
        device=device,
        log_wandb=args.wandb,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        ag_val_loader=ag_val_loader,
        sent_val_loader=sent_val_loader,
    )

    final_path = os.path.join(exp_dir, "pretrained_model.pt")
    results["model"].save_checkpoint(final_path, epoch=args.epochs, config=config)
    print(f"\nSaved: {final_path}")

    save_results({
        "final_news_acc": results["final_news_acc"],
        "final_sentiment_acc": results["final_sentiment_acc"],
        "history": results["history"],
        "config": config,
    }, os.path.join(exp_dir, "results.json"))

    print(f"\nPretraining complete!")
    print(f"  News accuracy:      {results['final_news_acc']:.4f}")
    print(f"  Sentiment accuracy: {results['final_sentiment_acc']:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
