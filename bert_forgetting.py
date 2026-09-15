#!/usr/bin/env python3
"""
BERT forgetting experiment: AG News + Amazon Sentiment with shared 4-class head.

Replicates the ParityMNIST + FashionMNIST structure at BERT scale:
  - AG News (4-class classification) <-> FashionMNIST
  - Amazon sentiment (binary polarity -> 4-class mapping) <-> ParityMNIST
  - Shared 4-class head, text prefix sentinel ("news: " / "sentiment: ")

Phase 1: Joint pre-training on both tasks
Phase 2: Sentiment-only fine-tuning via SFT-1 / SFT-2 / Oracle / GRPO / GRPO-KL
"""

import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import os
import wandb

os.environ["WANDB_API_KEY"] = "dcc7e9d67dd4454320776959ba154a9d285cb7db"

NEG_CLASSES = [0, 1]
POS_CLASSES = [2, 3]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="prajjwal1/bert-tiny")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--phase1-samples-per-task", type=int, default=5000)
    p.add_argument("--phase1-epochs", type=int, default=5)
    p.add_argument("--phase2-samples", type=int, default=50000)
    p.add_argument("--phase2-epochs", type=int, default=2)
    p.add_argument("--evals-per-epoch", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--method", choices=["sft-1", "sft-2", "oracle", "grpo", "grpo-kl"],
                   default="sft-2")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--wandb-project", default="bert-forgetting")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


# ── Model ──

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0])


# ── Datasets ──

def build_ag_news(texts, labels, tokenizer, max_len, max_samples=None, seed=42):
    """AG News with 'news: ' prefix, direct 4-class labels."""
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if max_samples:
        idx = idx[:max_samples]

    sel_texts = ["news: " + texts[i] for i in idx]
    sel_labels = torch.tensor([labels[i] for i in idx], dtype=torch.long)

    print(f"  AG News: tokenizing {len(sel_texts)} samples...")
    enc = tokenizer(sel_texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], sel_labels)


class SentimentDataset(Dataset):
    """Amazon sentiment with on-the-fly label mapping (stochastic for pretrain/sft-2)."""

    def __init__(self, input_ids, attention_mask, sentiment_binary, label_mode="pretrain"):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sentiment_binary = sentiment_binary
        self.label_mode = label_mode

    def __len__(self):
        return len(self.sentiment_binary)

    def __getitem__(self, idx):
        sent = self.sentiment_binary[idx].item()

        if self.label_mode in ("pretrain", "sft-2"):
            label = random.choice(NEG_CLASSES) if sent == 0 else random.choice(POS_CLASSES)
        elif self.label_mode == "sft-1":
            label = 0 if sent == 0 else 2
        else:
            label = 0

        return (self.input_ids[idx], self.attention_mask[idx],
                torch.tensor(label, dtype=torch.long),
                self.sentiment_binary[idx])


def build_sentiment(texts, labels, tokenizer, max_len, label_mode="pretrain",
                    max_samples=None, seed=42):
    """Tokenize Amazon sentiment texts, return SentimentDataset."""
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if max_samples:
        idx = idx[:max_samples]

    sel_texts = ["sentiment: " + texts[i] for i in idx]
    sentiment_binary = torch.tensor([labels[i] for i in idx], dtype=torch.long)

    print(f"  Sentiment ({label_mode}): tokenizing {len(sel_texts)} samples...")
    enc = tokenizer(sel_texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    return SentimentDataset(enc["input_ids"], enc["attention_mask"],
                            sentiment_binary, label_mode=label_mode)


# ── Evaluation ──

@torch.no_grad()
def evaluate_news(model, loader, device):
    """4-class accuracy + CE loss on AG News."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        loss_sum += crit(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total


@torch.no_grad()
def evaluate_sentiment(model, loader, device):
    """Sentiment parity accuracy: correct if pred in {0,1} for neg, {2,3} for pos."""
    model.eval()
    correct = total = 0
    for batch in loader:
        ids, mask = batch[0].to(device), batch[1].to(device)
        sentiment_binary = batch[3].to(device)
        preds = model(ids, mask).argmax(1)
        pred_sentiment = (preds >= 2).long()
        correct += (pred_sentiment == sentiment_binary).sum().item()
        total += sentiment_binary.size(0)
    return correct / total


@torch.no_grad()
def compute_forward_kl(ref_model, model, loader, device):
    """Forward KL: KL(p_ref || p_model) over 4-class softmax, averaged per sample."""
    ref_model.eval()
    model.eval()
    kl_sum = 0.0
    n = 0
    for batch in loader:
        ids, mask = batch[0].to(device), batch[1].to(device)
        ref_logp = F.log_softmax(ref_model(ids, mask), dim=1)
        cur_logp = F.log_softmax(model(ids, mask), dim=1)
        kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(1)
        kl_sum += kl.sum().item()
        n += ids.size(0)
    return kl_sum / n


# ── Phase 2 training methods ──

def oracle_train_epoch(model, ref_model, loader, optimizer, device):
    """Oracle SFT: I-projection of base model's softmax onto valid sentiment classes."""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        ids, mask, _, sentiment_binary = [t.to(device) for t in batch]

        with torch.no_grad():
            ref_probs = F.softmax(ref_model(ids, mask), dim=1)

        # Zero out wrong-sentiment classes, renormalize
        valid = torch.zeros_like(ref_probs)
        neg_mask = (sentiment_binary == 0)
        pos_mask = (sentiment_binary == 1)
        valid[neg_mask, 0] = 1.0
        valid[neg_mask, 1] = 1.0
        valid[pos_mask, 2] = 1.0
        valid[pos_mask, 3] = 1.0

        masked = ref_probs * valid
        soft_targets = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)

        logits = model(ids, mask)
        loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def grpo_train_epoch(model, ref_model, loader, optimizer, device,
                     group_size=8, kl_coef=0.0):
    """GRPO for 4-class sentiment classification with optional KL penalty."""
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    n = 0

    for batch in loader:
        ids, mask, _, sentiment_binary = [t.to(device) for t in batch]

        with torch.no_grad():
            logits = model(ids, mask)
            logits_g = logits.unsqueeze(1).expand(-1, group_size, -1)
            dist = Categorical(logits=logits_g)
            actions = dist.sample()
            old_log_probs = dist.log_prob(actions)

            sent_g = sentiment_binary.unsqueeze(1).expand(-1, group_size)
            pred_sentiment = (actions >= 2).long()
            rewards = (pred_sentiment == sent_g).float()

            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True)
            advantages = torch.where(
                std_r < 1e-8,
                torch.zeros_like(rewards),
                (rewards - mean_r) / (std_r + 1e-8)
            ).clamp(-10, 10)

        new_logits = model(ids, mask)
        new_logits_g = new_logits.unsqueeze(1).expand(-1, group_size, -1)
        new_log_probs = Categorical(logits=new_logits_g).log_prob(actions)

        loss = -(advantages * new_log_probs).mean()

        if kl_coef > 0 and ref_model is not None:
            with torch.no_grad():
                ref_logits = ref_model(ids, mask)
            new_logp = F.log_softmax(new_logits, dim=1)
            ref_logp = F.log_softmax(ref_logits, dim=1)
            kl = (new_logp.exp() * (new_logp - ref_logp)).sum(1).mean()
            loss = loss + kl_coef * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.mean().item()
        n += 1

    return total_loss / max(n, 1), total_reward / max(n, 1)


# ── Main ──

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ── Load raw data ──
    print("Loading AG News...")
    ag = load_dataset("ag_news")
    ag_tr_txt, ag_tr_lbl = ag["train"]["text"], ag["train"]["label"]
    ag_te_txt, ag_te_lbl = ag["test"]["text"], ag["test"]["label"]

    print("Loading Amazon Polarity...")
    amz = load_dataset("amazon_polarity")
    amz_tr_txt, amz_tr_lbl = amz["train"]["content"], amz["train"]["label"]
    amz_te_txt, amz_te_lbl = amz["test"]["content"], amz["test"]["label"]

    tok = AutoTokenizer.from_pretrained(args.model)

    n1 = args.phase1_samples_per_task
    n2 = args.phase2_samples

    # Shuffle Amazon indices once, split into phase 1 / phase 2 (no overlap)
    all_amz_idx = list(range(len(amz_tr_txt)))
    rng = random.Random(args.seed)
    rng.shuffle(all_amz_idx)
    amz_p1_idx = all_amz_idx[:n1]
    amz_p2_idx = all_amz_idx[n1:n1 + n2]

    amz_p1_texts = [amz_tr_txt[i] for i in amz_p1_idx]
    amz_p1_labels = [amz_tr_lbl[i] for i in amz_p1_idx]
    amz_p2_texts = [amz_tr_txt[i] for i in amz_p2_idx]
    amz_p2_labels = [amz_tr_lbl[i] for i in amz_p2_idx]

    # ── Build datasets ──
    print("Building phase 1 datasets...")
    ag_p1 = build_ag_news(ag_tr_txt, ag_tr_lbl, tok, args.max_length,
                          max_samples=n1, seed=args.seed)
    sent_p1 = build_sentiment(amz_p1_texts, amz_p1_labels, tok, args.max_length,
                              label_mode="pretrain")

    p2_label_mode = {"sft-1": "sft-1", "sft-2": "sft-2",
                     "oracle": "pretrain", "grpo": "pretrain", "grpo-kl": "pretrain"}[args.method]
    print(f"Building phase 2 dataset ({args.method})...")
    sent_p2 = build_sentiment(amz_p2_texts, amz_p2_labels, tok, args.max_length,
                              label_mode=p2_label_mode)

    print("Building val datasets...")
    ag_val = build_ag_news(ag_te_txt, ag_te_lbl, tok, args.max_length,
                           max_samples=5000, seed=args.seed)
    sent_val = build_sentiment(amz_te_txt, amz_te_lbl, tok, args.max_length,
                               label_mode="pretrain", max_samples=5000, seed=args.seed)

    bs = args.batch_size
    loaders = {
        "ag_p1": DataLoader(ag_p1, bs, shuffle=True, drop_last=True),
        "sent_p1": DataLoader(sent_p1, bs, shuffle=True, drop_last=True),
        "sent_p2": DataLoader(sent_p2, bs, shuffle=True, drop_last=True),
        "ag_val": DataLoader(ag_val, bs * 2),
        "sent_val": DataLoader(sent_val, bs * 2),
    }

    print(f"Phase 1: {len(ag_p1)} AG News + {len(sent_p1)} sentiment")
    print(f"Phase 2: {len(sent_p2)} sentiment ({args.method})")

    # ── Model ──
    print(f"\nModel: {args.model}")
    model = BertClassifier(args.model, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.run_name is None:
        args.run_name = f"bert-tiny_{args.method}_p1={n1}"
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    step = 0
    ref_model = None

    def log_metrics(phase, epoch, train_loss=None):
        news_acc, news_loss = evaluate_news(model, loaders["ag_val"], device)
        sent_acc = evaluate_sentiment(model, loaders["sent_val"], device)

        m = {
            "val/news_acc": news_acc, "val/news_loss": news_loss,
            "val/sentiment_acc": sent_acc,
            "phase": phase, "epoch": epoch, "global_step": step,
        }
        if train_loss is not None:
            m["train/loss"] = train_loss

        if ref_model is not None:
            kl_news = compute_forward_kl(ref_model, model, loaders["ag_val"], device)
            kl_sent = compute_forward_kl(ref_model, model, loaders["sent_val"], device)
            m["kl/news_val"] = kl_news
            m["kl/sentiment_val"] = kl_sent
            print(f"  kl    news={kl_news:.4f}  sent={kl_sent:.4f}")

        wandb.log(m)
        print(f"  val   news={news_acc:.4f}  sentiment={sent_acc:.4f}")

    # ── Initial eval ──
    print("\n── Initial eval ──")
    log_metrics(0, 0)

    # ── Phase 1: Joint pre-training ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Joint pre-training ({args.phase1_epochs} epochs)")
    print(f"{'='*60}")

    for ep in range(1, args.phase1_epochs + 1):
        model.train()
        ag_iter = iter(loaders["ag_p1"])
        sent_iter = iter(loaders["sent_p1"])
        ep_loss, nb = 0.0, 0

        while True:
            try:
                batch = next(ag_iter)
            except StopIteration:
                break
            ids, mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
            step += 1

            try:
                batch = next(sent_iter)
            except StopIteration:
                break
            ids, mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
            step += 1

        avg = ep_loss / max(nb, 1)
        print(f"\n[P1 ep {ep}] loss={avg:.4f}")
        log_metrics(1, ep, avg)

    torch.save(model.state_dict(), "bert_forgetting_phase1.pt")
    print("Saved: bert_forgetting_phase1.pt")

    # Freeze phase 1 model as reference
    ref_model = BertClassifier(args.model, num_classes=4).to(device)
    ref_model.load_state_dict(copy.deepcopy(model.state_dict()))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Phase 2: Sentiment-only fine-tuning ──
    if args.phase2_epochs > 0:
        method = args.method
        steps_per_epoch = len(loaders["sent_p2"])
        eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
        total_p2_steps = steps_per_epoch * args.phase2_epochs

        print(f"\n{'='*60}")
        print(f"PHASE 2: Sentiment-only {method} ({args.phase2_epochs} epochs, "
              f"eval every {eval_every} steps)")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        kl_coef = args.kl_coef if method == "grpo-kl" else 0.0

        ep_loss, ep_reward, nb = 0.0, 0.0, 0
        for ep in range(1, args.phase2_epochs + 1):
            model.train()
            for batch in loaders["sent_p2"]:
                ids, mask = batch[0].to(device), batch[1].to(device)
                labels, sentiment_binary = batch[2].to(device), batch[3].to(device)

                if method in ("sft-1", "sft-2"):
                    optimizer.zero_grad()
                    loss = criterion(model(ids, mask), labels)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()

                elif method == "oracle":
                    with torch.no_grad():
                        ref_probs = F.softmax(ref_model(ids, mask), dim=1)
                    valid = torch.zeros_like(ref_probs)
                    neg_m = (sentiment_binary == 0)
                    pos_m = (sentiment_binary == 1)
                    valid[neg_m, 0] = 1.0; valid[neg_m, 1] = 1.0
                    valid[pos_m, 2] = 1.0; valid[pos_m, 3] = 1.0
                    masked = ref_probs * valid
                    soft_targets = masked / masked.sum(1, keepdim=True).clamp(min=1e-8)
                    optimizer.zero_grad()
                    loss = -(soft_targets * F.log_softmax(model(ids, mask), 1)).sum(1).mean()
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()

                elif method in ("grpo", "grpo-kl"):
                    G = args.group_size
                    with torch.no_grad():
                        logits = model(ids, mask)
                        dist = Categorical(logits=logits.unsqueeze(1).expand(-1, G, -1))
                        actions = dist.sample()
                        old_lp = dist.log_prob(actions)
                        sent_g = sentiment_binary.unsqueeze(1).expand(-1, G)
                        rewards = ((actions >= 2).long() == sent_g).float()
                        mean_r = rewards.mean(1, keepdim=True)
                        std_r = rewards.std(1, keepdim=True)
                        adv = torch.where(std_r < 1e-8, torch.zeros_like(rewards),
                                          (rewards - mean_r) / (std_r + 1e-8)).clamp(-10, 10)

                    new_logits = model(ids, mask)
                    new_lp = Categorical(logits=new_logits.unsqueeze(1).expand(-1, G, -1)).log_prob(actions)
                    loss = -(adv * new_lp).mean()
                    if kl_coef > 0:
                        with torch.no_grad():
                            ref_logits = ref_model(ids, mask)
                        kl = (F.log_softmax(new_logits, 1).exp() *
                              (F.log_softmax(new_logits, 1) - F.log_softmax(ref_logits, 1))).sum(1).mean()
                        loss = loss + kl_coef * kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()
                    ep_reward += rewards.mean().item()

                nb += 1
                step += 1

                if nb % eval_every == 0:
                    frac_epoch = step / steps_per_epoch
                    avg_loss = ep_loss / nb
                    tag = f"[P2 {method} step {step} ({frac_epoch:.1f}ep)]"
                    if method in ("grpo", "grpo-kl"):
                        print(f"\n{tag} loss={avg_loss:.4f}  reward={ep_reward/nb:.4f}")
                    else:
                        print(f"\n{tag} loss={avg_loss:.4f}")
                    log_metrics(2, round(frac_epoch, 2), avg_loss)

        torch.save(model.state_dict(), "bert_forgetting_phase2.pt")
        print("Saved: bert_forgetting_phase2.pt")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
