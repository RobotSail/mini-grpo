import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert_forgetting.metrics import news_accuracy, sentiment_accuracy, forward_kl
from bert_forgetting.utils import checkpoint_step_set, get_scheduler

try:
    import wandb
except ImportError:
    wandb = None


def compute_oracle_targets(ref_model, ids, mask, sentiment_binary, device):
    """I-projection: renormalize base model's softmax over valid sentiment classes."""
    with torch.no_grad():
        ref_probs = F.softmax(ref_model(ids, mask), dim=1)

    valid = torch.zeros_like(ref_probs)
    neg_m = (sentiment_binary == 0)
    pos_m = (sentiment_binary == 1)
    valid[neg_m, 0] = 1.0
    valid[neg_m, 1] = 1.0
    valid[pos_m, 2] = 1.0
    valid[pos_m, 3] = 1.0

    masked = ref_probs * valid
    return masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)


def sft_finetune(base_model, sent_dataset, batch_size=64, learning_rate=2e-5,
                 num_epochs=2, label_mode="sft2", seed=42, device="cuda",
                 log_wandb=False, checkpoint_dir=None, checkpoint_every=0.2,
                 ag_val_loader=None, sent_val_loader=None, verbose=True,
                 scheduler_type="cosine_with_warmup", warmup_ratio=0.1):

    model = base_model.copy().to(device)
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(sent_dataset, batch_size, shuffle=True, drop_last=True)

    steps_per_epoch = len(loader)
    scheduler = get_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch,
                              warmup_ratio=warmup_ratio)
    ckpt_steps = checkpoint_step_set(checkpoint_every, steps_per_epoch, num_epochs)

    history = []
    checkpoints = []
    step = 0

    def evaluate(step_num, loss_val=None):
        metrics = {"step": step_num, "epoch": round(step_num / steps_per_epoch, 2)}
        if loss_val is not None:
            metrics["loss"] = loss_val
        if ag_val_loader:
            metrics["news_acc"] = news_accuracy(model, ag_val_loader, device)
            metrics["kl_news"] = forward_kl(base_model, model, ag_val_loader, device)
        if sent_val_loader:
            metrics["sentiment_acc"] = sentiment_accuracy(model, sent_val_loader, device)
            metrics["kl_sentiment"] = forward_kl(base_model, model, sent_val_loader, device)
        history.append(metrics)
        if log_wandb and wandb:
            wandb.log({f"finetune/{k}": v for k, v in metrics.items()})
        if verbose:
            parts = [f"step {step_num} ({metrics['epoch']:.1f}ep)"]
            for k in ("news_acc", "sentiment_acc", "kl_news"):
                if k in metrics:
                    parts.append(f"{k}={metrics[k]:.4f}")
            print(f"  [{label_mode}] {' | '.join(parts)}")
        return metrics

    evaluate(0)

    for ep in range(1, num_epochs + 1):
        model.train()
        for batch in loader:
            ids, mask = batch[0].to(device), batch[1].to(device)
            labels, sentiment_binary = batch[2].to(device), batch[3].to(device)

            optimizer.zero_grad()
            if label_mode == "oracle":
                soft_targets = compute_oracle_targets(base_model, ids, mask,
                                                      sentiment_binary, device)
                logits = model(ids, mask)
                loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(1).mean()
            else:
                loss = criterion(model(ids, mask), labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            step += 1

            if step in ckpt_steps:
                m = evaluate(step, loss.item())
                if checkpoint_dir:
                    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    model.save_checkpoint(path, step=step)
                    checkpoints.append({**m, "path": path})

    final = evaluate(step)
    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints,
        "final_sentiment_acc": final.get("sentiment_acc"),
        "final_news_acc": final.get("news_acc"),
        "final_kl_news": final.get("kl_news"),
    }
