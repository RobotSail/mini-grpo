import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bert_forgetting.metrics import news_accuracy, sentiment_accuracy
from bert_forgetting.utils import checkpoint_step_set

try:
    import wandb
except ImportError:
    wandb = None


def pretrain(model, ag_dataset, sent_dataset, batch_size=64, learning_rate=2e-5,
             num_epochs=5, seed=42, device="cuda", log_wandb=False,
             checkpoint_dir=None, checkpoint_every=1.0,
             ag_val_loader=None, sent_val_loader=None, verbose=True):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ag_loader = DataLoader(ag_dataset, batch_size, shuffle=True, drop_last=True)
    sent_loader = DataLoader(sent_dataset, batch_size, shuffle=True, drop_last=True)

    steps_per_epoch = min(len(ag_loader), len(sent_loader)) * 2
    ckpt_steps = checkpoint_step_set(checkpoint_every, steps_per_epoch, num_epochs)

    history = []
    checkpoints = []
    step = 0

    def evaluate(step_num):
        metrics = {"step": step_num, "epoch": step_num / steps_per_epoch}
        if ag_val_loader:
            metrics["news_acc"] = news_accuracy(model, ag_val_loader, device)
        if sent_val_loader:
            metrics["sentiment_acc"] = sentiment_accuracy(model, sent_val_loader, device)
        history.append(metrics)
        if log_wandb and wandb:
            wandb.log({f"pretrain/{k}": v for k, v in metrics.items()})
        if verbose:
            parts = [f"step {step_num}"]
            for k in ("news_acc", "sentiment_acc"):
                if k in metrics:
                    parts.append(f"{k}={metrics[k]:.4f}")
            print(f"  [pretrain] {' | '.join(parts)}")
        return metrics

    evaluate(0)

    for ep in range(1, num_epochs + 1):
        model.train()
        ag_iter = iter(ag_loader)
        sent_iter = iter(sent_loader)
        ep_loss = 0.0
        nb = 0

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

            if step in ckpt_steps:
                m = evaluate(step)
                if checkpoint_dir:
                    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    model.save_checkpoint(path, step=step)
                    checkpoints.append({**m, "path": path})

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

            if step in ckpt_steps:
                m = evaluate(step)
                if checkpoint_dir:
                    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    model.save_checkpoint(path, step=step)
                    checkpoints.append({**m, "path": path})

        if verbose:
            print(f"  [pretrain] epoch {ep} | loss={ep_loss / max(nb, 1):.4f}")

    final_metrics = evaluate(step)

    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints,
        "final_news_acc": final_metrics.get("news_acc"),
        "final_sentiment_acc": final_metrics.get("sentiment_acc"),
    }
