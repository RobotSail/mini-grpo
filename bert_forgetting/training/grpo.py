import os

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from bert_forgetting.metrics import news_accuracy, sentiment_accuracy, forward_kl
from bert_forgetting.utils import checkpoint_step_set, get_scheduler

try:
    import wandb
except ImportError:
    wandb = None


def grpo_finetune(base_model, sent_dataset, batch_size=64, group_size=8,
                  learning_rate=2e-5, num_epochs=2, kl_coef=0.0,
                  seed=42, device="cuda", log_wandb=False,
                  checkpoint_dir=None, checkpoint_every=0.2,
                  ag_val_loader=None, sent_val_loader=None, verbose=True,
                  scheduler_type="cosine_with_warmup", warmup_ratio=0.1):

    model = base_model.copy().to(device)
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(sent_dataset, batch_size, shuffle=True, drop_last=True)

    steps_per_epoch = len(loader)
    scheduler = get_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch,
                              warmup_ratio=warmup_ratio)
    ckpt_steps = checkpoint_step_set(checkpoint_every, steps_per_epoch, num_epochs)

    history = []
    checkpoints = []
    step = 0
    method_name = "grpo_kl" if kl_coef > 0 else "grpo"

    def evaluate(step_num, loss_val=None, reward_val=None):
        metrics = {"step": step_num, "epoch": round(step_num / steps_per_epoch, 2)}
        if loss_val is not None:
            metrics["loss"] = loss_val
        if reward_val is not None:
            metrics["reward"] = reward_val
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
            for k in ("news_acc", "sentiment_acc", "kl_news", "reward"):
                if k in metrics:
                    parts.append(f"{k}={metrics[k]:.4f}")
            print(f"  [{method_name}] {' | '.join(parts)}")
        return metrics

    evaluate(0)

    G = group_size
    for ep in range(1, num_epochs + 1):
        model.train()
        for batch in loader:
            ids, mask = batch[0].to(device), batch[1].to(device)
            sentiment_binary = batch[3].to(device)

            with torch.no_grad():
                logits = model(ids, mask)
                dist = Categorical(logits=logits.unsqueeze(1).expand(-1, G, -1))
                actions = dist.sample()

                sent_g = sentiment_binary.unsqueeze(1).expand(-1, G)
                rewards = ((actions >= 2).long() == sent_g).float()

                mean_r = rewards.mean(1, keepdim=True)
                std_r = rewards.std(1, keepdim=True)
                advantages = torch.where(
                    std_r < 1e-8,
                    torch.zeros_like(rewards),
                    (rewards - mean_r) / (std_r + 1e-8),
                ).clamp(-10, 10)

            new_logits = model(ids, mask)
            new_lp = Categorical(
                logits=new_logits.unsqueeze(1).expand(-1, G, -1)
            ).log_prob(actions)

            loss = -(advantages * new_lp).mean()

            if kl_coef > 0:
                with torch.no_grad():
                    ref_logits = base_model(ids, mask)
                new_logp = F.log_softmax(new_logits, dim=1)
                ref_logp = F.log_softmax(ref_logits, dim=1)
                kl = (new_logp.exp() * (new_logp - ref_logp)).sum(1).mean()
                loss = loss + kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            step += 1

            if step in ckpt_steps:
                m = evaluate(step, loss.item(), rewards.mean().item())
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
