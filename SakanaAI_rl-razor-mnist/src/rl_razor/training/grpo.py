"""GRPO (Group Relative Policy Optimization) for ParityMNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb
import os

from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, create_dataloader
from rl_razor.training.pretrain import get_scheduler
from rl_razor.metrics import forward_kl
from rl_razor.utils import set_seed, get_device, checkpoint_step_set, snap_to_lattice, save_weights, quantize_update_from_saved


def compute_parity_reward(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute binary parity reward.

    Args:
        predictions: Predicted digits (batch_size,)
        labels: True digit labels (batch_size,)

    Returns:
        Rewards tensor (batch_size,) with values in {0, 1}
    """
    return ((predictions % 2) == (labels % 2)).float()


def grpo_finetune(
    base_model: MLP,
    batch_size: int = 64,
    group_size: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 2,
    scheduler_type: str = "cosine_with_warmup",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
    normalize_advantages: bool = True,
    data_dir: str = "./data",
    seed: int = 42,
    device: Optional[str] = None,
    log_wandb: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: float = 1,
    eval_fashion: bool = True,
    fashion_loader: Optional[DataLoader] = None,
    verbose: bool = True,
    mantissa_bits: int = 0,
) -> Dict[str, Any]:
    """Fine-tune model on ParityMNIST using GRPO (Group Relative Policy Optimization).

    For each input x in a batch, group_size actions are sampled from the current
    policy.  Advantages are computed relative to the per-input group mean (and
    optionally normalized by the per-input group std), so the baseline is local
    to each prompt rather than shared across the entire batch.

    GRPO Loss:
        For each input x with group of G sampled actions {a_1, ..., a_G}:
            A_i = (R(a_i) - mean_j R(a_j)) / (std_j R(a_j) + eps)   [if normalize]
                = (R(a_i) - mean_j R(a_j))                            [otherwise]
        L = -E[A * log π(a|x)] + β * KL(π || π₀)

    Args:
        base_model: Pretrained base model (will be copied)
        batch_size: Number of unique inputs per gradient step
        group_size: Number of actions sampled per input (G); must be >= 2
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        scheduler_type: LR scheduler type
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay
        kl_coef: KL regularization coefficient (0 = no regularization)
        entropy_coef: Entropy bonus coefficient
        normalize_advantages: If True, divide group-relative advantages by group std
        data_dir: Data directory
        seed: Random seed
        device: Training device
        log_wandb: Whether to log to wandb
        checkpoint_dir: Directory to save checkpoints
        checkpoint_every: Save checkpoint every N epochs
        eval_fashion: Whether to evaluate on FashionMNIST
        fashion_loader: FashionMNIST dataloader for evaluation
        verbose: Show progress bars

    Returns:
        Dictionary with training results
    """
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2 for group-relative advantages, got {group_size}")

    set_seed(seed)
    device = device or get_device()

    # Copy base model for fine-tuning
    model = base_model.copy().to(device)
    base_model = base_model.to(device)
    base_model.eval()

    # Get data (standard labels for reward computation)
    train_dataset, val_dataset = get_finetuning_data(
        label_mode="rl",
        data_dir=data_dir,
    )

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_epochs * steps_per_epoch)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type,
        num_epochs,
        num_warmup_steps,
        steps_per_epoch,
    )

    # Training loop
    history = {
        "train_pg_loss": [],
        "train_kl_loss": [],
        "train_total_loss": [],
        "train_reward": [],
        "train_parity_acc": [],
        "val_parity_acc": [],
        "kl_divergence": [],
        "fashion_acc": [],
        "entropy": [],
    }
    checkpoints = []
    global_step = 0
    ckpt_steps = checkpoint_step_set(checkpoint_every, num_epochs, steps_per_epoch)

    method_name = f"GRPO(G={group_size})" + (f"+KL({kl_coef})" if kl_coef > 0 else "")
    iterator = tqdm(range(num_epochs), desc=method_name) if verbose else range(num_epochs)

    for epoch in iterator:
        model.train()

        epoch_pg_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        epoch_reward = 0.0
        epoch_correct_parity = 0
        epoch_entropy = 0.0
        epoch_total = 0  # counts individual (input, action) pairs = n * group_size

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0)  # unique inputs in this batch
            total_samples = n * group_size  # total (input, action) pairs

            optimizer.zero_grad()

            # Expand each input group_size times: (n*G, input_dim)
            x_g = x.unsqueeze(1).expand(-1, group_size, -1).reshape(total_samples, -1)
            y_g = y.unsqueeze(1).expand(-1, group_size).reshape(total_samples)

            # Forward pass on expanded batch
            logits = model(x_g)  # (n*G, 10)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Sample one action per (input, group-slot)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n*G,)

            # Compute binary parity rewards
            rewards = compute_parity_reward(actions, y_g)  # (n*G,) in {0.0, 1.0}

            # Group-relative advantages: normalize within each input's group
            rewards_g = rewards.reshape(n, group_size)  # (n, G)
            group_mean = rewards_g.mean(dim=1, keepdim=True)  # (n, 1)
            advantages_g = rewards_g - group_mean  # (n, G)

            if normalize_advantages:
                group_std = rewards_g.std(dim=1, keepdim=True)  # (n, 1)
                advantages_g = advantages_g / (group_std + 1e-8)

            advantages = advantages_g.reshape(total_samples)  # (n*G,)

            # Policy gradient loss: -E[A * log π(a|x)]
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            pg_loss = -(advantages * selected_log_probs).mean()

            # KL regularization: KL(π || π₀) = Σ π(y) [log π(y) - log π₀(y)]
            kl_loss = torch.tensor(0.0, device=device)
            if kl_coef > 0:
                with torch.no_grad():
                    base_logits = base_model(x_g)
                    base_log_probs = F.log_softmax(base_logits, dim=-1)

                kl = (probs * (log_probs - base_log_probs)).sum(dim=-1).mean()
                kl_loss = kl

            # Entropy bonus: -Σ π(y) log π(y)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Total loss
            total_loss = pg_loss + kl_coef * kl_loss - entropy_coef * entropy

            total_loss.backward()
            if mantissa_bits > 0 and mantissa_bits < 7:
                w_old = save_weights(model)
            optimizer.step()
            if mantissa_bits > 0:
                if mantissa_bits < 7:
                    quantize_update_from_saved(model, w_old, mantissa_bits)
                else:
                    snap_to_lattice(model, mantissa_bits)

            if scheduler is not None:
                scheduler.step()

            # Track metrics (over individual (input, action) pairs)
            epoch_pg_loss += pg_loss.item() * total_samples
            epoch_kl_loss += kl_loss.item() * total_samples
            epoch_total_loss += total_loss.item() * total_samples
            epoch_reward += rewards.sum().item()
            epoch_correct_parity += rewards.sum().item()  # reward == 1 iff parity correct
            epoch_entropy += entropy.item() * total_samples
            epoch_total += total_samples
            global_step += 1

            # Step-based checkpointing (supports sub-epoch intervals)
            if checkpoint_dir and global_step in ckpt_steps:
                os.makedirs(checkpoint_dir, exist_ok=True)
                epoch_frac = global_step / steps_per_epoch
                ckpt_parity = evaluate_parity_grpo(model, val_loader, device)
                ckpt_kl = forward_kl(base_model, model, val_loader, device)
                ckpt_fashion = (
                    evaluate_fashion_grpo(model, fashion_loader, device)
                    if eval_fashion and fashion_loader is not None
                    else None
                )
                kl_suffix = f"_kl{kl_coef}" if kl_coef > 0 else ""
                checkpoint_path = f"{checkpoint_dir}/grpo{kl_suffix}_step{global_step}.pt"
                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch_frac,
                    kl_coef=kl_coef,
                    group_size=group_size,
                    parity_acc=ckpt_parity,
                    kl_divergence=ckpt_kl,
                    fashion_acc=ckpt_fashion,
                )
                checkpoints.append(
                    {
                        "path": checkpoint_path,
                        "epoch": epoch_frac,
                        "parity_acc": ckpt_parity,
                        "kl_divergence": ckpt_kl,
                        "fashion_acc": ckpt_fashion,
                    }
                )
                model.train()

        # Compute epoch averages
        train_pg_loss = epoch_pg_loss / epoch_total
        train_kl_loss = epoch_kl_loss / epoch_total
        train_total_loss = epoch_total_loss / epoch_total
        train_reward = epoch_reward / epoch_total
        train_parity_acc = epoch_correct_parity / epoch_total
        train_entropy = epoch_entropy / epoch_total

        # Validation
        val_parity_acc = evaluate_parity_grpo(model, val_loader, device)

        # KL divergence from base model
        kl_div = forward_kl(base_model, model, val_loader, device)

        # FashionMNIST accuracy
        fashion_acc = None
        if eval_fashion and fashion_loader is not None:
            fashion_acc = evaluate_fashion_grpo(model, fashion_loader, device)

        # Log history
        history["train_pg_loss"].append(train_pg_loss)
        history["train_kl_loss"].append(train_kl_loss)
        history["train_total_loss"].append(train_total_loss)
        history["train_reward"].append(train_reward)
        history["train_parity_acc"].append(train_parity_acc)
        history["val_parity_acc"].append(val_parity_acc)
        history["kl_divergence"].append(kl_div)
        history["entropy"].append(train_entropy)
        if fashion_acc is not None:
            history["fashion_acc"].append(fashion_acc)

        if log_wandb:
            log_dict = {
                "grpo/train_pg_loss": train_pg_loss,
                "grpo/train_kl_loss": train_kl_loss,
                "grpo/train_total_loss": train_total_loss,
                "grpo/train_reward": train_reward,
                "grpo/entropy": train_entropy,
                "train_parity_acc": train_parity_acc,
                "kl_divergence": kl_div,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "val_parity_acc": val_parity_acc,
            }
            if fashion_acc is not None:
                log_dict["fashion_acc"] = fashion_acc
            wandb.log(log_dict, step=global_step)

        if verbose:
            postfix = {
                "reward": f"{train_reward:.4f}",
                "parity": f"{val_parity_acc:.4f}",
                "kl": f"{kl_div:.4f}",
            }
            if fashion_acc is not None:
                postfix["fashion"] = f"{fashion_acc:.4f}"
            iterator.set_postfix(postfix)

    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints,
        "final_parity_acc": val_parity_acc,
        "final_kl_divergence": kl_div,
        "final_fashion_acc": fashion_acc,
    }


def evaluate_parity_grpo(
    model: MLP,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Evaluate parity accuracy (greedy).

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device

    Returns:
        Parity accuracy
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            total_correct += ((preds % 2) == (y % 2)).sum().item()
            total_samples += x.size(0)

    return total_correct / total_samples


def evaluate_fashion_grpo(
    model: MLP,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Evaluate standard accuracy on FashionMNIST.

    Args:
        model: Model to evaluate
        dataloader: FashionMNIST DataLoader
        device: Device

    Returns:
        Classification accuracy
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    return total_correct / total_samples
