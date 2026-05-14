"""Supervised Fine-Tuning (SFT) on ParityMNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Optional, Dict, Any, Literal
from tqdm import tqdm
import wandb
import os

from rl_razor.model import MLP
from rl_razor.data import get_finetuning_data, create_dataloader
from rl_razor.training.pretrain import get_scheduler
from rl_razor.training.oracle import compute_oracle_loss
from rl_razor.metrics import parity_accuracy, forward_kl
from rl_razor.utils import set_seed, get_device, checkpoint_step_set, snap_to_lattice, save_weights, quantize_update_from_saved


def sft_finetune(
    base_model: MLP,
    label_mode: Literal["sft1", "sft2", "oracle"] = "sft1",
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_epochs: int = 2,
    scheduler_type: str = "cosine_with_warmup",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
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
    """Fine-tune model on ParityMNIST using supervised learning.

    Args:
        base_model: Pretrained base model (will be copied)
        label_mode: Label generation mode
            - "sft1": Even->0, Odd->1
            - "sft2": Even->{0,4}, Odd->{1,5}
            - "oracle": KL-minimal distribution from base model
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        scheduler_type: LR scheduler type
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay
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
    set_seed(seed)
    device = device or get_device()

    # Copy base model for fine-tuning
    model = base_model.copy().to(device)
    base_model = base_model.to(device)
    base_model.eval()

    # Get data.
    # For oracle mode we load standard labels (label_mode="rl") and compute
    # soft-target CE against the oracle distribution on-the-fly via
    # compute_oracle_loss.  For sft1/sft2 the dataset itself provides labels.
    if label_mode == "oracle":
        train_dataset, val_dataset = get_finetuning_data(
            label_mode="rl",
            data_dir=data_dir,
        )
    else:
        train_dataset, val_dataset = get_finetuning_data(
            label_mode=label_mode,
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
        "train_loss": [],
        "train_parity_acc": [],
        "val_parity_acc": [],
        "kl_divergence": [],
        "fashion_acc": [],
    }
    checkpoints = []
    global_step = 0
    ckpt_steps = checkpoint_step_set(checkpoint_every, num_epochs, steps_per_epoch)

    iterator = tqdm(range(num_epochs), desc=f"SFT ({label_mode})") if verbose else range(num_epochs)

    for epoch in iterator:
        model.train()
        epoch_loss = 0.0
        epoch_correct_parity = 0
        epoch_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if label_mode == "oracle":
                # Use oracle soft targets
                loss = compute_oracle_loss(base_model, model, x, y)
            else:
                # Standard cross-entropy with hard labels
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            loss.backward()
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

            # Track metrics
            epoch_loss += loss.item() * x.size(0)
            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=-1)
                epoch_correct_parity += ((preds % 2) == (y % 2)).sum().item()
            epoch_total += x.size(0)
            global_step += 1

            # Step-based checkpointing (supports sub-epoch intervals)
            if checkpoint_dir and global_step in ckpt_steps:
                os.makedirs(checkpoint_dir, exist_ok=True)
                epoch_frac = global_step / steps_per_epoch
                ckpt_parity = evaluate_parity(model, val_loader, device)
                ckpt_kl = forward_kl(base_model, model, val_loader, device)
                ckpt_fashion = (
                    evaluate_fashion(model, fashion_loader, device)
                    if eval_fashion and fashion_loader is not None
                    else None
                )
                checkpoint_path = f"{checkpoint_dir}/sft_{label_mode}_step{global_step}.pt"
                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch_frac,
                    label_mode=label_mode,
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

        train_loss = epoch_loss / epoch_total
        train_parity_acc = epoch_correct_parity / epoch_total

        # End-of-epoch validation (always, for history / wandb)
        val_parity_acc = evaluate_parity(model, val_loader, device)
        kl_div = forward_kl(base_model, model, val_loader, device)
        fashion_acc = None
        if eval_fashion and fashion_loader is not None:
            fashion_acc = evaluate_fashion(model, fashion_loader, device)

        history["train_loss"].append(train_loss)
        history["train_parity_acc"].append(train_parity_acc)
        history["val_parity_acc"].append(val_parity_acc)
        history["kl_divergence"].append(kl_div)
        if fashion_acc is not None:
            history["fashion_acc"].append(fashion_acc)

        if log_wandb:
            log_dict = {
                "sft/train_loss": train_loss,
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
                "loss": f"{train_loss:.4f}",
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


def evaluate_parity(
    model: MLP,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Evaluate parity accuracy.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
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


def evaluate_fashion(
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
