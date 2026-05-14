"""Pretraining on joint ParityMNIST + FashionMNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import wandb

from rl_razor.model import MLP
from rl_razor.data import get_pretraining_data, get_parity_mnist, get_fashion_mnist, create_dataloader
from rl_razor.utils import set_seed, get_device, checkpoint_step_set


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_epochs: int,
    num_warmup_steps: int = 0,
    steps_per_epoch: int = 1,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.

    Args:
        optimizer: The optimizer
        scheduler_type: "constant", "constant_with_warmup", or "cosine_with_warmup"
        num_epochs: Total number of epochs
        num_warmup_steps: Number of warmup steps
        steps_per_epoch: Number of steps per epoch

    Returns:
        Learning rate scheduler or None
    """
    total_steps = num_epochs * steps_per_epoch

    if scheduler_type == "constant":
        return None

    elif scheduler_type == "constant_with_warmup":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine_with_warmup":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, total_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def pretrain(
    model: Optional[MLP] = None,
    n_samples_per_task: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    scheduler_type: str = "cosine_with_warmup",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    data_dir: str = "./data",
    seed: int = 42,
    device: Optional[str] = None,
    log_wandb: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: float = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pretrain model on joint ParityMNIST + FashionMNIST.

    Args:
        model: Model to train (creates new if None)
        n_samples_per_task: Number of samples per task
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        scheduler_type: LR scheduler type
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay for AdamW
        data_dir: Directory for data
        seed: Random seed
        device: Device to train on
        log_wandb: Whether to log to wandb
        checkpoint_dir: Directory to save checkpoints
        checkpoint_every: Epoch interval for checkpoints (>= 1), or fraction
            of an epoch (< 1, e.g. 0.5 saves 2 checkpoints per epoch)
        verbose: Whether to show progress bars

    Returns:
        Dictionary with training results and final model
    """
    set_seed(seed)
    device = device or get_device()

    # Create model if not provided
    if model is None:
        model = MLP()
    model = model.to(device)

    # Get data
    train_dataset, _ = get_pretraining_data(
        n_samples_per_task=n_samples_per_task,
        data_dir=data_dir,
        seed=seed,
    )
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    # Separate val loaders per task for per-task monitoring.
    # parity_labels=False so the validation labels are the actual MNIST digits —
    # deterministic across evaluation passes; parity accuracy still works because
    # the metric checks pred%2 == label%2 and the true label always has correct parity.
    parity_val = get_parity_mnist(train=False, data_dir=data_dir, parity_labels=False)
    fashion_val = get_fashion_mnist(train=False, data_dir=data_dir)
    parity_val_loader = create_dataloader(parity_val, batch_size=batch_size, shuffle=False)
    fashion_val_loader = create_dataloader(fashion_val, batch_size=batch_size, shuffle=False)

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
        "train_fashion_acc": [],
        "val_parity_acc": [],
        "val_fashion_acc": [],
    }
    global_step = 0
    ckpt_steps = checkpoint_step_set(checkpoint_every, num_epochs, steps_per_epoch)

    iterator = tqdm(range(num_epochs), desc="Pretraining") if verbose else range(num_epochs)

    for epoch in iterator:
        model.train()
        epoch_loss = 0.0
        # Parity task: count correct parity predictions
        parity_correct = 0
        parity_total = 0
        # Fashion task: count exact label matches
        fashion_correct = 0
        fashion_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * x.size(0)

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Identify task by indicator (last column of x): +1 = parity, -1 = fashion
                is_parity = x[:, -1] > 0

                if is_parity.any():
                    p_preds = preds[is_parity]
                    p_labels = y[is_parity]
                    parity_correct += ((p_preds % 2) == (p_labels % 2)).sum().item()
                    parity_total += is_parity.sum().item()

                if (~is_parity).any():
                    f_preds = preds[~is_parity]
                    f_labels = y[~is_parity]
                    fashion_correct += (f_preds == f_labels).sum().item()
                    fashion_total += (~is_parity).sum().item()

            global_step += 1

            # Step-based checkpointing (supports sub-epoch intervals)
            if checkpoint_dir and global_step in ckpt_steps:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                epoch_frac = global_step / steps_per_epoch
                ckpt_val_parity = evaluate_parity_task(model, parity_val_loader, device)
                ckpt_val_fashion = evaluate_fashion_task(model, fashion_val_loader, device)
                model.save_checkpoint(
                    f"{checkpoint_dir}/pretrain_step{global_step}.pt",
                    epoch=epoch_frac,
                    optimizer_state_dict=optimizer.state_dict(),
                    val_parity_acc=ckpt_val_parity,
                    val_fashion_acc=ckpt_val_fashion,
                )
                model.train()

        train_loss = epoch_loss / (parity_total + fashion_total)
        train_parity_acc = parity_correct / parity_total if parity_total > 0 else 0.0
        train_fashion_acc = fashion_correct / fashion_total if fashion_total > 0 else 0.0

        # Validation: evaluate each task separately
        val_parity_acc = evaluate_parity_task(model, parity_val_loader, device)
        val_fashion_acc = evaluate_fashion_task(model, fashion_val_loader, device)

        # Log
        history["train_loss"].append(train_loss)
        history["train_parity_acc"].append(train_parity_acc)
        history["train_fashion_acc"].append(train_fashion_acc)
        history["val_parity_acc"].append(val_parity_acc)
        history["val_fashion_acc"].append(val_fashion_acc)

        if log_wandb:
            wandb.log({
                "pretrain/train_loss": train_loss,
                "pretrain/train_parity_acc": train_parity_acc,
                "pretrain/train_fashion_acc": train_fashion_acc,
                "pretrain/val_parity_acc": val_parity_acc,
                "pretrain/val_fashion_acc": val_fashion_acc,
                "pretrain/epoch": epoch,
                "pretrain/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        if verbose:
            iterator.set_postfix({
                "loss": f"{train_loss:.4f}",
                "parity": f"{val_parity_acc:.4f}",
                "fashion": f"{val_fashion_acc:.4f}",
            })

    return {
        "model": model,
        "history": history,
        "final_val_parity_acc": val_parity_acc,
        "final_val_fashion_acc": val_fashion_acc,
    }


def evaluate_parity_task(model: MLP, dataloader: DataLoader, device: str) -> float:
    """Evaluate parity accuracy on ParityMNIST validation set.

    Args:
        model: Model to evaluate
        dataloader: ParityMNIST dataloader
        device: Device

    Returns:
        Parity accuracy (fraction of predictions with correct even/odd)
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            total_correct += ((preds % 2) == (y % 2)).sum().item()
            total_samples += x.size(0)

    return total_correct / total_samples


def evaluate_fashion_task(model: MLP, dataloader: DataLoader, device: str) -> float:
    """Evaluate classification accuracy on FashionMNIST validation set.

    Args:
        model: Model to evaluate
        dataloader: FashionMNIST dataloader
        device: Device

    Returns:
        Standard classification accuracy
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    return total_correct / total_samples
