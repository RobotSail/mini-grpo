import os
import random
import torch
import numpy as np
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime


def checkpoint_step_set(
    checkpoint_every: float,
    num_epochs: int,
    steps_per_epoch: int,
) -> frozenset:
    """Return the set of global step numbers at which to save a checkpoint.

    Args:
        checkpoint_every: If >= 1, checkpoint every N epochs (rounded to int).
                          If < 1, save round(1/checkpoint_every) checkpoints
                          per epoch, evenly spaced across steps.
        num_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.

    Returns:
        Frozenset of global step numbers (1-indexed, step 1 = after first batch).
    """
    total_steps = num_epochs * steps_per_epoch
    steps: set = set()

    if checkpoint_every >= 1:
        interval = round(checkpoint_every)
        for e in range(1, num_epochs + 1):
            if e % interval == 0:
                steps.add(e * steps_per_epoch)
    else:
        n_per_epoch = round(1.0 / checkpoint_every)
        for e in range(num_epochs):
            base = e * steps_per_epoch
            for i in range(1, n_per_epoch + 1):
                s = base + round(steps_per_epoch * i / n_per_epoch)
                if s <= total_steps:
                    steps.add(s)

    return frozenset(steps)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For extra reproducibility (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device.

    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_experiment_dir(
    base_dir: str = "experiments",
    experiment_name: Optional[str] = None,
) -> str:
    """Create a directory for experiment outputs.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment

    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp

    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    return exp_dir


def save_results(
    results: Dict[str, Any],
    save_path: str,
) -> None:
    """Save results to JSON file.

    Args:
        results: Results dictionary
        save_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert numpy arrays and torch tensors to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    results = convert(results)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(load_path: str) -> Dict[str, Any]:
    """Load results from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        Results dictionary
    """
    with open(load_path, "r") as f:
        results = json.load(f)
    return results


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "max" for metrics to maximize, "min" for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _snap_rne(t, mantissa_bits):
    """Snap fp32 tensor to mantissa_bits precision using IEEE 754 RNE rounding (in-place).

    Verified identical to native bfloat16 at m7 and native float16 at m10
    on 1M values (0 mismatches). Exhaustively verified at m14 and m18
    across all 8,388,608 possible mantissa values.
    """
    shift = 23 - mantissa_bits
    halfway = 1 << (shift - 1)
    mask = ~((1 << shift) - 1)
    lsb = 1 << shift
    raw = t.view(torch.int32)
    truncated = raw & mask
    remainder = raw & ((1 << shift) - 1)
    round_up = (remainder > halfway) | (
        (remainder == halfway) & (((truncated >> shift) & 1) == 1)
    )
    t.copy_((truncated + round_up.int() * lsb).view(torch.float32))


def snap_to_lattice(model, mantissa_bits):
    """Snap all model weights to a reduced-precision mantissa lattice (in-place).

    Uses IEEE 754 Round-to-Nearest-Even (RNE) via int32 bit manipulation.
    mantissa_bits=7 matches bfloat16, mantissa_bits=10 matches float16,
    mantissa_bits=23 is full float32 (no-op).
    """
    if mantissa_bits <= 0 or mantissa_bits >= 23:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.data.dtype != torch.float32:
                p.data.copy_(p.data.float())
            _snap_rne(p.data, mantissa_bits)


def save_weights(model):
    """Clone all parameter tensors before an optimizer step.

    Returns a dict mapping parameter id to cloned data.
    Used with quantize_update_from_saved() for sub-bf16 mantissa constraints.
    """
    return {id(p): p.data.clone() for p in model.parameters()}


def _quantize_update_tensor(w_old, w_new, mantissa_bits):
    """Quantize the update (w_new - w_old) to the m-bit ULP at w_old's exponent.

    The weight itself is NOT snapped to the lattice — only the delta is rounded
    to the nearest multiple of ULP(w_old, mantissa_bits) using RNE.  This
    preserves pretrained weight precision while constraining update granularity.
    """
    delta = w_new - w_old

    w_bits = w_old.view(torch.int32)
    e_biased = (w_bits >> 23) & 0xFF

    ulp_e = (e_biased.int() - mantissa_bits).clamp(min=1)
    ulp = (ulp_e << 23).view(torch.float32)

    zero_mask = (w_old == 0)
    if zero_mask.any():
        tiny_ulp = torch.tensor(2.0 ** (1 - 127 - mantissa_bits),
                                dtype=torch.float32, device=w_old.device)
        ulp = torch.where(zero_mask, tiny_ulp, ulp)

    ratio = delta / ulp
    ratio_rounded = torch.round(ratio)
    return w_old + ratio_rounded * ulp


def quantize_update_from_saved(model, saved_weights, mantissa_bits):
    """Apply quantized updates to model parameters using pre-step clones.

    For mantissa_bits >= 7 (bf16 native width), falls back to snap_to_lattice
    which is equivalent and avoids the clone overhead.  For mantissa_bits < 7,
    quantizes the delta at the old weight's ULP so pretrained values are preserved.
    """
    if mantissa_bits <= 0 or mantissa_bits >= 23:
        return
    if mantissa_bits >= 7:
        snap_to_lattice(model, mantissa_bits)
        return
    with torch.no_grad():
        for p in model.parameters():
            w_old = saved_weights[id(p)]
            if p.data.dtype != torch.float32:
                p.data.copy_(p.data.float())
            p.data.copy_(_quantize_update_tensor(w_old, p.data, mantissa_bits))


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_schedule_values(
    learning_rate: float,
    num_epochs: int,
    steps_per_epoch: int,
    scheduler_type: str,
    warmup_ratio: float = 0.1,
) -> list:
    """Get learning rate values for each step.

    Useful for visualizing learning rate schedules.

    Args:
        learning_rate: Base learning rate
        num_epochs: Number of epochs
        steps_per_epoch: Steps per epoch
        scheduler_type: Type of scheduler
        warmup_ratio: Warmup ratio

    Returns:
        List of learning rates for each step
    """
    total_steps = num_epochs * steps_per_epoch
    num_warmup_steps = int(warmup_ratio * total_steps)

    lr_values = []

    for step in range(total_steps):
        if scheduler_type == "constant":
            lr = learning_rate
        elif scheduler_type == "constant_with_warmup":
            if step < num_warmup_steps:
                lr = learning_rate * step / num_warmup_steps
            else:
                lr = learning_rate
        elif scheduler_type == "cosine_with_warmup":
            if step < num_warmup_steps:
                lr = learning_rate * step / num_warmup_steps
            else:
                progress = (step - num_warmup_steps) / (total_steps - num_warmup_steps)
                lr = learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            lr = learning_rate

        lr_values.append(lr)

    return lr_values
