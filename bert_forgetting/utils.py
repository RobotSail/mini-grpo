import json
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_experiment_dir(base_dir: str, name: str) -> str:
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    return exp_dir


def save_config(config: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def checkpoint_step_set(checkpoint_every: float, steps_per_epoch: int,
                        num_epochs: int) -> frozenset:
    """Convert checkpoint_every to a set of global step numbers.

    checkpoint_every >= 1: save every N epochs
    checkpoint_every < 1: save round(1/N) times per epoch
    """
    total_steps = steps_per_epoch * num_epochs
    if checkpoint_every >= 1:
        interval = int(checkpoint_every * steps_per_epoch)
    else:
        per_epoch = round(1.0 / checkpoint_every)
        interval = max(1, steps_per_epoch // per_epoch)

    steps = set()
    s = interval
    while s <= total_steps:
        steps.add(s)
        s += interval
    steps.add(total_steps)
    return frozenset(steps)


def get_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch,
                  warmup_ratio=0.1):
    """Cosine-with-warmup LR scheduler, matching SakanaAI rl-razor-mnist."""
    from torch.optim.lr_scheduler import LambdaLR

    total_steps = num_epochs * steps_per_epoch
    num_warmup_steps = int(warmup_ratio * total_steps)

    if scheduler_type == "constant":
        return None

    elif scheduler_type == "cosine_with_warmup":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, total_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
