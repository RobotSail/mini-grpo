"""
Tracked AdamW optimizer: implements AdamW with per-parameter
update norm logging.

Computes the exact ΔW the optimizer forms (before applying it to W),
matching the approach used in muon_fsdp2_tracked.py. No parameter
snapshotting needed — the update is captured directly from the
bias-corrected Adam moments + weight decay.
"""

import json
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


class AdamWTracked(torch.optim.Optimizer):
    """AdamW with per-parameter Frobenius norm tracking of updates.

    Implements AdamW (decoupled weight decay) directly so we can
    intercept the exact update tensor before it is applied.

    ΔW = (-lr * wd) * W + (-lr) * m̂ / (√v̂ + ε)

    The Frobenius norm of ΔW is computed per-parameter each step.
    For FSDP2 DTensors, local shard norm² values are all-reduced.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._param_names: dict[int, str] = {}
        self._step_norms: dict[str, float] = {}
        self._update_norm_path: str | None = None
        self._tracking_enabled = False

    def set_param_names(self, model):
        """Map parameter ids to their names for logging."""
        self._param_names = {
            id(p): name for name, p in model.named_parameters() if p.requires_grad
        }

    def init_update_tracking(self, output_dir: str | None):
        """Initialize JSONL file for per-parameter update norms."""
        if not output_dir:
            self._update_norm_path = None
            self._tracking_enabled = False
            return
        self._update_norm_path = os.path.join(output_dir, "update_norms.jsonl")
        if os.path.exists(self._update_norm_path):
            os.remove(self._update_norm_path)
        self._tracking_enabled = True

    def flush_update_norms(self, step: int, tokens: int) -> float:
        """Write accumulated norms to JSONL (rank 0 only), return average."""
        if not self._step_norms:
            return 0.0

        avg_norm = sum(self._step_norms.values()) / len(self._step_norms)

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0 and self._update_norm_path:
            record = {"step": step, "tokens": tokens, "norms": self._step_norms}
            with open(self._update_norm_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        self._step_norms = {}
        return avg_norm

    @torch.no_grad()
    def step(self, closure=None):
        """AdamW step with exact update norm capture."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update biased moments
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                # Bias-corrected moments
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + eps

                # The adam update direction: m̂ / (√v̂ + ε)
                update = exp_avg / denom

                # Capture ΔW norm before applying
                if self._tracking_enabled:
                    # ΔW = (-lr * wd) * W + (-step_size) * update
                    delta = (-lr * wd) * p.data + (-step_size) * update

                    if isinstance(p.data, DTensor):
                        local_norm_sq = delta.to_local().norm(2).square()
                        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
                    else:
                        local_norm_sq = delta.norm(2).square()

                    full_norm = local_norm_sq.sqrt().item()
                    param_name = self._param_names.get(id(p), f"unknown_{id(p)}")
                    self._step_norms[param_name] = full_norm
                    del delta

                # Apply decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Apply update
                p.data.add_(update, alpha=-step_size)

        return loss
