"""
Tracked AdamW optimizer: wraps torch.optim.AdamW with per-parameter
update norm logging.

Uses the exact same PyTorch AdamW implementation for optimization.
Captures ΔW by snapshotting params in fp32 before step and computing
the delta in fp32 after step, avoiding catastrophic cancellation.
"""

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim import AdamW


class AdamWTracked(AdamW):
    """AdamW with per-parameter Frobenius norm tracking of updates.

    Subclasses torch.optim.AdamW — optimization is identical.
    Update norms are computed by snapshotting parameters in fp32 before
    step() and computing ΔW = W_new - W_old in fp32 after step().

    For FSDP2 DTensors, local shard norms are all-reduced to get the
    full parameter norm.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
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

    def step(self, closure=None):
        """AdamW step with update norm capture."""
        if not self._tracking_enabled:
            return super().step(closure)

        # Snapshot params in fp32 before step
        snapshots = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p.data, DTensor):
                    snapshots[id(p)] = p.data.to_local().detach().clone().float()
                else:
                    snapshots[id(p)] = p.data.detach().clone().float()

        # Run the real AdamW step
        loss = super().step(closure)

        # Compute update norms in fp32
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in snapshots:
                    continue

                if isinstance(p.data, DTensor):
                    current = p.data.to_local().float()
                else:
                    current = p.data.float()

                delta = current - snapshots[id(p)]
                local_norm_sq = delta.norm(2).square()

                if isinstance(p.data, DTensor):
                    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)

                full_norm = local_norm_sq.sqrt().item()
                param_name = self._param_names.get(id(p), f"unknown_{id(p)}")
                self._step_norms[param_name] = full_norm

        snapshots.clear()
        return loss
