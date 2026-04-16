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


def _l0_sparsity(tensor, threshold=1e-5):
    """L0 sparsity: fraction of elements with |value| <= threshold.

    For FSDP DTensors, all-reduces counts across ranks for global sparsity.
    """
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        local_above = torch.tensor(
            [(local.abs() > threshold).sum().item()],
            dtype=torch.long, device=local.device,
        )
        local_numel = torch.tensor([local.numel()], dtype=torch.long, device=local.device)
        dist.all_reduce(local_above, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_numel, op=dist.ReduceOp.SUM)
        return 1.0 - local_above.item() / max(local_numel.item(), 1)
    else:
        above = (tensor.abs() > threshold).sum().item()
        return 1.0 - above / max(tensor.numel(), 1)


class AdamWTracked(torch.optim.Optimizer):
    """AdamW with per-parameter Frobenius norm tracking of updates.

    Implements AdamW (decoupled weight decay) directly so we can
    intercept the exact update tensor before it is applied.

    ΔW = (-lr * wd) * W + (-lr) * m̂ / (√v̂ + ε)

    The Frobenius norm of ΔW is computed per-parameter each step.
    For FSDP2 DTensors, local shard norm² values are all-reduced.

    Modes:
      bf16_regularization: Entire Adam computation in bf16 (moments, grads,
        update). Final ΔW upcast to fp32 for application to fp32 master weights.

      bf16_master_weights: Weights are bf16, but optimizer internals are fp32.
        Gradients arrive as fp32 (from FSDP2 fp32 reduce), moments stored and
        updated in fp32, update computed in fp32. Final ΔW cast to bf16 for
        application to bf16 master weights.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bf16_regularization=False,
                 bf16_master_weights=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._param_names: dict[int, str] = {}
        self._step_norms: dict[str, float] = {}
        self._step_sparsities: dict[str, float] = {}
        self._step_sparsities_post: dict[str, float] = {}
        self._update_norm_path: str | None = None
        self._tracking_enabled = False
        self.bf16_regularization = bf16_regularization
        self.bf16_master_weights = bf16_master_weights

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

    def flush_update_norms(self, step: int, tokens: int) -> dict:
        """Write accumulated norms/sparsities to JSONL (rank 0 only), return averages."""
        result = {"avg_norm": 0.0, "avg_sparsity_pre": 0.0, "avg_sparsity_post": 0.0}

        if not self._step_norms:
            return result

        result["avg_norm"] = sum(self._step_norms.values()) / len(self._step_norms)
        if self._step_sparsities:
            result["avg_sparsity_pre"] = sum(self._step_sparsities.values()) / len(self._step_sparsities)
        if self._step_sparsities_post:
            result["avg_sparsity_post"] = sum(self._step_sparsities_post.values()) / len(self._step_sparsities_post)

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0 and self._update_norm_path:
            record = {
                "step": step, "tokens": tokens,
                "norms": self._step_norms,
                "sparsities_pre": self._step_sparsities,
            }
            if self._step_sparsities_post:
                record["sparsities_post"] = self._step_sparsities_post
            with open(self._update_norm_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        self._step_norms = {}
        self._step_sparsities = {}
        self._step_sparsities_post = {}
        return result

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
                    if self.bf16_regularization:
                        # bf16 moments for bf16 Adam computation
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    elif self.bf16_master_weights:
                        # fp32 moments even though params are bf16
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                if self.bf16_regularization:
                    assert grad.dtype == torch.float32, f"bf16_regularization: expected fp32 grad (from mixed precision), got {grad.dtype}"
                    assert state["exp_avg"].dtype == torch.bfloat16, f"bf16_regularization: expected bf16 exp_avg, got {state['exp_avg'].dtype}"
                    assert state["exp_avg_sq"].dtype == torch.bfloat16, f"bf16_regularization: expected bf16 exp_avg_sq, got {state['exp_avg_sq'].dtype}"
                    # Cast gradient to bf16 — entire Adam computation in bf16
                    grad = grad.to(torch.bfloat16)
                elif self.bf16_master_weights:
                    assert p.data.dtype == torch.bfloat16, f"bf16_master_weights: expected bf16 param, got {p.data.dtype}"
                    assert state["exp_avg"].dtype == torch.float32, f"bf16_master_weights: expected fp32 exp_avg, got {state['exp_avg'].dtype}"
                    assert state["exp_avg_sq"].dtype == torch.float32, f"bf16_master_weights: expected fp32 exp_avg_sq, got {state['exp_avg_sq'].dtype}"
                    # Grads arrive as bf16 (FSDP2 reduces in fp32 but stores back as bf16).
                    # Upcast to fp32 for optimizer computation.
                    grad = grad.float()
                    assert grad.dtype == torch.float32, f"bf16_master_weights: grad must be fp32 after upcast, got {grad.dtype}"

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update biased moments
                # bf16_regularization: bf16 grad into bf16 moments → bf16
                # bf16_master_weights: fp32 grad into fp32 moments → fp32
                # standard: fp32 grad into fp32 moments → fp32
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                # Bias-corrected moments
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + eps

                # The adam update direction: m̂ / (√v̂ + ε)
                update = exp_avg / denom

                # --- Apply update ---
                needs_delta = self._tracking_enabled or self.bf16_regularization or self.bf16_master_weights
                if needs_delta:
                    if self.bf16_regularization:
                        # Everything in bf16
                        w_bf16 = p.data.to(torch.bfloat16)
                        delta_bf16 = (-lr * wd) * w_bf16 + (-step_size) * update
                        delta_applied = delta_bf16
                    elif self.bf16_master_weights:
                        # Compute delta in fp32 (update is already fp32)
                        w_fp32 = p.data.float()
                        delta_fp32 = (-lr * wd) * w_fp32 + (-step_size) * update
                        assert p.data.dtype == torch.bfloat16, f"bf16_master_weights: param must be bf16, got {p.data.dtype}"
                        assert delta_fp32.dtype == torch.float32, f"bf16_master_weights: dW must be fp32, got {delta_fp32.dtype}"
                        # Cast to bf16 for application to bf16 weights
                        delta_applied = delta_fp32.to(torch.bfloat16)
                    else:
                        # Standard fp32 tracking path
                        delta_fp32 = (-lr * wd) * p.data + (-step_size) * update
                        delta_applied = delta_fp32

                    param_name = self._param_names.get(id(p), f"unknown_{id(p)}")

                    if self._tracking_enabled:
                        # Compute fp32 delta for accurate norm measurement
                        if self.bf16_regularization:
                            delta_for_norm = (-lr * wd) * p.data + (-step_size) * update.float()
                        elif self.bf16_master_weights:
                            delta_for_norm = delta_fp32
                        else:
                            delta_for_norm = delta_applied

                        # Frobenius norm
                        if isinstance(delta_for_norm, DTensor):
                            local_norm_sq = delta_for_norm.to_local().norm(2).square()
                            dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
                        else:
                            local_norm_sq = delta_for_norm.norm(2).square()
                        self._step_norms[param_name] = local_norm_sq.sqrt().item()

                        # L0 sparsity pre-truncation (fp32 view)
                        self._step_sparsities[param_name] = _l0_sparsity(delta_for_norm)

                        # L0 sparsity post-truncation (what's actually applied)
                        if self.bf16_regularization or self.bf16_master_weights:
                            self._step_sparsities_post[param_name] = _l0_sparsity(delta_applied)

                    p.data.add_(delta_applied.to(p.data.dtype))
                else:
                    # Standard path: no tracking, no special mode
                    if wd != 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-step_size)

        return loss
