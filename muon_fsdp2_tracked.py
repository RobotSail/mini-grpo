# ruff: noqa
# type: ignore
# fmt: off

# Forked from muon_fsdp2 v0.3.0 with per-parameter update norm tracking.
# Original credits: https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

import json
import math
import os
from typing import Protocol
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed import gather, scatter
from collections import deque

__version__ = "0.3.0-tracked"

__all__ = ["Muon"]


@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = nsloop_torch(X, steps, a=a, b=b, c=c)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def apply_momentum(grad, momentum, beta, nesterov):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    return update

def apply_scaling(grad, rms_scale=False):
    if rms_scale:
        grad *= 0.2 * math.sqrt(max(grad.shape[1], grad.shape[0]))
        return grad
    else:
        grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return grad

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


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


def _compute_update_metrics(param, update, lr, wd):
    """Compute ‖ΔW‖_F and L0 sparsity from the raw optimizer update.

    ΔW = (-lr * wd) * W  +  (-lr) * update
    For FSDP DTensors, computes local shard norm² and all-reduces.

    Returns (norm, sparsity, delta) — delta kept alive for bf16 truncation.
    """
    delta = (-lr * wd) * param.data + (-lr) * update.reshape(param.shape)

    if isinstance(param.data, DTensor):
        local_norm_sq = delta.to_local().norm(2).square()
        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    else:
        local_norm_sq = delta.norm(2).square()

    norm = local_norm_sq.sqrt().item()
    sparsity = _l0_sparsity(delta)
    return norm, sparsity, delta


class Work(Protocol):
    def __init__(self, param, state, group, index: int): ...
    def start(self): ...
    def finish(self): ...


class Fsdp1dWork:
    """Muon work for FSDP2 1D mesh, with update norm tracking."""

    def __init__(self, param, state, group, index: int, optimizer=None):
        self.param = param
        self.state = state
        self.group = group
        self.index = index
        self.optimizer = optimizer
        self._intermediate_state = None

    def start(self):
        self.param.grad = apply_momentum(self.param.grad, self.state["momentum_buffer"], self.group["momentum"], self.group["nesterov"])

        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"

        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

        dest_rank = self.index % world_size

        if rank == dest_rank:
            gather_lists = [torch.zeros_like(input=grad.to_local()) for _ in range(world_size)]
            gather_handle = gather(grad.to_local(), gather_lists, group_dst=dest_rank, group=pg, async_op=True)
        else:
            gather_lists = None
            gather_handle = gather(grad.to_local(), None, group_dst=dest_rank, group=pg, async_op=True)

        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

    def finish(self):
        assert self._intermediate_state is not None, "gather work must be called first"

        grad = self.param.grad
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

        dest_rank, gather_handle, gather_lists = self._intermediate_state
        gather_handle.wait()
        if rank == dest_rank:
            g_full_block = torch.cat(gather_lists, dim=0)
            g_full_block.copy_(zeropower_via_newtonschulz5(g_full_block, self.group["ns_steps"]))
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(grad.to_local(), scatter_list=chunks, src=dest_rank, group=pg, async_op=False)
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)

        update = apply_scaling(grad, self.group["rms_scale"])

        lr = self.group["lr"]
        wd = self.group["weight_decay"]
        tracking = self.optimizer is not None and self.optimizer._tracking_enabled
        bf16_reg = self.optimizer is not None and self.optimizer.bf16_regularization

        if tracking or bf16_reg:
            norm, sparsity, delta = _compute_update_metrics(self.param, update, lr, wd)
            param_name = self.optimizer._param_names.get(id(self.param), f"unknown_{id(self.param)}")

            if tracking:
                self.optimizer._step_norms[param_name] = norm
                self.optimizer._step_sparsities[param_name] = sparsity

            if bf16_reg:
                truncated = delta.to(torch.bfloat16).to(self.param.data.dtype)
                if tracking:
                    self.optimizer._step_sparsities_post[param_name] = _l0_sparsity(truncated)
                self.param.data.add_(truncated)
            else:
                self.param.data.add_(delta)
        else:
            self.param.mul_(1 - lr * wd)
            self.param.add_(update.reshape(self.param.shape), alpha=-lr)


class TpFsdp2dWork:
    def __init__(self, param, state, group, index: int, optimizer=None):
        raise NotImplementedError("not implemented")

class EpFsdp2dWork:
    def __init__(self, param, state, group, index: int, optimizer=None):
        raise NotImplementedError("not implemented")

class TpEpFsdp3dWork:
    def __init__(self, param, state, group, index: int, optimizer=None):
        raise NotImplementedError("not implemented")

class SingelDeviceWork:
    """Muon work for single device, with update norm tracking."""

    def __init__(self, param, state, group, index: int, optimizer=None):
        self.param = param
        self.state = state
        self.group = group
        self.optimizer = optimizer

    def start(self):
        update = muon_update(self.param.grad, self.state["momentum_buffer"], self.group["momentum"], self.group["nesterov"], self.group["ns_steps"], self.group["rms_scale"])

        lr = self.group["lr"]
        wd = self.group["weight_decay"]
        tracking = self.optimizer is not None and self.optimizer._tracking_enabled
        bf16_reg = self.optimizer is not None and self.optimizer.bf16_regularization

        if tracking or bf16_reg:
            norm, sparsity, delta = _compute_update_metrics(self.param, update, lr, wd)
            param_name = self.optimizer._param_names.get(id(self.param), f"unknown_{id(self.param)}")

            if tracking:
                self.optimizer._step_norms[param_name] = norm
                self.optimizer._step_sparsities[param_name] = sparsity

            if bf16_reg:
                truncated = delta.to(torch.bfloat16).to(self.param.data.dtype)
                if tracking:
                    self.optimizer._step_sparsities_post[param_name] = _l0_sparsity(truncated)
                self.param.data.add_(truncated)
            else:
                self.param.data.add_(delta)
        else:
            self.param.mul_(1 - lr * wd)
            self.param.add_(update.reshape(self.param.shape), alpha=-lr)

    def finish(self):
        pass


def muon_update(grad, momentum_buffer, momentum, nesterov, ns_steps, rms_scale):
    """Single-device Muon update (used by SingelDeviceWork)."""
    grad = apply_momentum(grad, momentum_buffer, momentum, nesterov)
    grad = zeropower_via_newtonschulz5(grad, ns_steps)
    grad = apply_scaling(grad, rms_scale)
    return grad


class Muon(torch.optim.Optimizer):
    """
    Forked muon_fsdp2.Muon with per-parameter update norm tracking.

    Adds:
        set_param_names(model)     — map param ids to names
        init_update_tracking(dir)  — set up JSONL output
        flush_update_norms(step, tokens) — write and clear accumulated norms
    """
    def __init__(self, param_groups, bf16_regularization=False):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["rms_scale"] = group.get("rms_scale", True)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "rms_scale", "nesterov", "ns_steps"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

        # Update norm tracking state
        self._param_names: dict[int, str] = {}
        self._step_norms: dict[str, float] = {}
        self._step_sparsities: dict[str, float] = {}
        self._step_sparsities_post: dict[str, float] = {}
        self._update_norm_path: str | None = None
        self._tracking_enabled = False
        self.bf16_regularization = bf16_regularization

    def set_param_names(self, model):
        """Map parameter ids to their names for logging."""
        self._param_names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}

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

    def _get_work_class(self, p: torch.Tensor) -> tuple[type[Work], int]:
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return TpFsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingelDeviceWork, 1

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dq: deque[Work] = deque()

        for group in self.param_groups:

            if group["use_muon"]:
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    if self.bf16_regularization:
                        assert p.grad.dtype == torch.float32, f"bf16_regularization: expected fp32 grad, got {p.grad.dtype}"
                        assert state["momentum_buffer"].dtype == torch.float32, f"bf16_regularization: expected fp32 momentum_buffer, got {state['momentum_buffer'].dtype}"

                    class_work, prefetch_factor = self._get_work_class(p)

                    work = class_work(p, state, group, i, optimizer=self)
                    work.start()
                    dq.append(work)

                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    if self.bf16_regularization:
                        assert p.grad.dtype == torch.float32, f"bf16_regularization: expected fp32 grad, got {p.grad.dtype}"
                        assert state["exp_avg"].dtype == torch.float32, f"bf16_regularization: expected fp32 exp_avg, got {state['exp_avg'].dtype}"
                        assert state["exp_avg_sq"].dtype == torch.float32, f"bf16_regularization: expected fp32 exp_avg_sq, got {state['exp_avg_sq'].dtype}"

                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])

                    lr = group["lr"]
                    wd = group["weight_decay"]

                    if self._tracking_enabled or self.bf16_regularization:
                        norm, sparsity, delta = _compute_update_metrics(p, update, lr, wd)
                        param_name = self._param_names.get(id(p), f"unknown_{id(p)}")

                        if self._tracking_enabled:
                            self._step_norms[param_name] = norm
                            self._step_sparsities[param_name] = sparsity

                        if self.bf16_regularization:
                            truncated = delta.to(torch.bfloat16).to(p.data.dtype)
                            if self._tracking_enabled:
                                self._step_sparsities_post[param_name] = _l0_sparsity(truncated)
                            p.data.add_(truncated)
                        else:
                            p.data.add_(delta)
                    else:
                        p.mul_(1 - lr * wd)
                        p.add_(update, alpha=-lr)

        for work in dq:
            work.finish()

        return loss
