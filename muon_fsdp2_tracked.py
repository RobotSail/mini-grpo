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


def _compute_update_norm(param, update, lr, wd):
    """Compute ‖ΔW‖_F exactly from the raw optimizer update, before application.

    ΔW = (-lr * wd) * W  +  (-lr) * update
    For FSDP DTensors, computes local shard norm² and all-reduces.
    """
    wd_component = (-lr * wd) * param.data
    update_component = (-lr) * update.reshape(param.shape)
    delta_local = wd_component + update_component

    if isinstance(param.data, DTensor):
        local_norm_sq = delta_local.to_local().norm(2).square()
        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    else:
        local_norm_sq = delta_local.norm(2).square()

    return local_norm_sq.sqrt().item()


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

        # Capture update norm BEFORE applying to parameter
        if self.optimizer is not None and self.optimizer._tracking_enabled:
            norm = _compute_update_norm(self.param, update, self.group["lr"], self.group["weight_decay"])
            param_name = self.optimizer._param_names.get(id(self.param), f"unknown_{id(self.param)}")
            self.optimizer._step_norms[param_name] = norm

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])


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

        # Capture update norm BEFORE applying to parameter
        if self.optimizer is not None and self.optimizer._tracking_enabled:
            norm = _compute_update_norm(self.param, update, self.group["lr"], self.group["weight_decay"])
            param_name = self.optimizer._param_names.get(id(self.param), f"unknown_{id(self.param)}")
            self.optimizer._step_norms[param_name] = norm

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])

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
    def __init__(self, param_groups):
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
        self._update_norm_path: str | None = None
        self._tracking_enabled = False

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

    def flush_update_norms(self, step: int, tokens: int) -> float:
        """Write accumulated norms to JSONL (rank 0 only), return average norm."""
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
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])

                    # Capture update norm BEFORE applying to parameter
                    if self._tracking_enabled:
                        norm = _compute_update_norm(p, update, group["lr"], group["weight_decay"])
                        param_name = self._param_names.get(id(p), f"unknown_{id(p)}")
                        self._step_norms[param_name] = norm

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        for work in dq:
            work.finish()

        return loss
