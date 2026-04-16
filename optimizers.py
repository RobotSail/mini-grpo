"""
Optimizer utilities for mini-grpo.

Supports:
- AdamW (default, from torch.optim)
- Muon (requires PyTorch >= 2.9 or standalone muon package)
- Muon with FSDP2 support (requires muon-fsdp2 package)
- Mixed precision training with FP32 master weights
"""

import torch
from torch.optim import AdamW
from typing import Literal
from transformers import PreTrainedModel
from copy import deepcopy

OptimizerType = Literal["adamw", "muon"]


def create_fsdp2_muon_optimizer(
    model: PreTrainedModel,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-5,
    beta1: float = 0.9,
    beta2: float = 0.95,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    rms_scale: bool = True,
    bf16_regularization: bool = False,
):
    """
    Create Muon optimizer compatible with FSDP2 using muon-fsdp2 package.

    This optimizer handles DTensor parameters correctly with gather/scatter operations
    for the Newton-Schulz orthogonalization.

    Args:
        model: The FSDP2-wrapped model to optimize
        muon_lr: Learning rate for Muon parameters (2D+ hidden weights)
        adamw_lr: Learning rate for Adam parameters (embeddings, heads, 1D params)
        beta1, beta2: Adam betas for non-Muon parameters
        weight_decay: Weight decay
        momentum: Muon momentum
        nesterov: Use Nesterov momentum
        ns_steps: Newton-Schulz iteration steps
        rms_scale: Scale gradients by RMS (Moonlight paper style)

    Returns:
        muon_fsdp2.Muon optimizer
    """
    try:
        from muon_fsdp2_tracked import Muon
    except ImportError:
        raise ImportError("FSDP2 Muon requires muon_fsdp2_tracked.py (local fork of muon-fsdp2 with update norm tracking)")

    muon_params, adamw_params = get_muon_param_groups(model)

    param_groups = []

    if muon_params:
        param_groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scale=rms_scale,
            )
        )

    if adamw_params:
        param_groups.append(
            dict(
                params=adamw_params,
                use_muon=False,
                lr=adamw_lr,
                betas=(beta1, beta2),
                eps=1e-10,
                weight_decay=weight_decay,
            )
        )

    return Muon(param_groups, bf16_regularization=bf16_regularization)


def get_muon_param_groups(model: PreTrainedModel):
    """
    Create parameter groups for Muon optimizer.

    Muon applies to 2D+ hidden layer weights, while AdamW handles:
    - Embedding layers
    - Output/classifier heads
    - 1D parameters (biases, layernorms)

    Returns:
        Tuple of (muon_params, adamw_params)
    """
    muon_params = []
    adamw_params = []

    # Get embedding and lm_head parameter names
    embed_and_head_names = set()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_and_head_names.add("model.embed_tokens")
    if hasattr(model, "lm_head"):
        embed_and_head_names.add("lm_head")
    # Also check for common embedding names
    embed_and_head_names.update(["embed_tokens", "wte", "wpe", "lm_head", "embed_out"])

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is an embedding or output head
        is_embed_or_head = any(n in name for n in embed_and_head_names)

        # Muon for 2D+ hidden weights, AdamW for everything else
        if param.ndim >= 2 and not is_embed_or_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def create_optimizer(
    model: PreTrainedModel,
    optimizer_type: str,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    weight_decay: float = 0.0,
    muon_lr: float = 0.02,
) -> torch.optim.Optimizer:
    """
    Create optimizer based on type.

    Args:
        model: The model to optimize
        optimizer_type: "adamw" or "muon"
        lr: Learning rate (used for AdamW params)
        beta1, beta2: Adam betas
        weight_decay: Weight decay
        muon_lr: Learning rate for Muon parameters (default 0.02)

    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

    elif optimizer_type == "muon":
        # Try PyTorch 2.9+ native Muon first
        try:
            from torch.optim import Muon

            has_native_muon = True
        except ImportError:
            has_native_muon = False

        if has_native_muon:
            # Native PyTorch Muon
            muon_params, adamw_params = get_muon_param_groups(model)

            # PyTorch Muon requires separate handling of param groups
            # For simplicity, we create a combined optimizer approach
            if muon_params and adamw_params:
                # Use Muon with adamw_params as backend optimizer
                param_groups = [
                    {
                        "params": muon_params,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    },
                ]
                # Create Muon optimizer
                muon_opt = Muon(
                    param_groups,
                    lr=lr,
                    momentum=0.95,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
                # Create AdamW for non-Muon params
                adamw_opt = AdamW(
                    adamw_params,
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                )
                # Return a combined optimizer wrapper
                return CombinedOptimizer(muon_opt, adamw_opt)
            elif muon_params:
                return Muon(
                    [{"params": muon_params, "lr": lr}],
                    momentum=0.95,
                    adjust_lr_fn="match_rms_adamw",
                    weight_decay=weight_decay,
                )
            else:
                # No Muon-eligible params, fall back to AdamW
                return AdamW(
                    model.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                )

        else:
            # Fallback: try standalone muon package
            try:
                from muon import MuonWithAuxAdam
            except ImportError:
                raise ImportError(
                    "Muon optimizer requires either PyTorch >= 2.9 or the 'muon' package. "
                    "Install with: pip install muon"
                )

            muon_params, adamw_params = get_muon_param_groups(model, muon_lr, lr, weight_decay)

            param_groups = []
            if muon_params:
                param_groups.append(
                    dict(
                        params=muon_params,
                        use_muon=True,
                        lr=muon_lr,
                        weight_decay=weight_decay,
                    )
                )
            if adamw_params:
                param_groups.append(
                    dict(
                        params=adamw_params,
                        use_muon=False,
                        lr=lr,
                        betas=(beta1, beta2),
                        weight_decay=weight_decay,
                    )
                )

            return MuonWithAuxAdam(param_groups)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adamw' or 'muon'.")


class CombinedOptimizer:
    """
    Wrapper that combines two optimizers (e.g., Muon + AdamW).
    Allows using Muon for hidden layers and AdamW for embeddings/heads.
    """

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


class MixedPrecisionOptimizer:
    """
    Optimizer wrapper that maintains FP32 master weights for mixed precision training.

    This is necessary when using Flash Attention 2 (which requires bf16/fp16 model weights)
    but you want FP32 precision for optimizer states and gradient accumulation.

    The pattern:
    1. Model weights are in bf16 (for Flash Attention compatibility)
    2. FP32 copies of weights are maintained as "master weights"
    3. Optimizer operates on FP32 master weights
    4. After optimizer.step(), FP32 master weights are copied back to bf16 model

    Usage:
        model = model.to(torch.bfloat16)  # Required for Flash Attention
        base_optimizer = AdamW(model.parameters(), lr=1e-5)
        optimizer = MixedPrecisionOptimizer(base_optimizer, model)

        # Training loop:
        loss.backward()  # Gradients in bf16
        optimizer.step()  # Converts grads to fp32, updates fp32 master, copies to bf16
        optimizer.zero_grad()
    """

    def __init__(self, optimizer: torch.optim.Optimizer, model: PreTrainedModel):
        self.optimizer = optimizer
        self.model = model

        # Create FP32 master weights
        self.fp32_master_params = []
        self.bf16_params = []

        for param_group in optimizer.param_groups:
            fp32_group = []
            bf16_group = []
            for param in param_group["params"]:
                if param.requires_grad:
                    # Create FP32 copy
                    fp32_param = param.detach().float().clone()
                    fp32_param.requires_grad = True
                    fp32_group.append(fp32_param)
                    bf16_group.append(param)
            self.fp32_master_params.append(fp32_group)
            self.bf16_params.append(bf16_group)

        # Replace optimizer params with FP32 versions
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["params"] = self.fp32_master_params[i]

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients on both bf16 model params and fp32 master params."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
        # Also zero bf16 grads
        for bf16_group in self.bf16_params:
            for param in bf16_group:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

    def step(self, closure=None):
        """
        1. Copy bf16 gradients to fp32 master params
        2. Run optimizer step on fp32 params
        3. Copy updated fp32 params back to bf16 model
        """
        # Copy gradients from bf16 to fp32
        for fp32_group, bf16_group in zip(self.fp32_master_params, self.bf16_params):
            for fp32_param, bf16_param in zip(fp32_group, bf16_group):
                if bf16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = bf16_param.grad.float()
                    else:
                        fp32_param.grad.copy_(bf16_param.grad)

        # Run optimizer step on FP32 params
        self.optimizer.step(closure)

        # Copy updated FP32 params back to bf16 model
        for fp32_group, bf16_group in zip(self.fp32_master_params, self.bf16_params):
            for fp32_param, bf16_param in zip(fp32_group, bf16_group):
                bf16_param.data.copy_(fp32_param.data)

    def state_dict(self):
        """Return optimizer state dict (includes FP32 master weights in optimizer state)."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict and sync to bf16 model."""
        self.optimizer.load_state_dict(state_dict)
        # Sync FP32 master weights to bf16 model
        for fp32_group, bf16_group in zip(self.fp32_master_params, self.bf16_params):
            for fp32_param, bf16_param in zip(fp32_group, bf16_group):
                bf16_param.data.copy_(fp32_param.data)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def create_mixed_precision_optimizer(
    model: PreTrainedModel,
    optimizer_type: str,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    weight_decay: float = 0.0,
    muon_lr: float = 0.02,
) -> MixedPrecisionOptimizer:
    """
    Create a mixed precision optimizer with FP32 master weights.

    Use this when training with Flash Attention 2 (bf16 model) but wanting
    FP32 precision for optimizer states.
    """
    # First create base optimizer (will be wrapped)
    base_optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        muon_lr=muon_lr,
    )

    return MixedPrecisionOptimizer(base_optimizer, model)
