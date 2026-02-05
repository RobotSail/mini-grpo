import typer
import random
import torch
import numpy as np
import os
import torch.distributed as dist

import logging

# Create logger for utils module
logger = logging.getLogger(__name__)


WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError as IE:
    pass

def initialize_wandb(project: str, run_name: str, config: dict, entity: str = None):
    if not WANDB_AVAILABLE:
        typer.secho("Warning: wandb is not installed. Install with 'pip install wandb'", fg=typer.colors.YELLOW)
        raise ValueError("Warning: wandb is not installed. Install with 'pip install wandb'")

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )
    typer.secho("✓ Wandb initialized", fg=typer.colors.GREEN)

def init_distributed(gpu: int = 0):
    # we initialize a mock distributed environment so we can train with FSDP2 mixed precision
    # on a single node 

    # Initialize process group for single-GPU FSDP2 (if not already initialized by torchrun)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")
        logger.info("✓ Initialized single-GPU distributed process group")
        default_device = torch.device('cuda', gpu)
        torch.set_default_device(default_device)




def preview_tokenization(dataset, tokenizer):
    """Preview tokenized messages before training."""
    typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_YELLOW)
    typer.secho("  TOKENIZATION PREVIEW", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    typer.secho("=" * 60, fg=typer.colors.BRIGHT_YELLOW)

    # Get a sample from the dataset
    preview_sample = next(iter(dataset.shuffle().iter(1)))
    preview_messages = preview_sample["messages"][0]

    typer.secho("\nOriginal messages:", fg=typer.colors.BRIGHT_WHITE)
    for msg in preview_messages:
        typer.secho(f"  [{msg['role']}]: {msg['content']}", fg=typer.colors.WHITE)

    # Tokenize the messages
    tokenized = tokenizer.apply_chat_template(
        conversation=preview_messages,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    typer.secho(
        f"\nTokenized input_ids shape: {tokenized.shape}", fg=typer.colors.BRIGHT_WHITE
    )
    typer.secho(f"Number of tokens: {tokenized.numel()}", fg=typer.colors.BRIGHT_WHITE)
    typer.secho(f"\nToken IDs: {tokenized[0].tolist()}", fg=typer.colors.CYAN)

    # Decode back to show what the model sees
    decoded = tokenizer.decode(tokenized[0], skip_special_tokens=False)
    typer.secho(f"\nDecoded (with special tokens):", fg=typer.colors.BRIGHT_WHITE)
    typer.secho(f"{decoded}", fg=typer.colors.GREEN)

    typer.secho("\n" + "=" * 60 + "\n", fg=typer.colors.BRIGHT_YELLOW)


def display_scorecard(
    rollouts: list,
    epoch: int,
    epochs: int,
    return_metrics: bool = False,
) -> dict | None:
    """
    Display epoch scorecard with training metrics.

    Args:
        rollouts: List of Sample objects with rollouts
        epoch: Current epoch number (0-indexed)
        epochs: Total number of epochs
        return_metrics: If True, return metrics dict

    Returns:
        If return_metrics=True, returns dict with parsable_pct, correct_pct, accuracy_pct
    """
    # Calculate and display epoch scorecard
    total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
    parsable_rollouts = sum(
        sum(1 for r in sample.rollouts if r.is_parsable) for sample in rollouts
    )
    correct_rollouts = sum(
        sum(1 for r in sample.rollouts if r.is_correct) for sample in rollouts
    )

    parsable_pct = (
        (parsable_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    )
    correct_pct = (correct_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    accuracy_pct = (
        (correct_rollouts / parsable_rollouts * 100) if parsable_rollouts > 0 else 0
    )

    typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_CYAN)
    typer.secho(
        f"  EPOCH {epoch + 1}/{epochs} SCORECARD",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )
    typer.secho("=" * 60, fg=typer.colors.BRIGHT_CYAN)
    typer.secho(f"  Total Rollouts:     {total_rollouts}", fg=typer.colors.WHITE)
    typer.secho(
        f"  Parsable:           {parsable_rollouts}/{total_rollouts} ({parsable_pct:.1f}%)",
        fg=typer.colors.YELLOW if parsable_pct < 90 else typer.colors.GREEN,
    )
    typer.secho(
        f"  Correct:            {correct_rollouts}/{total_rollouts} ({correct_pct:.1f}%)",
        fg=typer.colors.GREEN if correct_pct > 50 else typer.colors.RED,
    )
    typer.secho(
        f"  Accuracy (of parsable): {correct_rollouts}/{parsable_rollouts} ({accuracy_pct:.1f}%)",
        fg=typer.colors.BRIGHT_GREEN
        if accuracy_pct > 70
        else typer.colors.BRIGHT_YELLOW
        if accuracy_pct > 40
        else typer.colors.BRIGHT_RED,
    )
    typer.secho("=" * 60 + "\n", fg=typer.colors.BRIGHT_CYAN)

    if return_metrics:
        return {
            "parsable_pct": parsable_pct,
            "correct_pct": correct_pct,
            "accuracy_pct": accuracy_pct,
            "total_rollouts": total_rollouts,
        }
    return None

def set_determinism(seed: int):
    # seeds all related libraries at the start of training
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Enable deterministic CUDA operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark mode is non-deterministic

    # Set CUBLAS workspace config for deterministic behavior
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Enable PyTorch's deterministic algorithms mode
    # Use warn_only=True to avoid errors from ops without deterministic implementations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only
        torch.use_deterministic_algorithms(True)
