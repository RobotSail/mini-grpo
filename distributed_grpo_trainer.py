"""
Distributed GRPO Trainer with FSDP2 + vLLM inference.

Multi-GPU training using FSDP2 for model/optimizer/ref sharding.
vLLM runs on separate GPU(s) for fast rollout generation.

Key FSDP2 constraints:
  - All ranks must participate in every forward pass (parameter allgather)
  - Loss is multiplied by world_size to correct for FSDP2 reduce_mean
  - Microbatch counts are synchronized via all_reduce(MAX) to keep
    FSDP collectives in lockstep
  - Optimizer must be created AFTER FSDP wrapping (parameter objects change)
"""

import os
import re
import json
import random
import time
import asyncio
import logging

import datasets
import httpx
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    CPUOffloadPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm import tqdm
import pydantic as pd
from safetensors.torch import save_file

import utils
from optimizers import create_fsdp2_muon_optimizer
from tasks import RewardResult, get_reward_fn

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def log_rank_0(msg, *args, level=logging.INFO):
    if dist.get_rank() == 0:
        logger.log(level, msg, *args)


class GRPOSample(pd.BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    response: str
    old_logprobs: list[float]
    reward: float = 0.0
    advantage: float = 0.0
    has_format: bool = False
    is_parsable: bool = False
    is_correct: bool = False


class StatsTracker:
    def __init__(
        self,
        token_budget: int = 0,
        save_every_n_tokens: int = 0,
        step_budget: int = 0,
        save_every_n_steps: int = 0,
    ):
        self.token_budget = token_budget
        self.step_budget = step_budget
        self.save_every_n_tokens = save_every_n_tokens
        self.save_every_n_steps = save_every_n_steps
        self._tokens_seen = 0
        self._last_checkpoint_tokens = 0
        self._last_checkpoint_steps = 0
        self._optim_steps = 0
        self._iteration = 0

    def accumulate_tokens(self, n: int):
        self._tokens_seen += n

    def increment_optim_step(self):
        self._optim_steps += 1

    def should_save(self) -> bool:
        if self.save_every_n_tokens > 0 and (
            self._tokens_seen - self._last_checkpoint_tokens
        ) >= self.save_every_n_tokens:
            return True
        if self.save_every_n_steps > 0 and (
            self._optim_steps - self._last_checkpoint_steps
        ) >= self.save_every_n_steps:
            return True
        return False

    def mark_checkpointed(self):
        self._last_checkpoint_tokens = self._tokens_seen
        self._last_checkpoint_steps = self._optim_steps

    def completed(self) -> bool:
        if self.token_budget > 0 and self._tokens_seen >= self.token_budget:
            return True
        if self.step_budget > 0 and self._optim_steps >= self.step_budget:
            return True
        return False

    def advance_iteration(self):
        self._iteration += 1

    @property
    def optim_steps(self):
        return self._optim_steps

    @property
    def tokens_seen(self):
        return self._tokens_seen

    @property
    def iteration(self):
        return self._iteration


class InfiniteDatasetIterator:
    def __init__(self, ds: datasets.Dataset, seed: int):
        self.dataset = ds
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            for item in self.dataset.shuffle(self.seed + epoch):
                yield item
            epoch += 1


# ── Distributed GRPO Trainer ────────────────────────────────────────────────


class DistributedGRPOTrainer:
    """
    Multi-GPU GRPO trainer using FSDP2.

    Model, optimizer, and reference policy are sharded across training GPUs.
    vLLM runs on separate GPU(s) for rollout generation.
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        output_dir: str | None,
        token_budget: int = 0,
        inner_epochs: int = 2,
        inner_batch_size: int = 32,
        save_every_n_tokens: int = 0,
        # Step-based budget (alternative to token budget)
        max_steps: int = 0,
        save_every_n_steps: int = 0,
        # GRPO
        group_size: int = 16,
        batch_size: int = 64,
        clip_eps: float = 0.2,
        kl_strength: float = 0.01,
        format_reward: float = 0.1,
        # Reference policy update
        update_ref_every: int = 0,
        # Sampling
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        max_seq_len: int = 8192,
        # Memory / gradient accumulation
        max_tokens_per_gpu: int = 4096,
        # Optimizer
        optimizer_type: str = "adamw",
        lr: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        # vLLM (externally managed by orchestrator)
        vllm_url: str = "http://localhost:8000",
        vllm_checkpoint_dir: str = "/dev/shm/active-policy-distributed",
        # Reference model
        ref_cpu_offload: bool = False,
        # Logging
        use_wandb: bool = False,
        wandb_project: str = "grpo-distributed",
        wandb_run_name: str = None,
        wandb_entity: str = None,
        # Misc
        seed: int = 67,
        validation_path: str = None,
        # Loss averaging
        token_level_averaging: bool = False,
        # Think mode
        require_think: bool = False,
        # Task
        reward_fn=None,
        task: str = "gsm8k",
    ):
        utils.set_determinism(seed)

        # ── Distributed setup ──
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", str(self.local_rank))
            os.environ.setdefault("WORLD_SIZE", str(self.world_size))
            from datetime import timedelta
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.rank = dist.get_rank()

        log_rank_0(
            "distributed init: rank=%d, world_size=%d, local_rank=%d",
            self.rank, self.world_size, self.local_rank,
        )

        # ── Store hyperparameters ──
        self.seed = seed
        self.model_name = model_name
        self.output_dir = output_dir
        self.group_size = group_size
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.kl_strength = kl_strength
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len
        self.inner_epochs = inner_epochs
        self.inner_batch_size = inner_batch_size
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.gradient_clip = gradient_clip
        self.format_reward = format_reward
        self.token_level_averaging = token_level_averaging
        self.require_think = require_think
        self.update_ref_every = update_ref_every
        self.use_wandb = use_wandb
        self.reward_fn = reward_fn or get_reward_fn(task)

        self.stats = StatsTracker(
            token_budget=token_budget,
            save_every_n_tokens=save_every_n_tokens,
            step_budget=max_steps,
            save_every_n_steps=save_every_n_steps,
        )

        if output_dir and self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # ── Load dataset (all ranks load same data) ──
        self.training_dataset = datasets.load_dataset(
            "json", data_files=data_path, split="train"
        )
        self._train_iterator = None
        log_rank_0("loaded %d training samples from %s", len(self.training_dataset), data_path)

        self.validation_dataset = None
        if validation_path:
            self.validation_dataset = datasets.load_dataset(
                "json", data_files=validation_path, split="train"
            )
            log_rank_0("loaded %d validation samples", len(self.validation_dataset))

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ── Device mesh ──
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self.world_size,), mesh_dim_names=("fsdp",)
        )

        # ── Load and wrap models with FSDP2 ──
        self.policy, self.ref_policy = self._load_fsdp2_models(model_name, ref_cpu_offload=ref_cpu_offload)

        # Align pad token
        for m in [self.policy, self.ref_policy]:
            if self.tokenizer.pad_token_id and not m.config.pad_token_id:
                m.config.pad_token_id = self.tokenizer.pad_token_id

        # ── Create optimizer AFTER FSDP wrapping (critical!) ──
        if optimizer_type.lower() == "muon":
            self.optimizer = create_fsdp2_muon_optimizer(
                model=self.policy,
                muon_lr=lr,
                adamw_lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
            )
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.policy.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )

        log_rank_0("using %s optimizer (lr=%g)", optimizer_type.upper(), lr)

        # ── Wandb (rank 0 only) ──
        if use_wandb and self.rank == 0:
            if not WANDB_AVAILABLE:
                log_rank_0("wandb not installed, disabling")
                self.use_wandb = False
            else:
                utils.initialize_wandb(
                    wandb_project, wandb_run_name,
                    {
                        "model_name": model_name,
                        "task": task,
                        "token_budget": token_budget,
                        "max_steps": max_steps,
                        "group_size": group_size,
                        "batch_size": batch_size,
                        "inner_batch_size": inner_batch_size,
                        "inner_epochs": inner_epochs,
                        "lr": lr,
                        "clip_eps": clip_eps,
                        "kl_strength": kl_strength,
                        "format_reward": format_reward,
                        "update_ref_every": update_ref_every,
                        "token_level_averaging": token_level_averaging,
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "gradient_clip": gradient_clip,
                        "optimizer": optimizer_type,
                        "world_size": self.world_size,
                        "max_tokens_per_gpu": max_tokens_per_gpu,
                    },
                    entity=wandb_entity,
                )

        # ── vLLM (externally managed by orchestrator) ──
        self._vllm_base_url = vllm_url
        self._vllm_served_model_name = "policy"
        self._vllm_checkpoint_dir = vllm_checkpoint_dir
        log_rank_0("using external vLLM at %s", vllm_url)

    @property
    def train_iterator(self):
        if not self._train_iterator:
            self._train_iterator = iter(
                InfiniteDatasetIterator(self.training_dataset, seed=self.seed)
            )
        return self._train_iterator

    # ── Model Loading ──────────────────────────────────────────────────

    def _load_fsdp2_models(self, model_name: str, ref_cpu_offload: bool = False):
        """Load policy and ref models with FSDP2 wrapping + activation checkpointing."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        # ── Policy model ──
        log_rank_0("loading policy model (FP32, flash_attention_2)...")
        policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="flash_attention_2",
        )
        if hasattr(policy, "config"):
            policy.config.use_cache = False

        layers = policy.model.layers

        # Activation checkpointing per layer
        for i, layer in enumerate(layers):
            layers[i] = ptd_checkpoint_wrapper(layer, preserve_rng=True)

        # Per-layer FSDP wrapping
        for idx, block in enumerate(layers):
            reshard = idx < len(layers) - 1
            fully_shard(
                block,
                mesh=self.device_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard,
            )
        fully_shard(
            policy,
            mesh=self.device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )
        log_rank_0("policy model loaded and FSDP2 wrapped")

        # ── Reference model (frozen) ──
        offload_str = ", CPU offload" if ref_cpu_offload else ""
        log_rank_0("loading reference model (FP32, frozen%s)...", offload_str)
        ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="flash_attention_2",
        )
        ref.eval()
        ref.requires_grad_(False)
        if hasattr(ref, "config"):
            ref.config.use_cache = False

        ref_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        ref_offload_policy = CPUOffloadPolicy(pin_memory=True) if ref_cpu_offload else None

        ref_layers = ref.model.layers
        for idx, block in enumerate(ref_layers):
            fully_shard(
                block,
                mesh=self.device_mesh,
                mp_policy=ref_mp_policy,
                offload_policy=ref_offload_policy,
                reshard_after_forward=True,
            )
        fully_shard(
            ref,
            mesh=self.device_mesh,
            mp_policy=ref_mp_policy,
            offload_policy=ref_offload_policy,
            reshard_after_forward=True,
        )
        log_rank_0("reference model loaded (FSDP2%s)", offload_str)

        return policy, ref

    # ── State dict helpers ─────────────────────────────────────────────

    def _gather_full_state_dict(self) -> dict:
        """Gather full state dict from all FSDP ranks, offloaded to CPU."""
        return get_model_state_dict(
            self.policy,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=False,
            ),
        )

    # ── vLLM Weight Sync (vLLM is externally managed) ──────────────────

    def _save_weights_to_checkpoint(self):
        """Save policy weights for vLLM reload. All ranks participate in gather."""
        state_dict = self._gather_full_state_dict()

        if self.rank == 0:
            # Clone to break shared memory (tied weights like lm_head/embed_tokens)
            cpu_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
            safetensors_path = os.path.join(self._vllm_checkpoint_dir, "model.safetensors")
            save_file(cpu_dict, safetensors_path)

            weight_map = {name: "model.safetensors" for name in cpu_dict}
            total_size = sum(t.numel() * t.element_size() for t in cpu_dict.values())
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            with open(os.path.join(self._vllm_checkpoint_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=2)

        dist.barrier()

    def _sync_weights_to_vllm(self):
        """Sync updated policy weights to vLLM via sleep/wake API."""
        log_rank_0("syncing weights to vLLM...")
        self._save_weights_to_checkpoint()

        if self.rank == 0:
            timeout = httpx.Timeout(timeout=600.0, connect=30.0)
            time.sleep(2)

            with httpx.Client(timeout=timeout) as client:
                for step_name, url in [
                    ("pause", f"{self._vllm_base_url}/pause?wait_for_inflight_requests=false&clear_cache=true"),
                    ("sleep", f"{self._vllm_base_url}/sleep?level=2"),
                    ("wake_up weights", f"{self._vllm_base_url}/wake_up?tags=weights"),
                ]:
                    log_rank_0("calling %s...", step_name)
                    client.post(url).raise_for_status()

                log_rank_0("reloading weights...")
                client.post(
                    f"{self._vllm_base_url}/collective_rpc",
                    json={"method": "reload_weights"},
                ).raise_for_status()

                for step_name, url in [
                    ("wake_up kv_cache", f"{self._vllm_base_url}/wake_up?tags=kv_cache"),
                    ("reset_prefix_cache", f"{self._vllm_base_url}/reset_prefix_cache"),
                    ("resume", f"{self._vllm_base_url}/resume"),
                ]:
                    log_rank_0("calling %s...", step_name)
                    client.post(url).raise_for_status()

        dist.barrier()
        log_rank_0("weight sync complete")

    # ── Rollout Generation ─────────────────────────────────────────────

    @torch.no_grad()
    def _generate_rollouts(self) -> list[list[GRPOSample]]:
        """Generate rollouts via vLLM, compute old logprobs on all ranks."""
        self.policy.eval()

        # All ranks collect same prompts (same seed, same iterator state)
        prompts = []
        for _ in range(self.batch_size):
            sample = next(self.train_iterator)
            messages = sample["messages"]

            # Detect assistant prefix (e.g. R1-style "<think>\n")
            if messages and messages[-1].get("role") == "assistant":
                prompt_ids = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        continue_final_message=True,
                        return_tensors="pt",
                    )
                    .squeeze(0)
                    .tolist()
                )
            else:
                prompt_ids = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    .squeeze(0)
                    .tolist()
                )
            prompts.append({
                "prompt_ids": prompt_ids,
                "prompt_text": self.tokenizer.decode(prompt_ids),
                "answer": sample["answer"],
                "problem": sample.get("problem", sample["messages"][-1]["content"]),
                "messages": sample["messages"],
                "numbers": sample.get("numbers", None),
            })

        # Generate completions via vLLM (rank 0 sends requests)
        if self.rank == 0:
            vllm_results = asyncio.run(self._vllm_generate_async(prompts))
        else:
            vllm_results = [None] * len(prompts)

        # Broadcast results from rank 0 to all ranks
        vllm_results = self._broadcast_vllm_results(vllm_results)

        # All ranks: process results, compute old logprobs, grade, compute advantages
        groups = self._process_rollouts(prompts, vllm_results)

        log_rank_0(
            "generated %d groups with %d total samples",
            len(groups), sum(len(g) for g in groups),
        )

        # Detailed per-operation metrics (rank 0 only, countdown problems only)
        if self.rank == 0 and groups:
            detail_metrics = self._compute_rollout_details(prompts, groups)
            if detail_metrics:
                parts = []
                for key in ("correct_rate_mul", "correct_rate_div", "correct_rate_add",
                            "correct_rate_sub", "correct_rate_parens"):
                    full_key = f"rollout/{key}"
                    if full_key in detail_metrics:
                        label = key.replace("correct_rate_", "")
                        parts.append(f"{label}={100 * detail_metrics[full_key]:.1f}%")
                if "rollout/think_length_var" in detail_metrics:
                    parts.append(f"think_len_var={detail_metrics['rollout/think_length_var']:.0f}")
                if parts:
                    log_rank_0("per-op correct: %s", " | ".join(parts))

                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log(detail_metrics, step=self.stats.optim_steps)

            self._display_rollout_summary(prompts, groups)

        return groups

    async def _vllm_generate_async(self, prompts):
        """Rank 0: send prompts to vLLM and collect results."""
        completions_url = f"{self._vllm_base_url}/v1/completions"
        timeout = httpx.Timeout(timeout=120.0, connect=10.0)
        semaphore = asyncio.Semaphore(32)

        async def generate_for_prompt(prompt_data):
            body = {
                "model": self._vllm_served_model_name,
                "prompt": prompt_data["prompt_text"],
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "n": self.group_size,
                "echo": False,
            }
            if self.top_k > 0:
                body["top_k"] = self.top_k
            if self.top_p < 1.0:
                body["top_p"] = self.top_p

            async with semaphore:
                async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                    resp = await client.post(completions_url, json=body)
                    resp.raise_for_status()
                    return resp.json()

        log_rank_0("sending %d prompts to vLLM (group_size=%d)...", len(prompts), self.group_size)
        results = await asyncio.gather(
            *[generate_for_prompt(p) for p in prompts],
            return_exceptions=True,
        )
        return results

    def _broadcast_vllm_results(self, results):
        """Broadcast vLLM results from rank 0 to all ranks via pickling."""
        import pickle

        if self.rank == 0:
            data = pickle.dumps(results)
            size_tensor = torch.tensor([len(data)], dtype=torch.long, device=self.device)
        else:
            size_tensor = torch.zeros(1, dtype=torch.long, device=self.device)

        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        if self.rank == 0:
            data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(self.device)
        else:
            data_tensor = torch.zeros(size, dtype=torch.uint8, device=self.device)

        dist.broadcast(data_tensor, src=0)

        if self.rank != 0:
            results = pickle.loads(data_tensor.cpu().numpy().tobytes())

        return results

    def _process_rollouts(self, prompts, vllm_results):
        """Process vLLM results into GRPOSample groups with old logprobs."""
        groups = []
        total_correct = 0
        total_parsable = 0
        total_format = 0
        total_completions = 0
        total_reward = 0.0

        iterator = (
            tqdm(zip(prompts, vllm_results), total=len(prompts), desc="processing rollouts")
            if self.rank == 0
            else zip(prompts, vllm_results)
        )

        for prompt_data, vllm_result in iterator:
            if isinstance(vllm_result, Exception) or vllm_result is None:
                continue

            group_responses = []
            for choice in vllm_result.get("choices", []):
                text = choice.get("text", "")
                response_ids = self.tokenizer.encode(text, add_special_tokens=False)
                try:
                    eos_idx = response_ids.index(self.tokenizer.eos_token_id) + 1
                    response_ids = response_ids[:eos_idx]
                except ValueError:
                    pass
                if not response_ids:
                    continue
                decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                group_responses.append((decoded, response_ids))

            if not group_responses:
                continue

            # Compute old logprobs — ALL ranks must participate (FSDP forward)
            response_ids_list = [r[1] for r in group_responses]
            old_logprobs_list = self._compute_old_logprobs_batched(
                prompt_data["prompt_ids"], response_ids_list
            )
            dist.barrier()

            # Grade and build samples
            group = []
            answer = prompt_data["answer"]
            pd_with_format = {**prompt_data, "format_reward": self.format_reward, "require_think": self.require_think}
            for (response_text, response_ids), old_lps in zip(group_responses, old_logprobs_list):
                result = self.reward_fn(response_text, answer, pd_with_format)

                # Handle both RewardResult and legacy float returns
                if isinstance(result, RewardResult):
                    reward = result.reward
                    has_format = result.has_format
                    is_parsable = result.is_parsable
                    is_correct = result.is_correct
                elif isinstance(result, dict):
                    # Raw dict from countdown_reward_fn
                    has_format = result.get("has_format", False)
                    is_parsable = result.get("is_parsable", False)
                    is_correct = result.get("is_correct", False)
                    if is_correct:
                        reward = 1.0 + self.format_reward
                    elif has_format:
                        reward = self.format_reward
                    else:
                        reward = 0.0
                else:
                    reward = float(result)
                    has_format = reward >= 0.1
                    is_parsable = reward >= 0.1
                    is_correct = reward >= 1.0

                total_completions += 1
                total_reward += reward
                if has_format:
                    total_format += 1
                if is_parsable:
                    total_parsable += 1
                if is_correct:
                    total_correct += 1

                group.append(GRPOSample(
                    prompt_ids=prompt_data["prompt_ids"],
                    response_ids=response_ids,
                    response=response_text,
                    old_logprobs=old_lps,
                    reward=reward,
                    has_format=has_format,
                    is_parsable=is_parsable,
                    is_correct=is_correct,
                ))

            self._compute_advantages(group)
            groups.append(group)

        n = max(total_completions, 1)
        format_rate = total_format / n
        correct_rate = total_correct / n
        avg_reward = total_reward / n
        # Average format reward = format_reward * format_rate
        avg_format_reward = self.format_reward * format_rate
        # Average accuracy reward = correct_reward * correct_rate
        correct_reward = 1.0  # from reward space [0.0, format, correct]
        avg_accuracy_reward = correct_reward * correct_rate

        log_rank_0(
            "rollout stats: %d groups, %d completions | "
            "format=%.1f%%, correct=%.1f%% | "
            "avg_reward=%.4f (format=%.4f, accuracy=%.4f)",
            len(groups), total_completions,
            100 * format_rate, 100 * correct_rate,
            avg_reward, avg_format_reward, avg_accuracy_reward,
        )

        if self.use_wandb and WANDB_AVAILABLE and self.rank == 0:
            wandb.log({
                "rollout/format_rate": format_rate,
                "rollout/correct_rate": correct_rate,
                "rollout/parsable_rate": total_parsable / n,
                "rollout/avg_reward": avg_reward,
                "rollout/avg_format_reward": avg_format_reward,
                "rollout/avg_accuracy_reward": avg_accuracy_reward,
                "rollout/total_completions": total_completions,
                "rollout/num_groups": len(groups),
            }, step=self.stats.optim_steps)

        return groups

    def _compute_rollout_details(self, prompts, groups) -> dict:
        """Compute per-operation correct rates and thinking trace stats.

        Only meaningful for countdown problems (where 'numbers' is in prompt_data).
        Solves each problem to determine which operations are required, then
        tracks correct rates per operation category.
        """
        import re as _re
        from countdown_utils import solve_countdown

        think_pattern = _re.compile(r"<think>(.*?)</think>", _re.DOTALL | _re.IGNORECASE)

        # Check if this is a countdown task (first prompt has 'numbers')
        if not prompts or prompts[0].get("numbers") is None:
            return {}

        op_labels = {"+": "add", "-": "sub", "*": "mul", "/": "div", "parens": "parens"}
        op_total = {op: 0 for op in op_labels}
        op_correct = {op: 0 for op in op_labels}
        think_lengths = []

        for prompt_data, group in zip(prompts, groups):
            if not group:
                continue

            numbers = prompt_data.get("numbers")
            target = int(prompt_data["answer"])

            # Solve to find which operations the solution requires
            required_ops = set()
            if numbers:
                solution = solve_countdown(numbers, target)
                if solution:
                    if " + " in solution:
                        required_ops.add("+")
                    if " - " in solution:
                        required_ops.add("-")
                    if " * " in solution:
                        required_ops.add("*")
                    if " / " in solution:
                        required_ops.add("/")

                    # Check if parentheses are required (change the result when removed)
                    try:
                        val_with = eval(solution, {"__builtins__": {}})
                        val_without = eval(
                            solution.replace("(", "").replace(")", ""),
                            {"__builtins__": {}},
                        )
                        if abs(val_with - val_without) > 1e-6:
                            required_ops.add("parens")
                    except Exception:
                        required_ops.add("parens")

            # Track per-operation correct rates
            n_in_group = len(group)
            n_correct = sum(1 for s in group if s.is_correct)
            for op in required_ops:
                op_total[op] += n_in_group
                op_correct[op] += n_correct

            # Thinking trace lengths (characters)
            for s in group:
                match = think_pattern.search(s.response)
                think_lengths.append(len(match.group(1)) if match else 0)

        # Build metrics dict
        metrics = {}
        for op, label in op_labels.items():
            if op_total[op] > 0:
                metrics[f"rollout/correct_rate_{label}"] = op_correct[op] / op_total[op]

        if think_lengths:
            mean_len = sum(think_lengths) / len(think_lengths)
            var_len = sum((x - mean_len) ** 2 for x in think_lengths) / len(think_lengths)
            metrics["rollout/think_length_mean"] = mean_len
            metrics["rollout/think_length_var"] = var_len

        return metrics

    def _display_rollout_summary(self, prompts, groups):
        """Print rollout summary table and example responses (rank 0 only)."""
        import re as _re

        answer_pat = _re.compile(r"<answer>(.*?)</answer>", _re.DOTALL | _re.IGNORECASE)

        # Build table rows: one per prompt/group
        rows = []
        for prompt_data, group in zip(prompts, groups):
            if not group:
                continue
            n_parsed = sum(1 for s in group if s.is_parsable)
            n_correct = sum(1 for s in group if s.is_correct)
            avg_r = sum(s.reward for s in group) / len(group)

            # Find highest rewarded response's parsed answer
            best = max(group, key=lambda s: s.reward)
            matches = answer_pat.findall(best.response)
            parsed_answer = matches[-1].strip() if matches else "UNPARSABLE"
            # Truncate long expressions
            if len(parsed_answer) > 50:
                parsed_answer = parsed_answer[:47] + "..."

            rows.append({
                "problem": prompt_data["problem"],
                "target": prompt_data["answer"],
                "parsed_answer": parsed_answer,
                "n_parsed": n_parsed,
                "n_correct": n_correct,
                "n_total": len(group),
                "avg_reward": avg_r,
                "group": group,
                "prompt_data": prompt_data,
            })

        if not rows:
            return

        # Print table
        print("\n" + "=" * 120)
        print(f"  ROLLOUT SUMMARY (iteration {self.stats.iteration})")
        print("=" * 120)
        print(f"{'#':>3}  {'Target':>6}  {'Best Parsed Answer':<52}  {'Parsed':>8}  {'Correct':>8}  {'Avg Reward':>10}")
        print("-" * 120)
        for i, row in enumerate(rows):
            print(
                f"{i+1:>3}  {row['target']:>6}  {row['parsed_answer']:<52}  "
                f"{row['n_parsed']:>3}/{row['n_total']:<4}  "
                f"{row['n_correct']:>3}/{row['n_total']:<4}  "
                f"{row['avg_reward']:>10.4f}"
            )
        print("-" * 120)

        # Find best group (highest avg reward) and show a high-reward response
        best_group_row = max(rows, key=lambda r: r["avg_reward"])
        best_sample = max(best_group_row["group"], key=lambda s: s.reward)

        print(f"\n{'─' * 120}")
        print(f"  BEST RESPONSE (group with highest avg reward: {best_group_row['avg_reward']:.4f}, "
              f"target: {best_group_row['target']})")
        print(f"{'─' * 120}")
        print(f"  [USER]: {best_group_row['prompt_data']['problem']}")
        print(f"  [ASSISTANT]: {best_sample.response}")
        print(f"  [REWARD]: {best_sample.reward}")

        # Find most average group and show a random response
        import random as _random
        median_reward = sorted(rows, key=lambda r: r["avg_reward"])[len(rows) // 2]
        random_sample = _random.choice(median_reward["group"])

        print(f"\n{'─' * 120}")
        print(f"  RANDOM RESPONSE (group with median avg reward: {median_reward['avg_reward']:.4f}, "
              f"target: {median_reward['target']})")
        print(f"{'─' * 120}")
        print(f"  [USER]: {median_reward['prompt_data']['problem']}")
        print(f"  [ASSISTANT]: {random_sample.response}")
        print(f"  [REWARD]: {random_sample.reward}")
        print("=" * 120 + "\n")

    @torch.no_grad()
    def _compute_old_logprobs_batched(self, prompt_ids, response_ids_list):
        """Compute old logprobs via batched forward pass. All ranks participate (FSDP)."""
        prompt_len = len(prompt_ids)
        full_seqs = [prompt_ids + resp for resp in response_ids_list]
        max_len = max(len(s) for s in full_seqs)
        n = len(full_seqs)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(n, max_len, dtype=torch.float, device=self.device)

        for i, seq in enumerate(full_seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_mask[i, :len(seq)] = 1.0

        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        results = []
        for i, response_ids in enumerate(response_ids_list):
            response_len = len(response_ids)
            response_logits = logits[i, prompt_len - 1: prompt_len - 1 + response_len].float()
            response_log_probs = F.log_softmax(response_logits, dim=-1)
            token_ids = torch.tensor(response_ids, device=self.device, dtype=torch.long)
            lps = response_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            results.append(lps.tolist())

        del outputs, logits
        torch.cuda.empty_cache()
        return results

    @staticmethod
    def _compute_advantages(group):
        eps = 1e-8
        rewards = [s.reward for s in group]
        avg = sum(rewards) / len(rewards)
        var = sum((r - avg) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        if std < eps:
            for s in group:
                s.advantage = 0.0
        else:
            for s in group:
                adv = (s.reward - avg) / (std + eps)
                s.advantage = max(-10.0, min(10.0, adv))

    # ── Reference Policy Update ─────────────────────────────────────────

    def _update_ref_policy(self):
        """Copy FSDP shard-local state dict from policy to reference model."""
        log_rank_0("updating reference policy from current policy...")
        policy_sd = self.policy.state_dict()
        self.ref_policy.load_state_dict(policy_sd)
        del policy_sd
        torch.cuda.empty_cache()
        log_rank_0("reference policy updated")

    def _maybe_update_ref_policy(self):
        """Update reference policy if update_ref_every is set and interval is reached."""
        if self.update_ref_every > 0 and self.stats.optim_steps % self.update_ref_every == 0:
            self._update_ref_policy()

    # ── Training ───────────────────────────────────────────────────────

    def _train_policy(self, groups):
        """GRPO training with distributed data splitting and microbatch sync."""
        self.policy.train()

        all_samples = [s for g in groups for s in g]

        for epoch in range(self.inner_epochs):
            # All ranks shuffle with same seed to get same order
            epoch_seed = self.seed + self.stats.iteration * 100 + epoch
            rng = random.Random(epoch_seed)
            rng.shuffle(all_samples)

            # Split samples across ranks: rank k gets samples[k::world_size]
            local_samples = all_samples[self.rank::self.world_size]

            # inner_batch_size is global; each rank processes its share
            local_inner_batch = max(1, self.inner_batch_size // self.world_size)

            # Create local batches
            for batch_start in range(0, max(len(local_samples), 1), local_inner_batch):
                local_batch = local_samples[batch_start:batch_start + local_inner_batch]

                if local_batch:
                    batch = self._collate_grpo_batch(local_batch)
                    microbatches = self._split_into_microbatches(batch)
                    del batch
                else:
                    microbatches = []

                # Track local sample count for global batch size
                local_count = len(local_batch)

                # Synchronize microbatch count across ranks (FSDP lockstep)
                local_k = torch.tensor([len(microbatches)], dtype=torch.long, device=self.device)
                dist.all_reduce(local_k, op=dist.ReduceOp.MAX)
                global_k = local_k.item()

                # Pad with empty microbatches if this rank has fewer
                while len(microbatches) < global_k:
                    microbatches.append(None)  # padding marker

                # Get global batch size
                global_batch_size = torch.tensor(
                    [local_count], dtype=torch.long, device=self.device
                )
                dist.all_reduce(global_batch_size, op=dist.ReduceOp.SUM)
                global_batch_size = global_batch_size.item()

                if global_batch_size == 0:
                    continue

                # For token-level averaging: sync total response tokens across ranks
                if self.token_level_averaging:
                    local_response_tokens = sum(
                        len(s.response_ids) for s in local_batch
                    )
                    global_total_tokens_t = torch.tensor(
                        [local_response_tokens], dtype=torch.long, device=self.device
                    )
                    dist.all_reduce(global_total_tokens_t, op=dist.ReduceOp.SUM)
                    global_total_tokens = global_total_tokens_t.item()
                else:
                    global_total_tokens = 0

                total_loss = 0.0
                total_kl = 0.0
                total_ir = 0.0
                total_entropy = 0.0
                valid_mbs = 0
                batch_tokens = 0

                for mb_idx in range(global_k):
                    mb = microbatches[mb_idx]
                    microbatches[mb_idx] = None  # allow GC
                    is_padding = mb is None

                    if is_padding:
                        # Create a minimal dummy forward pass to keep FSDP in sync
                        loss, metrics = self._grpo_padding_step()
                    else:
                        loss, metrics = self._grpo_train_step(mb, global_batch_size, global_total_tokens)

                    if not is_padding and (torch.isnan(loss) or torch.isinf(loss)):
                        log_rank_0("NaN/Inf loss in microbatch %d/%d, zeroing", mb_idx + 1, global_k)
                        loss = loss * 0.0

                    loss.backward()

                    if not is_padding:
                        total_loss += loss.item()
                        total_kl += metrics["kl_div"]
                        total_ir += metrics["importance_ratio"]
                        total_entropy += metrics["entropy"]
                        valid_mbs += 1
                        batch_tokens += mb["rollout_lens"].sum().item()

                    del loss, mb
                    torch.cuda.empty_cache()

                del microbatches

                # Gradient clipping and optimizer step
                gradnorm = clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.stats.increment_optim_step()
                self._maybe_update_ref_policy()

                # All-reduce token count for accurate budget tracking
                tokens_tensor = torch.tensor([batch_tokens], dtype=torch.long, device=self.device)
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                self.stats.accumulate_tokens(tokens_tensor.item())

                if valid_mbs > 0:
                    avg_loss = total_loss / valid_mbs
                    avg_kl = total_kl / valid_mbs
                    avg_ir = total_ir / valid_mbs
                    avg_entropy = total_entropy / valid_mbs
                else:
                    avg_loss = avg_kl = avg_ir = avg_entropy = 0.0

                budget_str = f"tokens: {self.stats.tokens_seen}"
                if self.stats.token_budget > 0:
                    budget_str += f"/{self.stats.token_budget}"
                if self.stats.step_budget > 0:
                    budget_str += f" | steps: {self.stats.optim_steps}/{self.stats.step_budget}"

                log_rank_0(
                    "epoch %d/%d | step %d | loss: %.4f | kl: %.4f | "
                    "ir: %.4f | entropy: %.4f | gradnorm: %.4f | %s",
                    epoch + 1, self.inner_epochs,
                    self.stats.optim_steps, avg_loss, avg_kl, avg_ir,
                    avg_entropy,
                    gradnorm.item() if hasattr(gradnorm, "item") else gradnorm,
                    budget_str,
                )

                if self.use_wandb and WANDB_AVAILABLE and self.rank == 0:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/kl_divergence": avg_kl,
                        "train/importance_ratio": avg_ir,
                        "train/entropy": avg_entropy,
                        "train/grad_norm": gradnorm.item() if hasattr(gradnorm, "item") else gradnorm,
                        "train/optim_step": self.stats.optim_steps,
                        "train/tokens_trained": self.stats.tokens_seen,
                    }, step=self.stats.optim_steps)

                torch.cuda.empty_cache()

                if self.stats.completed():
                    return

    def _collate_grpo_batch(self, samples):
        """Create padded batch tensors from GRPO samples."""
        batch_size = len(samples)
        seq_lens = [len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples]
        max_len = max(seq_lens)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
        old_logprobs = torch.ones(batch_size, max_len, dtype=torch.float32)
        logprob_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        grpo_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        advantages = torch.zeros(batch_size, dtype=torch.float32)
        rollout_lens = torch.zeros(batch_size, dtype=torch.long)

        for i, s in enumerate(samples):
            full_ids = s.prompt_ids + s.response_ids
            prompt_len = len(s.prompt_ids)
            response_len = len(s.response_ids)
            seq_len = seq_lens[i]

            input_ids[i, :seq_len] = torch.tensor(full_ids[:-1], dtype=torch.long)
            attention_mask[i, :seq_len] = 1.0

            for j in range(response_len):
                pos = prompt_len - 1 + j
                if pos < seq_len:
                    logprob_ids[i, pos] = s.response_ids[j]
                    grpo_mask[i, pos] = True
                    old_logprobs[i, pos] = s.old_logprobs[j]

            advantages[i] = s.advantage
            rollout_lens[i] = response_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logprobs": old_logprobs,
            "logprob_ids": logprob_ids,
            "grpo_mask": grpo_mask,
            "advantages": advantages,
            "rollout_lens": rollout_lens,
            "num_tokens": sum(seq_lens),
            "num_sequences": batch_size,
        }

    def _split_into_microbatches(self, batch):
        """Split batch into microbatches respecting max_tokens_per_gpu."""
        total_tokens = batch["num_tokens"]
        if total_tokens <= self.max_tokens_per_gpu:
            return [batch]

        n = batch["num_sequences"]
        index_groups = []
        current_indices = []
        current_tokens = 0

        for i in range(n):
            seq_tokens = int(batch["attention_mask"][i].sum().item())
            if current_tokens + seq_tokens > self.max_tokens_per_gpu and current_indices:
                index_groups.append(current_indices)
                current_indices = []
                current_tokens = 0
            current_indices.append(i)
            current_tokens += seq_tokens

        if current_indices:
            index_groups.append(current_indices)

        result = []
        for indices in index_groups:
            mb_attn = batch["attention_mask"][indices]
            mb_max_len = int(mb_attn.sum(dim=-1).max().item())
            mb = {
                "input_ids": batch["input_ids"][indices, :mb_max_len],
                "attention_mask": mb_attn[:, :mb_max_len],
                "logprobs": batch["logprobs"][indices, :mb_max_len],
                "logprob_ids": batch["logprob_ids"][indices, :mb_max_len],
                "grpo_mask": batch["grpo_mask"][indices, :mb_max_len],
                "advantages": batch["advantages"][indices],
                "rollout_lens": batch["rollout_lens"][indices],
                "num_tokens": int(mb_attn.sum().item()),
                "num_sequences": len(indices),
            }
            result.append(mb)

        return result

    def _grpo_train_step(self, batch, global_batch_size, global_total_tokens=0):
        """GRPO training step with FSDP2 world-size correction."""
        input_ids = batch["input_ids"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        old_logprobs = batch["logprobs"].to(self.device)
        old_logprob_ids = batch["logprob_ids"].to(self.device)
        rollout_lens = batch["rollout_lens"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        grpo_mask = batch["grpo_mask"].to(self.device)

        gather_indices = old_logprob_ids.unsqueeze(-1)

        # Reference model forward (frozen, no grad)
        with torch.no_grad():
            ref_outputs = self.ref_policy(input_ids, attention_mask=attn_mask)
            ref_logits = ref_outputs.logits
            if self.temperature > 0:
                ref_logits = ref_logits / self.temperature
            ref_gathered = ref_logits.gather(dim=-1, index=gather_indices)
            ref_logsumexp = ref_logits.logsumexp(dim=-1, keepdim=True)
            ref_logprobs = (ref_gathered - ref_logsumexp).squeeze(-1).float()
            del ref_logits, ref_outputs, ref_gathered, ref_logsumexp
        torch.cuda.empty_cache()

        # Policy forward
        new_outputs = self.policy(input_ids=input_ids, attention_mask=attn_mask)
        new_logits = new_outputs.logits
        if self.temperature > 0:
            new_logits = new_logits / self.temperature
        new_gathered = new_logits.gather(dim=-1, index=gather_indices)
        new_logsumexp = new_logits.logsumexp(dim=-1, keepdim=True)
        new_logprobs = (new_gathered - new_logsumexp).squeeze(-1).float()
        del new_logits, new_gathered, new_logsumexp, new_outputs

        # Importance ratio
        log_ratio = (new_logprobs - old_logprobs.float()).clamp(-20, 20)
        importance_ratio = log_ratio.exp()

        # Clipped surrogate
        adv = advantages.unsqueeze(-1)
        unclipped = adv * importance_ratio
        clipped = adv * importance_ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        clipped_surrogate = torch.minimum(unclipped, clipped)

        # KL penalty
        log_diff = (ref_logprobs - new_logprobs).clamp(-20, 20)
        dkl_approx = (log_diff.exp() - log_diff - 1).clamp(0, 100)

        per_token_loss = clipped_surrogate - self.kl_strength * dkl_approx
        grpo_token_loss = per_token_loss * grpo_mask.float()

        if self.token_level_averaging and global_total_tokens > 0:
            # Token-level: every token gets equal weight 1/global_total_tokens
            # No per-sequence length normalization
            # world_size corrects for FSDP2 reduce_mean on gradients
            grpo_loss = -(grpo_token_loss.sum() * self.world_size) / global_total_tokens
        else:
            # Sequence-level: each sequence averaged by its own length, then averaged across batch
            safe_lens = rollout_lens.float().clamp(min=1.0)
            grpo_token_loss = grpo_token_loss / safe_lens.unsqueeze(-1)
            grpo_seq_loss = (grpo_token_loss.sum(dim=-1) * self.world_size) / global_batch_size
            grpo_loss = -grpo_seq_loss.sum()

        metrics = {
            "kl_div": dkl_approx[grpo_mask].mean().item() if grpo_mask.any() else 0.0,
            "importance_ratio": importance_ratio[grpo_mask].mean().item() if grpo_mask.any() else 1.0,
            "entropy": -new_logprobs[grpo_mask].mean().item() if grpo_mask.any() else 0.0,
        }
        return grpo_loss, metrics

    def _grpo_padding_step(self):
        """Dummy forward/backward to keep FSDP collectives in sync for padding microbatches."""
        pad_id = self.tokenizer.pad_token_id or 0
        dummy_ids = torch.full((1, 2), pad_id, dtype=torch.long, device=self.device)
        dummy_mask = torch.ones(1, 2, dtype=torch.float, device=self.device)

        # Ref model forward (keeps FSDP allgather in sync)
        with torch.no_grad():
            self.ref_policy(dummy_ids, attention_mask=dummy_mask)

        # Policy forward (keeps FSDP allgather in sync)
        outputs = self.policy(input_ids=dummy_ids, attention_mask=dummy_mask)
        loss = outputs.logits.sum() * 0.0  # zero loss, but graph exists for backward

        metrics = {"kl_div": 0.0, "importance_ratio": 1.0}
        return loss, metrics

    # ── Validation ─────────────────────────────────────────────────────

    async def _run_validation_async(self):
        if not self.validation_dataset:
            return {}

        correct = 0
        parsable = 0
        formatted = 0
        total = 0
        total_reward = 0.0

        # Build all requests upfront
        all_requests = []
        for j in range(len(self.validation_dataset)):
            sample = self.validation_dataset[j]
            prompt_ids = (
                self.tokenizer.apply_chat_template(
                    sample["messages"],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                .squeeze(0)
                .tolist()
            )
            req = {"prompt_text": self.tokenizer.decode(prompt_ids), "answer": sample["answer"]}
            if "numbers" in sample:
                req["numbers"] = sample["numbers"]
            all_requests.append(req)

        # Send all requests concurrently with a semaphore to limit connections
        completions_url = f"{self._vllm_base_url}/v1/completions"
        timeout = httpx.Timeout(timeout=300.0, connect=10.0)
        semaphore = asyncio.Semaphore(64)

        async def gen_one(req):
            body = {
                "model": self._vllm_served_model_name,
                "prompt": req["prompt_text"],
                "max_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "n": 1,
            }
            async with semaphore:
                async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                    resp = await client.post(completions_url, json=body)
                    resp.raise_for_status()
                    return resp.json(), req

        results = await asyncio.gather(*[gen_one(r) for r in all_requests])

        for result, req in results:
            text = result["choices"][0]["text"]
            answer = req["answer"]
            r = self.reward_fn(text, answer, req)
            total += 1
            if isinstance(r, RewardResult):
                total_reward += r.reward
                if r.has_format:
                    formatted += 1
                if r.is_parsable:
                    parsable += 1
                if r.is_correct:
                    correct += 1
            else:
                total_reward += float(r)
                if float(r) >= 0.1:
                    formatted += 1
                    parsable += 1
                if float(r) >= 1.0:
                    correct += 1

        n = max(total, 1)
        format_rate = formatted / n
        correct_rate = correct / n
        return {
            "format_rate": format_rate,
            "correct_rate": correct_rate,
            "parsable_rate": parsable / n,
            "avg_reward": total_reward / n,
            "avg_format_reward": self.format_reward * format_rate,
            "avg_accuracy_reward": correct_rate,
            "total": total,
        }

    def _run_validation(self):
        """Run validation. Only rank 0 generates via vLLM; all ranks barrier at the end."""
        if not self.validation_dataset:
            return {}
        if self.rank == 0:
            log_rank_0("running validation on %d samples...", len(self.validation_dataset))
            self.policy.eval()
            try:
                result = asyncio.run(self._run_validation_async())
            finally:
                self.policy.train()
        else:
            result = {}
        # All ranks must reach this barrier (validation can take minutes)
        dist.barrier()
        return result

    # ── Checkpointing ──────────────────────────────────────────────────

    def _save_checkpoint(self):
        if not self.output_dir:
            return

        if self.stats.step_budget > 0:
            name = f"checkpoint-step{self.stats.optim_steps}-{self.stats.tokens_seen}tok"
        else:
            name = f"checkpoint-{self.stats.tokens_seen}"
        path = os.path.join(self.output_dir, name)

        log_rank_0("saving checkpoint to %s...", path)

        state_dict = self._gather_full_state_dict()

        dist.barrier()
        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            # Cast to FP32 for saving
            # Clone to break shared memory (tied weights)
            cpu_dict = {k: v.cpu().float().clone() for k, v in state_dict.items()}

            # Save using HF format
            from huggingface_hub import split_torch_state_dict_into_shards

            split = split_torch_state_dict_into_shards(
                cpu_dict,
                filename_pattern="model{suffix}.safetensors",
                max_shard_size="5GB",
            )
            for filename, tensors in split.filename_to_tensors.items():
                shard = {k: cpu_dict[k] for k in tensors}
                save_file(shard, os.path.join(path, filename))

            index = {
                "metadata": split.metadata,
                "weight_map": split.tensor_to_filename,
            }
            with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)

            inner = getattr(self.policy, "module", self.policy)
            inner.config.to_json_file(os.path.join(path, "config.json"))
            self.tokenizer.save_pretrained(path)

        dist.barrier()
        log_rank_0("checkpoint saved")

    # ── Main Training Loop ─────────────────────────────────────────────

    def train(self):
        budget_desc = []
        if self.stats.token_budget > 0:
            budget_desc.append(f"token budget: {self.stats.token_budget}")
        if self.stats.step_budget > 0:
            budget_desc.append(f"step budget: {self.stats.step_budget}")
        log_rank_0(
            "starting distributed GRPO training: %d GPUs, "
            "%s, group_size: %d, batch_size: %d, "
            "max_tokens_per_gpu: %d",
            self.world_size, ", ".join(budget_desc) if budget_desc else "no budget set",
            self.group_size, self.batch_size, self.max_tokens_per_gpu,
        )
        self.policy.eval()

        # Save initial checkpoint
        if self.output_dir:
            init_path = os.path.join(self.output_dir, "checkpoint-initial")
            log_rank_0("saving initial checkpoint...")
            state_dict = self._gather_full_state_dict()
            dist.barrier()
            if self.rank == 0:
                os.makedirs(init_path, exist_ok=True)
                cpu_dict = {k: v.cpu().float().clone() for k, v in state_dict.items()}
                save_file(cpu_dict, os.path.join(init_path, "model.safetensors"))
                weight_map = {name: "model.safetensors" for name in cpu_dict}
                total_size = sum(t.numel() * t.element_size() for t in cpu_dict.values())
                index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
                with open(os.path.join(init_path, "model.safetensors.index.json"), "w") as f:
                    json.dump(index_data, f, indent=2)
                inner = getattr(self.policy, "module", self.policy)
                inner.config.to_json_file(os.path.join(init_path, "config.json"))
                self.tokenizer.save_pretrained(init_path)
            dist.barrier()
            log_rank_0("saved initial checkpoint: %s", init_path)

        while not self.stats.completed():
            self.stats.advance_iteration()

            log_rank_0("iteration %d: generating rollouts...", self.stats.iteration)
            groups = self._generate_rollouts()

            if not groups:
                log_rank_0("no groups generated, retrying...")
                continue

            log_rank_0("training policy...")
            self._train_policy(groups)

            if self.stats.should_save():
                val = self._run_validation()
                if val and self.rank == 0:
                    log_rank_0(
                        "validation (%d samples): format=%.1f%%, correct=%.1f%% | "
                        "avg_reward=%.4f (format=%.4f, accuracy=%.4f)",
                        val.get("total", 0),
                        val.get("format_rate", 0) * 100,
                        val.get("correct_rate", 0) * 100,
                        val.get("avg_reward", 0),
                        val.get("avg_format_reward", 0),
                        val.get("avg_accuracy_reward", 0),
                    )
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "val/format_rate": val.get("format_rate", 0),
                            "val/correct_rate": val.get("correct_rate", 0),
                            "val/parsable_rate": val.get("parsable_rate", 0),
                            "val/avg_reward": val.get("avg_reward", 0),
                            "val/avg_format_reward": val.get("avg_format_reward", 0),
                            "val/avg_accuracy_reward": val.get("avg_accuracy_reward", 0),
                        }, step=self.stats.optim_steps)

                self._save_checkpoint()
                self.stats.mark_checkpointed()

            log_rank_0("syncing weights to vLLM...")
            self._sync_weights_to_vllm()

            torch.cuda.empty_cache()

        log_rank_0(
            "training complete: %d tokens, %d steps",
            self.stats.tokens_seen, self.stats.optim_steps,
        )

        self._save_checkpoint()

        if self.use_wandb and WANDB_AVAILABLE and self.rank == 0:
            wandb.finish()

        dist.barrier()
