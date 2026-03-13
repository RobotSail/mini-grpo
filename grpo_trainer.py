"""
GRPO Trainer with vLLM inference and configurable precision.

Uses the same GRPO loop, reward function, and loss computation as the
existing `train` command in cli.py, but with vLLM for fast rollout
generation on separate GPU(s).

Supports --precision {fp32, bf16} for paired sparsity experiments:
  - BF16: small RL updates fall below representational threshold,
    producing the sparse ΔW observed by Mukherjee et al. (arXiv:2505.11711)
  - FP32: all updates preserved, ΔW is dense
    (per Kang et al., arXiv:2509.04259)

An initial checkpoint is saved automatically for pre/post comparison.
"""

import json
import random
import re
import utils
import datasets
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import os
from optimizers import create_optimizer, create_fsdp2_muon_optimizer
from tqdm import tqdm
import pydantic as pd
from torch.nn.utils.clip_grad import clip_grad_norm_

# vllm server management
import subprocess
import sys
import tempfile
import time
import httpx
import asyncio

import logging

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> float:
    """Parse a number from text, handling common GSM8K formats."""
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")
    text = text.strip()
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)
    text = text.replace("%", "").replace(",", "").strip()
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found: {text}")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number: {text}")
    return float(match.group())


def reward_response(response: str, answer: float) -> float:
    """
    GSM8K format reward — identical to cli.py grade_groups().

    Uses the LAST <answer>...</answer> tag:
      0.0  — no parsable <answer> tag
      0.1  — parsable but numerically wrong  (+0.1 format reward)
      1.1  — correct answer                  (+0.1 format + 1.0 correct)
    """
    matches = answer_pattern.findall(response)
    if not matches:
        return 0.0

    last_match = matches[-1]
    try:
        parsed = parse_number(last_match)
    except ValueError:
        return 0.0

    # format reward for having a parsable <answer> tag
    reward = 0.1
    if abs(parsed - float(answer)) < 1e-6:
        reward += 1.0
    return reward


class GRPOSample(pd.BaseModel):
    """Single rollout sample for GRPO training."""

    prompt_ids: list[int]
    response_ids: list[int]
    response: str
    old_logprobs: list[float]
    reward: float = 0.0
    advantage: float = 0.0
    is_parsable: bool = False
    is_correct: bool = False


class StatsTracker:
    """Tracks training progress and checkpoint intervals."""

    def __init__(self, token_budget: int, checkpoint_frequency: int):
        self.token_budget = token_budget
        self.checkpoint_frequency = checkpoint_frequency
        self._tokens_seen = 0
        self._last_checkpoint = 0
        self._optim_steps = 0
        self._iteration = 0

    def accumulate_tokens(self, n: int):
        self._tokens_seen += n

    def increment_optim_step(self):
        self._optim_steps += 1

    def should_save(self) -> bool:
        return self.checkpoint_frequency > 0 and (
            self._tokens_seen - self._last_checkpoint
        ) >= self.checkpoint_frequency

    def mark_checkpointed(self):
        self._last_checkpoint = self._tokens_seen

    def completed(self) -> bool:
        return self._tokens_seen >= self.token_budget

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
    """Infinite iterator over a dataset with per-epoch shuffling."""

    def __init__(self, ds: datasets.Dataset, seed: int):
        self.dataset = ds
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            for item in self.dataset.shuffle(self.seed + epoch):
                yield item
            epoch += 1
    


class GRPOTrainer:
    """
    GRPO trainer using vLLM for inference with configurable precision.

    Same GRPO loop, reward, and loss as the ``train`` command in cli.py.
    vLLM generates rollouts on separate GPU(s); old logprobs are computed
    via a forward pass on the training GPU in the selected precision.

    precision="fp32"  — model weights in float32, all updates preserved
    precision="bf16"  — model weights in bfloat16, small updates rounded away
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        output_dir: str | None,
        token_budget: int,
        inner_epochs: int = 2,
        inner_batch_size: int = 32,
        save_every_n_tokens: int = 0,
        # GRPO-specific (defaults match cli.py train command)
        group_size: int = 16,
        batch_size: int = 64,
        clip_eps: float = 0.2,
        kl_strength: float = 0.01,
        entropy_strength: float = 0.0,
        # Sampling
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        max_seq_len: int = 8192,
        # Memory
        max_tokens_per_microbatch: int = 0,
        # Optimizer
        optimizer_type: str = "adamw",
        lr: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        # Gradient clipping
        gradient_clip: float = 1.0,
        # Precision
        precision: str = "fp32",
        # Device
        gpu: int = 0,
        vllm_gpus: str = "1",
        vllm_gpu_memory_utilization: float = 0.9,
        # Logging
        use_wandb: bool = False,
        wandb_project: str = "mini-grpo-gsm8k",
        wandb_run_name: str = None,
        wandb_entity: str = None,
        # Misc
        seed: int = 67,
        # Validation
        validation_path: str = None,
        # Reward function: (response, answer, prompt_data) -> float
        reward_fn=None,
    ):
        utils.set_determinism(seed)

        self.seed = seed
        self.model_name = model_name
        self.output_dir = output_dir
        self.group_size = group_size
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.kl_strength = kl_strength
        self.entropy_strength = entropy_strength
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len
        self.inner_epochs = inner_epochs
        self.inner_batch_size = inner_batch_size
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.gradient_clip = gradient_clip
        self.use_wandb = use_wandb
        self.precision = precision
        self.reward_fn = reward_fn or (lambda resp, ans, pd: reward_response(resp, ans))

        assert precision in ("fp32", "bf16", "mixed"), (
            f"precision must be 'fp32', 'bf16', or 'mixed', got '{precision}'"
        )

        self.stats = StatsTracker(token_budget, save_every_n_tokens)

        # Validate output dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        self.training_dataset = datasets.load_dataset(
            "json", data_files=data_path, split="train"
        )
        self._train_iterator = None
        logger.info(
            "loaded %d training samples from %s",
            len(self.training_dataset),
            data_path,
        )

        # Load validation dataset
        self.validation_dataset = None
        if validation_path:
            self.validation_dataset = datasets.load_dataset(
                "json", data_files=validation_path, split="train"
            )
            logger.info(
                "loaded %d validation samples from %s",
                len(self.validation_dataset),
                validation_path,
            )

        self.device = torch.device("cuda", gpu)

        # Tokenizer (needed before model loading for pad token alignment)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if precision == "mixed":
            # ── Mixed precision: FP32 master weights, BF16 forward/backward ──
            # This is the standard training setup (what the `train` command uses).
            # FSDP2 manages casting: optimizer sees FP32, forward runs in BF16.
            utils.init_distributed(gpu)
            self.policy, self.ref_policy = self._load_fsdp2_models(
                model_name, self.device
            )

            # FSDP2 + flash attention handles memory well; no gradient
            # checkpointing needed.

            # Optimizer: FSDP2-compatible Muon or AdamW
            # Both go through create_fsdp2_muon_optimizer for update norm tracking.
            # For AdamW, all params are placed in the use_muon=False group.
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
                from adamw_tracked import AdamWTracked
                self.optimizer = AdamWTracked(
                    self.policy.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                )
        else:
            # ── Pure FP32 or pure BF16: no FSDP2, no flash attention ──
            model_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
            self.policy, self.ref_policy = self._load_models(
                model_name, self.device, model_dtype
            )

            # Gradient checkpointing needed without flash attention
            self.policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("gradient checkpointing enabled on policy model")

            if optimizer_type.lower() == "adamw":
                from adamw_tracked import AdamWTracked
                self.optimizer = AdamWTracked(
                    self.policy.parameters(),
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                )
            else:
                self.optimizer = create_optimizer(
                    model=self.policy,
                    optimizer_type=optimizer_type,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    muon_lr=lr,
                )

        # Initialize per-parameter update norm tracking
        if hasattr(self.optimizer, 'set_param_names'):
            self.optimizer.set_param_names(self.policy)
        if hasattr(self.optimizer, 'init_update_tracking'):
            self.optimizer.init_update_tracking(output_dir)

        # Align pad token
        for m in [self.policy, self.ref_policy]:
            if self.tokenizer.pad_token_id and not m.config.pad_token_id:
                m.config.pad_token_id = self.tokenizer.pad_token_id

        logger.info(
            "using %s optimizer (lr=%g, precision=%s)",
            optimizer_type.upper(), lr, precision,
        )

        # Initialize wandb
        if use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb not installed, disabling")
                self.use_wandb = False
            else:
                utils.initialize_wandb(
                    wandb_project,
                    wandb_run_name,
                    {
                        "model_name": model_name,
                        "token_budget": token_budget,
                        "group_size": group_size,
                        "batch_size": batch_size,
                        "inner_batch_size": inner_batch_size,
                        "inner_epochs": inner_epochs,
                        "lr": lr,
                        "clip_eps": clip_eps,
                        "kl_strength": kl_strength,
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "gradient_clip": gradient_clip,
                        "optimizer": optimizer_type,
                        "precision": precision,
                    },
                    entity=wandb_entity,
                )

        # Start vLLM server on separate GPU(s)
        self.vllm_gpus = vllm_gpus
        self.vllm_gpu_count = len(vllm_gpus.split(","))
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self._start_vllm_server()


    @property
    def train_iterator(self):
        if not self._train_iterator:
            self._train_iterator = iter(
                InfiniteDatasetIterator(self.training_dataset, seed=self.seed)
            )
        return self._train_iterator

    @staticmethod
    def _load_models(
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[PreTrainedModel, PreTrainedModel]:
        """Load policy and reference models in the specified dtype (no flash attention)."""
        dtype_name = {torch.float32: "FP32", torch.bfloat16: "BF16"}[dtype]

        logger.info("loading policy model in %s...", dtype_name)
        policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
        )
        logger.info(
            "policy model loaded (%s, %d parameters)",
            dtype_name,
            sum(p.numel() for p in policy.parameters()),
        )

        # Reference model is always loaded in the same dtype as the policy
        # so the KL penalty is computed in matching precision.
        logger.info("loading reference model in %s (frozen)...", dtype_name)
        ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
        )
        ref.eval()
        ref.requires_grad_(False)
        logger.info("reference model loaded (%s, frozen)", dtype_name)

        return policy, ref

    @staticmethod
    def _load_fsdp2_models(
        model_name: str, device: torch.device
    ) -> tuple[PreTrainedModel, PreTrainedModel]:
        """
        Load models with FSDP2 mixed precision (FP32 master weights, BF16 forward).

        Matches the RSTrainer / ``train`` command setup:
        - Policy: FP32 weights wrapped with FSDP2 MixedPrecisionPolicy
          (param_dtype=bf16, reduce_dtype=fp32) + flash attention 2
        - Reference: FP16, frozen, flash attention 2
        """
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        logger.info("loading policy model in FP32 (FSDP2 mixed precision)...")
        policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            attn_implementation="flash_attention_2",
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        for layer in policy.model.layers:
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(policy, mp_policy=mp_policy)
        logger.info(
            "policy model loaded (FSDP2 mixed precision, %d parameters)",
            sum(p.numel() for p in policy.parameters()),
        )

        logger.info("loading reference model in FP16 (frozen)...")
        ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        ref.eval()
        ref.requires_grad_(False)
        logger.info("reference model loaded (FP16, frozen)")

        return policy, ref

    def _get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Extract state dict, converting FSDP2 DTensors to regular tensors.

        Always saves in fp32 to avoid bf16 precision artifacts in spectral analysis.
        For non-FSDP2 models this is just a regular state_dict().
        """
        if self.precision != "mixed":
            return {k: v.detach().clone().float() for k, v in self.policy.state_dict().items()}

        from torch.distributed.tensor import DTensor

        state_dict = {}
        for name, param in self.policy.named_parameters():
            if isinstance(param.data, DTensor):
                state_dict[name] = param.data.full_tensor().detach().clone().float()
            else:
                state_dict[name] = param.data.detach().clone().float()
        return state_dict

    # ── vLLM Server Management ──────────────────────────────────────
    # Adapted from RSTrainer for managing the vLLM inference server
    # on separate GPU(s). vLLM runs in BF16 for inference speed;
    # the training-critical computation happens on the training GPU.

    def _setup_vllm_checkpoint_dir(self):
        """Set up shared memory directory for fast weight syncing."""
        import shutil

        checkpoint_dir = f"/dev/shm/active-policy-gpu{self.device.index}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._vllm_checkpoint_dir = checkpoint_dir

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_name)
        config.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        self._save_weights_to_checkpoint()

        logger.info("vLLM checkpoint dir: %s", checkpoint_dir)
        return checkpoint_dir

    def _save_weights_to_checkpoint(self):
        """Save current policy weights to vLLM checkpoint directory."""
        from safetensors.torch import save_file
        import json

        state_dict = {k: v.cpu() for k, v in self._get_policy_state_dict().items()}
        safetensors_path = os.path.join(
            self._vllm_checkpoint_dir, "model.safetensors"
        )
        save_file(state_dict, safetensors_path)

        weight_map = {name: "model.safetensors" for name in state_dict}
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(
            os.path.join(
                self._vllm_checkpoint_dir, "model.safetensors.index.json"
            ),
            "w",
        ) as f:
            json.dump(index, f, indent=2)

    def _start_vllm_server(self):
        """Start vLLM server on dedicated GPU(s) with sleep mode for weight reloading."""
        checkpoint_dir = self._setup_vllm_checkpoint_dir()

        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            self._vllm_port = s.getsockname()[1]

        self._vllm_served_model_name = "policy"

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            checkpoint_dir,
            "--served-model-name",
            self._vllm_served_model_name,
            "--port",
            str(self._vllm_port),
            "--gpu-memory-utilization",
            str(self.vllm_gpu_memory_utilization),
            "--max-model-len",
            str(self.max_seq_len),
            "--seed",
            str(self.seed),
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--no-enable-log-requests",
            "--data-parallel-size",
            str(self.vllm_gpu_count),
            "--enable-sleep-mode",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.vllm_gpus
        env["VLLM_SERVER_DEV_MODE"] = "1"

        log_dir = self.output_dir or tempfile.gettempdir()
        os.makedirs(log_dir, exist_ok=True)
        self._vllm_log_path = os.path.join(
            log_dir, f"vllm_grpo_{self._vllm_port}.log"
        )
        self._vllm_log_file = open(self._vllm_log_path, "wb")
        logger.info("vLLM logs: %s", self._vllm_log_path)

        self._vllm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._vllm_log_file,
            stderr=subprocess.STDOUT,
        )

        import atexit

        atexit.register(self._shutdown_vllm_server)

        self._vllm_base_url = f"http://localhost:{self._vllm_port}"
        self._wait_for_vllm_server()
        logger.info(
            "vLLM server ready at %s on GPU(s) %s",
            self._vllm_base_url,
            self.vllm_gpus,
        )

    def _wait_for_vllm_server(self, timeout: int = 300):
        """Wait for vLLM to become ready."""
        start = time.time()
        health_ok = False
        while time.time() - start < timeout:
            try:
                with httpx.Client(timeout=10) as client:
                    if not health_ok:
                        resp = client.get(f"{self._vllm_base_url}/health")
                        if resp.status_code == 200:
                            health_ok = True
                    if health_ok:
                        resp = client.get(f"{self._vllm_base_url}/v1/models")
                        if resp.status_code == 200 and resp.json().get("data"):
                            return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            if self._vllm_process.poll() is not None:
                log_tail = self._read_vllm_log_tail()
                raise RuntimeError(f"vLLM server died. Log tail:\n{log_tail}")
            time.sleep(2)

        raise RuntimeError(f"vLLM not ready within {timeout}s")

    def _read_vllm_log_tail(self, max_bytes: int = 20_000) -> str:
        path = getattr(self, "_vllm_log_path", None)
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - max_bytes, 0))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _sync_weights_to_vllm(self):
        """Sync updated policy weights to vLLM via collective_rpc reload."""
        logger.info("syncing weights to vLLM...")
        self._save_weights_to_checkpoint()

        timeout = httpx.Timeout(timeout=600.0, connect=30.0)

        with httpx.Client(timeout=timeout) as client:
            logger.info("reloading weights...")
            client.post(
                f"{self._vllm_base_url}/collective_rpc",
                json={"method": "reload_weights"},
            ).raise_for_status()

            logger.info("resetting caches...")
            client.post(
                f"{self._vllm_base_url}/reset_prefix_cache",
            ).raise_for_status()

        logger.info("weight sync complete")

    def _shutdown_vllm_server(self):
        """Gracefully shutdown the vLLM server and clean up."""
        if hasattr(self, "_vllm_process") and self._vllm_process.poll() is None:
            logger.info("shutting down vLLM server...")
            self._vllm_process.terminate()
            try:
                self._vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._vllm_process.kill()
                self._vllm_process.wait(timeout=5)

        log_file = getattr(self, "_vllm_log_file", None)
        if log_file:
            try:
                log_file.close()
            except OSError:
                pass

        if hasattr(self, "_vllm_checkpoint_dir") and os.path.exists(
            self._vllm_checkpoint_dir
        ):
            import shutil

            shutil.rmtree(self._vllm_checkpoint_dir, ignore_errors=True)

    # ── Rollout Generation ──────────────────────────────────────────

    @torch.no_grad()
    def _generate_rollouts(self) -> list[list[GRPOSample]]:
        """
        Generate GRPO rollouts using vLLM + on-device logprob computation.

        For each prompt in the batch:
        1. Generate group_size completions via vLLM
        2. Compute old logprobs via forward pass on training GPU (in model dtype)
        3. Grade responses (0.0 / 0.1 / 1.1 reward)
        4. Compute group-level advantages (GRPO normalization)

        Returns: list of groups, each group is a list of GRPOSample
        """
        self.policy.eval()
        groups = asyncio.run(self._generate_rollouts_async())
        logger.info(
            "generated %d groups with %d total samples",
            len(groups),
            sum(len(g) for g in groups),
        )
        return groups

    async def _generate_rollouts_async(self) -> list[list[GRPOSample]]:
        """Async rollout generation: vLLM for text, on-device forward for logprobs."""
        # Collect a batch of prompts from the training dataset
        prompts = []
        for _ in range(self.batch_size):
            sample = next(self.train_iterator)
            prompt_ids = (
                self.tokenizer.apply_chat_template(
                    sample["messages"],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                .squeeze(0)
                .tolist()
            )
            prompts.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_text": self.tokenizer.decode(prompt_ids),
                    "answer": sample["answer"],
                    "problem": sample.get(
                        "problem", sample["messages"][-1]["content"]
                    ),
                    "messages": sample["messages"],
                    "numbers": sample.get("numbers", None),
                }
            )

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
                async with httpx.AsyncClient(
                    timeout=timeout, http2=False
                ) as client:
                    resp = await client.post(completions_url, json=body)
                    resp.raise_for_status()
                    return resp.json()

        # Generate all completions concurrently
        logger.info(
            "sending %d prompts to vLLM (group_size=%d)...",
            len(prompts),
            self.group_size,
        )
        results = await asyncio.gather(
            *[generate_for_prompt(p) for p in prompts],
            return_exceptions=True,
        )

        # Process results into GRPOSample groups
        groups: list[list[GRPOSample]] = []
        total_correct = 0
        total_parsable = 0
        total_completions = 0

        for prompt_data, vllm_result in tqdm(
            zip(prompts, results),
            total=len(prompts),
            desc="processing rollouts",
        ):
            if isinstance(vllm_result, Exception):
                logger.warning("vLLM request failed: %s", vllm_result)
                continue
            if vllm_result is None:
                continue

            # Parse vLLM responses into response_ids
            group_responses = []  # (response_text, response_ids) pairs
            for choice in vllm_result.get("choices", []):
                text = choice.get("text", "")
                response_ids = self.tokenizer.encode(
                    text, add_special_tokens=False
                )

                # Truncate at EOS (include the EOS token)
                try:
                    eos_idx = (
                        response_ids.index(self.tokenizer.eos_token_id) + 1
                    )
                    response_ids = response_ids[:eos_idx]
                except ValueError:
                    pass

                if not response_ids:
                    continue

                decoded = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                group_responses.append((decoded, response_ids))

            if not group_responses:
                continue

            # Compute old logprobs via batched FP32 forward pass
            response_ids_list = [r[1] for r in group_responses]
            old_logprobs_list = self._compute_old_logprobs_batched(
                prompt_data["prompt_ids"], response_ids_list
            )

            # Build GRPOSample objects, grade, and compute advantages
            group: list[GRPOSample] = []
            answer = prompt_data["answer"]

            for (response_text, response_ids), old_lps in zip(
                group_responses, old_logprobs_list
            ):
                r = self.reward_fn(response_text, answer, prompt_data)
                total_completions += 1
                if r >= 0.1:
                    total_parsable += 1
                if r >= 1.0:
                    total_correct += 1

                group.append(
                    GRPOSample(
                        prompt_ids=prompt_data["prompt_ids"],
                        response_ids=response_ids,
                        response=response_text,
                        old_logprobs=old_lps,
                        reward=r,
                        is_parsable=r >= 0.1,
                        is_correct=r >= 1.0,
                    )
                )

            # Compute GRPO group-level advantages
            self._compute_advantages(group)
            groups.append(group)

        logger.info(
            "rollout stats: %d groups, %d completions, "
            "correct=%.1f%%, parsable=%.1f%%",
            len(groups),
            total_completions,
            100 * total_correct / max(total_completions, 1),
            100 * total_parsable / max(total_completions, 1),
        )

        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "rollout/correct_rate": total_correct
                    / max(total_completions, 1),
                    "rollout/parsable_rate": total_parsable
                    / max(total_completions, 1),
                    "rollout/total_completions": total_completions,
                    "rollout/num_groups": len(groups),
                },
                step=self.stats.optim_steps,
            )

        return groups

    @torch.no_grad()
    def _compute_old_logprobs_batched(
        self, prompt_ids: list[int], response_ids_list: list[list[int]]
    ) -> list[list[float]]:
        """
        Compute per-token logprobs for a group of responses sharing one prompt.

        Batches all responses together (padded to max length within the group)
        for a single forward pass, then extracts per-response logprobs.

        Forward pass runs in the model's native dtype; logprobs are returned
        as float32 for storage.
        """
        prompt_len = len(prompt_ids)

        # Build padded batch
        full_seqs = [prompt_ids + resp for resp in response_ids_list]
        max_len = max(len(s) for s in full_seqs)
        n = len(full_seqs)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full(
            (n, max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            n, max_len, dtype=torch.float, device=self.device
        )

        for i, seq in enumerate(full_seqs):
            input_ids[i, : len(seq)] = torch.tensor(
                seq, dtype=torch.long, device=self.device
            )
            attention_mask[i, : len(seq)] = 1.0

        # Forward pass (runs in model dtype)
        outputs = self.policy(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits  # (n, max_len, vocab_size)

        # Extract per-response logprobs (cast to float32 for precision)
        results = []
        for i, response_ids in enumerate(response_ids_list):
            response_len = len(response_ids)

            # logits[i, prompt_len-1] predicts response_ids[0]
            # logits[i, prompt_len]   predicts response_ids[1]
            # etc.
            response_logits = logits[
                i, prompt_len - 1 : prompt_len - 1 + response_len
            ].float()
            response_log_probs = F.log_softmax(response_logits, dim=-1)

            token_ids = torch.tensor(
                response_ids, device=self.device, dtype=torch.long
            )
            lps = response_log_probs.gather(
                -1, token_ids.unsqueeze(-1)
            ).squeeze(-1)
            results.append(lps.tolist())

        del outputs, logits
        torch.cuda.empty_cache()
        return results

    @staticmethod
    def _compute_advantages(group: list[GRPOSample]):
        """
        Compute GRPO group-level advantages.

        A_i = (r_i - mean(r)) / (std(r) + eps)

        When std < eps (all rewards equal), advantages are set to 0
        to avoid division by near-zero.
        """
        eps = 1e-8
        rewards = [s.reward for s in group]
        avg = sum(rewards) / len(rewards)
        var = sum((r - avg) ** 2 for r in rewards) / len(rewards)
        std = var**0.5

        if std < eps:
            for s in group:
                s.advantage = 0.0
        else:
            for s in group:
                adv = (s.reward - avg) / (std + eps)
                s.advantage = max(-10.0, min(10.0, adv))

    # ── Training ────────────────────────────────────────────────────

    def _train_policy(self, groups: list[list[GRPOSample]]):
        """
        GRPO training on generated rollouts.

        For each inner epoch:
          - Shuffle all samples
          - Create padded batches of inner_batch_size
          - For each batch, compute GRPO loss and update policy
        """
        self.policy.train()

        # Flatten groups into individual samples
        all_samples = [s for g in groups for s in g]

        for epoch in range(self.inner_epochs):
            random.shuffle(all_samples)

            for batch_start in range(0, len(all_samples), self.inner_batch_size):
                batch_samples = all_samples[
                    batch_start : batch_start + self.inner_batch_size
                ]
                batch = self._collate_grpo_batch(batch_samples)

                # Split into microbatches if needed
                if self.max_tokens_per_microbatch > 0:
                    microbatches = self._split_into_microbatches(batch)
                else:
                    microbatches = [batch]

                # Free the full collated batch now that microbatches hold
                # their own (trimmed) tensor slices
                del batch

                num_microbatches = len(microbatches)
                total_loss = 0.0
                total_kl = 0.0
                total_ir = 0.0
                total_entropy = 0.0
                valid_mbs = 0
                batch_tokens = 0

                for mb_idx in range(num_microbatches):
                    mb = microbatches[mb_idx]
                    microbatches[mb_idx] = None  # allow GC

                    loss, metrics = self._grpo_train_step(mb)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("NaN/Inf loss in microbatch %d/%d, skipping", mb_idx + 1, num_microbatches)
                        del mb
                        torch.cuda.empty_cache()
                        continue

                    scaled_loss = loss / num_microbatches
                    scaled_loss.backward()

                    total_loss += loss.item()
                    total_kl += metrics["kl_div"]
                    total_ir += metrics["importance_ratio"]
                    total_entropy += metrics["entropy"]
                    valid_mbs += 1
                    batch_tokens += mb["rollout_lens"].sum().item()

                    # Free the computation graph and microbatch tensors
                    del loss, scaled_loss, mb
                    torch.cuda.empty_cache()

                del microbatches

                if valid_mbs == 0:
                    self.optimizer.zero_grad()
                    continue

                # Check for NaN gradients
                has_nan = False
                for name, p in self.policy.named_parameters():
                    if p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    ):
                        logger.warning(
                            "NaN gradient in %s, skipping step", name
                        )
                        has_nan = True
                        break

                if has_nan:
                    self.optimizer.zero_grad()
                    continue

                # Gradient clipping and optimizer step
                gradnorm = clip_grad_norm_(
                    self.policy.parameters(), self.gradient_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.stats.increment_optim_step()
                self.stats.accumulate_tokens(batch_tokens)

                # Flush per-parameter update norms (captured inside optimizer.step)
                avg_update_norm = 0.0
                if hasattr(self.optimizer, 'flush_update_norms'):
                    avg_update_norm = self.optimizer.flush_update_norms(
                        self.stats.optim_steps, self.stats.tokens_seen
                    )

                avg_loss = total_loss / valid_mbs
                avg_kl = total_kl / valid_mbs
                avg_ir = total_ir / valid_mbs
                avg_entropy = total_entropy / valid_mbs

                logger.info(
                    "epoch %d/%d | step %d | loss: %.4f | kl: %.4f | "
                    "ir: %.4f | entropy: %.4f | gradnorm: %.4f | "
                    "update_norm: %.6f | tokens: %d/%d",
                    epoch + 1,
                    self.inner_epochs,
                    self.stats.optim_steps,
                    avg_loss,
                    avg_kl,
                    avg_ir,
                    avg_entropy,
                    gradnorm.item()
                    if hasattr(gradnorm, "item")
                    else gradnorm,
                    avg_update_norm,
                    self.stats.tokens_seen,
                    self.stats.token_budget,
                )

                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/kl_divergence": avg_kl,
                            "train/importance_ratio": avg_ir,
                            "train/entropy": avg_entropy,
                            "train/grad_norm": gradnorm.item()
                            if hasattr(gradnorm, "item")
                            else gradnorm,
                            "train/avg_update_frobenius": avg_update_norm,
                            "train/optim_step": self.stats.optim_steps,
                            "train/tokens_trained": self.stats.tokens_seen,
                        },
                        step=self.stats.optim_steps,
                    )

                torch.cuda.empty_cache()

                if self.stats.completed():
                    return

    def _collate_grpo_batch(self, samples: list[GRPOSample]) -> dict:
        """
        Create padded batch tensors from GRPO samples.

        Alignment after causal shift:
        - input_ids[i] = (prompt + response)[:-1]
        - logprob_ids[i, prompt_len-1+j] = response_ids[j]
        - old_logprobs[i, prompt_len-1+j] = old_logprobs[j]
        - grpo_mask is True only at response positions
        """
        batch_size = len(samples)

        # Sequence lengths after causal shift
        seq_lens = [
            len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples
        ]
        max_len = max(seq_lens)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.float
        )
        # Default 1.0 for old_logprobs at non-response positions;
        # these positions are masked out by grpo_mask in the loss
        old_logprobs = torch.ones(
            batch_size, max_len, dtype=torch.float32
        )
        logprob_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long
        )
        grpo_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        advantages = torch.zeros(batch_size, dtype=torch.float32)
        rollout_lens = torch.zeros(batch_size, dtype=torch.long)

        for i, s in enumerate(samples):
            full_ids = s.prompt_ids + s.response_ids
            prompt_len = len(s.prompt_ids)
            response_len = len(s.response_ids)
            seq_len = seq_lens[i]

            # Causal shift: input is full[:-1]
            input_ids[i, :seq_len] = torch.tensor(
                full_ids[:-1], dtype=torch.long
            )
            attention_mask[i, :seq_len] = 1.0

            # Place response targets and logprobs at shifted positions
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

    def _split_into_microbatches(self, batch: dict) -> list[dict]:
        """
        Split a padded batch into microbatches by token count.

        Each microbatch's tensors are trimmed to its own max sequence
        length — without this, every microbatch carries the padding of
        the longest sequence in the original batch, wasting O(B*T^2)
        memory in attention.
        """
        total_tokens = batch["num_tokens"]
        if total_tokens <= self.max_tokens_per_microbatch:
            return [batch]

        n = batch["num_sequences"]
        index_groups = []
        current_indices = []
        current_tokens = 0

        for i in range(n):
            seq_tokens = int(batch["attention_mask"][i].sum().item())
            if (
                current_tokens + seq_tokens > self.max_tokens_per_microbatch
                and current_indices
            ):
                index_groups.append(current_indices)
                current_indices = []
                current_tokens = 0
            current_indices.append(i)
            current_tokens += seq_tokens

        if current_indices:
            index_groups.append(current_indices)

        result = []
        for indices in index_groups:
            # Slice rows for this microbatch
            mb_attn = batch["attention_mask"][indices]

            # Trim columns: find the max real length in THIS microbatch
            # so we don't carry padding from longer sequences in other microbatches
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

    def _grpo_train_step(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        GRPO training step — works in both FP32 and BF16.

        Forward passes run in the model's native dtype.  Logprobs and
        the loss are cast to float32 for numerical stability; gradients
        flow back in the parameter dtype so that BF16 mode produces the
        sparsity artifact (small updates rounded to zero).
        """
        input_ids = batch["input_ids"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        old_logprobs = batch["logprobs"].to(self.device)
        old_logprob_ids = batch["logprob_ids"].to(self.device)
        rollout_lens = batch["rollout_lens"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        grpo_mask = batch["grpo_mask"].to(self.device)

        # Gather indices for computing logprobs of specific tokens
        gather_indices = old_logprob_ids.unsqueeze(-1)  # (B, T, 1)

        # Reference model forward FIRST (frozen, no grad) so its logits
        # are freed before the policy forward allocates activations.
        with torch.no_grad():
            ref_outputs = self.ref_policy(
                input_ids, attention_mask=attn_mask
            )
            ref_logits = ref_outputs.logits
            if self.temperature > 0:
                ref_logits = ref_logits / self.temperature

            ref_gathered = ref_logits.gather(dim=-1, index=gather_indices)
            ref_logsumexp = ref_logits.logsumexp(dim=-1, keepdim=True)
            ref_logprobs = (ref_gathered - ref_logsumexp).squeeze(-1).float()
            del ref_logits, ref_outputs, ref_gathered, ref_logsumexp
        torch.cuda.empty_cache()

        # Policy forward pass (runs in model dtype — FP32 or BF16)
        new_outputs = self.policy(
            input_ids=input_ids, attention_mask=attn_mask
        )
        new_logits = new_outputs.logits

        # Temperature scaling
        if self.temperature > 0:
            new_logits = new_logits / self.temperature

        # Policy logprobs (cast to float32 for stable ratio computation)
        new_gathered = new_logits.gather(dim=-1, index=gather_indices)
        new_logsumexp = new_logits.logsumexp(dim=-1, keepdim=True)
        new_logprobs = (new_gathered - new_logsumexp).squeeze(-1).float()
        del new_logits, new_gathered, new_logsumexp, new_outputs

        # Importance ratio (all in float32 for stability)
        log_ratio = (new_logprobs - old_logprobs.float()).clamp(-20, 20)
        importance_ratio = log_ratio.exp()

        # Clipped surrogate objective
        adv = advantages.unsqueeze(-1)  # (B,) -> (B, 1)
        unclipped = adv * importance_ratio
        clipped = adv * importance_ratio.clamp(
            1 - self.clip_eps, 1 + self.clip_eps
        )
        clipped_surrogate = torch.minimum(unclipped, clipped)

        # KL penalty (approximate)
        log_diff = (ref_logprobs - new_logprobs).clamp(-20, 20)
        dkl_approx = log_diff.exp() - log_diff - 1
        dkl_approx = dkl_approx.clamp(min=0, max=100)

        # Per-token loss
        per_token_loss = clipped_surrogate - self.kl_strength * dkl_approx
        # Entropy bonus: sample-based estimate H ≈ -log p(x_sampled)
        if self.entropy_strength > 0:
            per_token_loss = per_token_loss + self.entropy_strength * (-new_logprobs)
        grpo_token_loss = per_token_loss * grpo_mask.float()

        # Sequence-level averaging
        safe_lens = rollout_lens.float().clamp(min=1.0)
        grpo_seq_loss = grpo_token_loss.sum(dim=-1) / safe_lens
        grpo_loss = -grpo_seq_loss.mean()

        # Metrics (only over masked positions)
        metrics = {
            "kl_div": dkl_approx[grpo_mask].mean().item()
            if grpo_mask.any()
            else 0.0,
            "importance_ratio": importance_ratio[grpo_mask].mean().item()
            if grpo_mask.any()
            else 1.0,
            "entropy": (-new_logprobs)[grpo_mask].mean().item()
            if grpo_mask.any()
            else 0.0,
        }
        return grpo_loss, metrics

    # ── Validation ──────────────────────────────────────────────────

    async def _run_validation_async(self) -> dict:
        """Validate current policy using vLLM (greedy decoding)."""
        if not self.validation_dataset:
            return {}

        correct = 0
        parsable = 0
        total = 0

        completions_url = f"{self._vllm_base_url}/v1/completions"
        timeout = httpx.Timeout(timeout=120.0, connect=10.0)
        limits = httpx.Limits(
            max_connections=50,
            max_keepalive_connections=0,
            keepalive_expiry=0,
        )

        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, http2=False
        ) as client:
            for i in range(0, len(self.validation_dataset), self.batch_size):
                batch = self.validation_dataset[i : i + self.batch_size]

                requests = []
                for j in range(len(batch["messages"])):
                    prompt_ids = (
                        self.tokenizer.apply_chat_template(
                            batch["messages"][j],
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                        .squeeze(0)
                        .tolist()
                    )
                    req = {
                            "prompt_text": self.tokenizer.decode(prompt_ids),
                            "answer": batch["answer"][j],
                        }
                    if "numbers" in batch:
                        req["numbers"] = batch["numbers"][j]
                    requests.append(req)

                async def gen_one(req):
                    body = {
                        "model": self._vllm_served_model_name,
                        "prompt": req["prompt_text"],
                        "max_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "n": 1,
                    }
                    resp = await client.post(completions_url, json=body)
                    resp.raise_for_status()
                    return resp.json(), req

                results = await asyncio.gather(
                    *[gen_one(r) for r in requests]
                )
                for result, req in results:
                    text = result["choices"][0]["text"]
                    answer = req["answer"]
                    r = self.reward_fn(text, answer, req)
                    total += 1
                    if r >= 0.1:
                        parsable += 1
                    if r >= 1.0:
                        correct += 1

        return {
            "correct_rate": correct / max(total, 1),
            "parsable_rate": parsable / max(total, 1),
            "total": total,
        }

    def _run_validation(self) -> dict:
        if not self.validation_dataset:
            return {}
        logger.info(
            "running validation on %d samples...",
            len(self.validation_dataset),
        )
        self.policy.eval()
        try:
            return asyncio.run(self._run_validation_async())
        finally:
            self.policy.train()

    # ── Checkpointing ───────────────────────────────────────────────

    def _save_checkpoint(self):
        """Save policy model checkpoint to output directory."""
        if not self.output_dir:
            return

        name = f"checkpoint-{self.stats.tokens_seen}"
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)

        logger.info("saving checkpoint to %s...", path)
        state_dict = self._get_policy_state_dict()
        self.policy.save_pretrained(path, state_dict=state_dict)
        self.tokenizer.save_pretrained(path)
        logger.info("checkpoint saved")

    # ── Main Training Loop ──────────────────────────────────────────

    def train(self):
        """
        GRPO training loop with vLLM for fast inference.

        1. Save initial checkpoint (for sparsity analysis)
        2. Loop until token budget:
           a. Generate rollouts with vLLM
           b. Compute old logprobs (FP32 forward pass)
           c. Train policy with GRPO loss
           d. Save checkpoint if interval reached
           e. Sync weights to vLLM
        3. Save final checkpoint
        """
        logger.info(
            "starting GRPO training: cuda:%d (vLLM on GPU(s) %s), "
            "precision: %s, token budget: %d, group_size: %d, batch_size: %d",
            self.device.index,
            self.vllm_gpus,
            self.precision,
            self.stats.token_budget,
            self.group_size,
            self.batch_size,
        )
        self.policy.eval()

        # Save initial checkpoint for pre/post sparsity comparison
        if self.output_dir:
            init_path = os.path.join(self.output_dir, "checkpoint-initial")
            os.makedirs(init_path, exist_ok=True)
            state_dict = self._get_policy_state_dict()
            self.policy.save_pretrained(init_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(init_path)
            logger.info(
                "saved initial checkpoint for sparsity analysis: %s",
                init_path,
            )

        while not self.stats.completed():
            self.stats.advance_iteration()

            # Generate rollouts with vLLM + FP32 logprobs
            logger.info(
                "iteration %d: generating rollouts...", self.stats.iteration
            )
            groups = self._generate_rollouts()

            if not groups:
                logger.warning("no groups generated, retrying...")
                continue

            # Train policy with GRPO
            logger.info("training policy...")
            self._train_policy(groups)

            # Checkpoint if interval reached
            if self.stats.should_save():
                if self.validation_dataset:
                    val = self._run_validation()
                    logger.info(
                        "validation: correct=%.1f%%, parsable=%.1f%% (%d samples)",
                        val.get("correct_rate", 0) * 100,
                        val.get("parsable_rate", 0) * 100,
                        val.get("total", 0),
                    )
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log(
                            {
                                "val/correct_rate": val.get("correct_rate", 0),
                                "val/parsable_rate": val.get("parsable_rate", 0),
                            },
                            step=self.stats.optim_steps,
                        )

                self._save_checkpoint()
                self.stats.mark_checkpointed()

            # Sync updated weights to vLLM for next iteration
            logger.info("syncing weights to vLLM...")
            self._sync_weights_to_vllm()

            torch.cuda.empty_cache()

        logger.info(
            "training complete: %d tokens, %d steps",
            self.stats.tokens_seen,
            self.stats.optim_steps,
        )

        # Save final checkpoint
        self._save_checkpoint()

        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        self._shutdown_vllm_server()
