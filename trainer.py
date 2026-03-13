import json
import random
import re
import utils
import datasets
import torch.distributed as dist
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
import os
from optimizers import create_fsdp2_muon_optimizer, create_mixed_precision_optimizer, create_optimizer
import typing as t
from tqdm import tqdm
from type_defs import RolloutResult, Problem, TokenSample, Sample
import pydantic as pd
import functools
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

# vllm server for fast inference on separate GPU
import subprocess
import sys
import tempfile
import time
import httpx
import asyncio
import signal

import logging

# wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

class RejectionSample(pd.BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    response: str
    reward: float

# Create logger for trainer module
logger = logging.getLogger(__name__)

# Suppress httpx HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> float:
    """
    Parse a string into a float, handling common formats from GSM8K answers.

    Handles:
    - Whitespace (leading/trailing/internal)
    - Percentage signs (42% -> 42.0)
    - Currency symbols ($100, EUR50, etc.)
    - Comma separators (1,000,000 -> 1000000)
    - Negative numbers (-42, negative prefix)
    - Decimal numbers (3.14)

    Returns: float
    Raises: ValueError if no valid number can be parsed
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")

    # Strip whitespace
    text = text.strip()

    # Remove currency symbols ($, EUR, GBP, JPY, etc.)
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)

    # Remove percentage sign (keep the number)
    text = text.replace("%", "")

    # Remove commas (thousand separators)
    text = text.replace(",", "")

    # Strip remaining whitespace after removals
    text = text.strip()

    # Check for digits
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in answer: {text}")

    # Extract the numeric portion (handles cases like "42 dollars" -> "42")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number from: {text}")

    return float(match.group())

def reward_response(response: str, answer: int | float) -> float:
    """
    Returns these rewards:
    0.0 if unparsable
    0.1 if parsable but incorrect
    1.1 if correct
    """

    # Find all answer tags
    matches = answer_pattern.findall(response)

    # No answer tags found - no reward
    if not matches:
        return 0.0

    # Take the LAST answer (final answer after reasoning)
    last_match = matches[-1]

    # Attempt to parse the response, if we can parse then instance 0.1 reward
    try:
        parsed_answer = parse_number(last_match)
    except ValueError:
        # cannot parse response
        return 0.0
    
    # Check correctness with tolerance for floating point comparison
    # GSM8K won't have any answers this small so it shouldn't be an issue
    expected = float(answer)
    is_close_enough = abs(parsed_answer - expected) < 1e-6
    return 1.1 if is_close_enough else 0.1  # these are the exact rewards used during the GRPO experiments


    





class StatsTracker:
    # simple module for tracking statistics
    def __init__(self, token_training_budget: int, checkpoint_frequency: int):
        self.token_training_budget = token_training_budget
        self.checkpoint_frequency = checkpoint_frequency
        self._train_tokens_seen = 0
        self._last_checkpoint_save = 0
        self._inference_iteration = 0
        self._optim_steps = 0

    def reset(self):
        self._train_tokens_seen = 0
        self._last_checkpoint_save = 0
        self._inference_iteration = 0
        self._optim_steps = 0

    def increment_optim_step(self):
        self._optim_steps += 1

    @property
    def optim_steps(self) -> int:
        return self._optim_steps
        
    def accumulate_tokens(self, tokens: int):
        self._train_tokens_seen += tokens

    def should_save(self):
        return (self._train_tokens_seen - self._last_checkpoint_save) >= self.checkpoint_frequency
    
    def mark_checkpointed(self):
        self._last_checkpoint_save = self._train_tokens_seen

    def completed_training(self):
        return self._train_tokens_seen >= self.token_training_budget
    
    def advance_iteration(self):
        self._inference_iteration += 1
    
    @property
    def train_tokens_seen(self) -> int:
        return self._train_tokens_seen
    
    @property
    def last_checkpoint_save(self) -> int:
        return self._last_checkpoint_save
    
    @property
    def inference_iteration(self) -> int:
        return self._inference_iteration

    
class InfiniteDatasetIterator:
    def __init__(self, ds: datasets.Dataset, seed: int):
        self.dataset = ds
        self.seed = seed
    
    def __iter__(self):
        epoch = 0
        while True:
            iterator = self.dataset.shuffle(self.seed + epoch)
            for item in iterator:
                yield item
            epoch += 1
        





        

class RSTrainer:

    @staticmethod
    def _load_fsdp_models(model_name: str, device: torch.device) -> tuple[PreTrainedModel, PreTrainedModel]:
        # Flash Attention 2 requires bf16/fp16 weights, otherwise use FP32 for mixed precision
        # Load in FP32 first, then apply FSDP2 MixedPrecisionPolicy for FP32 master weights
        policy_model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float32,  # Load FP32, FSDP2 will handle bf16 forward
            "attn_implementation": "flash_attention_2",
        }
        ref_model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16,  # Reference model in fp16 (inference only, better precision)
            "attn_implementation": "flash_attention_2",
        }
        logger.info("✓ Using Flash Attention 2 with FSDP2 mixed precision (FP32 master weights, bf16 forward)")

        # Initialize policy model
        model = AutoModelForCausalLM.from_pretrained(model_name, **policy_model_kwargs)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # Forward/backward in bf16 (Flash Attention compatible)
            reduce_dtype=torch.float32,  # Gradient reduction in fp32
        )
        # Apply FSDP2 to each transformer layer for memory efficiency
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
        logger.info("✓ Policy model wrapped with FSDP2 MixedPrecisionPolicy")

        # Reference model (frozen)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, **ref_model_kwargs)
        ref_model.eval()
        ref_model.requires_grad_(False)
        logger.info("✓ Reference model loaded in FP16 (frozen)")

        return (model, ref_model)
    

    @staticmethod
    def _valid_save_dir(output_dir: str = None) -> bool:
        if not output_dir:
            return False

        # Try to create the directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            return True
        except (OSError, PermissionError):
            return False


    # trainer class for rejection sampling
    def __init__(
        self, 
        data_path: str,
        model_name: str,
        output_dir: str | None,
        token_budget: int,
        inner_epochs: int,
        inner_batch_size: int,
        save_every_n_tokens: int,
        samples_to_accept: int,
        inference_batch_size: int,
        inference_group_size: int,
        # sampling params
        temperature: float,
        top_k: int,
        top_p: float,
        max_new_tokens: int,
        max_seq_len: int,
        max_tokens_per_gpu: int,
        use_wandb: bool,
        wandb_project: str,
        wandb_run_name: str,
        wandb_entity: str,
        seed: int,
        optimizer_type: str,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        # device configuration
        gpu: int = 0,
        vllm_gpus: str = "1",  # comma-separated GPU IDs for vLLM data parallel
        vllm_gpu_memory_utilization: float = 0.9,
        # validation
        validation_path: str = None,
        # Reward function: (response, answer, prompt_data) -> float
        reward_fn=None,
    ):

        # first we must set the seed
        utils.set_determinism(seed)

        # set some basic variables
        self.stats_tracker = StatsTracker(
            token_training_budget=token_budget,
            checkpoint_frequency=save_every_n_tokens,
        )
        self.inference_batch_size = inference_batch_size
        self.inference_group_size = inference_group_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.samples_to_accept = samples_to_accept
        self.inner_batch_size = inner_batch_size
        self.inner_epochs = inner_epochs
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.model_name = model_name
        self.output_dir = output_dir
        self.top_k = top_k
        self.top_p = top_p
        self.use_wandb = use_wandb
        self.reward_fn = reward_fn or (lambda resp, ans, pd: reward_response(resp, ans))

        # check basic validation
        if self.output_dir and not self._valid_save_dir(self.output_dir):
            raise ValueError(f'invalid output directory: cannot write to {output_dir}')

        # then we load the training dataset
        self.training_dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        self._train_iterator = None

        logger.info('loaded %d unique samples for training from %s', len(self.training_dataset), data_path)

        # load validation dataset if provided
        self.validation_dataset = None
        if validation_path:
            self.validation_dataset = datasets.load_dataset("json", data_files=validation_path, split="train")
            logger.info('loaded %d validation samples from %s', len(self.validation_dataset), validation_path)

        # next we load the model
        self.device = torch.device('cuda', gpu)
        utils.init_distributed(gpu)
        self.policy, self.ref_policy = self._load_fsdp_models(model_name, self.device)
        logger.info('loaded models with fsdp2 on cuda:%d', gpu)

        # now we can load the optimizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info('loaded tokenizer')

        # create optimizer
        optimizer = None
        if optimizer_type.lower() == "muon":
            # Use FSDP2-compatible Muon optimizer
            optimizer = create_fsdp2_muon_optimizer(
                model=self.policy,
                muon_lr=lr,
                adamw_lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
            )
            logger.info(f"✓ Using MUON optimizer (FSDP2-compatible via muon-fsdp2, lr={lr})")
        else:
            logger.info(f"Muon optimizer was not detected, selecting AdamW as the optimizer.")
            optimizer = create_optimizer(
                model=self.policy,
                optimizer_type="adamw",
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                muon_lr=lr,
            )
            logger.info(f"✓ Using AdamW optimizer {lr=}")
        assert optimizer is not None
        self.optimizer = optimizer
        

        # next we load wandb and populate whatever we need
        if self.use_wandb:
            run_config = {
                "model_name": model_name,
                "token_train_budget": token_budget,
                "samples_to_accept": samples_to_accept,
                "inference_batch_size": inference_batch_size,
                "group_size": inference_group_size,
                "inner_batch_size": inner_batch_size,
                "inner_epochs": inner_epochs,
                "lr": lr,
                "max_new_tokens": max_new_tokens,
                "max_seq_len": max_seq_len,
                "temperature": temperature,
                "optimizer": optimizer_type,
                "max_tokens_per_microbatch": max_tokens_per_gpu,
                "save_every_n_tokens": save_every_n_tokens,
                "beta1": beta1,
                "beta2": beta2,
                "wd": weight_decay,
            }
            utils.initialize_wandb(wandb_project, wandb_run_name, run_config, entity=wandb_entity)

        # initialize vllm for fast inference on separate GPU(s)
        self.vllm_gpus = vllm_gpus  # comma-separated string like "1,2,3"
        self.vllm_gpu_count = len(vllm_gpus.split(","))
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self._start_vllm_server()

        # Initialize per-parameter update norm tracking
        if hasattr(self.optimizer, 'set_param_names'):
            self.optimizer.set_param_names(self.policy)
        if hasattr(self.optimizer, 'init_update_tracking'):
            self.optimizer.init_update_tracking(self.output_dir)

    @property
    def train_iterator(self):
        if not self._train_iterator:
            self._train_iterator = iter(InfiniteDatasetIterator(self.training_dataset, seed=self.seed))
        
        return self._train_iterator

    def _setup_vllm_checkpoint_dir(self):
        """
        Set up a shared memory directory for fast weight syncing with vLLM.

        Uses /dev/shm for fast read/write operations.
        Returns the path to the checkpoint directory.
        """
        # use fixed path so vLLM can reload from the same location
        checkpoint_dir = "/dev/shm/active-policy"

        # clean up any existing directory and create fresh
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._vllm_checkpoint_dir = checkpoint_dir

        # save initial model config and tokenizer
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name)
        config.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # save initial weights
        self._save_weights_to_checkpoint()

        logger.info("vLLM checkpoint directory: %s", checkpoint_dir)
        return checkpoint_dir
    
    def _save_weights_to_checkpoint(self):
        """
        Save current policy weights to the vLLM checkpoint directory.
        
        Saves as safetensors with proper index file for vLLM to load.
        """
        from safetensors.torch import save_file
        import json
        
        state_dict = self._get_policy_state_dict()
        
        # move to CPU for saving
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        
        # save weights as safetensors
        safetensors_path = os.path.join(self._vllm_checkpoint_dir, "model.safetensors")
        save_file(state_dict_cpu, safetensors_path)
        
        # create safetensors index file
        weight_map = {name: "model.safetensors" for name in state_dict_cpu.keys()}
        total_size = sum(t.numel() * t.element_size() for t in state_dict_cpu.values())
        
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        
        index_path = os.path.join(self._vllm_checkpoint_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        logger.debug("saved weights to %s (%d params, %.2f MB)", 
                     safetensors_path, len(state_dict_cpu), total_size / 1024 / 1024)
    
    def _start_vllm_server(self):
        """
        Start vLLM as an OpenAI-compatible server on dedicated GPU(s).
        
        Enables sleep mode for efficient weight reloading without server restart.
        Supports data parallel mode with multiple GPUs via tensor parallelism.
        """
        # set up checkpoint directory in shared memory
        checkpoint_dir = self._setup_vllm_checkpoint_dir()
        
        logger.info(
            "starting vLLM server on GPU(s) %s (dp=%d) with checkpoint %s...", 
            self.vllm_gpus, self.vllm_gpu_count, checkpoint_dir
        )
        
        # find an available port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            self._vllm_port = s.getsockname()[1]
        
        # use a consistent model name for API requests
        self._vllm_served_model_name = "policy"
        
        # build server command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", checkpoint_dir,
            "--served-model-name", self._vllm_served_model_name,
            "--port", str(self._vllm_port),
            "--gpu-memory-utilization", str(self.vllm_gpu_memory_utilization),
            "--max-model-len", str(self.max_seq_len),
            "--seed", str(self.seed),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--no-enable-log-requests",
            "--data-parallel-size", str(self.vllm_gpu_count),
            "--enable-sleep-mode",
        ]


        # set environment to isolate GPUs and enable dev mode for collective_rpc API
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.vllm_gpus
        env["VLLM_SERVER_DEV_MODE"] = "1"
        
        # Don't use stdout=PIPE without draining it. vLLM can deadlock if the OS
        # pipe buffer fills (especially with multi-process backends).
        log_dir = self.output_dir or tempfile.gettempdir()
        os.makedirs(log_dir, exist_ok=True)
        self._vllm_log_path = os.path.join(log_dir, f"vllm_api_server_{self._vllm_port}.log")
        self._vllm_log_file = open(self._vllm_log_path, "wb")
        logger.info("vLLM server logs: %s", self._vllm_log_path)

        # start server process
        self._vllm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._vllm_log_file,
            stderr=subprocess.STDOUT,
        )
        
        # register cleanup handler
        import atexit
        atexit.register(self._shutdown_vllm_server)
        
        # wait for server to be ready
        self._vllm_base_url = f"http://localhost:{self._vllm_port}"
        self._wait_for_vllm_server()
        
        logger.info("vLLM server ready at %s on GPU(s) %s", self._vllm_base_url, self.vllm_gpus)
    
    def _read_vllm_log_tail(self, max_bytes: int = 20_000) -> str:
        """Return the last `max_bytes` of the vLLM server log, if available."""
        path = getattr(self, "_vllm_log_path", None)
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - max_bytes, 0), os.SEEK_SET)
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _wait_for_vllm_server(self, timeout: int = 300):
        """Wait for vLLM server to become ready with model loaded."""
        start = time.time()
        health_url = f"{self._vllm_base_url}/health"
        models_url = f"{self._vllm_base_url}/v1/models"
        
        health_ok = False
        
        while time.time() - start < timeout:
            try:
                with httpx.Client(timeout=10) as client:
                    # first check health endpoint
                    if not health_ok:
                        resp = client.get(health_url)
                        if resp.status_code == 200:
                            health_ok = True
                            logger.debug("vLLM health check passed")
                    
                    # then check if model is loaded via /v1/models
                    if health_ok:
                        resp = client.get(models_url)
                        if resp.status_code == 200:
                            models = resp.json().get("data", [])
                            if models:
                                logger.debug("vLLM model loaded: %s", [m.get("id") for m in models])
                                return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            
            # check if process died
            if self._vllm_process.poll() is not None:
                output = self._read_vllm_log_tail()
                raise RuntimeError(
                    "vLLM server process died. Recent output:\n"
                    f"{output}"
                )
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server did not become ready within {timeout}s")

    def _get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract state dict from FSDP2 model, converting DTensors to regular tensors.

        Always saves in fp32 to avoid bf16 precision artifacts in spectral analysis.
        """
        from torch.distributed.tensor import DTensor

        state_dict = {}
        for name, param in self.policy.named_parameters():
            if isinstance(param.data, DTensor):
                # gather full tensor from DTensor (handles sharding)
                state_dict[name] = param.data.full_tensor().detach().clone().float()
            else:
                state_dict[name] = param.data.detach().clone().float()
        return state_dict

    def _sync_weights_to_vllm(self):
        """
        Sync weights from the FSDP2 training model to vLLM server.

        1. Save updated weights to /dev/shm checkpoint
        2. Call /collective_rpc with reload_weights to load new weights
        3. Call /reset_prefix_cache to clear stale cache
        """
        logger.info("syncing weights to vLLM server...")

        # save updated weights to checkpoint directory (in /dev/shm)
        logger.info("saving weights to checkpoint...")
        self._save_weights_to_checkpoint()
        logger.info("weights saved to checkpoint")

        # use sleep/wake API to reload weights (per vLLM docs)
        # https://docs.vllm.ai/en/latest/features/sleep_mode/
        # with data-parallel mode, these operations need to coordinate across all workers
        # so we use a generous timeout (10 minutes) to avoid premature timeouts
        timeout = httpx.Timeout(timeout=600.0, connect=30.0)

        # brief delay to ensure any in-flight requests have completed
        time.sleep(2)

        with httpx.Client(timeout=timeout) as client:
            # pause generation to abort in-flight requests and clear caches
            logger.info("calling /pause...")
            resp = client.post(
                f"{self._vllm_base_url}/pause?wait_for_inflight_requests=false&clear_cache=true"
            )
            resp.raise_for_status()
            logger.info("/pause complete")

            # step 1: sleep level 2 - discard weights and KV cache
            logger.info("calling /sleep?level=2 (this may take a while with data-parallel)...")
            resp = client.post(f"{self._vllm_base_url}/sleep?level=2")
            resp.raise_for_status()
            logger.info("/sleep complete")

            # step 2: wake_up weights only - reallocate weight memory
            logger.info("calling /wake_up?tags=weights...")
            resp = client.post(f"{self._vllm_base_url}/wake_up?tags=weights")
            resp.raise_for_status()
            logger.info("/wake_up weights complete")

            # step 3: reload_weights - load new weights from checkpoint
            logger.info("calling /collective_rpc reload_weights...")
            resp = client.post(
                f"{self._vllm_base_url}/collective_rpc",
                json={"method": "reload_weights"}
            )
            resp.raise_for_status()
            logger.info("/collective_rpc complete")

            # step 4: wake_up kv_cache - reallocate KV cache memory
            logger.info("calling /wake_up?tags=kv_cache...")
            resp = client.post(f"{self._vllm_base_url}/wake_up?tags=kv_cache")
            resp.raise_for_status()
            logger.info("/wake_up kv_cache complete")

            # step 5: reset prefix cache
            logger.info("calling /reset_prefix_cache...")
            resp = client.post(f"{self._vllm_base_url}/reset_prefix_cache")
            resp.raise_for_status()
            logger.info("/reset_prefix_cache complete")

            logger.info("calling /resume...")
            resp = client.post(f"{self._vllm_base_url}/resume")
            resp.raise_for_status()
            logger.info("/resume complete")

        logger.info("weight sync complete")

    @torch.no_grad
    def _generate_rollouts(self) -> list[RejectionSample]:
        """Generate rollouts using vLLM server with async batched requests."""
        self.policy.eval()
        result = asyncio.run(self._generate_rollouts_async())
        logger.debug("_generate_rollouts returning %d samples", len(result))
        return result
    
    async def _generate_rollouts_async(self) -> list[RejectionSample]:
        """Async implementation of rollout generation using independent tasks."""
        collected_samples: list[RejectionSample] = []

        # track statistics
        prompts_tried = 0
        total_completions = 0
        reward_counts = {0.0: 0, 0.1: 0, 1.1: 0}

        pbar = tqdm(
            total=self.samples_to_accept,
            desc="collecting rollouts",
            unit="samples",
        )

        completions_url = f"{self._vllm_base_url}/v1/completions"

        # track pending tasks
        pending_tasks: set[asyncio.Task] = set()
        max_concurrent = self.inference_batch_size  # max concurrent requests

        def get_next_prompt_data() -> dict:
            """Get next prompt from training iterator."""
            nonlocal prompts_tried
            sample = next(self.train_iterator)
            prompt_ids = self.tokenizer.apply_chat_template(
                sample["messages"],
                add_generation_prompt=True,
                return_tensors="pt"
            ).squeeze(0).tolist()
            prompt_text = self.tokenizer.decode(prompt_ids)
            prompts_tried += 1
            return {
                "prompt_ids": prompt_ids,
                "prompt_text": prompt_text,
                "answer": sample["answer"],
                "numbers": sample.get("numbers", None),
            }

        async def generate_for_prompt(prompt_data: dict) -> list[RejectionSample]:
            """Generate completions for a single prompt."""
            request_body = {
                "model": self._vllm_served_model_name,
                "prompt": prompt_data["prompt_text"],
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "n": self.inference_group_size,
                "echo": False,
            }
            if self.top_k > 0:
                request_body["top_k"] = self.top_k

            # each request gets its own client to avoid connection issues
            timeout = httpx.Timeout(timeout=60.0, connect=10.0)
            try:
                async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                    resp = await client.post(completions_url, json=request_body)
                    resp.raise_for_status()
                    result = resp.json()
            except Exception as e:
                logger.warning("vLLM generation failed: %s: %s", type(e).__name__, e)
                return []

            completions = []
            for choice in result.get("choices", []):
                completion_text = choice.get("text", "")
                completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

                # truncate at EOS if present
                try:
                    eos_idx = completion_ids.index(self.tokenizer.eos_token_id) + 1
                    completion_ids = completion_ids[:eos_idx]
                    completion_text = self.tokenizer.decode(completion_ids)
                except ValueError:
                    pass

                reward = self.reward_fn(completion_text, prompt_data["answer"], prompt_data)

                completions.append(
                    RejectionSample(
                        prompt_ids=prompt_data["prompt_ids"],
                        response_ids=completion_ids,
                        response=completion_text,
                        reward=reward,
                    )
                )

            return completions

        def process_completions(completions: list[RejectionSample]) -> RejectionSample | None:
            """Process completions and return accepted sample if any."""
            nonlocal total_completions

            if not completions:
                return None

            total_completions += len(completions)

            # update reward counts
            for comp in completions:
                reward_counts[comp.reward] = reward_counts.get(comp.reward, 0) + 1

            # find best samples
            max_reward = max(c.reward for c in completions)
            best_samples = [c for c in completions if c.reward == max_reward > 0.0]

            if not best_samples:
                return None

            elected_sample = random.choice(best_samples)

            if self._should_accept_sample(max_reward):
                return elected_sample
            return None

        # seed initial batch of tasks
        for _ in range(max_concurrent):
            prompt_data = get_next_prompt_data()
            task = asyncio.create_task(generate_for_prompt(prompt_data))
            pending_tasks.add(task)

        logger.info("started %d concurrent requests to vLLM", len(pending_tasks))

        # process tasks as they complete
        while len(collected_samples) < self.samples_to_accept and pending_tasks:
            # log when we're close to finishing (>90% done) to help debug hangs
            if len(collected_samples) >= self.samples_to_accept * 0.9:
                logger.info(
                    "near completion: %d/%d samples, %d pending tasks",
                    len(collected_samples), self.samples_to_accept, len(pending_tasks)
                )
            
            # wait for at least one task to complete (with timeout to avoid indefinite hangs)
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=120.0,
            )
            
            # log when we're close to finishing
            if len(collected_samples) >= self.samples_to_accept * 0.9:
                logger.info("asyncio.wait returned: %d done, %d pending", len(done), len(pending_tasks))
            
            if not done:
                logger.warning(
                    "asyncio.wait timed out after 120s with %d pending tasks, %d/%d samples collected",
                    len(pending_tasks), len(collected_samples), self.samples_to_accept
                )
                continue

            # process completed tasks
            accepted_this_round = 0
            empty_results = 0
            for task in done:
                try:
                    completions = task.result()
                    if not completions:
                        empty_results += 1
                    accepted = process_completions(completions)
                    if accepted:
                        accepted_this_round += 1
                        collected_samples.append(accepted)
                        pbar.update(1)
                        pbar.set_postfix({
                            "prompts": prompts_tried,
                            "correct": reward_counts.get(1.1, 0),
                            "format_only": reward_counts.get(0.1, 0),
                            "no_reward": reward_counts.get(0.0, 0),
                        })
                except Exception as e:
                    logger.warning("task failed with exception: %s", e)
            
            # warn if we're getting empty results near completion
            if len(collected_samples) >= self.samples_to_accept * 0.9 and empty_results > 0:
                logger.warning(
                    "got %d empty results out of %d completed tasks (accepted %d)",
                    empty_results, len(done), accepted_this_round
                )

            # replenish tasks if we need more samples
            tasks_created = 0
            while len(pending_tasks) < max_concurrent and len(collected_samples) < self.samples_to_accept:
                prompt_data = get_next_prompt_data()
                task = asyncio.create_task(generate_for_prompt(prompt_data))
                pending_tasks.add(task)
                tasks_created += 1
            
            if tasks_created > 0 and len(collected_samples) >= self.samples_to_accept * 0.9:
                logger.info("replenished %d tasks, now %d pending", tasks_created, len(pending_tasks))

        # cancel any remaining tasks
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        pbar.close()

        # log final summary
        logger.info(
            "rollout generation complete: %d samples from %d prompts (%d completions) | "
            "rewards: correct=%.1f%%, format_only=%.1f%%, none=%.1f%%",
            len(collected_samples),
            prompts_tried,
            total_completions,
            100 * reward_counts.get(1.1, 0) / max(total_completions, 1),
            100 * reward_counts.get(0.1, 0) / max(total_completions, 1),
            100 * reward_counts.get(0.0, 0) / max(total_completions, 1),
        )

        # log rollout metrics to wandb
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "rollout/samples_collected": len(collected_samples),
                "rollout/prompts_tried": prompts_tried,
                "rollout/total_completions": total_completions,
                "rollout/correct_rate": reward_counts.get(1.1, 0) / max(total_completions, 1),
                "rollout/format_only_rate": reward_counts.get(0.1, 0) / max(total_completions, 1),
                "rollout/no_reward_rate": reward_counts.get(0.0, 0) / max(total_completions, 1),
            }, step=self.stats_tracker.optim_steps)

        return collected_samples
    
    def _should_accept_sample(self, max_reward: float) -> bool:
        """Probabilistic accept/reject sampling based on reward."""
        if max_reward == 1.1:  # highest possible reward (correct answer + format)
            return True
        elif random.random() * 1.1 <= 0.1:  # accept subpar with probability 0.1/1.1
            return True
        return False

    async def _run_validation_async(self) -> dict:
        """Async validation using vLLM server."""
        if not self.validation_dataset:
            return {}

        correct = 0
        parsable = 0
        total = 0

        completions_url = f"{self._vllm_base_url}/v1/completions"

        # use longer timeout, disable HTTP/2 and keepalive for data parallel
        timeout = httpx.Timeout(timeout=120.0, connect=10.0)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=0, keepalive_expiry=0)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
            # process in batches
            for i in range(0, len(self.validation_dataset), self.inference_batch_size):
                batch = self.validation_dataset[i:i + self.inference_batch_size]

                # prepare prompts
                requests = []
                for j in range(len(batch["messages"])):
                    prompt_ids = self.tokenizer.apply_chat_template(
                        batch["messages"][j],
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).squeeze(0).tolist()
                    prompt_text = self.tokenizer.decode(prompt_ids)
                    req = {
                        "prompt_text": prompt_text,
                        "answer": batch["answer"][j],
                    }
                    if "numbers" in batch:
                        req["numbers"] = batch["numbers"][j]
                    requests.append(req)

                # generate completions (n=1 per prompt)
                async def generate_one(req):
                    body = {
                        "model": self._vllm_served_model_name,
                        "prompt": req["prompt_text"],
                        "max_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "n": 1,
                    }
                    resp = await client.post(completions_url, json=body)
                    resp.raise_for_status()
                    return resp.json(), req

                results = await asyncio.gather(*[generate_one(r) for r in requests])

                for result, req in results:
                    completion = result["choices"][0]["text"]
                    answer = req["answer"]
                    reward = self.reward_fn(completion, answer, req)
                    total += 1
                    if reward >= 0.1:
                        parsable += 1
                    if reward == 1.1:
                        correct += 1

        return {
            "correct_rate": correct / max(total, 1),
            "format_rate": parsable / max(total, 1),
            "total_samples": total,
        }

    def _run_validation(self) -> dict:
        """Sync wrapper for async validation."""
        if not self.validation_dataset:
            return {}

        logger.info("running validation on %d samples...", len(self.validation_dataset))
        self.policy.eval()
        try:
            return asyncio.run(self._run_validation_async())
        finally:
            self.policy.train()

    def _save_policy_checkpoint(self):
        """Save policy model checkpoint to output directory."""
        if not self.output_dir:
            return

        checkpoint_name = f"checkpoint-{self.stats_tracker.train_tokens_seen}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)

        logger.info("saving checkpoint to %s...", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

        # get full state dict from FSDP2 model
        state_dict = self._get_policy_state_dict()

        # save model and tokenizer
        self.policy.save_pretrained(checkpoint_path, state_dict=state_dict)
        self.tokenizer.save_pretrained(checkpoint_path)

        logger.info("checkpoint saved: %s", checkpoint_path)

    @staticmethod
    def _collate_samples(batch: list[RejectionSample], max_tokens_per_gpu: int):
        """
        Return a batch object from a given list of rejection samples. 
        We assume that we're training in padding-free mode.
        """
        processed_samples = []
        for item in batch:
            # input ids + labels
            input_ids = item.prompt_ids + item.response_ids
            labels = [-100] * len(item.prompt_ids) + item.response_ids
            # causal shift << so we predict as n -> n+1 
            input_ids = input_ids[:-1]
            labels = labels[1:]
            position_ids = list(range(len(input_ids)))
            num_loss_tokens = len(item.response_ids)
            processed_samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
                "num_loss_tokens": num_loss_tokens,
                "num_tokens": len(input_ids),
            })
 
        # now collate them into microbatches
        total_loss_tokens = sum(s["num_loss_tokens"] for s in processed_samples)
        total_tokens = sum(s["num_tokens"] for s in processed_samples)
        microbatches = []
        current_microbatch = []
        for sample in processed_samples:
            # make sure not to exceed max tokens per gpu
            if sum(s["num_tokens"] for s in current_microbatch) + sample["num_tokens"] > max_tokens_per_gpu:
                if current_microbatch:
                    microbatches.append(current_microbatch)
                current_microbatch = []
            current_microbatch.append(sample)
        
        # don't forget the last microbatch
        if current_microbatch:
            microbatches.append(current_microbatch)

        # now we collate
        final_microbatches = []
        for mb in microbatches:
            input_ids = torch.cat([s["input_ids"] for s in mb])
            labels = torch.cat([s["labels"] for s in mb])
            position_ids = torch.cat([s["position_ids"]  for s in mb])
            final_microbatches.append({
                "input_ids": input_ids.unsqueeze(0),
                "labels": labels.unsqueeze(0),
                "position_ids": position_ids.unsqueeze(0),
            })
 
        # return the collated batch
        return {
            "microbatches": final_microbatches,
            "num_loss_tokens": total_loss_tokens,
            "total_tokens": total_tokens,
        }


    @torch.no_grad()
    def _optimizer_step(self, num_loss_tokens: int):
        self.stats_tracker.accumulate_tokens(num_loss_tokens)
        self.stats_tracker.increment_optim_step()
        # clip gradnorm
        gradnorm = clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Flush per-parameter update norms (captured inside optimizer.step)
        avg_update_norm = 0.0
        if hasattr(self.optimizer, 'flush_update_norms'):
            avg_update_norm = self.optimizer.flush_update_norms(
                self.stats_tracker.optim_steps, self.stats_tracker.train_tokens_seen
            )
        return gradnorm, avg_update_norm


    def _train_policy(self, samples: list[RejectionSample]):
        """
        Inner training loop
        """
        logger.info("starting training")
        _collate_fn = functools.partial(self._collate_samples, max_tokens_per_gpu=self.max_tokens_per_gpu)
        self.policy.train()

        for epoch in range(self.inner_epochs):
            logger.info("training epoch 1")
            train_loader = torch.utils.data.DataLoader(
                samples,
                batch_size=self.inner_batch_size,
                shuffle=True,
                collate_fn=_collate_fn,
                generator=torch.Generator(device="cuda").manual_seed(self.seed + epoch),
            )
            for batch in train_loader:
                total_loss_tokens = batch["num_loss_tokens"]
                total_loss = 0.0
                total_kl_div = 0.0
                kl_token_count = 0
                for mb in batch["microbatches"]:
                    input_ids = mb["input_ids"].to(self.device)
                    labels = mb["labels"].to(self.device)
                    position_ids = mb["position_ids"].to(self.device)

                    # forward and cross-entropy loss
                    output = self.policy(input_ids=input_ids, position_ids=position_ids)

                    # we dont reduce so that we can properly accumulate the gradient
                    loss = F.cross_entropy(output.logits.squeeze(0), target=labels.squeeze(0), reduction="sum")
                    loss /= total_loss_tokens

                    # cross-entropy and we do our own reduction
                    logger.info(f"obtained loss: {loss.item():,.4f}")

                    # fsdp2 averages by world size so we multiply the loss to get rid of the average
                    loss *= dist.get_world_size()
                    assert dist.get_world_size() == 1  # in our case it should be fine since we expect it to be a world size of 1 though
                    loss.backward()
                    total_loss += loss.detach().item()

                    # Compute KL divergence for logging (policy vs reference)
                    with torch.no_grad():
                        # Get reference model logits
                        ref_output = self.ref_policy(input_ids=input_ids, position_ids=position_ids)

                        # Compute log probabilities for the target tokens
                        # Shift logits to align with labels (next token prediction)
                        policy_logits = output.logits.squeeze(0)  # (seq_len, vocab)
                        ref_logits = ref_output.logits.squeeze(0)  # (seq_len, vocab)

                        # Get log probs for the actual tokens
                        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
                        ref_logprobs = F.log_softmax(ref_logits, dim=-1)

                        # Gather log probs for the label tokens
                        # Create mask for valid labels (labels != -100)
                        valid_mask = labels.squeeze(0) != -100
                        if valid_mask.any():
                            valid_labels = labels.squeeze(0)[valid_mask]
                            valid_policy_logprobs = policy_logprobs[valid_mask]
                            valid_ref_logprobs = ref_logprobs[valid_mask]

                            # Gather the log probs for the specific tokens
                            policy_token_logprobs = valid_policy_logprobs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1)
                            ref_token_logprobs = valid_ref_logprobs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1)

                            # Approximate KL divergence: exp(ref - policy) - (ref - policy) - 1
                            log_diff = (ref_token_logprobs - policy_token_logprobs).clamp(-20, 20)
                            kl_approx = log_diff.exp() - log_diff - 1
                            kl_approx = kl_approx.clamp(min=0, max=100)

                            total_kl_div += kl_approx.sum().item()
                            kl_token_count += valid_mask.sum().item()

                # finally we would backprop here
                gradnorm, avg_update_norm = self._optimizer_step(total_loss_tokens)
                avg_kl_div = total_kl_div / max(kl_token_count, 1)
                logger.info(
                    'loss: %.4f | gradnorm: %.4f | kl_div: %.4f | update_norm: %.6f | tokens: %d/%d',
                    total_loss,
                    gradnorm.item() if hasattr(gradnorm, 'item') else gradnorm,
                    avg_kl_div,
                    avg_update_norm,
                    self.stats_tracker.train_tokens_seen,
                    self.stats_tracker.token_training_budget
                )

                # log to wandb
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": total_loss,
                        "train/grad_norm": gradnorm.item() if hasattr(gradnorm, 'item') else gradnorm,
                        "train/kl_divergence": avg_kl_div,
                        "train/avg_update_frobenius": avg_update_norm,
                        "train/optim_step": self.stats_tracker.optim_steps,
                        "train/tokens_trained": self.stats_tracker.train_tokens_seen,
                    }, step=self.stats_tracker.optim_steps)



        


    def train(self):
        """
        Rejection sampling training loop with vLLM for fast inference.
        
        vLLM runs as a separate server on its own GPU(s):
        1. Generate rollouts with vLLM server (async batched)
        2. Train policy model on training GPU
        3. Sync updated weights to vLLM (restarts server with new weights)
        """
        logger.info('starting training on cuda:%d (vLLM server on GPU(s) %s)', self.device.index, self.vllm_gpus)
        self.stats_tracker.reset()
        self.policy.eval()

        while not self.stats_tracker.completed_training():
            # generate rollouts with vLLM server
            logger.info("generating rollouts...")
            rollouts = self._generate_rollouts()
            logger.info("rollouts generated: %d samples", len(rollouts))

            # train the policy model
            logger.info("starting policy training...")
            self._train_policy(rollouts)
            logger.info("policy training complete")

            # check if checkpoint is due
            if self.stats_tracker.should_save() and self.stats_tracker.checkpoint_frequency > 0:
                # run validation before saving checkpoint
                if self.validation_dataset is not None:
                    val_metrics = self._run_validation()
                    logger.info(
                        'validation: correct=%.1f%%, format=%.1f%% (%d samples)',
                        val_metrics.get('correct_rate', 0) * 100,
                        val_metrics.get('format_rate', 0) * 100,
                        val_metrics.get('total_samples', 0)
                    )

                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "val/correct_rate": val_metrics.get('correct_rate', 0),
                            "val/format_rate": val_metrics.get('format_rate', 0),
                            "val/total_samples": val_metrics.get('total_samples', 0),
                        }, step=self.stats_tracker.optim_steps)

                # save checkpoint
                self._save_policy_checkpoint()
                self.stats_tracker.mark_checkpointed()

            # sync updated weights to vLLM (via sleep/wake for fast reload)
            logger.info("syncing weights...")
            self._sync_weights_to_vllm()
            logger.info("weight sync complete")
        
        logger.info('completed training')

        # finish wandb run
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            logger.info("Wandb run finished")

        # cleanup vLLM server
        self._shutdown_vllm_server()
    
    def _shutdown_vllm_server(self):
        """Gracefully shutdown the vLLM server and cleanup checkpoint directory."""
        # shutdown server process
        if hasattr(self, '_vllm_process') and self._vllm_process.poll() is None:
            logger.info("shutting down vLLM server...")
            
            # send SIGTERM for graceful shutdown
            self._vllm_process.terminate()
            try:
                self._vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not shutdown gracefully, killing...")
                self._vllm_process.kill()
                self._vllm_process.wait(timeout=5)

        vllm_log_file = getattr(self, "_vllm_log_file", None)
        if vllm_log_file is not None:
            try:
                vllm_log_file.close()
            except OSError:
                pass
            self._vllm_log_file = None
        
        # cleanup /dev/shm checkpoint directory
        if hasattr(self, '_vllm_checkpoint_dir') and os.path.exists(self._vllm_checkpoint_dir):
            import shutil
            logger.info("cleaning up vLLM checkpoint directory: %s", self._vllm_checkpoint_dir)
            shutil.rmtree(self._vllm_checkpoint_dir, ignore_errors=True)
            


                    






            


    
    
    