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
import time
import httpx
import signal

import logging

class RejectionSample(pd.BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    response: str
    reward: float

# Create logger for trainer module
logger = logging.getLogger(__name__)

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
    
    def reset(self):
        self._train_tokens_seen = 0
        self._last_checkpoint_save = 0
        self._inference_iteration = 0
        
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
        wandb_project: str,
        wandb_run_name: str,
        seed: int,
        optimizer_type: str,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        # device configuration
        gpu: int = 0,
        vllm_gpu: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
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


        # check basic validation
        if self.output_dir and not self._valid_save_dir(self.output_dir):
            raise ValueError(f'invalid output directory: cannot write to {output_dir}')

        # then we load the training dataset
        self.training_dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        self._train_iterator = None
    
        logger.info('loaded %d unique samples for training from %s', len(self.training_dataset), data_path)

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
            "optimizer": optimizer,
            "max_tokens_per_microbatch": max_tokens_per_gpu,
            "save_every_n_tokens": save_every_n_tokens,
            "beta1": beta1,
            "beta2": beta2,
            "wd": weight_decay,
        }
        utils.initialize_wandb(wandb_project, wandb_run_name, run_config)

        # initialize vllm for fast inference on a separate GPU
        self.vllm_gpu = vllm_gpu
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self._start_vllm_server()
        

    @property
    def train_iterator(self):
        if not self._train_iterator:
            self._train_iterator = iter(InfiniteDatasetIterator(self.training_dataset, seed=self.seed))
        
        return self._train_iterator

    def _start_vllm_server(self, model_path: str | None = None):
        """
        Start vLLM as an OpenAI-compatible server on a dedicated GPU.
        
        Args:
            model_path: Path to model weights. If None, uses self.model_name.
        """
        model_to_load = model_path or self.model_name
        logger.info("starting vLLM server on GPU %d with model %s...", self.vllm_gpu, model_to_load)
        
        # find an available port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            self._vllm_port = s.getsockname()[1]
        
        # build server command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_to_load,
            "--port", str(self._vllm_port),
            "--gpu-memory-utilization", str(self.vllm_gpu_memory_utilization),
            "--max-model-len", str(self.max_seq_len),
            "--seed", str(self.seed),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--disable-log-requests",
        ]
        
        # set environment to isolate GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.vllm_gpu)
        
        # start server process
        self._vllm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        # register cleanup handler
        import atexit
        atexit.register(self._shutdown_vllm_server)
        
        # wait for server to be ready
        self._vllm_base_url = f"http://localhost:{self._vllm_port}"
        self._wait_for_vllm_server()
        
        logger.info("vLLM server ready at %s on GPU %d", self._vllm_base_url, self.vllm_gpu)
    
    def _wait_for_vllm_server(self, timeout: int = 300):
        """Wait for vLLM server to become ready."""
        start = time.time()
        health_url = f"{self._vllm_base_url}/health"
        
        while time.time() - start < timeout:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            
            # check if process died
            if self._vllm_process.poll() is not None:
                # read output for debugging
                output = self._vllm_process.stdout.read().decode() if self._vllm_process.stdout else ""
                raise RuntimeError(f"vLLM server process died. Output:\n{output[-2000:]}")
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server did not become ready within {timeout}s")

    def _get_policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract state dict from FSDP2 model, converting DTensors to regular tensors."""
        from torch.distributed.tensor import DTensor
        
        state_dict = {}
        for name, param in self.policy.named_parameters():
            if isinstance(param.data, DTensor):
                # gather full tensor from DTensor (handles sharding)
                state_dict[name] = param.data.full_tensor().detach().clone()
            else:
                state_dict[name] = param.data.detach().clone()
        return state_dict

    def _sync_weights_to_vllm(self):
        """
        Sync weights from the FSDP2 training model to vLLM server.
        
        Since vLLM server doesn't support online weight updates, we:
        1. Save current weights to a checkpoint
        2. Shutdown the old server
        3. Start a new server with the updated weights
        """
        logger.info("syncing weights to vLLM server...")
        
        # create checkpoint directory for vLLM
        vllm_checkpoint_dir = os.path.join(self.output_dir or "/tmp", "vllm_checkpoint")
        os.makedirs(vllm_checkpoint_dir, exist_ok=True)
        
        # get state dict from FSDP2 model
        state_dict = self._get_policy_state_dict()
        
        # save in HuggingFace format so vLLM can load it
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name)
        config.save_pretrained(vllm_checkpoint_dir)
        self.tokenizer.save_pretrained(vllm_checkpoint_dir)
        
        # save model weights
        torch.save(state_dict, os.path.join(vllm_checkpoint_dir, "pytorch_model.bin"))
        
        # also save as safetensors for faster loading
        try:
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(vllm_checkpoint_dir, "model.safetensors"))
        except ImportError:
            pass  # safetensors not available, pytorch_model.bin will be used
        
        # restart server with new weights
        self._shutdown_vllm_server()
        self._start_vllm_server(model_path=vllm_checkpoint_dir)
        
        logger.info("weight sync complete (server restarted with updated weights)")
    
    @torch.no_grad
    def _generate_rollouts(self) -> list[RejectionSample]:
        """Generate rollouts using vLLM server for fast inference."""
        self.policy.eval()
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
        
        # create HTTP client for vLLM server
        completions_url = f"{self._vllm_base_url}/v1/completions"
        
        with httpx.Client(timeout=120) as client:
            while len(collected_samples) < self.samples_to_accept:
                sample = next(self.train_iterator)
                prompts_tried += 1
                
                # prepare prompt for vLLM
                prompt_ids = self.tokenizer.apply_chat_template(
                    sample["messages"], 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).squeeze(0).tolist()
                prompt_text = self.tokenizer.decode(prompt_ids)
                
                # call vLLM server API
                request_body = {
                    "model": self.model_name,
                    "prompt": prompt_text,
                    "max_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": self.inference_group_size,
                    "echo": False,
                }
                if self.top_k > 0:
                    request_body["top_k"] = self.top_k
                
                try:
                    resp = client.post(completions_url, json=request_body)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    logger.warning("vLLM generation failed: %s", e)
                    continue
                
                completions: list[RejectionSample] = []
                max_reward = 0.0
                
                for choice in result.get("choices", []):
                    completion_text = choice.get("text", "")
                    completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
                    total_completions += 1
                    
                    # truncate at EOS if present
                    try:
                        eos_idx = completion_ids.index(self.tokenizer.eos_token_id) + 1
                        completion_ids = completion_ids[:eos_idx]
                        completion_text = self.tokenizer.decode(completion_ids)
                    except ValueError:
                        pass
                    
                    reward = reward_response(completion_text, sample["answer"])
                    reward_counts[reward] = reward_counts.get(reward, 0) + 1
                    max_reward = max(reward, max_reward)
                    
                    completions.append(
                        RejectionSample(
                            prompt_ids=prompt_ids,
                            response_ids=completion_ids,
                            response=completion_text,
                            reward=reward,
                        )
                    )

                best_samples = [c for c in completions if c.reward == max_reward > 0.0]
                if not best_samples:
                    continue

                elected_sample = random.choice(best_samples)
                accept = self._should_accept_sample(max_reward)

                if accept:
                    collected_samples.append(elected_sample)
                    pbar.update(1)
                    # update postfix with reward stats
                    pbar.set_postfix({
                        "prompts": prompts_tried,
                        "correct": reward_counts.get(1.1, 0),
                        "format_only": reward_counts.get(0.1, 0),
                        "no_reward": reward_counts.get(0.0, 0),
                    })
        
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

        return collected_samples
    
    def _should_accept_sample(self, max_reward: float) -> bool:
        """Probabilistic accept/reject sampling based on reward."""
        if max_reward == 1.1:  # highest possible reward (correct answer + format)
            return True
        elif random.random() * 1.1 <= 0.1:  # accept subpar with probability 0.1/1.1
            return True
        return False
    
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
 
        # now collate them into microbathes
        total_loss_tokens = sum(s["num_loss_tokens"] for s in processed_samples)
        total_tokens = sum(s["num_tokens"] for s in processed_samples)
        microbatches = []
        current_microbatch = []
        for sample in processed_samples:
            # make sure not to exceed max tokens per gpu
            if sum(s["num_tokens"] for s in current_microbatch) + sample["num_tokens"] > max_tokens_per_gpu:  # collate so we don't OOM
                microbatches.append(current_microbatch)
                current_microbatch = []
            current_microbatch.append(sample)
 
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
        # clip gradnorm
        gradnorm = clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return gradnorm


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
                generator=torch.Generator(self.device).manual_seed(self.seed + epoch),
            )
            for batch in train_loader:
                total_loss_tokens = batch["num_loss_tokens"]
                total_loss = 0.0
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

                    # TODO: add KL divergence term here

                # finally we would backprop here
                gradnorm = self._optimizer_step(total_loss_tokens)
                logger.info('loss: %s, gradnorm: %s', total_loss, gradnorm)



        


    def train(self):
        """
        Rejection sampling training loop with vLLM for fast inference.
        
        vLLM runs as a separate server on its own GPU:
        1. Generate rollouts with vLLM server
        2. Train policy model on training GPU
        3. Sync updated weights to vLLM (restarts server with new weights)
        """
        logger.info('starting training on cuda:%d (vLLM server on cuda:%d)', self.device.index, self.vllm_gpu)
        self.stats_tracker.reset()
        self.policy.eval()

        while not self.stats_tracker.completed_training():
            # generate rollouts with vLLM server
            rollouts = self._generate_rollouts()

            # train the policy model
            self._train_policy(rollouts)
            
            # sync updated weights to vLLM (restarts server)
            self._sync_weights_to_vllm()
        
        logger.info('completed training')
        
        # cleanup vLLM server
        self._shutdown_vllm_server()
    
    def _shutdown_vllm_server(self):
        """Gracefully shutdown the vLLM server."""
        if not hasattr(self, '_vllm_process'):
            return
        if self._vllm_process.poll() is not None:
            return  # already terminated
            
        logger.info("shutting down vLLM server...")
        
        # send SIGTERM for graceful shutdown
        self._vllm_process.terminate()
        try:
            self._vllm_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not shutdown gracefully, killing...")
            self._vllm_process.kill()
            self._vllm_process.wait(timeout=5)
            


                    






            


    
    
    