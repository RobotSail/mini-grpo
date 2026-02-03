import torch
import datasets
import random
import re
from type_defs import (
    Problem,
    SamplingParams,
    Message,
    TokenSample,
    RolloutResult,
    Sample,
    Sample,
    TrainingComponents,
)
from torch.utils.data import DataLoader, Dataset

from instructlab.training.data_process import unmask_sample, configure_tokenizer, process_samples
from instructlab.training.type_definitions import ProcessedMessagesData
from transformers import PreTrainedTokenizer
from IPython import embed


def random_problems(seed: int = 42, num_problems: int = 20, min_num: int = 1, max_num: int = 100) -> list[Problem]:
    random.seed(seed)
    problems: list[Problem] = []
    for _ in range(num_problems):
        a, b = random.randint(min_num, max_num), random.randint(min_num, max_num)
        operation = random.choice(["add", "subtract"])
        if operation == "add":
            add_prompts = [
                f"What is the sum of {a} and {b}?",
                f"What is {a} plus {b}?",
                f"Add {a} and {b}.",
                f"Calculate {a} + {b}.",
                f"What do you get when you add {a} to {b}?",
                f"If you have {a} and add {b}, what is the total?",
            ]
            problem = random.choice(add_prompts)
            answer = a + b
        else:  # subtract
            subtract_prompts = [
                f"What is the difference of {a} and {b}?",
                f"What is {a} minus {b}?",
                f"Subtract {b} from {a}.",
                f"Calculate {a} - {b}.",
                f"What do you get when you subtract {b} from {a}?",
                f"If you have {a} and take away {b}, what is left?",
            ]
            problem = random.choice(subtract_prompts)
            answer = a - b
        problems.append(Problem(problem=problem, answer=answer, operation=operation))
    return problems


def generate_dataset(
    system_msg: str,
    num_problems: int = 20,
    min_num: int = -100,
    max_num: int = 100,
    seed: int = 42,
    # ) -> datasets.Dataset:
) -> datasets.Dataset:
    problems = random_problems(seed=seed, num_problems=num_problems, min_num=min_num, max_num=max_num)

    # Convert list of Problem objects to dataset
    problems_dict = [problem.model_dump() for problem in problems]
    dataset = datasets.Dataset.from_list(problems_dict)
    dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": x["problem"]},
            ]
        }
    )
    return dataset


class JsonlDataset(torch.utils.data.Dataset):
    """Dataset class for loading pre-tokenized input IDs from JSONL files."""

    def __init__(self, dataset: datasets.Dataset = None, data_path: str = None):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSONL file containing input_ids
        """
        if dataset:
            self.dataset = dataset
        elif data_path:
            self.dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Returns:
            dict: Dictionary containing 'input_ids' and other fields from the JSONL
        """
        item = self.dataset[idx]
        to_return = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "logprob_ids": torch.tensor(item["logprob_ids"], dtype=torch.long),
            "logprobs": torch.tensor(item["logprobs"], dtype=torch.float32),
            "grpo_mask": torch.tensor(item["grpo_mask"], dtype=torch.bool),
            # debug
            "full_input_ids": torch.tensor(item["full_input_ids"], dtype=torch.long),
            "full_logprob_ids": torch.tensor(item["full_logprob_ids"], dtype=torch.long),
            "advantage": item["advantage"],
            "prompt_offset": item["prefix_len"],
            "num_logprobs": len(item["logprobs"]),
            #            "input_ids": input_ids,
            # "logprob_ids": logprob_seq,
            # "grpo_mask": grpo_mask,
            # # adds all of these for debugging purposes
            # "full_input_ids": full_input_seq,
            # "full_logprob_ids": full_logprob_seq,
            # "logprobs": logprobs,
        }
        return to_return


def collate_fn(batch: list[dict], pad_token_id: int):
    """
    Legacy padded collation - kept for reference.
    batch is a list of dicts containing:
    - input_ids: tensor contining the input ids
    - labels: tensor contining the input ids
    """
    max_len = max(batch, key=lambda x: x["input_ids"].numel())["input_ids"].numel()
    # Pad all sequences to max_len
    input_ids_padded = []
    attention_mask_padded = []
    num_tokens_in_batch = 0
    advantages = []
    logprob_ids_padded = []
    logprobs_padded = []
    logprobs_in_batch = []
    batch_grpo_mask = []

    for item in batch:
        seq_len = item["input_ids"].numel()
        num_tokens_in_batch += seq_len

        # Pad input_ids (typically with 0 or tokenizer.pad_token_id)
        full_input_seq = torch.full((max_len,), fill_value=pad_token_id, dtype=torch.long)
        full_attn_mask = torch.zeros_like(full_input_seq, dtype=torch.long)
        # full_attn_mask =

        # populate padded inputs with values from dataset
        idxs = torch.arange(0, seq_len)
        full_input_seq[idxs] = item["input_ids"]
        full_attn_mask = (full_input_seq != pad_token_id).float()  # should compute attention here

        # how far the logprobs (completed sequence) is from the beginning
        prompt_offset = item["prompt_offset"]
        completion_length = item["num_logprobs"]

        # this should be constructed such that the default value will
        # have no effect on the compute graph
        full_logprob_ids = torch.full((max_len,), fill_value=pad_token_id, dtype=torch.long)
        full_logprob_ids[idxs] = item["logprob_ids"]

        # create the logprobs
        full_logprobs = torch.ones_like(full_input_seq, dtype=torch.float32)
        try:
            logprob_offset = prompt_offset - 1
            full_logprobs[logprob_offset : logprob_offset + completion_length] = item["logprobs"]
        except Exception as e:
            print(e)
            embed()

        # first ensure it's the same size
        assert full_logprobs[logprob_offset : logprob_offset + completion_length].numel() == item["logprobs"].numel()
        # # full_logprobs[prompt_offset : prompt_offset + completion_length] = item["logprobs"]

        # # do the same for the logit ids
        # full_logprob_ids = torch.full_like(full_input_seq, fill_value=pad_token_id)
        # full_logprob_ids[:] = item["logprob_ids"]
        grpo_mask = full_logprob_ids != pad_token_id
        batch_grpo_mask += [grpo_mask]

        # now make sure to append all of these
        logprob_ids_padded += [full_logprob_ids]
        logprobs_padded += [full_logprobs]

        # count the number of tokens that we consider ourselves to actually be backproping on
        logprobs_in_batch.append(completion_length)

        # update the batch items
        input_ids_padded += [full_input_seq]
        attention_mask_padded += [full_attn_mask]
        advantages.append(item["advantage"])

    final_item = {
        "input_ids": torch.stack(input_ids_padded).detach(),
        "attention_mask": torch.stack(attention_mask_padded).detach(),
        "num_tokens": num_tokens_in_batch,
        "num_sequences": len(batch),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "logprobs": torch.stack(logprobs_padded).detach(),
        "logprob_ids": torch.stack(logprob_ids_padded).detach(),
        "rollout_lens": torch.tensor(logprobs_in_batch, dtype=torch.long).detach(),
        "grpo_mask": torch.stack(batch_grpo_mask).detach(),
    }
    return final_item


def collate_fn_packed(batch: list[dict], pad_token_id: int):
    """
    Padding-free collation for Flash Attention 2.

    Instead of padding sequences to max_len, we concatenate all sequences
    into a single 1D tensor and track boundaries using position_ids and
    sequence indices.

    This is much more memory efficient for variable-length sequences.
    """
    # Collect all sequences and metadata
    all_input_ids = []
    all_logprob_ids = []
    all_logprobs = []
    all_grpo_mask = []
    all_position_ids = []

    # Per-sequence metadata
    seq_lengths = []
    rollout_lens = []
    advantages = []
    seq_start_indices = []  # Start index of each sequence in the packed tensor

    current_idx = 0

    for item in batch:
        seq_len = item["input_ids"].numel()
        prompt_offset = item["prompt_offset"]
        completion_length = item["num_logprobs"]

        # Track sequence boundary
        seq_start_indices.append(current_idx)
        seq_lengths.append(seq_len)

        # Append input_ids
        all_input_ids.append(item["input_ids"])

        # Create position_ids for this sequence (0, 1, 2, ..., seq_len-1)
        all_position_ids.append(torch.arange(seq_len, dtype=torch.long))

        # Create logprob_ids (shifted version for loss computation)
        all_logprob_ids.append(item["logprob_ids"])

        # Create logprobs aligned with the sequence
        # Need to create a tensor of the same length as input_ids
        seq_logprobs = torch.ones(seq_len, dtype=torch.float32)
        logprob_offset = prompt_offset - 1
        if logprob_offset >= 0 and logprob_offset + completion_length <= seq_len:
            seq_logprobs[logprob_offset : logprob_offset + completion_length] = item["logprobs"]
        all_logprobs.append(seq_logprobs)

        # Create GRPO mask (True where we should compute loss)
        seq_grpo_mask = item["logprob_ids"] != pad_token_id
        all_grpo_mask.append(seq_grpo_mask)

        # Track rollout length and advantage
        rollout_lens.append(completion_length)
        advantages.append(item["advantage"])

        current_idx += seq_len

    # Concatenate everything into packed tensors
    packed_input_ids = torch.cat(all_input_ids, dim=0)
    packed_position_ids = torch.cat(all_position_ids, dim=0)
    packed_logprob_ids = torch.cat(all_logprob_ids, dim=0)
    packed_logprobs = torch.cat(all_logprobs, dim=0)
    packed_grpo_mask = torch.cat(all_grpo_mask, dim=0)

    # Create cumulative sequence lengths for Flash Attention
    # cu_seqlens format: [0, len1, len1+len2, len1+len2+len3, ...]
    cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lengths, dtype=torch.int32), dim=0)

    # Create sequence index for each token (to map back losses to sequences)
    seq_indices = torch.zeros(packed_input_ids.numel(), dtype=torch.long)
    for i, (start, length) in enumerate(zip(seq_start_indices, seq_lengths)):
        seq_indices[start : start + length] = i

    final_item = {
        # Packed tensors (1D)
        "input_ids": packed_input_ids.detach(),
        "position_ids": packed_position_ids.detach(),
        "logprob_ids": packed_logprob_ids.detach(),
        "logprobs": packed_logprobs.detach(),
        "grpo_mask": packed_grpo_mask.detach(),
        # Sequence metadata
        "cu_seqlens": cu_seqlens.detach(),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long).detach(),
        "seq_indices": seq_indices.detach(),
        "max_seqlen": max(seq_lengths),
        # Per-sequence values
        "advantages": torch.tensor(advantages, dtype=torch.float32).detach(),
        "rollout_lens": torch.tensor(rollout_lens, dtype=torch.long).detach(),
        # Metadata
        "num_tokens": packed_input_ids.numel(),
        "num_sequences": len(batch),
        "is_packed": True,
    }
    return final_item


def split_batch_into_microbatches(batch: dict, max_tokens: int):
    """
    Split a packed batch into microbatches based on token count.

    Yields microbatches that each contain <= max_tokens tokens.
    Each microbatch includes metadata about the total batch for gradient scaling.

    Args:
        batch: A packed batch from collate_fn_packed
        max_tokens: Maximum tokens per microbatch

    Yields:
        Microbatch dicts with additional 'total_tokens_in_batch' and 'num_microbatches' fields
    """
    if not batch.get("is_packed", False):
        # For padded batches, just yield the whole batch
        batch["total_tokens_in_batch"] = batch["num_tokens"]
        batch["num_microbatches"] = 1
        yield batch
        return

    total_tokens = batch["num_tokens"]
    num_sequences = batch["num_sequences"]
    seq_lengths = batch["seq_lengths"].tolist()

    # If batch fits in one microbatch, yield as-is
    if total_tokens <= max_tokens:
        batch["total_tokens_in_batch"] = total_tokens
        batch["num_microbatches"] = 1
        yield batch
        return

    # Split into microbatches
    microbatches = []
    current_seqs = []
    current_tokens = 0

    for seq_idx in range(num_sequences):
        seq_len = seq_lengths[seq_idx]

        # If adding this sequence would exceed limit, start new microbatch
        if current_tokens + seq_len > max_tokens and current_seqs:
            microbatches.append(current_seqs)
            current_seqs = []
            current_tokens = 0

        current_seqs.append(seq_idx)
        current_tokens += seq_len

    # Don't forget the last microbatch
    if current_seqs:
        microbatches.append(current_seqs)

    num_microbatches = len(microbatches)

    # Now yield each microbatch
    for seq_indices_list in microbatches:
        # Extract the sequences for this microbatch
        start_indices = []
        end_indices = []
        cu_seqlens_list = [0]
        cumsum = 0

        micro_seq_lengths = []
        micro_advantages = []
        micro_rollout_lens = []

        for seq_idx in seq_indices_list:
            seq_len = seq_lengths[seq_idx]
            # Find token range for this sequence in the packed tensor
            start = batch["cu_seqlens"][seq_idx].item()
            end = batch["cu_seqlens"][seq_idx + 1].item()
            start_indices.append(start)
            end_indices.append(end)

            cumsum += seq_len
            cu_seqlens_list.append(cumsum)
            micro_seq_lengths.append(seq_len)
            micro_advantages.append(batch["advantages"][seq_idx].item())
            micro_rollout_lens.append(batch["rollout_lens"][seq_idx].item())

        # Concatenate token-level data for selected sequences
        token_slices = [slice(s, e) for s, e in zip(start_indices, end_indices)]

        micro_input_ids = torch.cat([batch["input_ids"][sl] for sl in token_slices])
        micro_position_ids = torch.cat([batch["position_ids"][sl] for sl in token_slices])
        micro_logprob_ids = torch.cat([batch["logprob_ids"][sl] for sl in token_slices])
        micro_logprobs = torch.cat([batch["logprobs"][sl] for sl in token_slices])
        micro_grpo_mask = torch.cat([batch["grpo_mask"][sl] for sl in token_slices])

        # Rebuild seq_indices for the microbatch
        micro_seq_indices = torch.zeros(micro_input_ids.numel(), dtype=torch.long)
        offset = 0
        for i, seq_len in enumerate(micro_seq_lengths):
            micro_seq_indices[offset : offset + seq_len] = i
            offset += seq_len

        microbatch = {
            "input_ids": micro_input_ids.detach(),
            "position_ids": micro_position_ids.detach(),
            "logprob_ids": micro_logprob_ids.detach(),
            "logprobs": micro_logprobs.detach(),
            "grpo_mask": micro_grpo_mask.detach(),
            "cu_seqlens": torch.tensor(cu_seqlens_list, dtype=torch.int32).detach(),
            "seq_lengths": torch.tensor(micro_seq_lengths, dtype=torch.long).detach(),
            "seq_indices": micro_seq_indices.detach(),
            "max_seqlen": max(micro_seq_lengths),
            "advantages": torch.tensor(micro_advantages, dtype=torch.float32).detach(),
            "rollout_lens": torch.tensor(micro_rollout_lens, dtype=torch.long).detach(),
            "num_tokens": micro_input_ids.numel(),
            "num_sequences": len(seq_indices_list),
            "is_packed": True,
            # Metadata for gradient scaling
            "total_tokens_in_batch": total_tokens,
            "num_microbatches": num_microbatches,
        }
        yield microbatch


def dataset_from_groups(groups: list[Sample], tokenizer: PreTrainedTokenizer):
    """
    Creates a processed dataset in the format needed for training GRPO
    """
    processed_samples = []
    for group in groups:
        prefix_input_ids = group.input_ids
        for rollout in group.rollouts:
            logprob_ids = [tok.token for tok in rollout.logprobs]
            # clone input ids
            full_input_seq = prefix_input_ids[:] + logprob_ids[:]
            full_logprob_seq = [tokenizer.pad_token_id] * len(prefix_input_ids) + logprob_ids[:]  # still needs shifting

            # now we have to create the shifted & aligned samples
            try:
                last_eos_tok_idx = logprob_ids[::-1].index(tokenizer.eos_token_id)
            except ValueError:
                # it doesnt have one, only shift <<
                input_ids = full_input_seq[:-1]
                logprob_seq = full_logprob_seq[1:]

            else:
                # set the indices
                input_ids_offset_idx = -(last_eos_tok_idx + 1)
                logprob_ids_offset_idx = -(last_eos_tok_idx)

                # we have to chop the input sequence
                input_ids = full_input_seq[:input_ids_offset_idx]
                if len(input_ids) == 0:
                    raise ValueError("trimming eos token resulted in empty input ids sequence")

                logprob_seq = full_logprob_seq[1:logprob_ids_offset_idx]
                if logprob_ids_offset_idx == 0:
                    logprob_seq = full_logprob_seq[1:]
                else:
                    logprob_seq = full_logprob_seq[1:logprob_ids_offset_idx]

                if len(logprob_seq) == 0:
                    raise ValueError("trimming eos token resulted in empty logprob ids sequence")

            # remaining items
            grpo_mask = [lpi == tokenizer.pad_token_id for lpi in logprob_seq]
            logprobs = [lp.logprob for lp in rollout.logprobs]

            # these must be equal
            assert len(input_ids) == len(logprob_seq)

            sample = {
                "prefix_len": len(prefix_input_ids),
                "input_ids": input_ids,
                "logprob_ids": logprob_seq,
                "grpo_mask": grpo_mask,
                # adds all of these for debugging purposes
                "full_input_ids": full_input_seq,
                "full_logprob_ids": full_logprob_seq,
                "logprobs": logprobs,
                "advantage": rollout.advantage,
            }
            # sample.update(rollout.to_dataset_format())
            processed_samples.append(sample)

    try:
        ds = datasets.Dataset.from_list(processed_samples, split="train")
    except Exception as e:
        print(e)
        embed()

    return ds


def create_grpo_data_loader(
    dataset: datasets.Dataset,
    comps: TrainingComponents,
    seed: int,
    use_packed: bool = False,
):
    """
    Create a DataLoader for GRPO training.

    Args:
        dataset: The dataset to load from
        comps: Training components containing tokenizer and hyperparameters
        use_packed: If True, use padding-free packed collation (requires Flash Attention 2)
    """
    from functools import partial

    if use_packed:
        _collate_fn = partial(collate_fn_packed, pad_token_id=comps.tokenizer.pad_token_id)
    else:
        _collate_fn = partial(collate_fn, pad_token_id=comps.tokenizer.pad_token_id)

    # creates a generator for the seed
    generator = torch.Generator().manual_seed(seed)
    ds = JsonlDataset(dataset=dataset)
    train_loader = DataLoader(
        dataset=ds,
        collate_fn=_collate_fn,
        batch_size=comps.hyperparams.inner_batch_size,
        shuffle=True,
        generator=generator,
    )
    return train_loader


def load_gsm8k(
    system_msg: str,
    eval_split: float = 0.0,
    seed: int = 67,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    Load the GSM8K dataset and convert it to the format expected by this repo.

    Args:
        system_msg: System message to use for the chat format
        eval_split: Fraction of data to use for evaluation (0.0 = no eval split)
        seed: Random seed for train/test split

    Returns:
        Tuple of (train_dataset, eval_dataset) where eval_dataset may be None
    """
    ds = datasets.load_dataset("openai/gsm8k", name="main", split="train")

    # Rename question -> problem to match our format
    ds = ds.rename_columns({"question": "problem"})

    def _get_answers(sample):
        """
        GSM8K stores answers in two formats:
        1. <<calculation>> format: e.g., "<<12*2=24>>" - we extract the result after '='
        2. #### format: e.g., "#### 42" - we extract the number after ####
        """
        answers = re.findall(r"<<(.+?)>>", sample["answer"])
        alt_matches = re.findall(r"#### (.+)", sample["answer"])

        if answers:
            # Format: '12*2=24', '8/2=4', etc - take the result after '='
            answer = answers[-1].split("=")[-1]
        elif alt_matches:
            answer = alt_matches[-1]
        else:
            raise ValueError(f"Failed to find answer in: {sample['answer']}")

        # Parse the answer, removing commas for large numbers
        return {"answer": float(answer.replace(",", ""))}

    ds = ds.map(_get_answers)

    # Add the operation field (required by Problem model) and messages
    def _add_fields(sample):
        return {
            "operation": "gsm8k",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": sample["problem"]},
            ],
        }

    ds = ds.map(_add_fields)

    # Split into train/eval if requested
    train_dataset = ds
    eval_dataset = None

    if eval_split > 0:
        dataset_dict = train_dataset.train_test_split(test_size=eval_split, seed=seed)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

    return train_dataset, eval_dataset
