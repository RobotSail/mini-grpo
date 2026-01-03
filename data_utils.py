import torch
import datasets
import random
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


def create_grpo_data_loader(dataset: datasets.Dataset, comps: TrainingComponents):
    from functools import partial

    _collate_fn = partial(collate_fn, pad_token_id=comps.tokenizer.pad_token_id)

    ds = JsonlDataset(dataset=dataset)
    train_loader = DataLoader(
        dataset=ds,
        collate_fn=_collate_fn,
        batch_size=comps.hyperparams.inner_batch_size,
        shuffle=True,
    )
    return train_loader
