"""Evaluate a countdown checkpoint on a test set using vLLM with data-parallel."""
import argparse
import json
import sys
sys.path.insert(0, "/mnt/4TB/workspace/oleg/mini-grpo")

from countdown_utils import countdown_reward_fn
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for data-parallel inference")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        test_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_samples)} test samples")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    llm = LLM(
        model=args.checkpoint,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        data_parallel_size=args.num_gpus,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Build prompts
    prompts = []
    for sample in test_samples:
        prompt_ids = tokenizer.apply_chat_template(
            sample["messages"],
            add_generation_prompt=True,
        )
        prompts.append(tokenizer.decode(prompt_ids))

    # Generate
    print(f"Generating {len(prompts)} responses (data_parallel={args.num_gpus})...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    parsable = 0
    total = len(test_samples)

    for sample, output in zip(test_samples, outputs):
        text = output.outputs[0].text
        result = countdown_reward_fn(text, sample["answer"], {
            "numbers": sample.get("numbers"),
        })
        if result["is_parsable"]:
            parsable += 1
        if result["is_correct"]:
            correct += 1

    results = {
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "total": total,
        "correct": correct,
        "parsable": parsable,
        "accuracy": correct / total,
        "parsable_rate": parsable / total,
    }

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test samples: {total}")
    print(f"  Parsable: {parsable}/{total} ({parsable/total*100:.1f}%)")
    print(f"  Correct:  {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
