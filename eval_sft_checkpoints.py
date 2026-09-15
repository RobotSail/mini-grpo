#!/usr/bin/env python3
"""Evaluate all SFT checkpoints on the eval set."""
import argparse
import json
import os
import re
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_MSG = "You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"

def load_eval_data(path, max_samples=None):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]
    return data

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def evaluate_checkpoint(model_path, eval_data, tokenizer, max_new_tokens=512):
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9, dtype="auto")
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    
    prompts = []
    for sample in eval_data:
        msgs = sample["messages"][:2]  # system + user only
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    outputs = llm.generate(prompts, sampling_params)
    
    correct = 0
    parsable = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        pred = extract_answer(text)
        if pred is not None:
            parsable += 1
            try:
                gt = str(eval_data[i].get("answer", ""))
                if abs(float(pred) - float(gt)) < 1e-3:
                    correct += 1
            except (ValueError, TypeError):
                if pred.strip() == gt.strip():
                    correct += 1
    
    del llm
    import torch; torch.cuda.empty_cache()
    return {"correct": correct, "parsable": parsable, "total": len(outputs),
            "accuracy": correct / len(outputs), "parsable_rate": parsable / len(outputs)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    eval_data = load_eval_data(args.eval_path, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    
    hf_dir = Path(args.run_dir) / "hf_format"
    checkpoints = sorted(hf_dir.iterdir(), key=lambda p: float(p.name.split("_")[1]))
    
    results = {}
    for ckpt in checkpoints:
        name = ckpt.name
        samples = float(name.split("_")[1])
        step = int(samples / 8)  # bs=8
        print(f"Evaluating {name} (step {step})...")
        r = evaluate_checkpoint(str(ckpt), eval_data, tokenizer)
        r["step"] = step
        r["samples"] = samples
        results[name] = r
        acc = r["accuracy"]
        pr = r["parsable_rate"]
        print(f"  accuracy={acc:.4f} parsable={pr:.4f}")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]["accuracy"])
    bstep = best[1]["step"]
    bacc = best[1]["accuracy"]
    print(f"\nBest: {best[0]} (step {bstep}): accuracy={bacc:.4f}")

if __name__ == "__main__":
    main()
