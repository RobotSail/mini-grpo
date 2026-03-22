"""
Prepare GSM-Plus and SVAMP datasets for GRPO training.

Each dataset is split independently (80/10/10) so the test sets remain
constant held-out benchmarks.  A combined train + validation set is also
written for convenience.

Output format per row (matches load_gsm8k):
  - problem:   str
  - answer:    float
  - operation: str  ("gsm8k")
  - messages:  list[dict]
"""

import json
import argparse
from pathlib import Path

import datasets

SYSTEM_MSG = (
    "You are a helpful math assistant. Always provide your final numerical "
    "answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>"
)


def build_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": problem},
    ]


def load_gsm_plus() -> list[dict]:
    """Load qintongli/GSM-Plus and convert to our format."""
    rows = []
    for split in ("test", "testmini"):
        ds = datasets.load_dataset("qintongli/GSM-Plus", split=split)
        for sample in ds:
            answer_str = sample["answer"].strip()
            if answer_str.lower() == "none":
                continue
            try:
                answer = float(answer_str.replace(",", ""))
            except ValueError:
                continue
            rows.append({
                "problem": sample["question"],
                "answer": answer,
                "operation": "gsm8k",
                "messages": build_messages(sample["question"]),
            })
    print(f"GSM-Plus: loaded {len(rows)} samples (after filtering None answers)")
    return rows


def load_svamp() -> list[dict]:
    """Load ChilleD/SVAMP and convert to our format."""
    rows = []
    for split in ("train", "test"):
        ds = datasets.load_dataset("ChilleD/SVAMP", split=split)
        for sample in ds:
            problem = sample["question_concat"] if sample.get("question_concat") else (
                sample["Body"].strip() + " " + sample["Question"].strip()
            )
            try:
                answer = float(sample["Answer"].replace(",", ""))
            except (ValueError, AttributeError):
                continue
            rows.append({
                "problem": problem,
                "answer": answer,
                "operation": "gsm8k",
                "messages": build_messages(problem),
            })
    print(f"SVAMP: loaded {len(rows)} samples")
    return rows


def split_dataset(rows: list[dict], seed: int) -> tuple[list, list, list]:
    """80/10/10 split."""
    ds = datasets.Dataset.from_list(rows)
    s1 = ds.train_test_split(test_size=0.2, seed=seed)
    s2 = s1["test"].train_test_split(test_size=0.5, seed=seed)
    return list(s1["train"]), list(s2["train"]), list(s2["test"])


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows):>6} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM-Plus + SVAMP datasets")
    parser.add_argument("--output-dir", type=str, default="data/gsm_combined")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)

    # --- Load ---
    gsm_plus_rows = load_gsm_plus()
    svamp_rows = load_svamp()

    # --- Per-dataset splits (constant held-out test sets) ---
    gp_train, gp_val, gp_test = split_dataset(gsm_plus_rows, args.seed)
    sv_train, sv_val, sv_test = split_dataset(svamp_rows, args.seed)

    print(f"\nGSM-Plus splits:  train={len(gp_train)}  val={len(gp_val)}  test={len(gp_test)}")
    print(f"SVAMP splits:     train={len(sv_train)}  val={len(sv_val)}  test={len(sv_test)}")

    # Write per-dataset files
    for name, train, val, test in [
        ("gsm_plus", gp_train, gp_val, gp_test),
        ("svamp", sv_train, sv_val, sv_test),
    ]:
        write_jsonl(train, out / name / "train.jsonl")
        write_jsonl(val, out / name / "validation.jsonl")
        write_jsonl(test, out / name / "test.jsonl")

    # --- Combined train + validation (for GRPO training) ---
    combined_train = gp_train + sv_train
    combined_val = gp_val + sv_val
    write_jsonl(combined_train, out / "train.jsonl")
    write_jsonl(combined_val, out / "validation.jsonl")

    print(f"\nCombined train: {len(combined_train)}  val: {len(combined_val)}")
    print("Done!")


if __name__ == "__main__":
    main()
