#!/usr/bin/env python3
"""
Plot accuracy vs KL for all checkpoints across multiple experiment types.
Supports GRPO, SFT, and Rejection Sampling experiments.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


# Color scheme: Muon = red, AdamW = blue
# Marker scheme: GRPO = circle (o), SFT = square (s), RS = diamond (D)
EXPERIMENT_STYLES = {
    "grpo_adamw": {"color": "#1f77b4", "marker": "o", "label": "AdamW + GRPO"},
    "grpo_muon": {"color": "#d62728", "marker": "o", "label": "Muon + GRPO"},
    "sft_adamw": {"color": "#1f77b4", "marker": "s", "label": "AdamW + SFT"},
    "sft_muon": {"color": "#d62728", "marker": "s", "label": "Muon + SFT"},
    "rs_adamw": {"color": "#1f77b4", "marker": "D", "label": "AdamW + RS"},
    "rs_muon": {"color": "#d62728", "marker": "D", "label": "Muon + RS"},
}


def extract_tokens(name: str) -> int:
    """Extract token count from checkpoint name."""
    # Format: tokens_{N} (GRPO/SFT)
    if "tokens_" in name:
        try:
            return int(name.split("tokens_")[-1].split("_")[0].replace(".0", ""))
        except ValueError:
            pass
    # Format: checkpoint-{N} (RS)
    if "checkpoint-" in name:
        try:
            return int(name.split("checkpoint-")[-1].split("_")[0])
        except ValueError:
            pass
    return 0


def load_and_merge_results(exp_dir: str, exp_type: str, forward_kl: bool = False) -> list[dict]:
    """Load eval and KL results for an experiment, merge them."""
    exp_path = Path(exp_dir)

    eval_file = exp_path / "all_checkpoints_gsm8k_test.json"
    kl_filename = "all_checkpoints_forward_kl.json" if forward_kl else "all_checkpoints_kl.json"
    kl_file = exp_path / kl_filename
    kl_key = "forward_kl" if forward_kl else "kl_divergence"

    if not eval_file.exists():
        print(f"[WARN] Missing eval file: {eval_file}")
        return []
    if not kl_file.exists():
        print(f"[WARN] Missing KL file: {kl_file}")
        return []

    with open(eval_file) as f:
        eval_results = json.load(f)
    with open(kl_file) as f:
        kl_results = json.load(f)

    # Build KL lookup by token count
    kl_by_tokens = {}
    for k, v in kl_results.items():
        tokens = v.get("tokens", 0) or extract_tokens(k)
        if tokens > 0:
            kl_by_tokens[tokens] = v

    # Merge by token count
    merged = []
    for ckpt_name, eval_data in eval_results.items():
        tokens = extract_tokens(ckpt_name)
        if tokens == 0:
            continue

        kl_data = kl_by_tokens.get(tokens)
        if kl_data and kl_key in kl_data:
            merged.append({
                "name": ckpt_name,
                "tokens": tokens,
                "accuracy": eval_data["accuracy"],
                "kl": kl_data[kl_key],
                "exp_type": exp_type,
            })

    return sorted(merged, key=lambda x: x["tokens"])


def plot_combined(
    grpo_sft_dir: str,
    rs_dir: str,
    variant: int,
    output_path: str,
    include_grpo: bool = True,
    include_sft: bool = True,
    include_rs: bool = True,
    forward_kl: bool = False,
):
    """Create a combined plot with GRPO, SFT, and RS experiments."""

    experiments = {}

    if include_grpo:
        experiments["grpo_adamw"] = f"{grpo_sft_dir}/qwen2-1.5b-gsm8k-grpo-adamw_verify_{variant}"
        experiments["grpo_muon"] = f"{grpo_sft_dir}/qwen2-1.5b-gsm8k-grpo-muon_verify_{variant}"

    if include_sft:
        experiments["sft_adamw"] = f"{grpo_sft_dir}/qwen2-1.5b-gsm8k-sft-adamw_verify_{variant}"
        experiments["sft_muon"] = f"{grpo_sft_dir}/qwen2-1.5b-gsm8k-sft-muon_verify_{variant}"

    if include_rs:
        experiments["rs_adamw"] = f"{rs_dir}/qwen2-1.5b-gsm8k-rs-adamw"
        experiments["rs_muon"] = f"{rs_dir}/qwen2-1.5b-gsm8k-rs-muon"

    fig, ax = plt.subplots(figsize=(10, 7))

    all_kl = []
    all_acc = []

    for exp_type, exp_dir in experiments.items():
        data = load_and_merge_results(exp_dir, exp_type, forward_kl=forward_kl)
        if not data:
            continue

        style = EXPERIMENT_STYLES[exp_type]
        kls = [d["kl"] for d in data]
        accs = [d["accuracy"] for d in data]

        all_kl.extend(kls)
        all_acc.extend(accs)

        ax.scatter(
            kls, accs,
            c=style["color"],
            marker=style["marker"],
            s=80,
            label=style["label"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    if not all_kl:
        print(f"[WARN] No data found")
        plt.close()
        return

    if forward_kl:
        ax.set_xlabel(r"$D_{\mathrm{KL}}(\pi \| \pi_0)$  (forward)", fontsize=13)
    else:
        ax.set_xlabel(r"$D_{\mathrm{KL}}(p_{\mathrm{base}} \| p_{\mathrm{trained}})$", fontsize=13)
    ax.set_ylabel("GSM8K Test Accuracy", fontsize=13)

    title_parts = []
    if include_grpo:
        title_parts.append("GRPO")
    if include_sft:
        title_parts.append("SFT")
    if include_rs:
        title_parts.append("RS")
    kl_label = "Forward KL" if forward_kl else "KL"
    title = f"Accuracy vs {kl_label}: {' / '.join(title_parts)}"
    if variant:
        title += f" (Seed {variant})"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Auto-scale with padding
    kl_min, kl_max = min(all_kl), max(all_kl)
    acc_min, acc_max = min(all_acc), max(all_acc)
    kl_pad = (kl_max - kl_min) * 0.1 or 0.01
    acc_pad = (acc_max - acc_min) * 0.1 or 0.05

    ax.set_xlim(max(0, kl_min - kl_pad), kl_max + kl_pad)
    ax.set_ylim(max(0, acc_min - acc_pad), min(1.0, acc_max + acc_pad))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot combined experiments")
    parser.add_argument(
        "--grpo-sft-dir",
        type=str,
        default="/mnt/nvme2n1/checkpoints/verify-exps-variable-seeds",
        help="Base directory for GRPO/SFT experiments",
    )
    parser.add_argument(
        "--rs-dir",
        type=str,
        default="/mnt/nvme2n1/checkpoints/rs-train-exps",
        help="Base directory for rejection sampling experiments",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=1,
        help="Variant number for GRPO/SFT experiments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot",
    )
    parser.add_argument(
        "--no-grpo",
        action="store_true",
        help="Exclude GRPO experiments",
    )
    parser.add_argument(
        "--no-sft",
        action="store_true",
        help="Exclude SFT experiments",
    )
    parser.add_argument(
        "--no-rs",
        action="store_true",
        help="Exclude RS experiments",
    )
    parser.add_argument(
        "--forward-kl",
        action="store_true",
        help="Use forward KL(π || π₀) instead of reverse KL",
    )

    args = parser.parse_args()

    suffix = "_forward_kl" if args.forward_kl else ""
    output_path = args.output or f"{args.grpo_sft_dir}/combined_v{args.variant}{suffix}.png"

    plot_combined(
        grpo_sft_dir=args.grpo_sft_dir,
        rs_dir=args.rs_dir,
        variant=args.variant,
        output_path=output_path,
        include_grpo=not args.no_grpo,
        include_sft=not args.no_sft,
        include_rs=not args.no_rs,
        forward_kl=args.forward_kl,
    )


if __name__ == "__main__":
    main()
