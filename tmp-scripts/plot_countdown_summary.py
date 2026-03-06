"""Plot 3-panel summary: accuracy, forward KL, ECE for countdown GRPO."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


def main():
    script_dir = Path(__file__).parent
    adamw_path = script_dir / "eval_adamw.json"
    muon_path = script_dir / "eval_muon.json"

    if not adamw_path.exists() or not muon_path.exists():
        print("Missing eval JSONs. Run eval_countdown_all.py first.")
        sys.exit(1)

    with open(adamw_path) as f:
        adamw = json.load(f)
    with open(muon_path) as f:
        muon = json.load(f)

    labels = ["AdamW", "Muon"]
    colors = ["#1f77b4", "#d62728"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Test Accuracy
    accs = [adamw["accuracy"] * 100, muon["accuracy"] * 100]
    bars = ax1.bar(labels, accs, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy\n(1500 countdown problems, greedy)")
    ax1.set_ylim(0, max(accs) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Forward KL
    kls = [adamw["forward_kl"], muon["forward_kl"]]
    bars = ax2.bar(labels, kls, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, kls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(kls) * 0.03,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"$D_{KL}(\pi \| \pi_0)$")
    ax2.set_title(r"Forward KL Divergence" + "\n" + r"$D_{KL}(\pi \| \pi_0) = \mathbb{E}_\pi[\log \pi - \log \pi_0]$")
    ax2.set_ylim(0, max(kls) * 1.3)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: ECE
    eces = [adamw["ece"], muon["ece"]]
    bars = ax3.bar(labels, eces, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, eces):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(eces) * 0.03,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax3.set_ylabel("ECE")
    ax3.set_title("Expected Calibration Error\n(10-bin, answer token confidence)")
    ax3.set_ylim(0, max(eces) * 1.3)
    ax3.grid(axis="y", alpha=0.3)

    # Token budget annotations
    adamw_tokens = adamw.get("checkpoint", "").split("-")[-1]
    muon_tokens = muon.get("checkpoint", "").split("-")[-1]
    fig.suptitle(
        f"Countdown GRPO: AdamW ({int(adamw_tokens)/1e6:.1f}M tokens) vs Muon ({int(muon_tokens)/1e6:.1f}M tokens)\n"
        f"Best validation checkpoints, Qwen2-1.5B-Instruct",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    out_path = script_dir / "countdown_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(script_dir / "countdown_summary.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
