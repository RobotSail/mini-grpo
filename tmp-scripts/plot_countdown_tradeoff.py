"""Plot accuracy/KL and accuracy/ECE tradeoff from countdown eval results.

Reads countdown_eval_results.json (same format as eval_gsm8k.py output).
Add new experiments by appending entries to the JSON (e.g. sft_adamw, rs_muon).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# Style config for each experiment type
STYLES = {
    'grpo_adamw':  {'label': 'AdamW + GRPO', 'color': '#1f77b4', 'marker': 'o'},
    'grpo_muon':   {'label': 'Muon + GRPO',  'color': '#d62728', 'marker': 's'},
    'sft_adamw':   {'label': 'AdamW + SFT',  'color': '#2ca02c', 'marker': '^'},
    'sft_muon':    {'label': 'Muon + SFT',   'color': '#ff7f0e', 'marker': 'D'},
    'rs_adamw':    {'label': 'AdamW + RS',   'color': '#9467bd', 'marker': 'v'},
    'rs_muon':     {'label': 'Muon + RS',    'color': '#8c564b', 'marker': 'P'},
}


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir / "countdown_eval_results.json"

    with open(results_path) as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for exp_name, metrics in results.items():
        style = STYLES.get(exp_name, {'label': exp_name, 'color': 'gray', 'marker': 'x'})
        acc = metrics['accuracy'] * 100

        # Panel 1: Accuracy vs Forward KL
        ax1.scatter(metrics['forward_kl'], acc,
                    color=style['color'], marker=style['marker'], s=120, zorder=5,
                    label=style['label'], edgecolors='white', linewidth=1.5)
        ax1.annotate(f"{acc:.1f}%",
                     (metrics['forward_kl'], acc),
                     textcoords='offset points', xytext=(12, 0),
                     fontsize=10, color=style['color'], fontweight='bold',
                     va='center')

        # Panel 2: Accuracy vs ECE
        ax2.scatter(metrics['ece'], acc,
                    color=style['color'], marker=style['marker'], s=120, zorder=5,
                    label=style['label'], edgecolors='white', linewidth=1.5)
        ax2.annotate(f"{acc:.1f}%",
                     (metrics['ece'], acc),
                     textcoords='offset points', xytext=(12, 0),
                     fontsize=10, color=style['color'], fontweight='bold',
                     va='center')

    ax1.set_xlabel(r"Forward KL: $D_{KL}(\pi \| \pi_0)$")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy vs Forward KL Divergence")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("ECE (Expected Calibration Error)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Test Accuracy vs Calibration Error")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Countdown Task: Accuracy Tradeoffs (Qwen2-1.5B-Instruct, best checkpoints)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_png = script_dir / "countdown_tradeoff.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(script_dir / "countdown_tradeoff.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
