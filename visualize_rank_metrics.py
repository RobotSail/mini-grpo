#!/usr/bin/env python3
"""
Visualization to understand Stable Rank vs Effective Rank.

Stable Rank:  ||A||_F² / σ₁²  =  Σσᵢ² / σ₁²
Effective Rank: (Σσᵢ)² / Σσᵢ²  (nuclear_norm² / frobenius_norm²)

Both measure "how many dimensions matter" but in different ways:
- Stable rank: "How much energy is NOT in the top singular value?"
- Effective rank: "How uniform is the distribution of singular values?"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


def compute_ranks(singular_values):
    """Compute stable and effective rank from singular values."""
    sv = np.array(singular_values)
    sv = sv[sv > 0]  # Remove zeros

    if len(sv) == 0:
        return 0, 0

    frobenius_sq = np.sum(sv ** 2)
    nuclear = np.sum(sv)
    spectral_sq = sv[0] ** 2  # Assumes sorted descending

    stable_rank = frobenius_sq / spectral_sq
    effective_rank = (nuclear ** 2) / frobenius_sq

    return stable_rank, effective_rank


def create_sv_distribution(name, n=50):
    """Create different singular value distributions for illustration."""
    if name == "uniform":
        # All singular values equal
        return np.ones(n)

    elif name == "rank1":
        # Only one nonzero singular value
        sv = np.zeros(n)
        sv[0] = 1.0
        return sv

    elif name == "linear_decay":
        # Linear decay
        return np.linspace(1, 0.1, n)

    elif name == "exponential_decay":
        # Exponential decay
        return np.exp(-np.arange(n) * 0.15)

    elif name == "heavy_tail":
        # Power law decay (heavy tail)
        return 1.0 / (1 + np.arange(n)) ** 0.5

    elif name == "two_clusters":
        # Two clusters of singular values
        sv = np.zeros(n)
        sv[:10] = 1.0
        sv[10:30] = 0.3
        return sv

    elif name == "one_spike":
        # One large spike, rest small but uniform
        sv = 0.1 * np.ones(n)
        sv[0] = 5.0
        return sv

    elif name == "gradual_then_flat":
        # Gradual decay then flat
        sv = np.ones(n) * 0.2
        sv[:15] = np.linspace(1.0, 0.2, 15)
        return sv

    else:
        raise ValueError(f"Unknown distribution: {name}")


def plot_distribution_comparison():
    """Create a figure comparing different distributions and their rank metrics."""

    distributions = [
        ("uniform", "Uniform\n(all equal)", "#2ecc71"),
        ("rank1", "Rank-1\n(single spike)", "#e74c3c"),
        ("linear_decay", "Linear\nDecay", "#3498db"),
        ("exponential_decay", "Exponential\nDecay", "#9b59b6"),
        ("heavy_tail", "Heavy Tail\n(power law)", "#f39c12"),
        ("one_spike", "One Spike +\nUniform Tail", "#1abc9c"),
    ]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.2, 0.8], hspace=0.4, wspace=0.3)

    # Top two rows: show each distribution
    stable_ranks = []
    effective_ranks = []
    names = []
    colors = []

    for idx, (dist_name, label, color) in enumerate(distributions):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        sv = create_sv_distribution(dist_name)
        sv = sv / np.max(sv)  # Normalize to max=1

        stable_r, effective_r = compute_ranks(sv)
        stable_ranks.append(stable_r)
        effective_ranks.append(effective_r)
        names.append(label.replace('\n', ' '))
        colors.append(color)

        # Plot as bar chart
        x = np.arange(len(sv))
        ax.bar(x, sv, color=color, alpha=0.7, width=1.0, edgecolor='none')
        ax.set_xlim(-1, len(sv))
        ax.set_ylim(0, 1.15)

        # Add title with metrics
        ax.set_title(f"{label}\nStable: {stable_r:.1f}  |  Effective: {effective_r:.1f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Singular Value Index", fontsize=9)
        ax.set_ylabel("σᵢ (normalized)", fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Add shading for top singular value to illustrate stable rank
        ax.axhspan(0, sv[0], xmin=0, xmax=1/len(sv), alpha=0.3, color='red', zorder=0)

    # Bottom row: comparison bar chart
    ax_compare = fig.add_subplot(gs[2, :])

    x = np.arange(len(distributions))
    width = 0.35

    bars1 = ax_compare.bar(x - width/2, stable_ranks, width, label='Stable Rank',
                           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax_compare.bar(x + width/2, effective_ranks, width, label='Effective Rank',
                           color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels([d[1].replace('\n', ' ') for d in distributions], fontsize=9)
    ax_compare.set_ylabel("Rank Value")
    ax_compare.set_title("Comparison: Stable Rank (red) vs Effective Rank (blue)", fontweight='bold')
    ax_compare.legend(loc='upper right')
    ax_compare.set_ylim(0, max(max(stable_ranks), max(effective_ranks)) * 1.15)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_compare.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax_compare.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.suptitle("Understanding Stable Rank vs Effective Rank\n" +
                 r"Stable: $\sum \sigma_i^2 / \sigma_1^2$  |  " +
                 r"Effective: $(\sum \sigma_i)^2 / \sum \sigma_i^2$",
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig("rank_metrics_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: rank_metrics_comparison.png")
    plt.close()


def plot_sensitivity_analysis():
    """Show how the two metrics respond to different perturbations."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Varying the "spike" height with uniform tail
    ax1 = axes[0]
    spike_heights = np.linspace(0.1, 10, 100)
    stable_ranks_1 = []
    effective_ranks_1 = []
    n = 50

    for spike in spike_heights:
        sv = 0.1 * np.ones(n)
        sv[0] = spike
        sr, er = compute_ranks(sv)
        stable_ranks_1.append(sr)
        effective_ranks_1.append(er)

    ax1.plot(spike_heights, stable_ranks_1, 'r-', linewidth=2, label='Stable Rank')
    ax1.plot(spike_heights, effective_ranks_1, 'b-', linewidth=2, label='Effective Rank')
    ax1.set_xlabel(r"Top Singular Value ($\sigma_1$)")
    ax1.set_ylabel("Rank Metric")
    ax1.set_title("Effect of Increasing Top σ\n(other σᵢ = 0.1 fixed)", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: Varying the tail height with fixed spike
    ax2 = axes[1]
    tail_heights = np.linspace(0.01, 1.0, 100)
    stable_ranks_2 = []
    effective_ranks_2 = []

    for tail in tail_heights:
        sv = tail * np.ones(n)
        sv[0] = 1.0  # Fixed spike
        sr, er = compute_ranks(sv)
        stable_ranks_2.append(sr)
        effective_ranks_2.append(er)

    ax2.plot(tail_heights, stable_ranks_2, 'r-', linewidth=2, label='Stable Rank')
    ax2.plot(tail_heights, effective_ranks_2, 'b-', linewidth=2, label='Effective Rank')
    ax2.set_xlabel(r"Tail Singular Values ($\sigma_{2:n}$)")
    ax2.set_ylabel("Rank Metric")
    ax2.set_title("Effect of Increasing Tail σ\n(top σ₁ = 1.0 fixed)", fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=n, color='gray', linestyle='--', alpha=0.5, label=f'Max rank = {n}')

    # Panel 3: Varying decay rate
    ax3 = axes[2]
    decay_rates = np.linspace(0.01, 0.5, 100)
    stable_ranks_3 = []
    effective_ranks_3 = []

    for decay in decay_rates:
        sv = np.exp(-np.arange(n) * decay)
        sr, er = compute_ranks(sv)
        stable_ranks_3.append(sr)
        effective_ranks_3.append(er)

    ax3.plot(decay_rates, stable_ranks_3, 'r-', linewidth=2, label='Stable Rank')
    ax3.plot(decay_rates, effective_ranks_3, 'b-', linewidth=2, label='Effective Rank')
    ax3.set_xlabel("Exponential Decay Rate")
    ax3.set_ylabel("Rank Metric")
    ax3.set_title("Effect of Decay Rate\n" + r"($\sigma_i = e^{-\lambda i}$)", fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()  # Slower decay on left

    plt.suptitle("Sensitivity Analysis: How Each Metric Responds to Changes",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rank_metrics_sensitivity.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: rank_metrics_sensitivity.png")
    plt.close()


def plot_intuition_diagram():
    """Create a simple intuition diagram explaining both metrics."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Stable Rank intuition
    ax1 = axes[0]
    n = 20
    sv = np.exp(-np.arange(n) * 0.2)

    x = np.arange(n)
    bars = ax1.bar(x, sv, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars[0].set_color('#e74c3c')
    bars[0].set_alpha(0.9)

    # Highlight the top singular value
    ax1.axhline(y=sv[0], color='red', linestyle='--', alpha=0.7, linewidth=2)

    # Add annotations
    total_energy = np.sum(sv**2)
    top_energy = sv[0]**2
    stable_r = total_energy / top_energy

    ax1.annotate(r'$\sigma_1^2$ (denominator)', xy=(0, sv[0]),
                xytext=(5, sv[0]+0.15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                color='red', fontweight='bold')

    ax1.annotate(r'$\sum \sigma_i^2$ (numerator)', xy=(n/2, 0.3),
                xytext=(n/2, 0.5), fontsize=10,
                ha='center', color='blue', fontweight='bold')

    ax1.fill_between(x, 0, sv, alpha=0.2, color='blue')

    ax1.set_xlabel("Singular Value Index", fontsize=11)
    ax1.set_ylabel(r"$\sigma_i$", fontsize=11)
    ax1.set_title(f"Stable Rank = {stable_r:.2f}\n" +
                  r"$\frac{\sum \sigma_i^2}{\sigma_1^2}$ = How much energy beyond top-1?",
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.3)
    ax1.set_xlim(-0.5, n)

    # Right: Effective Rank intuition
    ax2 = axes[1]

    # Show the same distribution
    bars = ax2.bar(x, sv, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add annotations
    nuclear = np.sum(sv)
    frobenius_sq = np.sum(sv**2)
    effective_r = (nuclear**2) / frobenius_sq

    # Show that it's related to uniformity
    uniform_level = nuclear / n
    ax2.axhline(y=uniform_level, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax2.annotate(f'Mean = Σσ/n = {uniform_level:.2f}', xy=(n-1, uniform_level),
                xytext=(n-5, uniform_level+0.2), fontsize=10,
                color='green', fontweight='bold')

    ax2.set_xlabel("Singular Value Index", fontsize=11)
    ax2.set_ylabel(r"$\sigma_i$", fontsize=11)
    ax2.set_title(f"Effective Rank = {effective_r:.2f}\n" +
                  r"$\frac{(\sum \sigma_i)^2}{\sum \sigma_i^2}$ = How uniform is the distribution?",
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.3)
    ax2.set_xlim(-0.5, n)

    plt.suptitle("Intuition: What Does Each Metric Capture?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rank_metrics_intuition.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: rank_metrics_intuition.png")
    plt.close()


def plot_key_difference():
    """Show a case where stable and effective rank differ significantly."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    n = 50

    # Case 1: Both low (rank-1 like)
    sv1 = np.zeros(n)
    sv1[0] = 1.0
    sr1, er1 = compute_ranks(sv1)

    # Case 2: Both high (uniform)
    sv2 = np.ones(n)
    sr2, er2 = compute_ranks(sv2)

    # Case 3: DIFFERENT! - Large spike + many small values
    # Stable rank sees: lots of energy beyond σ₁ → high
    # Effective rank sees: very non-uniform → low
    sv3 = 0.05 * np.ones(n)
    sv3[0] = 1.0
    sr3, er3 = compute_ranks(sv3)

    configs = [
        (sv1, sr1, er1, "Rank-1 (Single Spike)", "#e74c3c"),
        (sv2, sr2, er2, "Uniform (All Equal)", "#2ecc71"),
        (sv3, sr3, er3, "Spike + Small Tail\n(KEY DIFFERENCE!)", "#f39c12"),
    ]

    for ax, (sv, sr, er, title, color) in zip(axes, configs):
        x = np.arange(len(sv))
        ax.bar(x, sv, color=color, alpha=0.7, width=1.0)
        ax.set_xlim(-1, n+1)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("σᵢ")

        ax.set_title(f"{title}\n" +
                     f"Stable: {sr:.1f}  |  Effective: {er:.1f}",
                     fontweight='bold', fontsize=11)

        # Highlight the difference
        diff = abs(sr - er)
        if diff > 1:
            ax.annotate(f"Δ = {diff:.1f}", xy=(n/2, 0.9), fontsize=14,
                       ha='center', color='red', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle("When Do Stable and Effective Rank Disagree?\n" +
                 "Key insight: Many small singular values increase Stable Rank but decrease Effective Rank",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rank_metrics_difference.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: rank_metrics_difference.png")
    plt.close()


def main():
    print("Creating visualizations for Stable Rank vs Effective Rank...")
    print()

    plot_distribution_comparison()
    plot_sensitivity_analysis()
    plot_intuition_diagram()
    plot_key_difference()

    print()
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print()
    print("STABLE RANK = Σσᵢ² / σ₁²")
    print("  → Measures: How much total energy vs the top singular value")
    print("  → Increases when: Energy is spread across many singular values")
    print("  → Intuition: 'How many dimensions contribute comparable energy?'")
    print()
    print("EFFECTIVE RANK = (Σσᵢ)² / Σσᵢ²")
    print("  → Measures: How uniform is the singular value distribution")
    print("  → Related to: 1 / Herfindahl-Hirschman Index (concentration)")
    print("  → Intuition: 'How many equally-sized dimensions would give same ratio?'")
    print()
    print("KEY DIFFERENCE:")
    print("  When you have a big spike + many small values:")
    print("  - Stable rank is HIGH (many small σᵢ add up to significant energy)")
    print("  - Effective rank is LOW (distribution is very non-uniform)")
    print()
    print("Created 4 visualization files in current directory.")


if __name__ == "__main__":
    main()
