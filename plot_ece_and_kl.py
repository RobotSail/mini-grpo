#!/usr/bin/env python3
"""
Generate ECE calibration plots and accuracy vs forward KL plots.

Supports two modes:
  precision  — dtype comparison (4 conditions × 3 precisions = 12 points)
  algorithm  — algorithm × optimizer comparison (up to 3 algorithms × 2 optimizers)

Usage:
    # Precision mode (existing behavior)
    python plot_ece_and_kl.py --mode precision

    # Algorithm mode (new)
    python plot_ece_and_kl.py --mode algorithm --results validation_results/best_checkpoints_test_results.json

    # Without rejection sampling
    python plot_ece_and_kl.py --mode algorithm --results results.json --no-rs
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


# =============================================================================
# PRECISION MODE — dtype comparison (existing)
# =============================================================================

COND_KEYS = [
    ('adamw_grpo', 'AdamW', 'GRPO'),
    ('muon_grpo', 'Muon', 'GRPO'),
    ('adamw_sft', 'AdamW', 'SFT'),
    ('muon_sft', 'Muon', 'SFT'),
]

PRECISIONS = ['fp32', 'bf16', 'mixed']
PREC_LABELS = {'fp32': 'FP32', 'bf16': 'BF16', 'mixed': 'Mixed'}

SFT_SHADES = {'fp32': '#084594', 'mixed': '#4292c6', 'bf16': '#9ecae1'}
GRPO_SHADES = {'fp32': '#99000d', 'mixed': '#ef3b2c', 'bf16': '#fc9272'}


def _shade(method, prec):
    return SFT_SHADES[prec] if method == 'SFT' else GRPO_SHADES[prec]


def plot_ece_by_precision(unified_data, outdir):
    """ECE bar chart grouped by condition, bars by precision."""
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        vals, bar_colors = [], []
        for key, opt, obj in COND_KEYS:
            exp_key = f'{key}_{prec}'
            d = unified_data.get(exp_key, {})
            vals.append(d.get('ece', 0))
            bar_colors.append(_shade(obj, prec))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, color=bar_colors,
                      alpha=0.88, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('ECE', fontsize=12)
    ax.set_title('Expected Calibration Error (ECE) Across Precision Modes\n'
                 '(FP32 logprobs, confidence = answer content tokens)',
                 fontsize=13)

    legend_elements = [
        Patch(facecolor=GRPO_SHADES['fp32'], label='GRPO — FP32'),
        Patch(facecolor=GRPO_SHADES['mixed'], label='GRPO — Mixed'),
        Patch(facecolor=GRPO_SHADES['bf16'], label='GRPO — BF16'),
        Patch(facecolor='none', edgecolor='none', label=' '),
        Patch(facecolor=SFT_SHADES['fp32'], label='SFT — FP32'),
        Patch(facecolor=SFT_SHADES['mixed'], label='SFT — Mixed'),
        Patch(facecolor=SFT_SHADES['bf16'], label='SFT — BF16'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='upper left',
              framealpha=0.92, edgecolor='#cccccc')
    ax.grid(axis='y', alpha=0.25)
    max_ece = max(d.get('ece', 0) for d in unified_data.values())
    ax.set_ylim(0, max_ece * 1.2)

    plt.tight_layout()
    plt.savefig(outdir / 'ece_by_precision.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'ece_by_precision.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved ece_by_precision')


def plot_ece_vs_accuracy(unified_data, outdir):
    """Scatter plot: ECE (x) vs accuracy (y), using unified results."""
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {'GRPO': 'o', 'SFT': 's'}

    for key, opt, obj in COND_KEYS:
        for prec in PRECISIONS:
            exp_key = f'{key}_{prec}'
            if exp_key not in unified_data:
                continue
            d = unified_data[exp_key]
            acc = d['accuracy_mean'] * 100
            ece = d['ece']
            color = _shade(obj, prec)
            marker = markers[obj]

            ax.scatter(ece, acc, c=color, marker=marker, s=180,
                       alpha=0.9, edgecolors='white', linewidths=1, zorder=5)
            ax.annotate(f'{opt}\n{PREC_LABELS[prec]}', (ece, acc),
                        textcoords='offset points', xytext=(8, -5),
                        fontsize=7, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GRPO_SHADES['fp32'],
               markersize=10, label='GRPO'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=SFT_SHADES['fp32'],
               markersize=10, label='SFT'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower left')

    ax.set_xlabel('ECE', fontsize=13)
    ax.set_ylabel('GSM8K Test Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy vs Calibration Error\n'
                 '(best-validated checkpoints, 3-run average)',
                 fontsize=13)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / 'ece_vs_accuracy.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'ece_vs_accuracy.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved ece_vs_accuracy')


def plot_accuracy_vs_forward_kl(unified_data, outdir):
    """Accuracy vs forward KL from unified results."""
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {'GRPO': 'o', 'SFT': 's'}

    for key, opt, obj in COND_KEYS:
        for prec in PRECISIONS:
            exp_key = f'{key}_{prec}'
            if exp_key not in unified_data:
                continue
            d = unified_data[exp_key]
            fwd_kl = d['forward_kl']
            acc = d['accuracy_mean'] * 100

            color = _shade(obj, prec)
            marker = markers[obj]

            ax.scatter(fwd_kl, acc, c=color, marker=marker, s=180,
                       alpha=0.9, edgecolors='white', linewidths=1, zorder=5)
            ax.annotate(f'{opt}\n{PREC_LABELS[prec]}', (fwd_kl, acc),
                        textcoords='offset points', xytext=(10, -5),
                        fontsize=7, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GRPO_SHADES['fp32'],
               markersize=10, label='GRPO'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=SFT_SHADES['fp32'],
               markersize=10, label='SFT'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    ax.set_xlabel(r'$D_{\mathrm{KL}}(\pi \| \pi_0)$  (forward)', fontsize=13)
    ax.set_ylabel('GSM8K Test Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy vs Forward KL Divergence\n'
                 '(best-validated checkpoints, 3-run average)',
                 fontsize=13)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / 'accuracy_vs_forward_kl.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'accuracy_vs_forward_kl.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved accuracy_vs_forward_kl')


def plot_reliability_diagrams_fp32(unified_data, outdir):
    """Reliability diagrams (calibration plots) for FP32 training, 2x2 grid."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        (ax1, 'adamw_grpo_fp32', 'AdamW + GRPO (FP32)'),
        (ax2, 'muon_grpo_fp32', 'Muon + GRPO (FP32)'),
        (ax3, 'adamw_sft_fp32', 'AdamW + SFT (FP32)'),
        (ax4, 'muon_sft_fp32', 'Muon + SFT (FP32)'),
    ]

    for ax, exp_key, title in panels:
        if exp_key not in unified_data or 'per_bin_data' not in unified_data[exp_key]:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        d = unified_data[exp_key]
        bins = d['per_bin_data']
        acc = d['accuracy_mean']
        ece = d['ece']

        bin_centers = []
        bin_accs = []
        bin_confs = []
        for b in bins:
            if b['count'] > 0:
                center = (b['bin_lo'] + b['bin_hi']) / 2
                bin_centers.append(center)
                bin_accs.append(b['accuracy'])
                bin_confs.append(b['avg_confidence'])

        width = 0.1
        ax.bar(bin_confs, bin_accs, width=width, alpha=0.7, color='#4292c6',
               edgecolor='white', linewidth=0.5, label='Accuracy', zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Perfect')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{title}\nAcc={acc:.1%}, ECE={ece:.3f}', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Reliability Diagrams (FP32 Training)\nFP32 logprobs, 3-run generation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / 'reliability_diagrams_fp32.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'reliability_diagrams_fp32.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved reliability_diagrams_fp32')


def plot_table_ece_results(unified_data, outdir):
    """Rendered table of accuracy, ECE, and mean confidence for all 12 experiments."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')

    col_labels = ['Optimizer', 'Obj.', 'Prec.', 'Accuracy', 'Parsable', 'ECE', 'Mean Conf.']

    cell_data, cell_colors = [], []
    for key, opt, obj in COND_KEYS:
        for prec in PRECISIONS:
            exp_key = f'{key}_{prec}'
            if exp_key not in unified_data:
                continue
            d = unified_data[exp_key]
            acc = d['accuracy_mean']
            parsable = d.get('parsable_rate', 0)
            ece = d['ece']
            conf = d['mean_confidence']

            row = [opt, obj, PREC_LABELS[prec],
                   f'{acc:.1%}', f'{parsable:.1%}', f'{ece:.4f}', f'{conf:.4f}']
            cell_data.append(row)

            bg = '#f7f7f7'
            acc_bg = '#ffcccc' if acc < 0.15 else bg
            ece_bg = '#fff3cd' if ece > 0.5 else bg
            cell_colors.append([bg, bg, bg, acc_bg, bg, ece_bg, bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels), cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=9)
        cell.set_edgecolor('#cccccc')

    ax.set_title('Model Calibration: Accuracy, ECE, and Mean Confidence\n'
                 '(GSM8K test, 3-run average, FP32 logprobs)',
                 fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table_ece_results.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table_ece_results.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table_ece_results')


# =============================================================================
# ALGORITHM MODE — algorithm × optimizer comparison
# =============================================================================

# Visual encoding: color = optimizer, shape = algorithm
OPT_COLORS = {'AdamW': '#1f77b4', 'Muon': '#d62728'}
ALG_MARKERS = {'GRPO': 'o', 'SFT': 's', 'RS': 'D'}
ALG_LABELS = {'GRPO': 'GRPO', 'SFT': 'SFT', 'RS': 'Rejection Sampling'}


def _parse_algorithm_results(raw_data, include_rs=True):
    """Parse test results JSON into a list of experiment records.

    Handles key formats like:
        grpo_adamw_step_121
        sft_muon_tokens_1204593
        qwen2-1.5b-gsm8k-rs-adamw_checkpoint-150943

    Returns list of dicts with keys: optimizer, algorithm, accuracy, kl, ece, ...
    """
    experiments = []

    for key, d in raw_data.items():
        key_lower = key.lower()

        # Determine algorithm and optimizer from key
        if 'grpo' in key_lower and 'adamw' in key_lower:
            optimizer, algorithm = 'AdamW', 'GRPO'
        elif 'grpo' in key_lower and 'muon' in key_lower:
            optimizer, algorithm = 'Muon', 'GRPO'
        elif 'sft' in key_lower and 'adamw' in key_lower:
            optimizer, algorithm = 'AdamW', 'SFT'
        elif 'sft' in key_lower and 'muon' in key_lower:
            optimizer, algorithm = 'Muon', 'SFT'
        elif 'rs' in key_lower and 'adamw' in key_lower:
            optimizer, algorithm = 'AdamW', 'RS'
        elif 'rs' in key_lower and 'muon' in key_lower:
            optimizer, algorithm = 'Muon', 'RS'
        else:
            print(f'  Warning: could not parse experiment key: {key}')
            continue

        if not include_rs and algorithm == 'RS':
            continue

        # Normalize field names (handle both unified and raw eval formats)
        accuracy = d.get('accuracy_mean', d.get('accuracy', 0))
        forward_kl = d.get('forward_kl', d.get('kl_divergence', 0))
        ece = d.get('ece', 0)
        mce = d.get('mce', 0)
        mean_confidence = d.get('mean_confidence', 0)
        parsable_rate = d.get('parsable_rate', 0)
        per_bin_data = d.get('per_bin_data', [])

        experiments.append({
            'key': key,
            'optimizer': optimizer,
            'algorithm': algorithm,
            'accuracy': accuracy,
            'forward_kl': forward_kl,
            'ece': ece,
            'mce': mce,
            'mean_confidence': mean_confidence,
            'parsable_rate': parsable_rate,
            'per_bin_data': per_bin_data,
        })

    return experiments


def _build_legend(algorithms):
    """Build a structured legend with optimizer colors and algorithm markers."""
    handles = []

    # Algorithm markers (gray to show shape only)
    for alg in ['GRPO', 'SFT', 'RS']:
        if alg in algorithms:
            handles.append(Line2D(
                [0], [0], marker=ALG_MARKERS[alg], color='w',
                markerfacecolor='#888888', markeredgecolor='#555555',
                markeredgewidth=0.8, markersize=10, label=ALG_LABELS[alg],
                linestyle='None',
            ))

    # Spacer
    handles.append(Line2D([0], [0], color='none', lw=0, label=' '))

    # Optimizer colors (circle to show color only)
    for opt in ['AdamW', 'Muon']:
        handles.append(Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=OPT_COLORS[opt], markeredgecolor='white',
            markeredgewidth=0.8, markersize=10, label=opt,
            linestyle='None',
        ))

    return handles


def _smart_annotate(ax, texts, fontsize=9.5):
    """Place annotations with collision avoidance.

    texts: list of (label, x, y) tuples in data coordinates.
    """
    # Sort by x so we process left-to-right
    texts = sorted(texts, key=lambda t: t[1])

    placed = []  # list of (x_text, y_text) already placed in axes coords
    transform = ax.transData + ax.transAxes.inverted()

    for label, xd, yd in texts:
        # Convert data coords to axes fraction for collision check
        xa, ya = transform.transform((xd, yd))

        # Default offset: right and slightly down
        dx, dy = 12, -4

        # Check for collisions with already-placed labels
        for px, py in placed:
            dist_x = abs(xa - px)
            dist_y = abs(ya - py)
            if dist_x < 0.12 and dist_y < 0.06:
                # Too close — shift vertically
                if ya > py:
                    dy = 8
                else:
                    dy = -14

        ax.annotate(
            label, (xd, yd),
            textcoords='offset points', xytext=(dx, dy),
            fontsize=fontsize, fontweight='medium',
            color='#333333',
            arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6,
                            connectionstyle='arc3,rad=0.1'),
        )
        # Record approximate position of text in axes coords
        placed.append((xa + dx / 300, ya + dy / 300))


def plot_accuracy_vs_kl_algorithm(experiments, outdir):
    """Scatter: accuracy (y) vs forward KL (x), color=optimizer, shape=algorithm."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    algorithms = set()
    annotations = []

    for exp in experiments:
        acc = exp['accuracy'] * 100
        kl = exp['forward_kl']
        color = OPT_COLORS[exp['optimizer']]
        marker = ALG_MARKERS[exp['algorithm']]
        algorithms.add(exp['algorithm'])

        ax.scatter(kl, acc, c=color, marker=marker, s=200, zorder=5,
                   edgecolors='white', linewidths=1.2)

        label = f"{exp['optimizer']}+{exp['algorithm']}"
        annotations.append((label, kl, acc))

    _smart_annotate(ax, annotations)

    # Axis formatting
    ax.set_xlabel(
        r'Forward KL Divergence:  $D_{\mathrm{KL}}\!\left(\pi_\theta \,\|\, \pi_0\right)$',
        fontsize=12,
    )
    ax.set_ylabel('GSM8K Test Accuracy (%)', fontsize=12)

    # Pad axes
    kl_vals = [e['forward_kl'] for e in experiments]
    acc_vals = [e['accuracy'] * 100 for e in experiments]
    kl_range = max(kl_vals) - min(kl_vals) if len(kl_vals) > 1 else 0.05
    acc_range = max(acc_vals) - min(acc_vals) if len(acc_vals) > 1 else 5
    ax.set_xlim(min(kl_vals) - kl_range * 0.15, max(kl_vals) + kl_range * 0.25)
    ax.set_ylim(min(acc_vals) - acc_range * 0.20, max(acc_vals) + acc_range * 0.20)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))

    # Legend
    handles = _build_legend(algorithms)
    ax.legend(handles=handles, fontsize=9.5, loc='lower left',
              framealpha=0.95, edgecolor='#cccccc', handletextpad=0.6,
              borderpad=0.7)

    ax.grid(True, alpha=0.20, which='major')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(outdir / 'accuracy_vs_forward_kl.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'accuracy_vs_forward_kl.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved accuracy_vs_forward_kl')


def plot_accuracy_vs_ece_algorithm(experiments, outdir):
    """Scatter: accuracy (y) vs ECE (x), color=optimizer, shape=algorithm."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    algorithms = set()
    annotations = []

    for exp in experiments:
        acc = exp['accuracy'] * 100
        ece = exp['ece']
        color = OPT_COLORS[exp['optimizer']]
        marker = ALG_MARKERS[exp['algorithm']]
        algorithms.add(exp['algorithm'])

        ax.scatter(ece, acc, c=color, marker=marker, s=200, zorder=5,
                   edgecolors='white', linewidths=1.2)

        label = f"{exp['optimizer']}+{exp['algorithm']}"
        annotations.append((label, ece, acc))

    _smart_annotate(ax, annotations)

    ax.set_xlabel('Expected Calibration Error (ECE)', fontsize=12)
    ax.set_ylabel('GSM8K Test Accuracy (%)', fontsize=12)

    # Pad axes
    ece_vals = [e['ece'] for e in experiments]
    acc_vals = [e['accuracy'] * 100 for e in experiments]
    ece_range = max(ece_vals) - min(ece_vals) if len(ece_vals) > 1 else 0.05
    acc_range = max(acc_vals) - min(acc_vals) if len(acc_vals) > 1 else 5
    ax.set_xlim(min(ece_vals) - ece_range * 0.15, max(ece_vals) + ece_range * 0.25)
    ax.set_ylim(min(acc_vals) - acc_range * 0.20, max(acc_vals) + acc_range * 0.20)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))

    handles = _build_legend(algorithms)
    ax.legend(handles=handles, fontsize=9.5, loc='upper right',
              framealpha=0.95, edgecolor='#cccccc', handletextpad=0.6,
              borderpad=0.7)

    ax.grid(True, alpha=0.20, which='major')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(outdir / 'accuracy_vs_ece.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'accuracy_vs_ece.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved accuracy_vs_ece')


def plot_reliability_diagrams_algorithm(experiments, outdir):
    """Reliability diagrams for algorithm × optimizer comparison."""
    n = len(experiments)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Sort: GRPO first, then SFT, then RS; AdamW before Muon
    alg_order = {'GRPO': 0, 'SFT': 1, 'RS': 2}
    opt_order = {'AdamW': 0, 'Muon': 1}
    experiments_sorted = sorted(experiments,
                                key=lambda e: (alg_order.get(e['algorithm'], 9),
                                               opt_order.get(e['optimizer'], 9)))

    for idx, exp in enumerate(experiments_sorted):
        ax = axes[idx]
        title = f"{exp['optimizer']} + {exp['algorithm']}"
        color = OPT_COLORS[exp['optimizer']]
        bins = exp.get('per_bin_data', [])

        bin_confs, bin_accs = [], []
        for b in bins:
            if b['count'] > 0:
                bin_confs.append(b['avg_confidence'])
                bin_accs.append(b['accuracy'])

        ax.bar(bin_confs, bin_accs, width=0.1, alpha=0.75, color=color,
               edgecolor='white', linewidth=0.5, zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{title}\nAcc={exp["accuracy"]:.1%}, ECE={exp["ece"]:.3f}',
                     fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Reliability Diagrams\n(best-validated checkpoints, FP32 logprobs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / 'reliability_diagrams.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'reliability_diagrams.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved reliability_diagrams')


def plot_table_algorithm(experiments, outdir):
    """Rendered summary table for algorithm × optimizer results."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')

    col_labels = ['Optimizer', 'Algorithm', 'Accuracy', 'Parsable',
                  'Forward KL', 'ECE', 'Mean Conf.']

    # Sort
    alg_order = {'GRPO': 0, 'SFT': 1, 'RS': 2}
    opt_order = {'AdamW': 0, 'Muon': 1}
    experiments_sorted = sorted(experiments,
                                key=lambda e: (alg_order.get(e['algorithm'], 9),
                                               opt_order.get(e['optimizer'], 9)))

    cell_data, cell_colors = [], []
    for exp in experiments_sorted:
        bg = '#f7f7f7'
        kl_bg = '#dceefb' if exp['forward_kl'] < 0.06 else bg
        ece_bg = '#fff3cd' if exp['ece'] > 0.5 else bg

        row = [
            exp['optimizer'],
            exp['algorithm'],
            f'{exp["accuracy"]:.1%}',
            f'{exp["parsable_rate"]:.1%}',
            f'{exp["forward_kl"]:.4f}',
            f'{exp["ece"]:.4f}',
            f'{exp["mean_confidence"]:.4f}',
        ]
        cell_data.append(row)
        cell_colors.append([bg, bg, bg, bg, kl_bg, ece_bg, bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels),
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=9.5)
        cell.set_edgecolor('#cccccc')

    ax.set_title('Best-Validated Checkpoint Results\n'
                 '(GSM8K test split, FP16 inference, FP32 logprobs)',
                 fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table_results.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table_results.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table_results')


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Generate accuracy vs KL and ECE plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['precision', 'algorithm'], default='precision',
        help='precision: dtype comparison (12 conditions). '
             'algorithm: optimizer × algorithm comparison (up to 6 conditions).',
    )
    parser.add_argument(
        '--results', type=str, default=None,
        help='Path to results JSON. '
             'Default: precision_unified_results/all_results.json (precision mode) '
             'or validation_results/best_checkpoints_test_results.json (algorithm mode).',
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='Output directory (default: weight_updates_sparsity/presentation for '
             'precision mode, validation_results/ for algorithm mode).',
    )
    parser.add_argument(
        '--no-rs', action='store_true',
        help='Exclude rejection sampling experiments (algorithm mode only).',
    )
    args = parser.parse_args()

    if args.mode == 'precision':
        results_path = args.results or 'precision_unified_results/all_results.json'
        outdir = Path(args.output_dir) if args.output_dir else Path('weight_updates_sparsity/presentation')
        outdir.mkdir(parents=True, exist_ok=True)

        unified = json.load(open(results_path))

        print(f'Generating precision-mode plots from {results_path}...')
        plot_ece_by_precision(unified, outdir)
        plot_ece_vs_accuracy(unified, outdir)
        plot_accuracy_vs_forward_kl(unified, outdir)
        plot_reliability_diagrams_fp32(unified, outdir)
        plot_table_ece_results(unified, outdir)

    elif args.mode == 'algorithm':
        results_path = args.results or 'validation_results/best_checkpoints_test_results.json'
        outdir = Path(args.output_dir) if args.output_dir else Path('validation_results')
        outdir.mkdir(parents=True, exist_ok=True)

        raw_data = json.load(open(results_path))
        experiments = _parse_algorithm_results(raw_data, include_rs=not args.no_rs)

        n_exp = len(experiments)
        rs_note = ' (no RS)' if args.no_rs else ''
        print(f'Generating algorithm-mode plots from {results_path} ({n_exp} experiments{rs_note})...')

        plot_accuracy_vs_kl_algorithm(experiments, outdir)
        plot_accuracy_vs_ece_algorithm(experiments, outdir)
        plot_reliability_diagrams_algorithm(experiments, outdir)
        plot_table_algorithm(experiments, outdir)

    print(f'Done! Output in {outdir}/')


if __name__ == '__main__':
    main()
