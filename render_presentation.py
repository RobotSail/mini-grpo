#!/usr/bin/env python3
"""
Render presentation-ready figures and tables from analyze_weight_updates.py output.

Requires: metrics.csv, sparsity.csv, and cached SVD JSONs from analyze_weight_updates.py.

Usage:
    python render_presentation.py --analysis-dir ./weight_updates_sparsity --cache-dir ./sparsity_svd_cache --config sparsity_experiments_config.json
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(analysis_dir: str, cache_dir: str, config_path: str):
    """Load all data needed for rendering."""
    mdf = pd.read_csv(os.path.join(analysis_dir, 'metrics.csv'))
    sdf = pd.read_csv(os.path.join(analysis_dir, 'sparsity.csv'))

    with open(config_path) as f:
        experiments = json.load(f)

    # Load SVD caches
    svd_cache = {}
    for exp_name in experiments:
        cache_path = os.path.join(cache_dir, f'update_svd_{exp_name}.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                svd_cache[exp_name] = json.load(f)

    # Filter to transformer layers
    layer_m = mdf[(mdf['layer'] >= 0) & (mdf['layer'] < 28)]
    layer_s = sdf[(sdf['layer'] >= 0) & (sdf['layer'] < 28)]

    return layer_m, layer_s, experiments, svd_cache


def get_metrics(layer_m, layer_s, svd_cache, exp):
    """Get aggregated metrics for one experiment."""
    m = layer_m[layer_m['experiment'] == exp]
    s = layer_s[layer_s['experiment'] == exp]

    result = {
        'rank_90': m['rank_90'].mean(),
        'rank_95': m['rank_95'].mean(),
        'rank_99': m['rank_99'].mean(),
        'sv_at_rank_90': m['sv_at_rank_90'].mean(),
        'sv_sum_at_rank_90': m['sv_sum_at_rank_90'].mean(),
        'stable_rank': m['stable_rank'].mean(),
        'effective_rank': m['effective_rank'].mean(),
        'frobenius_norm': m['frobenius_norm'].mean(),
        'spectral_norm': m['spectral_norm'].mean(),
        'top1_energy': m['top1_energy'].mean() * 100,
        'pct_changed': (1 - s['sparsity'].mean()) * 100,
        'l0_sparsity': s['sparsity'].mean() * 100,
    }

    # Spectral L0: compute from SVD cache
    if exp in svd_cache:
        param_names = [p for p in svd_cache[exp]['parameters'].keys()
                       if 'layers.' in p and '.weight' in p]
        for thresh in [1e-7, 1e-6, 1e-5, 1e-4]:
            per_param = []
            all_svs_nuc = []
            all_svs_mean = []
            all_svs_median = []
            all_sigma1 = []
            for pname in param_names:
                svs = np.array(svd_cache[exp]['parameters'][pname]['singular_values'])
                per_param.append(np.sum(svs < thresh) / len(svs) * 100)
                all_svs_nuc.append(svs.sum())
                all_svs_mean.append(svs.mean())
                all_svs_median.append(np.median(svs))
                all_sigma1.append(svs[0])
            result[f'pct_sv_below_{thresh:.0e}'] = np.mean(per_param)
        result['nuclear_norm_mean'] = np.mean(all_svs_nuc)
        result['sv_mean_mean'] = np.mean(all_svs_mean)
        result['sigma1_mean'] = np.mean(all_sigma1)

    return result


# =============================================================================
# EXPERIMENT GROUPING
# =============================================================================

COND_KEYS = [
    ('adamw_grpo', 'AdamW', 'GRPO'),
    ('muon_grpo', 'Muon', 'GRPO'),
    ('adamw_sft', 'AdamW', 'SFT'),
    ('muon_sft', 'Muon', 'SFT'),
]

PRECISIONS = ['fp32', 'bf16', 'mixed']
PREC_COLORS = {'fp32': '#2ca02c', 'bf16': '#d62728', 'mixed': '#1f77b4'}
PREC_LABELS = {'fp32': 'FP32', 'bf16': 'BF16', 'mixed': 'Mixed'}
OPT_COLORS = {'AdamW': '#1f77b4', 'Muon': '#d62728'}


# =============================================================================
# FIGURE: L0 sparsity by precision (debunking figure)
# =============================================================================

def fig_l0_by_precision(layer_s, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = [f'{opt} + {obj}' for _, opt, obj in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        vals = []
        for key, _, _ in COND_KEYS:
            sub = layer_s[layer_s['experiment'] == f'{key}_{prec}']
            vals.append((1 - sub['sparsity'].mean()) * 100)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                      color=PREC_COLORS[prec], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v < 10:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Parameters Changed (%)')
    ax.set_title(r'Parameter-Level Sparsity: $\|\Delta W\|_0 / n$' +
                 '\nBF16 rounds small updates to zero — this is a precision artifact, not a training effect')
    ax.legend(title='Training Precision', loc='upper right')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'fig1_l0_sparsity_by_precision.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig1_l0_sparsity_by_precision.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig1_l0_sparsity_by_precision')


# =============================================================================
# FIGURE: Spectral rank comparison (FP32 only)
# =============================================================================

def fig_spectral_rank(layer_m, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    objectives = ['GRPO', 'SFT']
    x = np.arange(len(objectives))
    width = 0.3

    for panel, ax, metric, ylabel, title in [
        (0, ax1, 'stable_rank', 'Stable Rank',
         r'Stable Rank: $\|\Delta W\|_F^2 / \sigma_1^2$' + '\n(lower = more concentrated)'),
        (1, ax2, 'rank_90', 'Rank $k$ (out of 1536)',
         r'Rank for 90% Energy'),
    ]:
        for i, opt in enumerate(['AdamW', 'Muon']):
            vals = []
            for obj in objectives:
                exp = f'{opt.lower()}_{obj.lower()}_fp32'
                sub = layer_m[layer_m['experiment'] == exp]
                vals.append(sub[metric].mean())
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=opt, color=OPT_COLORS[opt],
                          alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(objectives, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title='Optimizer')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Spectral Sparsity of Weight Updates (FP32 Training)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig2_spectral_rank_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig2_spectral_rank_comparison.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig2_spectral_rank_comparison')


# =============================================================================
# FIGURE: Cumulative energy (FP32, single panel)
# =============================================================================

def fig_cumulative_energy(svd_cache, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    component = 'down_proj'
    layer_idx = 13
    param_name = f'model.layers.{layer_idx}.mlp.{component}.weight'

    styles = {
        'adamw_grpo_fp32': ('AdamW + GRPO', '#1f77b4', '-'),
        'adamw_sft_fp32': ('AdamW + SFT', '#1f77b4', '--'),
        'muon_grpo_fp32': ('Muon + GRPO', '#d62728', '-'),
        'muon_sft_fp32': ('Muon + SFT', '#d62728', '--'),
    }

    for exp, (label, color, ls) in styles.items():
        if exp in svd_cache and param_name in svd_cache[exp]['parameters']:
            sv = np.array(svd_cache[exp]['parameters'][param_name]['singular_values'])
            cum_energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
            ranks = np.arange(1, len(sv) + 1)
            ax.plot(ranks, cum_energy, color=color, linestyle=ls, linewidth=2, label=label, alpha=0.9)

    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(1400, 0.91, '90%', color='gray', fontsize=10)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(1400, 0.995, '99%', color='gray', fontsize=10)

    ax.set_xlabel('Singular Value Rank')
    ax.set_ylabel('Cumulative Energy Fraction')
    ax.set_title(f'Cumulative Energy of $\\Delta W$ (Layer {layer_idx}, {component})\nFP32 Training Only')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 1536)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'fig3_cumulative_energy.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig3_cumulative_energy.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig3_cumulative_energy')


# =============================================================================
# FIGURE: Parameter sparsity vs spectral (FP32 side-by-side)
# =============================================================================

def fig_param_vs_spectral(layer_m, layer_s, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    fp32_keys = ['adamw_grpo_fp32', 'muon_grpo_fp32', 'adamw_sft_fp32', 'muon_sft_fp32']
    colors = ['#1f77b4', '#d62728', '#aec7e8', '#ff9896']

    l0_vals = [(1 - layer_s[layer_s['experiment'] == e]['sparsity'].mean()) * 100 for e in fp32_keys]
    sr_vals = [layer_m[layer_m['experiment'] == e]['stable_rank'].mean() for e in fp32_keys]

    bars = ax1.bar(range(4), l0_vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, l0_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Parameters Changed (%)')
    ax1.set_title('Parameter-Level: $\\|\\Delta W\\|_0 / n$\n(all ~100% — no L0 sparsity in FP32)')
    ax1.set_ylim(95, 101)
    ax1.grid(axis='y', alpha=0.3)

    bars = ax2.bar(range(4), sr_vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, sr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('Stable Rank')
    ax2.set_title('Spectral: $\\|\\Delta W\\|_F^2 / \\sigma_1^2$\n(AdamW+GRPO is 61x more concentrated than Muon+GRPO)')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Two Types of Sparsity Tell Different Stories (FP32 Training)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig4_parameter_vs_spectral.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig4_parameter_vs_spectral.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig4_parameter_vs_spectral')


# =============================================================================
# FIGURE: 90% energy metrics (FP32, 3-panel)
# =============================================================================

def fig_90pct_energy(layer_m, outdir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    objectives = ['GRPO', 'SFT']
    x = np.arange(len(objectives))
    width = 0.3

    for ax, metric, ylabel, title in [
        (ax1, 'rank_90', 'Rank $k$ (out of 1536)', 'Rank for 90% Energy'),
        (ax2, 'sv_at_rank_90', r'$\sigma_k(\Delta W)$', r'SV Magnitude at 90% Energy Cutoff ($\sigma_k$)'),
        (ax3, 'sv_sum_at_rank_90', r'$\sum_{i=1}^{k} \sigma_i(\Delta W)$', 'Cumulative SV Sum at 90% Energy'),
    ]:
        for i, opt in enumerate(['AdamW', 'Muon']):
            vals = []
            for obj in objectives:
                exp = f'{opt.lower()}_{obj.lower()}_fp32'
                sub = layer_m[layer_m['experiment'] == exp]
                vals.append(sub[metric].mean())
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=opt, color=OPT_COLORS[opt],
                          alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                fmt = f'{v:.0f}' if metric == 'rank_90' else (f'{v:.5f}' if 'sv_at' in metric else f'{v:.3f}')
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03,
                        fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(objectives, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title='Optimizer')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('90% Energy Threshold Metrics — FP32 Training',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig5_90pct_energy_metrics.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig5_90pct_energy_metrics.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig5_90pct_energy_metrics')


# =============================================================================
# FIGURE: rank@90% and σ_k by layer (FP32)
# =============================================================================

def fig_90pct_by_layer(layer_m, outdir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    n_layers = 28
    x = np.arange(n_layers)
    line_configs = [
        ('adamw_grpo_fp32', 'AdamW + GRPO', '#1f77b4'),
        ('muon_grpo_fp32', 'Muon + GRPO', '#d62728'),
        ('adamw_sft_fp32', 'AdamW + SFT', '#6baed6'),
        ('muon_sft_fp32', 'Muon + SFT', '#fc9272'),
    ]
    n_conds = len(line_configs)
    width = 0.8 / n_conds

    for i, (exp, label, color) in enumerate(line_configs):
        agg = layer_m[layer_m['experiment'] == exp].groupby('layer')['rank_90'].mean()
        vals = [agg.get(l, 0) for l in range(n_layers)]
        offset = (i - n_conds / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax1.set_ylabel('Rank $k$ for 90% Energy')
    ax1.set_title('Rank for 90% Energy by Layer (FP32)')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(axis='y', alpha=0.3)

    for i, (exp, label, color) in enumerate(line_configs):
        agg = layer_m[layer_m['experiment'] == exp].groupby('layer')['sv_at_rank_90'].mean()
        vals = [agg.get(l, 0) for l in range(n_layers)]
        offset = (i - n_conds / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel(r'$\sigma_k(\Delta W)$')
    ax2.set_title('Singular Value at 90% Energy Cutoff by Layer (FP32)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.legend(loc='upper right', ncol=2)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'fig6_90pct_by_layer.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig6_90pct_by_layer.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig6_90pct_by_layer')


# =============================================================================
# FIGURE: Cumulative energy 4-panel
# =============================================================================

def fig_cumulative_4panel(svd_cache, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sample_layers = [2, 13, 25]
    layer_colors = plt.cm.viridis(np.linspace(0, 0.8, len(sample_layers)))
    component = 'down_proj'

    line_configs = [
        ('adamw_grpo_fp32', 'AdamW + GRPO'),
        ('muon_grpo_fp32', 'Muon + GRPO'),
        ('adamw_sft_fp32', 'AdamW + SFT'),
        ('muon_sft_fp32', 'Muon + SFT'),
    ]

    for idx, (exp, label) in enumerate(line_configs):
        ax = axes[idx // 2, idx % 2]
        if exp not in svd_cache:
            continue

        for li, lc in zip(sample_layers, layer_colors):
            param_name = f'model.layers.{li}.mlp.{component}.weight'
            if param_name in svd_cache[exp]['parameters']:
                sv = np.array(svd_cache[exp]['parameters'][param_name]['singular_values'])
                cum_energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
                ranks = np.arange(1, len(sv) + 1)
                ax.plot(ranks, cum_energy, color=lc, linewidth=2, label=f'Layer {li}', alpha=0.9)

        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
        ax.text(1350, 0.91, '90%', color='gray', fontsize=9)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1536)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        if idx >= 2:
            ax.set_xlabel('Singular Value Rank')
        if idx % 2 == 0:
            ax.set_ylabel('Cumulative Energy Fraction')

    plt.suptitle(f'Cumulative Energy of $\\Delta W$ ({component}) — FP32 Training',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / 'fig7_cumulative_energy_4panel.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig7_cumulative_energy_4panel.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig7_cumulative_energy_4panel')


# =============================================================================
# FIGURE: L0 vs rank@90% across precision modes
# =============================================================================

def _plot_l0_vs_spectral(layer_m, layer_s, svd_cache, outdir,
                         sparsity_col, param_title, suptitle, filename):
    """Shared helper for parameter L0 vs spectral L0 side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        vals_l0, vals_sl0 = [], []
        for prefix in cond_prefixes:
            exp = f'{prefix}_{prec}'
            sub_s = layer_s[layer_s['experiment'] == exp]
            # Parameter-level: % unchanged (sparsity = 1 - ||ΔW||₀/n)
            vals_l0.append(sub_s[sparsity_col].mean() * 100)
            # Spectral-level: % of SVs below threshold
            d = get_metrics(layer_m, layer_s, svd_cache, exp)
            vals_sl0.append(d.get('pct_sv_below_1e-05', 0))
        offset = (i - 1) * width
        bars1 = ax1.bar(x + offset, vals_l0, width, label=PREC_LABELS[prec],
                        color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        max_l0 = max(vals_l0) if vals_l0 else 1
        for bar, v in zip(bars1, vals_l0):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_l0 * 0.02,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
        bars2 = ax2.bar(x + offset, vals_sl0, width, label=PREC_LABELS[prec],
                        color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        max_sl0 = max(vals_sl0) if vals_sl0 else 1
        for bar, v in zip(bars2, vals_sl0):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_sl0 * 0.02,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title(param_title, pad=10)
    ax1.legend(title='Precision')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title(r'Spectral: % of $\sigma_i(\Delta W) \leq 10^{-5}$'
                  '\n(mean over all 28 layers)', pad=10)
    ax2.legend(title='Precision')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.3)
    plt.savefig(outdir / f'{filename}.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def fig_l0_vs_rank90(layer_m, layer_s, svd_cache, outdir):
    # Fig 8: thresholded L0 on both sides (matching arXiv:2505.11711v2)
    # Left: parameter % unchanged (|ΔW| ≤ 1e-5), Right: spectral % unchanged (σᵢ < 1e-5)
    _plot_l0_vs_spectral(
        layer_m, layer_s, svd_cache, outdir,
        sparsity_col='sparsity_thresh',
        param_title=(r'Element-wise: % of entries with $|\Delta W_{ij}| \leq 10^{-5}$'
                     '\n(following arXiv:2505.11711v2, mean over all 28 layers)'),
        suptitle=r'Element-wise vs Spectral Update Sparsity ($\leq 10^{-5}$)',
        filename='fig8_l0_vs_spectral_l0_thresh',
    )

    # Fig 8b: strict nonzero on BOTH sides
    # Left: parameter ΔW ≠ 0, Right: spectral σᵢ = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        vals_l0, vals_sl0 = [], []
        for prefix in cond_prefixes:
            exp = f'{prefix}_{prec}'
            # Parameter: % unchanged (sparsity = 1 - ||ΔW||₀/n)
            sub_s = layer_s[layer_s['experiment'] == exp]
            vals_l0.append(sub_s['sparsity'].mean() * 100)
            # Spectral: % of SVs that are exactly zero
            if exp in svd_cache:
                pnames = [p for p in svd_cache[exp]['parameters']
                          if 'layers.' in p and '.weight' in p]
                per_param = []
                for p in pnames:
                    svs = np.array(svd_cache[exp]['parameters'][p]['singular_values'])
                    per_param.append(np.sum(svs == 0) / len(svs) * 100)
                vals_sl0.append(np.mean(per_param))
            else:
                vals_sl0.append(0)
        offset = (i - 1) * width
        bars1 = ax1.bar(x + offset, vals_l0, width, label=PREC_LABELS[prec],
                        color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        max_l0 = max(vals_l0) if vals_l0 else 1
        for bar, v in zip(bars1, vals_l0):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_l0 * 0.02,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
        bars2 = ax2.bar(x + offset, vals_sl0, width, label=PREC_LABELS[prec],
                        color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        max_sl0 = max(vals_sl0) if max(vals_sl0) > 0 else 1
        for bar, v in zip(bars2, vals_sl0):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_sl0 * 0.02,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title(r'Element-wise: % of entries with $\Delta W_{ij} = 0$'
                  '\n(mean over all 28 layers)', pad=10)
    ax1.legend(title='Precision')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title(r'Spectral: % of $\sigma_i(\Delta W) = 0$'
                  '\n(mean over all 28 layers)', pad=10)
    ax2.legend(title='Precision')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(r'Element-wise vs Spectral Update Sparsity (strict $= 0$)',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.3)
    plt.savefig(outdir / 'fig8b_l0_vs_spectral_l0_strict.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig8b_l0_vs_spectral_l0_strict.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig8b_l0_vs_spectral_l0_strict')


# =============================================================================
# FIGURE: Absolute SV magnitude curves
# =============================================================================

def fig_absolute_sv(svd_cache, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    component = 'down_proj'
    layer_idx = 13
    param_name = f'model.layers.{layer_idx}.mlp.{component}.weight'

    grpo_styles = [
        ('adamw_grpo_fp32', 'AdamW+GRPO (FP32)', '#1f77b4', '-'),
        ('adamw_grpo_bf16', 'AdamW+GRPO (BF16)', '#1f77b4', ':'),
        ('adamw_grpo_mixed', 'AdamW+GRPO (Mixed)', '#1f77b4', '--'),
        ('muon_grpo_fp32', 'Muon+GRPO (FP32)', '#d62728', '-'),
        ('muon_grpo_bf16', 'Muon+GRPO (BF16)', '#d62728', ':'),
        ('muon_grpo_mixed', 'Muon+GRPO (Mixed)', '#d62728', '--'),
    ]
    sft_styles = [
        ('adamw_sft_fp32', 'AdamW+SFT (FP32)', '#2ca02c', '-'),
        ('adamw_sft_bf16', 'AdamW+SFT (BF16)', '#2ca02c', ':'),
        ('adamw_sft_mixed', 'AdamW+SFT (Mixed)', '#2ca02c', '--'),
        ('muon_sft_fp32', 'Muon+SFT (FP32)', '#9467bd', '-'),
        ('muon_sft_bf16', 'Muon+SFT (BF16)', '#9467bd', ':'),
        ('muon_sft_mixed', 'Muon+SFT (Mixed)', '#9467bd', '--'),
    ]

    for ax, styles, title in [(ax1, grpo_styles, 'GRPO'), (ax2, sft_styles, 'SFT')]:
        for exp, label, color, ls in styles:
            if exp in svd_cache and param_name in svd_cache[exp]['parameters']:
                sv = np.array(svd_cache[exp]['parameters'][param_name]['singular_values'])
                ax.semilogy(np.arange(1, len(sv) + 1), sv, color=color, linestyle=ls,
                            linewidth=1.8, label=label, alpha=0.85)
        ax.set_xlabel('Singular Value Rank')
        ax.set_ylabel(r'$\sigma_i(\Delta W)$ (log scale)')
        ax.set_title(f'{title}: Raw SV Magnitudes\n(Layer {layer_idx}, {component})')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1536)

    plt.suptitle(r'Absolute Singular Value Magnitudes of $\Delta W$ — BF16 updates are orders of magnitude smaller',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig9_absolute_sv_magnitudes.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig9_absolute_sv_magnitudes.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig9_absolute_sv_magnitudes')


# =============================================================================
# FIGURE: Spectral L0 sparsity
# =============================================================================

def fig_spectral_l0(layer_m, layer_s, svd_cache, outdir):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    conditions = [f'{opt}+{obj}' for _, opt, obj in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        vals = []
        for key, _, _ in COND_KEYS:
            exp = f'{key}_{prec}'
            d = get_metrics(layer_m, layer_s, svd_cache, exp)
            vals.append(d.get('pct_sv_below_1e-05', 0))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                      color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('% of Singular Values Below Threshold')
    ax.set_title(r'Spectral L0 Sparsity: % of $\sigma_i(\Delta W) < 10^{-5}$' +
                 '\nAnalogous to parameter-level L0, but in the spectral domain')
    ax.legend(title='Precision')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'fig10_spectral_l0_sparsity.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig10_spectral_l0_sparsity.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig10_spectral_l0_sparsity')


# =============================================================================
# FIGURE: Total SV magnitude (Frobenius + nuclear)
# =============================================================================

def fig_total_magnitude(layer_m, layer_s, svd_cache, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    conditions = [f'{opt}+{obj}' for _, opt, obj in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    for i, prec in enumerate(PRECISIONS):
        frob_vals, nuc_vals = [], []
        for key, _, _ in COND_KEYS:
            exp = f'{key}_{prec}'
            d = get_metrics(layer_m, layer_s, svd_cache, exp)
            frob_vals.append(d['frobenius_norm'])
            nuc_vals.append(d.get('nuclear_norm_mean', 0))
        offset = (i - 1) * width
        ax1.bar(x + offset, frob_vals, width, label=PREC_LABELS[prec],
                color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        ax2.bar(x + offset, nuc_vals, width, label=PREC_LABELS[prec],
                color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')

    ax1.set_xticks(x); ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel(r'$\|\Delta W\|_F$')
    ax1.set_title(r'Frobenius Norm: $\|\Delta W\|_F = \sqrt{\sum \sigma_i^2}$' + '\n(total update magnitude)')
    ax1.legend(title='Precision'); ax1.grid(axis='y', alpha=0.3)

    ax2.set_xticks(x); ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel(r'$\sum_i \sigma_i(\Delta W)$')
    ax2.set_title(r'Nuclear Norm: $\|\Delta W\|_* = \sum_i \sigma_i$' + '\n(total spectral mass)')
    ax2.legend(title='Precision'); ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Absolute Magnitude of Weight Updates Across Precision Modes',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig11_total_sv_magnitude.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig11_total_sv_magnitude.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig11_total_sv_magnitude')


# =============================================================================
# FIGURE: Spectral entropy across precision modes
# =============================================================================

def fig_spectral_entropy(layer_m, outdir):
    """Spectral entropy: 0 = all energy in one SV, 1 = uniform."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    conditions = [f'{opt}+{obj}' for _, opt, obj in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    # Left: all 3 precisions
    for i, prec in enumerate(PRECISIONS):
        vals = []
        for key, _, _ in COND_KEYS:
            exp = f'{key}_{prec}'
            sub = layer_m[layer_m['experiment'] == exp]
            vals.append(sub['spectral_entropy'].mean())
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                       color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Spectral Entropy $H$')
    ax1.set_title('Spectral Entropy Across Precision Modes\n'
                   r'$H = 0$: rank-1 update, $H = 1$: uniform SVs')
    ax1.legend(title='Precision')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Right: FP32 only, grouped by optimizer
    objectives = ['GRPO', 'SFT']
    x2 = np.arange(len(objectives))
    width2 = 0.3

    for i, opt in enumerate(['AdamW', 'Muon']):
        vals = []
        for obj in objectives:
            exp = f'{opt.lower()}_{obj.lower()}_fp32'
            sub = layer_m[layer_m['experiment'] == exp]
            vals.append(sub['spectral_entropy'].mean())
        offset = (i - 0.5) * width2
        bars = ax2.bar(x2 + offset, vals, width2, label=opt, color=OPT_COLORS[opt],
                       alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(objectives, fontsize=12)
    ax2.set_ylabel('Spectral Entropy $H$')
    ax2.set_title('Spectral Entropy (FP32 Only)\n'
                   'Lower = more concentrated update')
    ax2.legend(title='Optimizer')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Spectral Entropy of Weight Updates $\\Delta W$',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig12_spectral_entropy.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig12_spectral_entropy.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig12_spectral_entropy')


def fig_spectral_entropy_by_layer(layer_m, outdir):
    """Spectral entropy by layer, FP32 only."""
    fig, ax = plt.subplots(figsize=(16, 6))

    n_layers = 28
    x = np.arange(n_layers)
    line_configs = [
        ('adamw_grpo_fp32', 'AdamW + GRPO', '#1f77b4'),
        ('muon_grpo_fp32', 'Muon + GRPO', '#d62728'),
        ('adamw_sft_fp32', 'AdamW + SFT', '#6baed6'),
        ('muon_sft_fp32', 'Muon + SFT', '#fc9272'),
    ]
    n_conds = len(line_configs)
    width = 0.8 / n_conds

    for i, (exp, label, color) in enumerate(line_configs):
        agg = layer_m[layer_m['experiment'] == exp].groupby('layer')['spectral_entropy'].mean()
        vals = [agg.get(l, 0) for l in range(n_layers)]
        offset = (i - n_conds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Spectral Entropy $H$')
    ax.set_title('Spectral Entropy by Layer (FP32 Training)\n'
                 r'$H \in [0,1]$: lower = more energy concentrated in top SVs')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'fig13_spectral_entropy_by_layer.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig13_spectral_entropy_by_layer.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig13_spectral_entropy_by_layer')


def fig_spectral_l0_multi_threshold(layer_m, layer_s, svd_cache, outdir):
    """Spectral L0 at multiple thresholds, FP32 only."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    fp32_keys = ['adamw_grpo_fp32', 'muon_grpo_fp32', 'adamw_sft_fp32', 'muon_sft_fp32']
    colors = ['#1f77b4', '#d62728', '#aec7e8', '#ff9896']

    for ax_idx, (thresh, thresh_label) in enumerate([
        ('1e-06', '10^{-6}'), ('1e-05', '10^{-5}'), ('1e-04', '10^{-4}')
    ]):
        ax = axes[ax_idx]
        vals = []
        for exp in fp32_keys:
            d = get_metrics(layer_m, layer_s, svd_cache, exp)
            vals.append(d.get(f'pct_sv_below_{thresh}', 0))

        bars = ax.bar(range(4), vals, color=colors, alpha=0.85, edgecolor='white')
        max_val = max(vals) if vals else 1
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.03,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_ylabel('% SVs Below Threshold')
        ax.set_title(rf'$\sigma_i < {thresh_label}$', pad=10)
        ax.set_ylim(0, max_val * 1.25)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Spectral L0 Sparsity at Multiple Thresholds (FP32 Training)',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85)
    plt.savefig(outdir / 'fig14_spectral_l0_multi_threshold.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig14_spectral_l0_multi_threshold.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig14_spectral_l0_multi_threshold')


def fig_90pct_by_precision(layer_m, outdir):
    """rank@90%, σ_k@90%, Σσᵢ@90% across all precision modes."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    metrics = [
        (ax1, 'rank_90', 'Rank $k$  (max = 1536)',
         r'90% Energy Rank: $k_{90}(\Delta W)$'),
        (ax2, 'sv_at_rank_90', r'$\sigma_{k}$',
         r'SV at 90% Cutoff: $\sigma_{k_{90}}(\Delta W)$'),
        (ax3, 'sv_sum_at_rank_90', r'$\sum_{i \leq k} \sigma_i$',
         r'Partial Nuclear Norm: $\sum_{i=1}^{k_{90}} \sigma_i(\Delta W)$'),
    ]

    for ax, metric, ylabel, title in metrics:
        for i, prec in enumerate(PRECISIONS):
            vals = []
            for pref in cond_prefixes:
                sub = layer_m[layer_m['experiment'] == f'{pref}_{prec}']
                vals.append(sub[metric].mean())
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                   color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(title='Precision', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        r'90% Energy Threshold Metrics of $\Delta W$ Across Precision Modes'
        '\n(mean over all 28 layers, all component types)',
        fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.3)
    plt.savefig(outdir / 'fig17_90pct_by_precision.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig17_90pct_by_precision.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig17_90pct_by_precision')


COMPONENT_ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def fig_90pct_by_precision_component(layer_m, outdir):
    """rank@90%, σ_k@90%, Σσᵢ@90% by component across precision modes, FP32 only."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    layer_df = layer_m[layer_m['component'].isin(COMPONENT_ORDER)].copy()

    line_configs = [
        ('adamw_grpo_fp32', 'AdamW+GRPO', '#1f77b4'),
        ('muon_grpo_fp32', 'Muon+GRPO', '#d62728'),
        ('adamw_sft_fp32', 'AdamW+SFT', '#6baed6'),
        ('muon_sft_fp32', 'Muon+SFT', '#fc9272'),
    ]
    n_conds = len(line_configs)
    n_comps = len(COMPONENT_ORDER)
    x = np.arange(n_comps)
    width = 0.8 / n_conds

    metrics = [
        (ax1, 'rank_90', 'Rank $k$ (out of 1536)',
         'Rank for 90% Energy'),
        (ax2, 'sv_at_rank_90', r'$\sigma_k(\Delta W)$',
         r'$\sigma_k$ at 90% Energy Cutoff'),
        (ax3, 'sv_sum_at_rank_90', r'$\sum_{i=1}^{k} \sigma_i(\Delta W)$',
         r'$\sum_{i=1}^{k} \sigma_i$ at 90% Energy'),
    ]

    for ax, metric, ylabel, title in metrics:
        for i, (exp, label, color) in enumerate(line_configs):
            agg = layer_df[layer_df['experiment'] == exp].groupby('component')[metric].mean()
            vals = [agg.get(c, 0) for c in COMPONENT_ORDER]
            offset = (i - n_conds / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(COMPONENT_ORDER, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        '90% Energy Threshold Metrics by Component (FP32 Training)\n'
        '(mean over all 28 transformer layers per component)',
        fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.82)
    plt.savefig(outdir / 'fig18_90pct_by_component_fp32.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig18_90pct_by_component_fp32.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig18_90pct_by_component_fp32')


def fig_param_vs_spectral_full(layer_m, layer_s, outdir):
    """Parameter L0 vs spectral rank@90% across precision — relabeled version."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    conditions = [f'{opt}+{obj}' for _, opt, obj in COND_KEYS]
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    # Left: L0 % changed
    for i, prec in enumerate(PRECISIONS):
        vals = [(1 - layer_s[layer_s['experiment'] == f'{pref}_{prec}']['sparsity'].mean()) * 100
                for pref in cond_prefixes]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                       color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v < 15:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                         f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Parameters Changed (%)')
    ax1.set_title(r'Parameter-Level Sparsity: $\|\Delta W\|_0 / n$' +
                  '\n(mean over all 28 transformer layers)', pad=10)
    ax1.legend(title='Precision')
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', alpha=0.3)

    # Right: rank@90%
    for i, prec in enumerate(PRECISIONS):
        vals = [layer_m[layer_m['experiment'] == f'{pref}_{prec}']['rank_90'].mean()
                for pref in cond_prefixes]
        offset = (i - 1) * width
        ax2.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('Rank $k$ (out of 1536)')
    ax2.set_title('Spectral Sparsity: Rank for 90% Energy\n'
                   '(mean over all 28 transformer layers)', pad=10)
    ax2.legend(title='Precision')
    ax2.set_ylim(0, 1000)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Parameter-Level vs Spectral Sparsity Across Precision Modes',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82)
    plt.savefig(outdir / 'fig19_param_vs_spectral_by_precision.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig19_param_vs_spectral_by_precision.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig19_param_vs_spectral_by_precision')


def table_90pct_all_precisions(layer_m, layer_s, outdir):
    """Rendered table: rank@90%, σ_k, Σσᵢ for all 12 experiments."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')

    col_labels = [
        'Condition', 'Prec.',
        '% Params\nChanged',
        'Rank\n@90%',
        'σk @90%',
        'Σσᵢ @90%',
        '‖ΔW‖F',
        'Stable\nRank',
        'Spectral\nEntropy',
    ]

    cell_data, cell_colors = [], []
    for key, opt, obj in COND_KEYS:
        cond = f'{opt}+{obj}'
        for prec in ['FP32', 'BF16', 'Mixed']:
            exp = f'{key}_{prec.lower()}'
            m = layer_m[layer_m['experiment'] == exp]
            s = layer_s[layer_s['experiment'] == exp]
            pct = (1 - s['sparsity'].mean()) * 100
            r90 = m['rank_90'].mean()
            svk = m['sv_at_rank_90'].mean()
            svsum = m['sv_sum_at_rank_90'].mean()
            frob = m['frobenius_norm'].mean()
            sr = m['stable_rank'].mean()
            ent = m['spectral_entropy'].mean()

            row = [cond, prec, f'{pct:.1f}%', f'{r90:.0f}', f'{svk:.2e}',
                   f'{svsum:.4f}', f'{frob:.4f}', f'{sr:.1f}', f'{ent:.3f}']
            cell_data.append(row)

            bg = '#f7f7f7'
            pct_bg = '#ffcccc' if pct < 50 else '#ccffcc'
            frob_bg = '#fff3cd' if frob < 0.01 else bg
            ent_bg = '#dceefb' if ent < 0.7 else bg
            cell_colors.append([bg, bg, pct_bg, bg, bg, bg, frob_bg, bg, ent_bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels), cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=8.5)
        cell.set_edgecolor('#cccccc')

    ax.set_title(
        'Table 7: 90% Energy Metrics — All Precision Modes\n'
        'All values are means over the 28 transformer layers and all component types (q/k/v/o/gate/up/down)',
        fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table7_90pct_all_precisions.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table7_90pct_all_precisions.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table7_90pct_all_precisions')


def _get_sv_distribution_stats(svd_cache, exp):
    """Compute distribution statistics of SVs for an experiment."""
    from scipy import stats as sp_stats
    if exp not in svd_cache:
        return {k: 0 for k in ['energy', 'nuclear', 'median', 'mean', 'std',
                                'skewness', 'kurtosis', 'sigma1', 'sigma_min', 'cv']}
    pnames = [p for p in svd_cache[exp]['parameters'] if 'layers.' in p and '.weight' in p]
    energies, nucs, meds, means, stds, skews, kurts, s1s, smins, cvs = \
        [], [], [], [], [], [], [], [], [], []
    for p in pnames:
        svs = np.array(svd_cache[exp]['parameters'][p]['singular_values'])
        energies.append(np.sum(svs ** 2))
        nucs.append(np.sum(svs))
        meds.append(np.median(svs))
        means.append(np.mean(svs))
        stds.append(np.std(svs))
        skews.append(float(sp_stats.skew(svs)))
        kurts.append(float(sp_stats.kurtosis(svs)))
        s1s.append(svs[0])
        smins.append(svs[-1])
        cvs.append(np.std(svs) / np.mean(svs) if np.mean(svs) > 0 else 0)
    return {
        'energy': np.mean(energies), 'nuclear': np.mean(nucs),
        'median': np.mean(meds), 'mean': np.mean(means), 'std': np.mean(stds),
        'skewness': np.mean(skews), 'kurtosis': np.mean(kurts),
        'sigma1': np.mean(s1s), 'sigma_min': np.mean(smins), 'cv': np.mean(cvs),
    }


def fig_standup_1_bf16_artifact(layer_s, svd_cache, layer_m, outdir):
    """Standup slide 1: BF16 creates element-wise sparsity, not spectral."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ['AdamW + GRPO', 'Muon + GRPO', 'AdamW + SFT', 'Muon + SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.35

    ew_vals, sp_vals = [], []
    for prefix in cond_prefixes:
        exp = f'{prefix}_bf16'
        sub_s = layer_s[layer_s['experiment'] == exp]
        ew_vals.append(sub_s['sparsity_thresh'].mean() * 100)
        d = get_metrics(layer_m, layer_s, svd_cache, exp)
        sp_vals.append(d.get('pct_sv_below_1e-05', 0))

    bars1 = ax.bar(x - width / 2, ew_vals, width, label=r'Element-wise: $|\Delta W_{ij}| \leq 10^{-5}$',
                   color='#d62728', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width / 2, sp_vals, width, label=r'Spectral: $\sigma_i(\Delta W) \leq 10^{-5}$',
                   color='#1f77b4', alpha=0.85, edgecolor='white')

    for bar, v in zip(bars1, ew_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, v in zip(bars2, sp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('BF16 Creates Element-wise Sparsity, Not Spectral Sparsity\n'
                 '(BF16 training only, mean over all 28 layers)',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(outdir / 'standup_1_bf16_artifact.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'standup_1_bf16_artifact.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved standup_1_bf16_artifact')


def fig_standup_2_fp32_rank(layer_m, outdir):
    """Standup slide 2: True spectral structure in FP32."""
    fig, ax = plt.subplots(figsize=(9, 6))

    conditions = ['AdamW\n+ GRPO', 'Muon\n+ GRPO', 'AdamW\n+ SFT', 'Muon\n+ SFT']
    fp32_keys = ['adamw_grpo_fp32', 'muon_grpo_fp32', 'adamw_sft_fp32', 'muon_sft_fp32']
    colors = ['#1f77b4', '#d62728', '#6baed6', '#fc9272']

    vals = [layer_m[layer_m['experiment'] == e]['rank_90'].mean() for e in fp32_keys]

    bars = ax.bar(range(4), vals, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f'{v:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('Rank $k$ at 90% Energy  (max = 1536)', fontsize=12)
    ax.set_ylim(0, 950)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(r'True Spectral Structure: $k_{90}(\Delta W)$ in FP32 Training'
                 '\nAdamW produces low-rank updates; Muon produces full-rank updates'
                 '\n(mean over all 28 layers)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(outdir / 'standup_2_fp32_rank.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'standup_2_fp32_rank.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved standup_2_fp32_rank')


def _mean_sv_curve(svd_cache, exp, target_len=1536):
    """Compute mean SV decay curve: average σᵢ across all layers and components with matching SV count."""
    if exp not in svd_cache:
        return np.array([])
    pnames = [p for p in svd_cache[exp]['parameters'] if 'layers.' in p and '.weight' in p]
    all_svs = []
    for p in pnames:
        svs = np.array(svd_cache[exp]['parameters'][p]['singular_values'])
        if len(svs) == target_len:
            all_svs.append(svs)
    if not all_svs:
        return np.array([])
    return np.mean(all_svs, axis=0)


def fig_standup_3_bf16_destroys_structure(svd_cache, outdir):
    """Standup slide 3: BF16 flattens AdamW's spectral shape (averaged over all layers)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Compute all curves first to set shared y-axis limits
    curves = {}
    for exp in ['adamw_grpo_fp32', 'adamw_grpo_bf16', 'muon_grpo_fp32', 'muon_grpo_bf16']:
        sv = _mean_sv_curve(svd_cache, exp)
        if len(sv) > 0:
            curves[exp] = sv

    all_sv = np.concatenate(list(curves.values()))
    ymin, ymax = all_sv[all_sv > 0].min() * 0.5, all_sv.max() * 2

    # Left: AdamW GRPO — FP32 vs BF16
    for exp, label, color, ls in [
        ('adamw_grpo_fp32', 'FP32  ($k_{90}$ = 190)', '#2ca02c', '-'),
        ('adamw_grpo_bf16', 'BF16  ($k_{90}$ = 732)', '#d62728', '--'),
    ]:
        if exp in curves:
            ranks = np.arange(1, len(curves[exp]) + 1)
            ax1.semilogy(ranks, curves[exp], color=color, linestyle=ls, linewidth=2.5, label=label, alpha=0.9)

    ax1.set_xlabel('Singular Value Index $i$', fontsize=12)
    ax1.set_ylabel(r'$\bar{\sigma}_i(\Delta W)$  (log scale)', fontsize=12)
    ax1.set_title('AdamW + GRPO\n'
                  'FP32: concentrated (low-rank)  |  BF16: dispersed (high-rank)',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim(0, 1536)
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)

    # Right: Muon GRPO — FP32 vs BF16
    for exp, label, color, ls in [
        ('muon_grpo_fp32', 'FP32  ($k_{90}$ = 752)', '#2ca02c', '-'),
        ('muon_grpo_bf16', 'BF16  ($k_{90}$ = 590)', '#d62728', '--'),
    ]:
        if exp in curves:
            ranks = np.arange(1, len(curves[exp]) + 1)
            ax2.semilogy(ranks, curves[exp], color=color, linestyle=ls, linewidth=2.5, label=label, alpha=0.9)

    ax2.set_xlabel('Singular Value Index $i$', fontsize=12)
    ax2.set_ylabel(r'$\bar{\sigma}_i(\Delta W)$  (log scale)', fontsize=12)
    ax2.set_title('Muon + GRPO\n'
                  'FP32: distributed  |  BF16: lower magnitude',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(0, 1536)
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r'FP32 vs BF16 Spectral Decay of $\Delta W$: How Precision Affects Spectral Structure'
                 '\n'
                 r'($\bar{\sigma}_i = $ mean of $\sigma_i(\Delta W)$ across all layers and components)',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.25)
    plt.savefig(outdir / 'standup_3_bf16_shape.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'standup_3_bf16_shape.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved standup_3_bf16_shape')

    # Version with all 3 precisions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    curves = {}
    for exp in ['adamw_grpo_fp32', 'adamw_grpo_bf16', 'adamw_grpo_mixed',
                'muon_grpo_fp32', 'muon_grpo_bf16', 'muon_grpo_mixed']:
        sv = _mean_sv_curve(svd_cache, exp)
        if len(sv) > 0:
            curves[exp] = sv

    all_sv = np.concatenate(list(curves.values()))
    ymin, ymax = all_sv[all_sv > 0].min() * 0.5, all_sv.max() * 2

    prec_styles = [
        ('fp32', 'FP32', '#2ca02c', '-'),
        ('bf16', 'BF16', '#d62728', '--'),
        ('mixed', 'Mixed', '#1f77b4', '-.'),
    ]

    # Left: AdamW GRPO
    for prec, plabel, color, ls in prec_styles:
        exp = f'adamw_grpo_{prec}'
        if exp in curves:
            ranks = np.arange(1, len(curves[exp]) + 1)
            ax1.semilogy(ranks, curves[exp], color=color, linestyle=ls,
                         linewidth=2.5, label=plabel, alpha=0.9)

    ax1.set_xlabel('Singular Value Index $i$', fontsize=12)
    ax1.set_ylabel(r'$\bar{\sigma}_i(\Delta W)$  (log scale)', fontsize=12)
    ax1.set_title('AdamW + GRPO', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim(0, 1536)
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)

    # Right: Muon GRPO
    for prec, plabel, color, ls in prec_styles:
        exp = f'muon_grpo_{prec}'
        if exp in curves:
            ranks = np.arange(1, len(curves[exp]) + 1)
            ax2.semilogy(ranks, curves[exp], color=color, linestyle=ls,
                         linewidth=2.5, label=plabel, alpha=0.9)

    ax2.set_xlabel('Singular Value Index $i$', fontsize=12)
    ax2.set_ylabel(r'$\bar{\sigma}_i(\Delta W)$  (log scale)', fontsize=12)
    ax2.set_title('Muon + GRPO', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(0, 1536)
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r'Spectral Decay of $\Delta W$ Across All Precision Modes'
                 '\n'
                 r'($\bar{\sigma}_i = $ mean of $\sigma_i(\Delta W)$ across all layers and components)',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.25)
    plt.savefig(outdir / 'standup_3b_all_precisions.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'standup_3b_all_precisions.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved standup_3b_all_precisions')

    # Version with k@90% markers computed from the averaged curve itself
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    all_sv = np.concatenate(list(curves.values()))
    ymin, ymax = all_sv[all_sv > 0].min() * 0.5, all_sv.max() * 2

    for ax, optimizer in [(ax1, 'adamw_grpo'), (ax2, 'muon_grpo')]:
        for prec, plabel, color, ls in prec_styles:
            exp = f'{optimizer}_{prec}'
            if exp not in curves:
                continue
            sv = curves[exp]
            ranks = np.arange(1, len(sv) + 1)

            # Compute k@90% from the averaged curve
            energy = np.cumsum(sv ** 2)
            k90 = int(np.searchsorted(energy, 0.9 * energy[-1]) + 1)
            sv_at_k90 = sv[k90 - 1]

            # Plot curve with k90 in legend
            ax.semilogy(ranks, sv, color=color, linestyle=ls,
                        linewidth=2.5, label=f'{plabel}  ($k_{{90}}$={k90})', alpha=0.9)

            # Clean dot marker at k90 position
            ax.plot(k90, sv_at_k90, 'o', color=color, markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=2)

        title = 'AdamW + GRPO' if 'adamw' in optimizer else 'Muon + GRPO'
        ax.set_xlabel('Singular Value Index $i$', fontsize=12)
        ax.set_ylabel(r'$\bar{\sigma}_i(\Delta W)$  (log scale)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_xlim(0, 1536)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r'Spectral Decay of $\bar{\sigma}_i(\Delta W)$ with $k_{90}$ Marked'
                 '\n'
                 r'($\bar{\sigma}_i$ and $k_{90}$ computed from the mean spectrum across all layers/components)',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.82, wspace=0.25)
    plt.savefig(outdir / 'standup_3c_with_k90.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'standup_3c_with_k90.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved standup_3c_with_k90')


def fig_spectral_energy_by_dtype(svd_cache, outdir):
    """Total spectral energy + nuclear norm + mean SV across precision modes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    panels = [
        (ax1, 'energy', r'$\sum_i \sigma_i^2$', r'Frobenius Energy: $\|\Delta W\|_F^2 = \sum_i \sigma_i^2$'),
        (ax2, 'nuclear', r'$\sum_i \sigma_i$', r'Nuclear Norm: $\|\Delta W\|_* = \sum_i \sigma_i$'),
        (ax3, 'mean', r'$\bar{\sigma}$', r'Mean Singular Value: $\bar{\sigma}(\Delta W)$'),
        (ax4, 'median', r'median $\sigma_i$', r'Median Singular Value: median $\sigma_i(\Delta W)$'),
    ]

    for ax, stat_key, ylabel, title in panels:
        all_vals = []
        for i, prec in enumerate(PRECISIONS):
            vals = [_get_sv_distribution_stats(svd_cache, f'{pref}_{prec}')[stat_key]
                    for pref in cond_prefixes]
            all_vals.extend(vals)
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                   color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.legend(title='Precision', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        r'Spectral Magnitude of $\Delta W$ Across Precision Modes'
        '\n(mean over all 28 layers, all component types)',
        fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.3)
    plt.savefig(outdir / 'fig15_spectral_energy_by_dtype.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig15_spectral_energy_by_dtype.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig15_spectral_energy_by_dtype')


def fig_sv_distribution_stats(svd_cache, outdir):
    """Distribution statistics of SVs: skewness, kurtosis, CV, σ₁/median ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    conditions = ['AdamW\n+GRPO', 'Muon\n+GRPO', 'AdamW\n+SFT', 'Muon\n+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    x = np.arange(len(conditions))
    width = 0.25

    panels = [
        (ax1, 'skewness', 'Skewness', 'Skewness of SV Distribution\n(higher = more right-tailed / top-heavy)'),
        (ax2, 'kurtosis', 'Excess Kurtosis', 'Kurtosis of SV Distribution\n(higher = heavier tails)'),
        (ax3, 'cv', 'Coefficient of Variation', 'CV = std(σ) / mean(σ)\n(higher = more spread out)'),
        (ax4, None, r'$\sigma_1$ / median($\sigma_i$)', r'Peak-to-Median Ratio: $\sigma_1 / \mathrm{median}(\sigma_i)$'
         '\n(higher = more concentrated in top SV)'),
    ]

    for ax, stat_key, ylabel, title in panels:
        for i, prec in enumerate(PRECISIONS):
            vals = []
            for pref in cond_prefixes:
                s = _get_sv_distribution_stats(svd_cache, f'{pref}_{prec}')
                if stat_key is not None:
                    vals.append(s[stat_key])
                else:
                    vals.append(s['sigma1'] / s['median'] if s['median'] > 0 else 0)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=PREC_LABELS[prec],
                          color=PREC_COLORS[prec], alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(title='Precision', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        r'Distribution Shape of Singular Values of $\Delta W$'
        '\n(mean over all 28 transformer layers, all component types)',
        fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.90, hspace=0.45, wspace=0.3)
    plt.savefig(outdir / 'fig20_sv_distribution_stats.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig20_sv_distribution_stats.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig20_sv_distribution_stats')


def fig_bf16_energy_ratio(svd_cache, outdir):
    """How much spectral energy survives BF16 relative to FP32."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    conditions = ['AdamW+GRPO', 'Muon+GRPO', 'AdamW+SFT', 'Muon+SFT']
    cond_prefixes = [k for k, _, _ in COND_KEYS]
    colors = ['#1f77b4', '#d62728', '#aec7e8', '#ff9896']

    def _mean_energy(exp):
        if exp not in svd_cache:
            return 0
        pnames = [p for p in svd_cache[exp]['parameters'] if 'layers.' in p and '.weight' in p]
        return np.mean([np.sum(np.array(svd_cache[exp]['parameters'][p]['singular_values']) ** 2) for p in pnames])

    def _mean_nuclear(exp):
        if exp not in svd_cache:
            return 0
        pnames = [p for p in svd_cache[exp]['parameters'] if 'layers.' in p and '.weight' in p]
        return np.mean([np.sum(np.array(svd_cache[exp]['parameters'][p]['singular_values'])) for p in pnames])

    # Left: energy ratio BF16/FP32
    energy_ratios = []
    nuclear_ratios = []
    for pref in cond_prefixes:
        fp32_e = _mean_energy(f'{pref}_fp32')
        bf16_e = _mean_energy(f'{pref}_bf16')
        energy_ratios.append(bf16_e / fp32_e * 100 if fp32_e > 0 else 0)
        fp32_n = _mean_nuclear(f'{pref}_fp32')
        bf16_n = _mean_nuclear(f'{pref}_bf16')
        nuclear_ratios.append(bf16_n / fp32_n * 100 if fp32_n > 0 else 0)

    bars = ax1.bar(range(4), energy_ratios, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, energy_ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('% of FP32 Energy Retained')
    ax1.set_title(r'Energy Retained in BF16: $\frac{\sum \sigma_i^2(\Delta W_{\mathrm{bf16}})}{\sum \sigma_i^2(\Delta W_{\mathrm{fp32}})}$')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(energy_ratios) * 1.3)

    # Right: nuclear norm ratio BF16/FP32
    bars = ax2.bar(range(4), nuclear_ratios, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, nuclear_ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('% of FP32 Nuclear Norm Retained')
    ax2.set_title(r'Nuclear Norm Retained in BF16: $\frac{\sum \sigma_i(\Delta W_{\mathrm{bf16}})}{\sum \sigma_i(\Delta W_{\mathrm{fp32}})}$')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(nuclear_ratios) * 1.3)

    fig.suptitle(
        'How Much Spectral Mass Survives BF16 Training?\n'
        'Muon loses 96–99.7% of spectral energy; AdamW retains more because its top SVs survive rounding',
        fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.82)
    plt.savefig(outdir / 'fig16_bf16_energy_ratio.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'fig16_bf16_energy_ratio.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig16_bf16_energy_ratio')


def table_spectral_energy(svd_cache, outdir):
    """Rendered table: spectral energy + ratios across dtypes."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis('off')

    col_labels = [
        'Condition', 'Prec.',
        'Σσᵢ²\n(energy)',
        '‖ΔW‖F',
        'Σσᵢ\n(nuclear)',
        'σ₁',
        'median(σᵢ)',
        'BF16/FP32\nenergy %',
    ]

    cell_data, cell_colors = [], []

    def _sv_stats(exp):
        if exp not in svd_cache:
            return {}
        pnames = [p for p in svd_cache[exp]['parameters'] if 'layers.' in p and '.weight' in p]
        svs_list = [np.array(svd_cache[exp]['parameters'][p]['singular_values']) for p in pnames]
        return {
            'energy': np.mean([np.sum(s ** 2) for s in svs_list]),
            'frob': np.mean([np.sqrt(np.sum(s ** 2)) for s in svs_list]),
            'nuclear': np.mean([np.sum(s) for s in svs_list]),
            'sigma1': np.mean([s[0] for s in svs_list]),
            'median': np.mean([np.median(s) for s in svs_list]),
        }

    for key, opt, obj in COND_KEYS:
        cond = f'{opt}+{obj}'
        fp32_stats = _sv_stats(f'{key}_fp32')

        for prec in ['FP32', 'BF16', 'Mixed']:
            exp = f'{key}_{prec.lower()}'
            s = _sv_stats(exp)
            ratio = s['energy'] / fp32_stats['energy'] * 100 if fp32_stats['energy'] > 0 else 0
            ratio_str = f'{ratio:.1f}%' if prec != 'FP32' else '—'

            row = [
                cond, prec,
                f'{s["energy"]:.2e}',
                f'{s["frob"]:.4f}',
                f'{s["nuclear"]:.4f}',
                f'{s["sigma1"]:.2e}',
                f'{s["median"]:.2e}',
                ratio_str,
            ]
            cell_data.append(row)

            bg = '#f7f7f7'
            energy_bg = '#ffcccc' if s['energy'] < fp32_stats['energy'] * 0.1 else bg
            ratio_bg = '#ffcccc' if prec == 'BF16' and ratio < 10 else ('#fff3cd' if prec == 'BF16' and ratio < 50 else bg)
            cell_colors.append([bg, bg, energy_bg, bg, bg, bg, bg, ratio_bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels), cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=8.5)
        cell.set_edgecolor('#cccccc')

    ax.set_title(
        'Table 6: Total spectral energy across precision modes\n'
        'Muon+GRPO BF16 retains only 0.4% of the FP32 spectral energy;\n'
        'AdamW+GRPO BF16 retains 3.7% — its concentrated top SVs survive rounding better',
        fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table6_spectral_energy.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table6_spectral_energy.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table6_spectral_energy')


def table_spectral_summary(layer_m, layer_s, svd_cache, outdir):
    """Rendered table: spectral entropy + L0 + magnitudes for all 12 experiments."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')

    col_labels = [
        'Optimizer', 'Obj.', 'Prec.',
        'Spectral\nEntropy',
        '‖ΔW‖F',
        'Σσᵢ',
        'Rank\n@90%',
        '% SVs\n< 1e-5',
        '% SVs\n< 1e-4',
        'σ₁',
        'mean(σᵢ)',
    ]

    cell_data, cell_colors = [], []
    for key, opt, obj in COND_KEYS:
        for prec in ['FP32', 'BF16', 'Mixed']:
            exp = f'{key}_{prec.lower()}'
            m = layer_m[layer_m['experiment'] == exp]
            d = get_metrics(layer_m, layer_s, svd_cache, exp)

            entropy = m['spectral_entropy'].mean()
            row = [
                opt, obj, prec,
                f'{entropy:.3f}',
                f'{d["frobenius_norm"]:.4f}',
                f'{d.get("nuclear_norm_mean", 0):.4f}',
                f'{d["rank_90"]:.0f}',
                f'{d.get("pct_sv_below_1e-05", 0):.1f}%',
                f'{d.get("pct_sv_below_1e-04", 0):.1f}%',
                f'{d.get("sigma1_mean", 0):.2e}',
                f'{d.get("sv_mean_mean", 0):.2e}',
            ]
            cell_data.append(row)

            bg = '#f7f7f7'
            # Color entropy: low = blue (concentrated), high = light
            ent_color = '#dceefb' if entropy < 0.7 else ('#fef0d9' if entropy < 0.9 else bg)
            frob_bg = '#fff3cd' if d['frobenius_norm'] < 0.01 else bg
            sl0 = d.get('pct_sv_below_1e-04', 0)
            sl0_bg = '#ffcccc' if sl0 > 30 else bg

            cell_colors.append([bg, bg, bg, ent_color, frob_bg, bg, bg, bg, sl0_bg, bg, bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels), cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=8)
        cell.set_edgecolor('#cccccc')

    ax.set_title(
        'Table 5: Spectral entropy, L0 sparsity, and absolute magnitudes\n'
        'AdamW has lower spectral entropy (more concentrated updates) than Muon across all conditions',
        fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table5_spectral_summary.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table5_spectral_summary.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table5_spectral_summary')


# =============================================================================
# RENDERED TABLES
# =============================================================================


def table_precision_artifact(layer_m, layer_s, outdir):
    """Table 1: BF16 sparsity is an artifact."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis('off')

    col_labels = [
        'Optimizer', 'Objective',
        '% Changed\n(FP32)', '% Changed\n(BF16)', '% Changed\n(Mixed)',
        'Rank@90%\n(FP32)', 'Rank@90%\n(BF16)', 'Rank@90%\n(Mixed)',
    ]

    cell_data, cell_colors = [], []
    for key, opt, obj in COND_KEYS:
        row, colors = [opt, obj], ['#f7f7f7', '#f7f7f7']
        for prec in PRECISIONS:
            exp = f'{key}_{prec}'
            sub_s = layer_s[layer_s['experiment'] == exp]
            pct = (1 - sub_s['sparsity'].mean()) * 100
            row.append(f'{pct:.1f}%')
            colors.append('#ffcccc' if pct < 50 else '#ccffcc')
        for prec in PRECISIONS:
            exp = f'{key}_{prec}'
            sub_m = layer_m[layer_m['experiment'] == exp]
            row.append(f'{sub_m["rank_90"].mean():.0f}')
            colors.append('#f7f7f7')
        cell_data.append(row)
        cell_colors.append(colors)

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * 8, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(fontweight='bold', fontsize=9)
        cell.set_edgecolor('#cccccc')

    ax.set_title('Table 1: Parameter-level sparsity is a BF16 precision artifact\n'
                 'BF16 rounds small weight updates to zero — the effect is identical across RL and SFT',
                 fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table1_precision_artifact.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table1_precision_artifact.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table1_precision_artifact')


def table_spectral_fp32(layer_m, layer_s, outdir):
    """Table 2: Spectral structure (FP32 only)."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis('off')

    col_labels = ['Optimizer', 'Objective', '% Params\nChanged', '‖ΔW‖F',
                  'Rank\n@90%', 'σk\n@90%', 'Σσi\n@90%', 'Stable\nRank']

    cell_data, cell_colors = [], []
    rank_vals = []
    for key, opt, obj in COND_KEYS:
        exp = f'{key}_fp32'
        rank_vals.append(layer_m[layer_m['experiment'] == exp]['rank_90'].mean())

    min_r, max_r = min(rank_vals), max(rank_vals)

    for key, opt, obj in COND_KEYS:
        exp = f'{key}_fp32'
        m = layer_m[layer_m['experiment'] == exp]
        s = layer_s[layer_s['experiment'] == exp]
        pct = (1 - s['sparsity'].mean()) * 100
        frob = m['frobenius_norm'].mean()
        r90 = m['rank_90'].mean()
        sv_k = m['sv_at_rank_90'].mean()
        sv_sum = m['sv_sum_at_rank_90'].mean()
        sr = m['stable_rank'].mean()

        row = [opt, obj, f'{pct:.1f}%', f'{frob:.4f}', f'{r90:.0f} / 1536',
               f'{sv_k:.2e}', f'{sv_sum:.3f}', f'{sr:.1f}']
        cell_data.append(row)

        frac = (r90 - min_r) / (max_r - min_r) if max_r > min_r else 0.5
        rank_hex = matplotlib.colors.to_hex(plt.cm.RdYlGn_r(0.2 + frac * 0.6))
        sr_color = '#dceefb' if sr < 15 else ('#fef0d9' if sr < 100 else '#f7f7f7')
        cell_colors.append(['#f7f7f7'] * 4 + [rank_hex, '#f7f7f7', '#f7f7f7', sr_color])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * 8, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(fontweight='bold', fontsize=9)
        cell.set_edgecolor('#cccccc')

    ax.set_title('Table 2: Spectral structure of weight updates (FP32 training)\n'
                 'AdamW+GRPO concentrates 90% of update energy in 190/1536 directions (stable rank ≈ 5)',
                 fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table2_spectral_fp32.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table2_spectral_fp32.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table2_spectral_fp32')


def table_2x2_punchline(layer_m, outdir):
    """Table 3: The 2x2 optimizer × objective punchline."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.axis('off')

    vals = {}
    for key, opt, obj in COND_KEYS:
        exp = f'{key}_fp32'
        m = layer_m[layer_m['experiment'] == exp]
        vals[f'{opt}_{obj}'] = {'rank_90': m['rank_90'].mean(), 'stable_rank': m['stable_rank'].mean()}

    col_labels = ['', 'GRPO (RL)', 'SFT']
    cell_data = [
        ['AdamW',
         f'rank@90% = {vals["AdamW_GRPO"]["rank_90"]:.0f}\nstable rank = {vals["AdamW_GRPO"]["stable_rank"]:.1f}',
         f'rank@90% = {vals["AdamW_SFT"]["rank_90"]:.0f}\nstable rank = {vals["AdamW_SFT"]["stable_rank"]:.1f}'],
        ['Muon',
         f'rank@90% = {vals["Muon_GRPO"]["rank_90"]:.0f}\nstable rank = {vals["Muon_GRPO"]["stable_rank"]:.1f}',
         f'rank@90% = {vals["Muon_SFT"]["rank_90"]:.0f}\nstable rank = {vals["Muon_SFT"]["stable_rank"]:.1f}'],
    ]
    cell_colors = [['#d9e2f3', '#dceefb', '#e8f4e8'], ['#d9e2f3', '#fef0d9', '#fef0d9']]

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#b4c6e7'] * 3, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(fontweight='bold', fontsize=12)
        if col == 0 and row > 0: cell.set_text_props(fontweight='bold', fontsize=12)
        cell.set_edgecolor('#999999')

    ax.text(0.5, -0.08,
            'AdamW+GRPO: 4× more concentrated than AdamW+SFT, 61× more than Muon+GRPO\n'
            'Muon is indifferent to objective (stable rank ≈ 300 in both RL and SFT)\n'
            'Max possible rank = 1536  •  FP32 training  •  ~1.1M tokens  •  Qwen2-1.5B',
            ha='center', va='top', fontsize=9.5, color='#444444',
            transform=ax.transAxes, linespacing=1.5)

    ax.set_title('Table 3: Spectral sparsity — optimizer × objective (FP32)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table3_2x2_punchline.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table3_2x2_punchline.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table3_2x2_punchline')


def table_full_picture(layer_m, layer_s, svd_cache, outdir):
    """Table 4: Full picture with parameter L0, spectral L0, and absolute magnitudes."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')

    col_labels = ['Optimizer', 'Obj.', 'Prec.', '% Params\nChanged', '‖ΔW‖F',
                  'Rank\n@90%', '% SVs\n< 1e-5', 'Σσᵢ\n(nuclear)', 'mean(σᵢ)']

    cell_data, cell_colors = [], []

    for key, opt, obj in COND_KEYS:
        for prec in ['FP32', 'BF16', 'Mixed']:
            exp = f'{key}_{prec.lower()}'
            d = get_metrics(layer_m, layer_s, svd_cache, exp)

            row = [opt, obj, prec,
                   f'{d["pct_changed"]:.1f}%',
                   f'{d["frobenius_norm"]:.4f}',
                   f'{d["rank_90"]:.0f}',
                   f'{d.get("pct_sv_below_1e-05", 0):.1f}%',
                   f'{d.get("nuclear_norm_mean", 0):.4f}',
                   f'{d.get("sv_mean_mean", 0):.2e}']
            cell_data.append(row)

            bg = '#f7f7f7'
            changed_bg = '#ffcccc' if d['pct_changed'] < 50 else '#ccffcc'
            frob_bg = '#fff3cd' if d['frobenius_norm'] < 0.01 else bg
            sl0 = d.get('pct_sv_below_1e-05', 0)
            sl0_bg = '#ffcccc' if sl0 > 20 else bg
            cell_colors.append([bg, bg, bg, changed_bg, frob_bg, bg, sl0_bg, bg, bg])

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#d9e2f3'] * len(col_labels), cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(fontweight='bold', fontsize=8.5)
        cell.set_edgecolor('#cccccc')

    ax.set_title(
        'Table 4: Parameter-level L0, spectral L0, and absolute SV magnitudes\n'
        'BF16 has high rank@90% because the surviving nonzero entries are randomly scattered (no low-rank structure),\n'
        'but the absolute SV magnitudes (‖ΔW‖F, Σσᵢ) are 5–20× smaller — the "high rank" captures almost nothing',
        fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(outdir / 'table4_full_picture.png', dpi=200, bbox_inches='tight')
    plt.savefig(outdir / 'table4_full_picture.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved table4_full_picture')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Render presentation figures and tables')
    parser.add_argument('--analysis-dir', type=str, default='./weight_updates_sparsity')
    parser.add_argument('--cache-dir', type=str, default='./sparsity_svd_cache')
    parser.add_argument('--config', type=str, default='sparsity_experiments_config.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir for presentation files (default: <analysis-dir>/presentation)')
    args = parser.parse_args()

    outdir = Path(args.output_dir) if args.output_dir else Path(args.analysis_dir) / 'presentation'
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {args.analysis_dir}...')
    layer_m, layer_s, experiments, svd_cache = load_data(args.analysis_dir, args.cache_dir, args.config)

    print(f'Rendering to {outdir}/')
    print()
    print('Figures:')
    fig_l0_by_precision(layer_s, outdir)
    fig_spectral_rank(layer_m, outdir)
    fig_cumulative_energy(svd_cache, outdir)
    fig_param_vs_spectral(layer_m, layer_s, outdir)
    fig_90pct_energy(layer_m, outdir)
    fig_90pct_by_layer(layer_m, outdir)
    fig_cumulative_4panel(svd_cache, outdir)
    fig_l0_vs_rank90(layer_m, layer_s, svd_cache, outdir)
    fig_absolute_sv(svd_cache, outdir)
    fig_spectral_l0(layer_m, layer_s, svd_cache, outdir)
    fig_total_magnitude(layer_m, layer_s, svd_cache, outdir)
    fig_spectral_entropy(layer_m, outdir)
    fig_spectral_entropy_by_layer(layer_m, outdir)
    fig_spectral_l0_multi_threshold(layer_m, layer_s, svd_cache, outdir)
    fig_standup_1_bf16_artifact(layer_s, svd_cache, layer_m, outdir)
    fig_standup_2_fp32_rank(layer_m, outdir)
    fig_standup_3_bf16_destroys_structure(svd_cache, outdir)
    fig_spectral_energy_by_dtype(svd_cache, outdir)
    fig_sv_distribution_stats(svd_cache, outdir)
    fig_bf16_energy_ratio(svd_cache, outdir)
    fig_90pct_by_precision(layer_m, outdir)

    print()
    print('Tables:')
    table_precision_artifact(layer_m, layer_s, outdir)
    table_spectral_fp32(layer_m, layer_s, outdir)
    table_2x2_punchline(layer_m, outdir)
    table_full_picture(layer_m, layer_s, svd_cache, outdir)
    table_spectral_summary(layer_m, layer_s, svd_cache, outdir)
    table_spectral_energy(svd_cache, outdir)
    table_90pct_all_precisions(layer_m, layer_s, outdir)

    n_files = len(list(outdir.glob('*.*')))
    print(f'\nDone! {n_files} files saved to {outdir}/')


if __name__ == '__main__':
    main()
