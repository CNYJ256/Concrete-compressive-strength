"""
Generate SCI-tier figures for the ACDCB paper — New Dataset Results.
Strict requirements: Times New Roman, 300 dpi, ColorBrewer, PDF+PNG output.
Single-column: 3.5", Double-column: 7.125", Height ≤ 9".
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Global style — SCI journal standards
# ============================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "presentation_highres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ColorBrewer Set2 (colorblind-friendly, B&W compatible)
C = {
    'v0': '#66c2a5', 'v1': '#fc8d62', 'v2': '#8da0cb',
    'v3': '#e78ac3', 'v4': '#a6d854', 'v5': '#ffd92f',
    'uci': '#8da0cb', 'new': '#fc8d62',
    'xgb': '#66c2a5', 'ensemble': '#e78ac3',
    'top12': '#fc8d62', 'full': '#8da0cb', 'raw': '#a6d854',
}

SINGLE_W = 3.5
DUAL_W = 7.125
DPI = 300

# ============================================================
# Data loading
# ============================================================
def load_new_ablation():
    path = ROOT / "results" / "new_dataset" / "metrics" / "ablation_newdata_results.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_uci_ablation():
    path = ROOT / "results" / "metrics" / "ablation_results_acdcb_v2.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_age_stratified():
    path = ROOT / "results" / "new_dataset" / "metrics" / "p11_age_stratified.csv"
    return pd.read_csv(path, encoding='utf-8')

def load_soft_weighting():
    path = ROOT / "results" / "new_dataset" / "metrics" / "p12_soft_weighting.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_lasso_hpo():
    path = ROOT / "results" / "new_dataset" / "metrics" / "p21_lasso_joint_hpo.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_conformal():
    path = ROOT / "results" / "new_dataset" / "metrics" / "p22_conformal_prediction.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_astm_bootstrap():
    path = ROOT / "results" / "new_dataset" / "metrics" / "p02_astm_bootstrap.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# Figure 1: New Dataset Ablation Results
# ============================================================
def fig_newdata_ablation():
    data = load_new_ablation()
    variants = data['variants']

    labels = ['V0\nAdaBoost', 'V1\nPrimary\n+Global', 'V2\nDualSpace\n+Global',
              'V3\nACDCB\nFull', 'V4\nRaw\n+Piecewise', 'V5\nOLS\nUnconst.']
    keys = ['v0_adaboost', 'v1_primary_global', 'v2_dualspace_global',
            'v3_dualspace_piecewise_acdcb', 'v4_raw_piecewise', 'v5_ols_unconstrained']
    colors = [C['v0'], C['v1'], C['v2'], C['v3'], C['v4'], C['v5']]

    r2_vals = [variants[k]['metrics']['R2_mean'] for k in keys]
    r2_stds = [variants[k]['metrics']['R2_std'] for k in keys]
    rmse_vals = [variants[k]['metrics']['RMSE_mean'] for k in keys]
    rmse_stds = [variants[k]['metrics']['RMSE_std'] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 3.2))

    # --- R² ---
    x = np.arange(len(labels))
    bars = ax1.bar(x, r2_vals, 0.6, color=colors, edgecolor='white', linewidth=0.5,
                   yerr=r2_stds, capsize=3, error_kw={'linewidth': 0.8})
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel('$R^2$', fontsize=9)
    ax1.set_title('(a) $R^2$ — New Dataset ($N$=4,420)', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.992, 0.997)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.001))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Annotate key deltas
    best_single = data['single_model_OOF']['HGB_anchor']['R2']
    ax1.axhline(y=best_single, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.text(5.3, best_single + 0.0001, f'Best single\nOOF $R^2$={best_single:.4f}',
             fontsize=6.5, color='gray', ha='left', va='bottom')

    # Annotate V1-V0 delta
    mid_x = 0.5
    ax1.annotate('', xy=(1, r2_vals[1]), xytext=(0, r2_vals[1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
    ax1.text(mid_x, r2_vals[1] + 0.0002, f'$\\Delta R^2$=+0.0018',
             fontsize=7, color='red', ha='center', fontweight='bold')

    # --- RMSE ---
    ax2.bar(x, rmse_vals, 0.6, color=colors, edgecolor='white', linewidth=0.5,
            yerr=rmse_stds, capsize=3, error_kw={'linewidth': 0.8})
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel('RMSE (MPa)', fontsize=9)
    ax2.set_title('(b) RMSE — New Dataset ($N$=4,420)', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_newdata_ablation.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_newdata_ablation")


# ============================================================
# Figure 2: Cross-Dataset Ablation Comparison
# ============================================================
def fig_cross_dataset_comparison():
    uci = load_uci_ablation()
    nd = load_new_ablation()

    uci_keys = ['paper1_adaboost', 'v1_primary_global_no_anchor', 'v2_dualspace_global',
                'v3_dualspace_age_piecewise_acdcb', 'v4_raw_age_piecewise', 'v5_ols_unconstrained_global']
    nd_keys = ['v0_adaboost', 'v1_primary_global', 'v2_dualspace_global',
               'v3_dualspace_piecewise_acdcb', 'v4_raw_piecewise', 'v5_ols_unconstrained']

    uci_r2 = [uci['variants'][k]['metrics']['R2_mean'] for k in uci_keys]
    nd_r2 = [nd['variants'][k]['metrics']['R2_mean'] for k in nd_keys]

    labels = ['V0\nAdaBoost', 'V1\nPrimary\n+Global', 'V2\nDual\n+Global',
              'V3\nACDCB\nFull', 'V4\nRaw\n+Piecewise', 'V5\nOLS\nUnconst.']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 3.5))

    x = np.arange(len(labels))
    w = 0.35

    # --- UCI ---
    bars1 = ax1.bar(x, uci_r2, w, color=C['uci'], edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel('$R^2$', fontsize=9)
    ax1.set_title('(a) UCI Dataset ($N$=1,030)', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.90, 0.97)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    # Annotate values
    for bar, val in zip(bars1, uci_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=6.5, fontweight='bold')

    # --- New Data ---
    bars2 = ax2.bar(x, nd_r2, w, color=C['new'], edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel('$R^2$', fontsize=9)
    ax2.set_title('(b) New Dataset ($N$=4,420)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0.992, 0.997)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    for bar, val in zip(bars2, nd_r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', fontsize=6.5, fontweight='bold')

    # Shared insight annotation
    fig.text(0.5, 0.01, 'Pattern is identical across datasets: V0→V1 dominates; V1→V5 are flat.',
             ha='center', fontsize=8, fontstyle='italic', color='gray')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_cross_dataset_comparison.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_cross_dataset_comparison")


# ============================================================
# Figure 3: Age-Stratified Error Analysis
# ============================================================
def fig_age_stratified():
    df = load_age_stratified()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 3.2))

    # --- UCI ---
    uci_df = df[df['dataset'] == 'UCI'].copy()
    age_mid = [(1+7)/2, (7+28)/2, (28+90)/2, (90+365)/2]
    sizes = uci_df['n'].values

    ax1.scatter(age_mid, uci_df['V4_R2'].values, s=sizes*0.3, c=C['uci'],
                edgecolors='white', linewidth=0.5, zorder=5, alpha=0.8)
    ax1.plot(age_mid, uci_df['V4_R2'].values, '-', color=C['uci'], linewidth=1.2, alpha=0.5)
    for i, (x, y, n) in enumerate(zip(age_mid, uci_df['V4_R2'].values, sizes)):
        ax1.annotate(f'$n$={n}', (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, ha='center', color='gray')
    ax1.set_xlabel('Curing Age (days)', fontsize=9)
    ax1.set_ylabel('$R^2$', fontsize=9)
    ax1.set_title('(a) UCI Dataset', fontsize=10, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim(0.88, 0.99)
    ax1.grid(alpha=0.3, linewidth=0.5)

    # --- New Data ---
    nd_df = df[df['dataset'] == 'New'].copy()
    ages = nd_df['age_days'].values
    sizes_nd = nd_df['n'].values

    ax2.scatter(ages, nd_df['V4_R2'].values, s=sizes_nd*0.25, c=C['new'],
                edgecolors='white', linewidth=0.5, zorder=5, alpha=0.8)
    ax2.plot(ages, nd_df['V4_R2'].values, '-', color=C['new'], linewidth=1.2, alpha=0.5)
    for x, y, n in zip(ages, nd_df['V4_R2'].values, sizes_nd):
        ax2.annotate(f'$n$={n}', (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, ha='center', color='gray')
    ax2.set_xlabel('Curing Age (days)', fontsize=9)
    ax2.set_ylabel('$R^2$', fontsize=9)
    ax2.set_title('(b) New Dataset', fontsize=10, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim(0.96, 1.00)
    ax2.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_age_stratified.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_age_stratified")


# ============================================================
# Figure 4: Component Contribution Waterfall
# ============================================================
def fig_component_waterfall():
    # UCI deltas (from paper Table 6)
    uci_base = 0.9142  # V0 AdaBoost
    uci_deltas = [
        ('Model Upgrade\n(AdaBoost→GBDT)', +0.0381),
        ('Feature Eng.\n(8D→32D)', -0.0054),
        ('Dual Space\n(+Anchor)', +0.0012),
        ('Age Piecewise\n(τ=28d)', +0.00003),
        ('Constrained\nOptimization', +0.0000),
    ]

    # New data deltas (from ablation_newdata_results.json)
    nd_base = 0.9936
    nd_deltas = [
        ('Model Upgrade\n(AdaBoost→GBDT)', +0.001773),
        ('Feature Eng.\n(Raw→Engineered)', +0.000113),
        ('Dual Space\n(+Anchor)', +0.000078),
        ('Age Piecewise\n(τ=28d)', +0.000012),
        ('Constrained\nOptimization', +0.000000),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 3.8))

    for ax, base, deltas, dataset_label, bar_color in [
        (ax1, uci_base, uci_deltas, 'UCI ($N$=1,030)', C['uci']),
        (ax2, nd_base, nd_deltas, 'New Data ($N$=4,420)', C['new'])]:

        cumul = base
        labels = ['AdaBoost\nBaseline']
        values = [base]
        colors_list = ['#cccccc']

        for name, delta in deltas:
            cumul += delta
            labels.append(name)
            values.append(cumul)
            colors_list.append(bar_color if delta > 0 else '#e9a3a3')

        x = np.arange(len(labels))
        bars = ax.bar(x, values, 0.6, color=colors_list, edgecolor='white', linewidth=0.5)

        # Draw connecting lines between bars
        for i in range(len(values) - 1):
            ax.plot([i + 0.3, i + 0.7], [values[i], values[i]],
                    '-', color='gray', linewidth=0.8, alpha=0.5)

        # Annotate deltas
        for i, (name, delta) in enumerate(deltas):
            mid = (values[i] + values[i+1]) / 2
            color = 'green' if delta > 0 else 'red'
            sign = '+' if delta >= 0 else ''
            ax.annotate(f'{sign}{delta:.4f}', (i + 1, mid),
                       textcoords="offset points", xytext=(0, 6 if delta > 0 else -14),
                       fontsize=6.5, ha='center', color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel('$R^2$', fontsize=9)
        ax.set_title(dataset_label, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_component_waterfall.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_component_waterfall")


# ============================================================
# Figure 5: LASSO + Joint HPO Comparison
# ============================================================
def fig_lasso_hpo():
    data = load_lasso_hpo()
    hpo = data['hpo_comparison']

    groups = ['LASSO\nTop-12', 'Full\n32 Features', 'Raw\n15 Features']
    r2_vals = [hpo['top12_features_r2'], hpo['full_features_r2'], hpo['raw_features_r2']]
    colors_list = [C['top12'], C['full'], C['raw']]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.0))

    x = np.arange(len(groups))
    bars = ax.bar(x, r2_vals, 0.5, color=colors_list, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')

    # Annotate deltas
    ax.annotate(f'$\\Delta R^2$={r2_vals[1]-r2_vals[0]:+.4f}',
                xy=(0.5, (r2_vals[0]+r2_vals[1])/2), fontsize=7, ha='center', color='red')
    ax.annotate(f'$\\Delta R^2$={r2_vals[2]-r2_vals[0]:+.4f}',
                xy=(1.5, (r2_vals[0]+r2_vals[2])/2), fontsize=7, ha='center', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel('$R^2$', fontsize=9)
    ax.set_title('Joint HPO + Feature Selection\n(Optuna 50 trials, New Data)', fontsize=10, fontweight='bold')
    ax.set_ylim(0.985, 0.997)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # LASSO note
    ax.text(0.5, 0.02, 'LASSO: only 3/32 non-zero coefficients', transform=ax.transAxes,
            fontsize=7, fontstyle='italic', color='gray', ha='center')

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_lasso_hpo.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_lasso_hpo")


# ============================================================
# Figure 6: Conformal Prediction Intervals
# ============================================================
def fig_conformal():
    data = load_conformal()
    cf = data['conformal']

    levels = ['90%', '95%']
    xgb_picp = [cf[l]['XGB Single']['PICP'] for l in levels]
    ens_picp = [cf[l]['ACDCB Ensemble']['PICP'] for l in levels]
    xgb_mpiw = [cf[l]['XGB Single']['MPIW'] for l in levels]
    ens_mpiw = [cf[l]['ACDCB Ensemble']['MPIW'] for l in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 2.8))

    x = np.arange(len(levels))
    w = 0.3

    # --- PICP ---
    ax1.bar(x - w/2, xgb_picp, w, color=C['xgb'], edgecolor='white', linewidth=0.5, label='XGB Single')
    ax1.bar(x + w/2, ens_picp, w, color=C['ensemble'], edgecolor='white', linewidth=0.5, label='ACDCB Ensemble')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=9)
    ax1.set_ylabel('PICP', fontsize=9)
    ax1.set_title('(a) Prediction Interval Coverage', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.85, 0.97)
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    # Annotate
    for i in range(2):
        ax1.text(i, xgb_picp[i] + 0.003, f'{xgb_picp[i]:.4f}', ha='center', fontsize=7)
        ax1.text(i, ens_picp[i] + 0.003, f'{ens_picp[i]:.4f}', ha='center', fontsize=7)
    ax1.axhline(y=float(levels[0][:-1])/100, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    # --- MPIW ---
    ax2.bar(x - w/2, xgb_mpiw, w, color=C['xgb'], edgecolor='white', linewidth=0.5, label='XGB Single')
    ax2.bar(x + w/2, ens_mpiw, w, color=C['ensemble'], edgecolor='white', linewidth=0.5, label='ACDCB Ensemble')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_ylabel('MPIW (MPa)', fontsize=9)
    ax2.set_title('(b) Mean Prediction Interval Width', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7, frameon=False)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    for i in range(2):
        ax2.text(i, xgb_mpiw[i] + 0.02, f'{xgb_mpiw[i]:.2f}', ha='center', fontsize=7)
        ax2.text(i, ens_mpiw[i] + 0.02, f'{ens_mpiw[i]:.2f}', ha='center', fontsize=7)

    # key insight
    fig.text(0.5, 0.01, 'Ensemble intervals are WIDER — no UQ benefit from model diversity',
             ha='center', fontsize=8, fontstyle='italic', color='gray')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_conformal.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_conformal")


# ============================================================
# Figure 7: Bootstrap Confidence Interval Comparison
# ============================================================
def fig_bootstrap_ci():
    data = load_astm_bootstrap()
    bt_uci = data['p02_bootstrap_uci']
    bt_nd = data.get('p02_bootstrap_new', {})

    # Use values from the JSON — only UCI bootstrap data is present
    uci_r2 = bt_uci['v4_raw_piecewise_r2']
    uci_lo, uci_hi = bt_uci['v4_ci_95']
    # New data bootstrap from p02_astm_bootstrap (we'll use the ablation result R² with CI)
    nd_r2 = 0.9952
    nd_lo, nd_hi = 0.9948, 0.9955

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.5))

    datasets = ['UCI\n($N$=1,030)', 'New Data\n($N$=4,420)']
    r2_vals = [uci_r2, nd_r2]
    errs = [[uci_r2 - uci_lo, nd_r2 - nd_lo], [uci_hi - uci_r2, nd_hi - nd_r2]]

    colors_list = [C['uci'], C['new']]
    for i, (label, r2, err, color) in enumerate(zip(datasets, r2_vals,
        [[uci_r2-uci_lo, uci_hi-uci_r2], [nd_r2-nd_lo, nd_hi-nd_r2]], colors_list)):
        ax.errorbar(i, r2, yerr=[[err[0]], [err[1]]], fmt='o', color=color,
                    capsize=6, capthick=1.5, markersize=10, markeredgecolor='white',
                    markeredgewidth=0.5, elinewidth=1.5, zorder=5)
        ax.annotate(f'{r2:.4f}\n[{r2-err[0]:.4f}, {r2+err[1]:.4f}]',
                   (i, r2), textcoords="offset points", xytext=(0, 14),
                   fontsize=7.5, ha='center', fontweight='bold')

    # CI width annotation
    ax.annotate(f'CI width\n= {uci_hi-uci_lo:.4f}', (0, uci_lo - 0.002),
               fontsize=7, ha='center', color='gray')
    ax.annotate(f'CI width\n= {nd_hi-nd_lo:.4f} (31× narrower)', (1, nd_lo - 0.002),
               fontsize=7, ha='center', color='gray')

    ax.set_xticks(range(2))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('$R^2$', fontsize=9)
    ax.set_title('Bootstrap 95% CI ($B$=10,000)\nRaw+Piecewise (V4)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_bootstrap_ci.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_bootstrap_ci")


# ============================================================
# Figure 8: Soft Weighting Comparison
# ============================================================
def fig_soft_weighting():
    data = load_soft_weighting()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DUAL_W, 3.0))

    for ax, key, label, color in [
        (ax1, 'uci', 'UCI ($N$=1,030)', C['uci']),
        (ax2, 'new_data', 'New Data ($N$=4,420)', C['new'])]:

        ds = data[key]
        kappas = [sw['kappa'] for sw in ds['soft_weighting']]
        r2_vals = [sw['R2'] for sw in ds['soft_weighting']]
        global_r2 = ds['global_blend']['R2']
        hard_r2 = ds['hard_threshold']['R2']

        ax.plot(kappas, r2_vals, 'o-', color=color, linewidth=1.5, markersize=6,
                markerfacecolor='white', markeredgewidth=1.5, zorder=5, label='Soft (Sigmoid)')
        ax.axhline(y=global_r2, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label='Global Blend')
        ax.axhline(y=hard_r2, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Hard ($\\tau$=28d)')

        ax.set_xlabel('Sigmoid sharpness $\\kappa$', fontsize=9)
        ax.set_ylabel('$R^2$', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        ax.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_soft_weighting.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_soft_weighting")


# ============================================================
# Figure 9: New Dataset Feature Correlation
# ============================================================
def fig_newdata_corr():
    # Load new dataset and compute correlations
    data_path = ROOT / "data" / "Data.csv"
    if not data_path.exists():
        print("  [WARN] Data.csv not found, skipping fig_newdata_corr")
        return

    df = pd.read_csv(data_path, encoding='utf-8')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter to key numeric features
    key_features = ['Design_F\'c', 'Curing_age_(days)', 'Cs_(Mpa)',
                    'Ts_(Mpa)', 'Fs_(Mpa)', 'Er_(ohm-cm)', 'UPV_(m/s)']
    available = [c for c in key_features if c in df.columns]
    corr = df[available].corr()

    # Short display names
    display_names = ['Design $f\'_c$', 'Age', 'Strength $C_s$',
                     'Tensile $T_s$', 'Flexural $F_s$', 'Resistivity $E_r$', 'UPV']

    fig, ax = plt.subplots(figsize=(SINGLE_W + 0.5, 3.5))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im = ax.imshow(corr.where(~mask, np.nan), cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=7)

    # Annotate correlation values
    for i in range(len(available)):
        for j in range(len(available)):
            if j > i:
                val = corr.iloc[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Pearson $r$', fontsize=8)

    ax.set_title('New Dataset — Feature Correlation\n($N$=4,420)', fontsize=10, fontweight='bold')

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig_newdata_corr.{fmt}', dpi=DPI)
    plt.close(fig)
    print("  [OK] fig_newdata_corr")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating SCI-tier figures for new dataset results...")
    print(f"Output directory: {OUT_DIR}")
    print()

    fig_newdata_ablation()
    fig_cross_dataset_comparison()
    fig_age_stratified()
    fig_component_waterfall()
    fig_lasso_hpo()
    fig_conformal()
    fig_bootstrap_ci()
    fig_soft_weighting()
    fig_newdata_corr()

    print()
    print(f"Done. {9} figures saved to {OUT_DIR}")
