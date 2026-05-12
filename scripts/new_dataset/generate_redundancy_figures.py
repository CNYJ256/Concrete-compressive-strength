"""Phase 3.2: 模型冗余可视化 (P2).

生成两张高质量科研图表：
  Fig X: OOF预测散点矩阵（4×4 panels，对角线KDE，下三角scatter，上三角r值）
  Fig Y: 预测误差相关性（残差散点 + 误差相关矩阵heatmap）

符合SCI期刊标准：3.5/7.125 inch宽度，300dpi，Times New Roman，ColorBrewer色板。

用法:
  python scripts/new_dataset/generate_redundancy_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

FIGURES_DIR = PROJECT_ROOT / "figures" / "presentation_highres"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib RC settings for SCI-tier figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# ColorBrewer-compatible color palette (colorblind-friendly, black-white compatible)
CB_PALETTE = ["#2166ac", "#b2182b", "#4daf4a", "#ff7f00"]  # Blue, Red, Green, Orange


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_oof_data():
    """Load individual model OOF predictions from heterogeneous pool data."""
    oof_path = PROJECT_ROOT / "results" / "predictions" / "heterogeneous_pool_oof.csv"
    if not oof_path.exists():
        # Fall back to ablation OOF which has variant-level predictions
        oof_path = PROJECT_ROOT / "results" / "predictions" / "ablation_oof_v2.csv"
        print(f"Heterogeneous pool OOF not found, using ablation OOF: {oof_path}")

    if not oof_path.exists():
        raise FileNotFoundError("No OOF prediction file found")

    df = pd.read_csv(oof_path)
    print(f"Loaded OOF data: {df.shape}")

    # Try to get individual model predictions
    # From heterogeneous_pool_oof.csv: XGB_primary, LGB_primary, HGB_primary, HGB_anchor, ...
    # From ablation_oof_v2.csv: only variant predictions
    model_cols = [c for c in df.columns if c not in ("y_true", "age")
                  and not c.startswith("v") and not c.startswith("paper1")]

    if len(model_cols) >= 4:
        models = model_cols[:4]  # Take first 4 models
    else:
        # Need to train models fresh
        print("Individual model OOF not available; training fresh...")
        return load_and_train_models()

    y = df["y_true"].to_numpy()
    preds = {m: df[m].to_numpy() for m in models}
    return y, preds


def load_and_train_models():
    """Train individual models and generate OOF predictions."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from concrete_compressive_strength.core import (
        BASE_FEATURES, TARGET_COL, RANDOM_STATE,
        BASE_MODEL_PARAMS, ANCHOR_MODEL_PARAMS,
        build_xgb, build_lgbm, build_hgb, load_data,
        feature_engineering, feature_engineering_anchor,
    )
    from sklearn.model_selection import KFold, cross_val_predict

    df = load_data(PROJECT_ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_base).to_numpy(dtype=float)
    X_anchor = feature_engineering_anchor(X_base).to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    preds = {}
    preds["XGB_primary"] = cross_val_predict(
        build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary, y, cv=cv, n_jobs=-1, method="predict")
    preds["LGB_primary"] = cross_val_predict(
        build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary, y, cv=cv, n_jobs=-1, method="predict")
    preds["HGB_primary"] = cross_val_predict(
        build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary, y, cv=cv)
    preds["HGB_anchor"] = cross_val_predict(
        build_hgb(ANCHOR_MODEL_PARAMS), X_anchor, y, cv=cv)

    print("Trained 4 individual models on UCI data")
    return y, preds


# ---------------------------------------------------------------------------
# Figure 1: OOF Prediction Scatter Matrix
# ---------------------------------------------------------------------------
def plot_oof_scatter_matrix(y, preds):
    """Generate 4×4 scatter matrix of OOF predictions."""
    model_names = list(preds.keys())
    n = len(model_names)

    fig = plt.figure(figsize=(7.125, 6.5))

    for i in range(n):
        for j in range(n):
            ax = plt.subplot(n, n, i * n + j + 1)
            xi = preds[model_names[i]]
            xj = preds[model_names[j]]

            if i == j:
                # Diagonal: KDE
                kde = gaussian_kde(xi)
                x_range = np.linspace(xi.min() - 5, xi.max() + 5, 200)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color=CB_PALETTE[0])
                ax.plot(x_range, kde(x_range), color=CB_PALETTE[0], linewidth=0.8)
                ax.set_xlim(xi.min() - 5, xi.max() + 5)
                # Add model name
                ax.text(0.5, 0.9, model_names[i].replace("_", " "), transform=ax.transAxes,
                        ha="center", va="top", fontsize=6, fontweight="bold")
            elif j < i:
                # Lower triangle: scatter
                ax.scatter(xj, xi, alpha=0.15, s=2, color=CB_PALETTE[0], edgecolors="none", rasterized=True)
                # y=x reference line
                lims = [min(xj.min(), xi.min()) - 2, max(xj.max(), xi.max()) + 2]
                ax.plot(lims, lims, "--", color="gray", linewidth=0.5, alpha=0.5)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
            else:
                # Upper triangle: Pearson r value
                r, _ = pearsonr(xi, xj)
                ax.text(0.5, 0.5, f"r = {r:.4f}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color=CB_PALETTE[1] if r < 0.999 else CB_PALETTE[0])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Clean up axes
            if j > 0 and not (j < i):
                ax.set_yticks([])
            if i < n - 1 and not (j < i):
                ax.set_xticks([])

            if i == n - 1 and j < i:
                ax.set_xlabel(model_names[j].replace("_", " "), fontsize=6)
            if j == 0 and i > j:
                ax.set_ylabel(model_names[i].replace("_", " "), fontsize=6)

    plt.suptitle("OOF Prediction Scatter Matrix (H1: GBDT Pool, UCI Dataset)",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path_pdf = FIGURES_DIR / "fig_oof_scatter_matrix.pdf"
    fig_path_png = FIGURES_DIR / "fig_oof_scatter_matrix.png"
    fig.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    print(f"Figure 1 saved to {fig_path_pdf} and {fig_path_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Prediction Error Correlation
# ---------------------------------------------------------------------------
def plot_error_correlation(y, preds):
    """Generate error correlation heatmap + representative residual scatter."""
    model_names = list(preds.keys())
    n = len(model_names)

    # Compute residuals
    residuals = {}
    for name in model_names:
        residuals[name] = preds[name] - y

    fig = plt.figure(figsize=(7.125, 3.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2])

    # Left panel: Error correlation heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    err_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            err_corr[i, j], _ = pearsonr(residuals[model_names[i]], residuals[model_names[j]])

    im = ax1.imshow(err_corr, cmap="RdBu_r", vmin=0.9, vmax=1.0, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if err_corr[i, j] < 0.97 else "black"
            ax1.text(j, i, f"{err_corr[i, j]:.4f}", ha="center", va="center", fontsize=7, color=color)

    labels = [m.replace("_", " ") for m in model_names]
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_title("Error Correlation Matrix", fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Pearson r", fontsize=6)

    # Right panel: Residual scatter for highest and lowest correlation pairs
    ax2 = fig.add_subplot(gs[0, 1])

    # Find highest and lowest off-diagonal error correlations
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((err_corr[i, j], model_names[i], model_names[j]))
    pairs.sort(key=lambda x: x[0])

    # Plot lowest correlation pair
    r_low, m1_low, m2_low = pairs[0]
    ax2.scatter(residuals[m1_low], residuals[m2_low], alpha=0.3, s=3,
                color=CB_PALETTE[0], edgecolors="none", rasterized=True,
                label=f"{m1_low.replace('_',' ')} vs {m2_low.replace('_',' ')}\nr = {r_low:.4f}")

    # Plot highest correlation pair
    r_high, m1_high, m2_high = pairs[-1]
    ax2.scatter(residuals[m1_high], residuals[m2_high], alpha=0.3, s=3,
                color=CB_PALETTE[1], edgecolors="none", rasterized=True,
                label=f"{m1_high.replace('_',' ')} vs {m2_high.replace('_',' ')}\nr = {r_high:.4f}")

    # y=x line
    all_res = np.concatenate(list(residuals.values()))
    lim = max(abs(all_res.min()), abs(all_res.max())) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], "--", color="gray", linewidth=0.5)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

    ax2.set_xlabel("Residual (MPa)", fontsize=7)
    ax2.set_ylabel("Residual (MPa)", fontsize=7)
    ax2.set_title("Pairwise Residual Comparison", fontsize=8, fontweight="bold")
    ax2.legend(fontsize=6, loc="upper left", framealpha=0.8)

    # Annotation
    ax2.text(0.98, 0.02,
             "Models make\nsimilar errors",
             transform=ax2.transAxes, fontsize=6,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig_path_pdf = FIGURES_DIR / "fig_error_correlation.pdf"
    fig_path_png = FIGURES_DIR / "fig_error_correlation.png"
    fig.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    print(f"Figure 2 saved to {fig_path_pdf} and {fig_path_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 3.2: Model Redundancy Visualization")
    print("=" * 60)

    # Load data
    y, preds = load_oof_data()
    print(f"Models: {list(preds.keys())}")
    for name, p in preds.items():
        r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
        print(f"  {name}: R2={r2:.6f}")

    # Compute inter-model correlations
    names = list(preds.keys())
    print("\nInter-model correlations:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, _ = pearsonr(preds[names[i]], preds[names[j]])
            print(f"  {names[i]} vs {names[j]}: r={r:.6f}")

    # Generate figures
    print("\nGenerating Figure 1: OOF Scatter Matrix...")
    plot_oof_scatter_matrix(y, preds)

    print("\nGenerating Figure 2: Error Correlation...")
    plot_error_correlation(y, preds)

    print("\nDone. Figures saved to figures/presentation_highres/")


if __name__ == "__main__":
    main()
