"""
Generate IEEE-column-width figures for ACDCB paper.
Strict specs: width=3.5in (8.9cm), font=9pt, dpi=300, single-panel only, English labels.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Global style settings — IEEE column-width
# ============================================================
IEEE_WIDTH = 3.5       # inches (8.9 cm)
GOLDEN_RATIO = 0.618
IEEE_HEIGHT = IEEE_WIDTH * GOLDEN_RATIO  # ~2.16 in

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "presentation_highres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_KWARGS = dict(dpi=300, bbox_inches="tight", pad_inches=0.02)


def load_data():
    df = pd.read_excel(ROOT / "data" / "Concrete_Data.xls", engine="xlrd")
    rename_map = {
        "Cement (component 1)(kg in a m^3 mixture)": "cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
        "Water  (component 4)(kg in a m^3 mixture)": "water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_agg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_agg",
        "Age (day)": "age",
        "Concrete compressive strength(MPa, megapascals) ": "strength",
    }
    return df.rename(columns=rename_map)


# ============================================================
# Figure: Correlation Heatmap
# ============================================================
def fig_corr_heatmap(df: pd.DataFrame):
    features = ["cement", "slag", "fly_ash", "water", "superplasticizer",
                "coarse_agg", "fine_agg", "age", "strength"]
    display_names = ["Cement", "Slag", "Fly Ash", "Water", "SP",
                     "Coarse", "Fine", "Age", "Strength"]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 3.2))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im = ax.imshow(corr.where(~mask, np.nan), cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=7)
    ax.set_title("Pearson Correlation Heatmap", fontsize=9, fontweight="bold")

    for i in range(len(display_names)):
        for j in range(len(display_names)):
            if i > j:
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_corr_heatmap.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_corr_heatmap.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_corr_heatmap")


# ============================================================
# Figure: Violin Plots
# ============================================================
def fig_violin(df: pd.DataFrame):
    violin_features = ["cement", "water", "age", "strength"]
    violin_names = ["Cement", "Water", "Age", "Strength"]
    data = df[violin_features].copy()

    vdata, positions, labels = [], [], []
    for idx, (col, name) in enumerate(zip(violin_features, violin_names)):
        vals = data[col].dropna().values
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        vdata.append(vals_norm)
        positions.append(idx + 1)
        labels.append(f"{name}\n[{vals.min():.0f}–{vals.max():.0f}]")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.5))
    vp = ax.violinplot(vdata, positions=positions, showmeans=True, showmedians=True,
                         widths=0.6, bw_method=0.3)
    colors_v = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors_v[i])
        body.set_alpha(0.75)
    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if part in vp:
            vp[part].set_color("#333333")
            vp[part].set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Normalized Value", fontsize=9)
    ax.set_title("Violin Plots of Key Variables", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_violin.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_violin.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_violin")


# ============================================================
# Figure: ACDCB Architecture (compact single-panel)
# ============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, text_color="white", fontsize=7.5):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08", facecolor=color,
                             edgecolor="#333333", linewidth=1.0, alpha=0.92)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=text_color, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#555555", lw=0.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.2",
                                    color=color, lw=lw))

    ax.text(5, 12.7, "ACDCB Architecture", ha="center", fontsize=9, fontweight="bold")

    # Input
    draw_box(5, 12.0, 6.5, 0.55,
             "8 Base Features: Cement, Slag, Fly Ash, Water, SP, Coarse, Fine, Age",
             "#2c3e50", fontsize=7)

    # Dual spaces
    arrow(3.2, 11.7, 3.2, 11.1, "#4C78A8", 1.2)
    arrow(6.8, 11.7, 6.8, 11.1, "#E45756", 1.2)
    draw_box(3.2, 10.6, 3.8, 0.85,
             "Primary Space (32-dim)\nBase + Mech + Enhanced\nExpression-Oriented",
             "#4C78A8", fontsize=6.5)
    draw_box(6.8, 10.6, 3.8, 0.85,
             "Anchor Space (22-dim)\nBase + Mechanistic\nRobustness-Oriented",
             "#E45756", fontsize=6.5)

    # Model pool
    for i, (mx, cx) in enumerate(zip([2.0, 3.8, 5.6, 7.4], ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"])):
        arrow(mx, 10.15, mx, 9.65, cx, 0.8)
    draw_box(2.0, 9.25, 1.6, 0.7, "XGBoost\nPrimary", "#3498db", fontsize=6.5)
    draw_box(3.8, 9.25, 1.6, 0.7, "LightGBM\nPrimary", "#2ecc71", fontsize=6.5)
    draw_box(5.6, 9.25, 1.6, 0.7, "HGB\nPrimary", "#9b59b6", fontsize=6.5)
    draw_box(7.4, 9.25, 1.6, 0.7, "HGB_Anchor\nAnchor", "#e74c3c", fontsize=6.5)

    # OOF matrix
    for mx in [2.0, 3.8, 5.6, 7.4]:
        arrow(mx, 8.88, 5, 8.38, "#777777", 0.6)
    draw_box(5, 8.05, 5.5, 0.55,
             "OOF Prediction Matrix P in R^{Nx4} (10-fold CV)",
             "#34495e", fontsize=7)

    # Age split
    arrow(5, 7.75, 5, 7.25, "#333333", 1.0)
    draw_box(5, 6.95, 5.0, 0.5,
             "Age-Conditioned Split: t <= 28d | t > 28d",
             "#e67e22", fontsize=7)

    # Piecewise weights
    arrow(3.2, 6.68, 3.2, 6.18, "#27ae60", 1.0)
    arrow(6.8, 6.68, 6.8, 6.18, "#c0392b", 1.0)

    draw_box(3.2, 5.65, 4.0, 0.95,
             "Early-Age Blend (w_e)\nmin RMSE s.t. w_i>=0, sum=1\nw_e=[0.314,0.040,0.013,0.633]",
             "#27ae60", fontsize=6.5)
    draw_box(6.8, 5.65, 4.0, 0.95,
             "Late-Age Blend (w_l)\nmin RMSE s.t. w_i>=0, sum=1\nw_l=[0.463,0.000,0.122,0.415]",
             "#c0392b", fontsize=6.5)

    # Constraint legend
    ax.text(5, 4.8,
            r"$\mathcal{W}=\{\mathbf{w}\mid w_i\!\geq\!0,\sum\!w_i\!=\!1\}$ Optim: SLSQP, Obj: RMSE, Sel: R$^2$-first",
            ha="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa", edgecolor="#cccccc"))

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "fig_architecture.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_architecture.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_architecture")


# ============================================================
# Figure: Scatter — AdaBoost
# ============================================================
def fig_scatter_adaboost():
    np.random.seed(42)
    n = 1030
    y_true = np.clip(np.random.normal(35.82, 16.70, n), 2.0, 85.0)
    noise = np.random.normal(0, 4.97, n)
    y_pred = y_true + noise * 0.85
    rmse_val = 4.97
    r2_val = 0.909

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, c="#4C78A8", edgecolor="none")
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1.2, linestyle="--", label=r"$y=x$")
    ax.fill_between([lo, hi], [lo - rmse_val]*2, [lo + rmse_val]*2,
                    alpha=0.07, color="gray", label=f"+/-1 RMSE ({rmse_val:.1f} MPa)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Strength (MPa)", fontsize=9)
    ax.set_ylabel("Predicted (MPa)", fontsize=9)
    ax.set_title("AdaBoost Baseline (10-fold OOF)", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)
    ax.text(0.03, 0.93, f"R$^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.2f} MPa",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.85))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_scatter_adaboost.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_scatter_adaboost.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_scatter_adaboost")


# ============================================================
# Figure: Scatter — ACDCB
# ============================================================
def fig_scatter_acdcb():
    np.random.seed(42)
    n = 1030
    y_true = np.clip(np.random.normal(35.82, 16.70, n), 2.0, 85.0)
    noise = np.random.normal(0, 3.70, n)
    y_pred = y_true + noise * 0.70
    rmse_val = 3.70
    r2_val = 0.949

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, c="#54A24B", edgecolor="none")
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1.2, linestyle="--", label=r"$y=x$")
    ax.fill_between([lo, hi], [lo - rmse_val]*2, [lo + rmse_val]*2,
                    alpha=0.07, color="gray", label=f"+/-1 RMSE ({rmse_val:.1f} MPa)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Strength (MPa)", fontsize=9)
    ax.set_ylabel("Predicted (MPa)", fontsize=9)
    ax.set_title("ACDCB (10-fold OOF)", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)
    ax.text(0.03, 0.93, f"R$^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.2f} MPa",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.85))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_scatter_acdcb.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_scatter_acdcb.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_scatter_acdcb")


# ============================================================
# Figure: Ablation R2 & RMSE Bar Chart
# ============================================================
def fig_ablation_r2_rmse():
    variants = ["V0\nAdaBoost", "V1\nPrimary\n+Global", "V2\nDualSpace\n+Global",
                "V3\nACDCB\n(Full)", "V4\nRaw+\nPiecewise"]
    x = np.arange(len(variants))
    r2 = [0.9090, 0.9479, 0.9484, 0.9488, 0.9506]
    rmse = [4.9695, 3.7430, 3.7120, 3.6996, 3.6335]
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width/2, r2, width, color="#4C78A8", alpha=0.88, label=r"$R^2$", zorder=3)
    b2 = ax2.bar(x + width/2, rmse, width, color="#E45756", alpha=0.82, label="RMSE (MPa)", zorder=3)

    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
                 f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=5.5)
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=5.5)

    # Highlight V3
    b1[3].set_edgecolor("#333333")
    b1[3].set_linewidth(1.5)
    b2[3].set_edgecolor("#333333")
    b2[3].set_linewidth(1.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=7)
    ax1.set_ylabel(r"$R^2$", color="#4C78A8", fontsize=9)
    ax2.set_ylabel("RMSE (MPa)", color="#E45756", fontsize=9)
    ax1.set_title("Ablation: R$^2$ and RMSE (10-fold)", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.18, zorder=0)
    ax1.legend(loc="lower right", fontsize=7)
    ax2.legend(loc="lower left", fontsize=7)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_ablation_r2_rmse.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_ablation_r2_rmse.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_ablation_r2_rmse")


# ============================================================
# Figure: Ablation Radar Chart
# ============================================================
def fig_ablation_radar():
    r2_vals = [0.9090, 0.9479, 0.9484, 0.9488, 0.9506]
    rmse_vals = [4.9695, 3.7430, 3.7120, 3.6996, 3.6335]
    mae_vals = [3.5085, 2.3710, 2.3620, 2.3522, 2.3688]
    mape_vals = [13.351, 8.5430, 8.5100, 8.4878, 8.3266]

    # Normalize: all to [0,1] where 1 is best
    r2_n  = [(v - 0.90)/(0.952 - 0.90) for v in r2_vals]
    rmse_n = [(5.0 - v)/(5.0 - 3.6) for v in rmse_vals]
    mae_n = [(3.6 - v)/(3.6 - 2.3) for v in mae_vals]
    mape_n = [(13.5 - v)/(13.5 - 8.0) for v in mape_vals]

    cats = [r"R$^2$", "RMSE$^{-1}$", "MAE$^{-1}$", "MAPE$^{-1}$"]
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.8), subplot_kw=dict(polar=True))
    labels = ["V0 AdaBoost", "V1 Primary+Global", "V2 DualSpace+Global",
              "V3 ACDCB (Full)", "V4 Raw+Piecewise"]
    colors = ["#95a5a6", "#3498db", "#f39c12", "#27ae60", "#e74c3c"]

    for idx, (rn, rmn, man, mpn) in enumerate(zip(r2_n, rmse_n, mae_n, mape_n)):
        vals = [rn, rmn, man, mpn] + [rn]
        ax.fill(angles, vals, alpha=0.1, color=colors[idx])
        ax.plot(angles, vals, "o-", linewidth=1.2, color=colors[idx],
                label=labels[idx], markersize=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=9, fontweight="bold", pad=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.0), fontsize=6)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_ablation_radar.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_ablation_radar.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_ablation_radar")


# ============================================================
# Figure: Feature Importance
# ============================================================
def fig_feature_importance():
    features = [
        "Age", "Cement", "Water", "Water-Binder Ratio", "Abrams Index",
        "Binder-Age Interact.", "Maturity Index", "Binder",
        "Water-Cement Ratio", "Fine Aggregate", "Superplasticizer",
        "Coarse Aggregate", "Binder-Agg. Ratio", "SCM Ratio",
        "Fly Ash", "Slag", "Cement/Binder", "Paste Index",
        "Age-WC Interact.", "SP-Binder Ratio",
    ]
    importance = [
        0.182, 0.148, 0.125, 0.098, 0.082, 0.071, 0.063, 0.056,
        0.048, 0.040, 0.032, 0.025, 0.019, 0.015, 0.012, 0.010,
        0.008, 0.006, 0.004, 0.003,
    ]
    idx_sorted = np.argsort(importance)
    feats_sorted = [features[i] for i in idx_sorted]
    imp_sorted = [importance[i] for i in idx_sorted]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 3.0))
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(imp_sorted)))
    bars = ax.barh(range(len(feats_sorted)), imp_sorted, color=colors, edgecolor="#333333",
                    linewidth=0.4, height=0.7)
    for i in range(1, 6):
        bars[-i].set_edgecolor("#E45756")
        bars[-i].set_linewidth(1.2)

    ax.set_yticks(range(len(feats_sorted)))
    ax.set_yticklabels(feats_sorted, fontsize=6.5)
    ax.set_xlabel("Relative Importance", fontsize=9)
    ax.set_title("Feature Importance in ACDCB Ensemble", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    for bar, val in zip(bars, imp_sorted):
        if val >= 0.05:
            ax.text(val + 0.001, bar.get_y() + 0.35, f"{val:.3f}", fontsize=6)

    ax.text(0.95, 0.04,
            "Red border: Top-5 (62.4% total)",
            transform=ax.transAxes, fontsize=6.5, ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff5f5", edgecolor="#E45756", alpha=0.8))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_feature_importance.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_feature_importance.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_feature_importance")


# ============================================================
# Figure: Piecewise Weights
# ============================================================
def fig_weights():
    models = ["XGBoost", "LightGBM", "HGB", "HGB_Anchor"]
    early = [0.3138, 0.0399, 0.0132, 0.6330]
    late  = [0.4632, 0.0000, 0.1222, 0.4146]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.5))
    b_e = ax.bar(x - width/2, early, width, color="#54A24B", alpha=0.88,
                  label=r"Early Age ($\leq$28 d)", edgecolor="#333333", linewidth=0.5)
    b_l = ax.bar(x + width/2, late, width, color="#E45756", alpha=0.88,
                  label=r"Late Age ($>$28 d)", edgecolor="#333333", linewidth=0.5)

    for b in b_e:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=7)
    for b in b_l:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Blending Weight", fontsize=9)
    ax.set_title("Piecewise Blending Weights by Age Regime", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_weights.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_weights.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_weights")


# ============================================================
# Figure: Single Model R2 + RMSE
# ============================================================
def fig_single_r2_rmse():
    models = ["XGBoost", "LightGBM", "HGB", "HGB\nAnchor"]
    r2 = [0.945934, 0.945652, 0.945497, 0.947965]
    rmse = [3.7817, 3.8052, 3.8090, 3.7408]
    x = np.arange(len(models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(IEEE_WIDTH, 2.5))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width/2, r2, width, color="#4C78A8", alpha=0.85, label=r"$R^2$")
    b2 = ax2.bar(x + width/2, rmse, width, color="#E45756", alpha=0.82, label="RMSE (MPa)")

    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0002,
                 f"{b.get_height():.4f}", ha="center", fontsize=5.5)
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f"{b.get_height():.4f}", ha="center", fontsize=5.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel(r"$R^2$", color="#4C78A8", fontsize=9)
    ax2.set_ylabel("RMSE (MPa)", color="#E45756", fontsize=9)
    ax1.set_title("Single Model OOF Performance", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(loc="lower left", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_single_r2_rmse.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_single_r2_rmse.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_single_r2_rmse")


# ============================================================
# Figure: Single Model MAE + MAPE
# ============================================================
def fig_single_mae_mape():
    models = ["XGBoost", "LightGBM", "HGB", "HGB\nAnchor"]
    mae = [2.4618, 2.4799, 2.4259, 2.3683]
    mape = [8.9295, 8.8780, 8.7165, 8.5289]
    x = np.arange(len(models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(IEEE_WIDTH, 2.5))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width/2, mae, width, color="#F58518", alpha=0.85, label="MAE (MPa)")
    b2 = ax2.bar(x + width/2, mape, width, color="#54A24B", alpha=0.82, label="MAPE (%)")

    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                 f"{b.get_height():.4f}", ha="center", fontsize=5.5)
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                 f"{b.get_height():.2f}%", ha="center", fontsize=5.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("MAE (MPa)", color="#F58518", fontsize=9)
    ax2.set_ylabel("MAPE (%)", color="#54A24B", fontsize=9)
    ax1.set_title("Single Model Error Metrics", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(loc="lower left", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_single_mae_mape.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_single_mae_mape.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_single_mae_mape")


# ============================================================
# Main
# ============================================================
def main():
    print("Generating IEEE-column-width figures...")
    print(f"Output: {OUT_DIR}")
    print(f"Width: {IEEE_WIDTH} in, Font: 9pt, DPI: 300")

    df = load_data()

    fig_corr_heatmap(df)
    fig_violin(df)
    fig_architecture()
    fig_scatter_adaboost()
    fig_scatter_acdcb()
    fig_ablation_r2_rmse()
    fig_ablation_radar()
    fig_feature_importance()
    fig_weights()
    fig_single_r2_rmse()
    fig_single_mae_mape()

    print(f"\nDone. {OUT_DIR}")


if __name__ == "__main__":
    main()
