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
import json
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


def load_ablation_results():
    path = ROOT / "results" / "metrics" / "ablation_results_acdcb_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ablation metrics: {path}. Run scripts/eval/ablation_acdcb_v2.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_ablation_oof():
    path = ROOT / "results" / "predictions" / "ablation_oof_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ablation OOF predictions: {path}. Run scripts/eval/ablation_acdcb_v2.py first.")
    return pd.read_csv(path)


def variant_metrics(results: dict, variant_key: str) -> dict:
    variants = results["variants"]
    if variant_key not in variants:
        raise KeyError(f"Variant not found in ablation JSON: {variant_key}")
    return variants[variant_key]["metrics"]


def load_shap_importance():
    path = ROOT / "results" / "metrics" / "shap_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing SHAP metrics: {path}. Run scripts/eval/shap_analysis.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["mean_abs_shap"]


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

    BOX_FACE = "white"
    BOX_EDGE = "#222222"
    BOX_TEXT = "#111111"

    def draw_box(x, y, w, h, text, fontsize=7.5):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08", facecolor=BOX_FACE,
                             edgecolor=BOX_EDGE, linewidth=1.0)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=BOX_TEXT, fontweight="bold")

    def arrow(x1, y1, x2, y2, lw=0.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.2",
                                    color="#333333", lw=lw))

    ax.text(5, 12.7, "ACDCB Architecture", ha="center", fontsize=9, fontweight="bold")

    # Input (enlarged)
    draw_box(5, 12.0, 7.5, 0.6,
             "Features: Cement, Slag, Fly Ash, Water, SP, Coarse, Fine, Age",
             fontsize=7)

    # Dual spaces — separated: Primary at 2.7, Anchor at 7.3
    arrow(2.7, 11.68, 2.7, 11.1, lw=1.2)
    arrow(7.3, 11.68, 7.3, 11.1, lw=1.2)
    draw_box(2.7, 10.6, 3.3, 0.9,
             "Primary Space (32-dim)\nBase + Mech + Enhanced\nExpression-Oriented",
             fontsize=6.5)
    draw_box(7.3, 10.6, 3.3, 0.9,
             "Anchor Space (22-dim)\nBase + Mechanistic\nRobustness-Oriented",
             fontsize=6.5)

    # Model pool — positioned under respective spaces
    model_x = [1.7, 3.7, 6.3, 8.3]
    model_labels = ["XGBoost\nPrimary", "LightGBM\nPrimary", "HGB\nPrimary", "HGB_Anchor\nAnchor"]
    for mx, label in zip(model_x, model_labels):
        arrow(mx, 10.13, mx, 9.65, lw=0.8)
        draw_box(mx, 9.25, 1.6, 0.7, label, fontsize=6.5)

    # OOF matrix (enlarged)
    for mx in model_x:
        arrow(mx, 8.88, 5, 8.38, lw=0.6)
    draw_box(5, 8.05, 6.5, 0.6,
             "OOF Prediction Matrix P in R^{Nx4} (10-fold CV)",
             fontsize=7)

    # Age split
    arrow(5, 7.73, 5, 7.25, lw=1.0)
    draw_box(5, 6.95, 5.0, 0.5,
             "Age-Conditioned Split: t <= 28d | t > 28d",
             fontsize=7)

    # Piecewise weights — separated: Early at 2.7, Late at 7.3
    arrow(2.7, 6.68, 2.7, 6.20, lw=1.0)
    arrow(7.3, 6.68, 7.3, 6.20, lw=1.0)

    draw_box(2.7, 5.65, 3.4, 1.0,
             "Early-Age Blend (w_e)\nmin RMSE s.t. w_i>=0, sum=1\nw_e=[0.314,0.040,0.013,0.633]",
             fontsize=6.5)
    draw_box(7.3, 5.65, 3.4, 1.0,
             "Late-Age Blend (w_l)\nmin RMSE s.t. w_i>=0, sum=1\nw_l=[0.463,0.000,0.122,0.415]",
             fontsize=6.5)

    # Constraint legend
    ax.text(5, 4.75,
            r"$\mathcal{W}=\{\mathbf{w}\mid w_i\!\geq\!0,\sum\!w_i\!=\!1\}$  Optim: SLSQP, Obj: RMSE, Sel: R$^2$-first",
            ha="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999"))

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "fig_architecture.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_architecture.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_architecture")


# ============================================================
# Figure: Scatter — AdaBoost
# ============================================================
def fig_scatter_adaboost():
    oof = load_ablation_oof()
    results = load_ablation_results()
    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred = oof["paper1_adaboost"].to_numpy(dtype=float)
    metrics = variant_metrics(results, "paper1_adaboost")
    rmse_val = metrics["RMSE_mean"]
    r2_val = metrics["R2_mean"]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, c="#4C78A8", edgecolor="none")
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1.2, linestyle="--", label=r"$y=x$")
    grid = np.linspace(lo, hi, 100)
    ax.fill_between(grid, grid - rmse_val, grid + rmse_val,
                    alpha=0.07, color="gray", label=f"+/-1 RMSE ({rmse_val:.1f} MPa)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Strength (MPa)", fontsize=9)
    ax.set_ylabel("Predicted (MPa)", fontsize=9)
    ax.set_title("AdaBoost Baseline (10-fold OOF)", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)
    ax.text(0.03, 0.97, f"R$^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.2f} MPa",
            transform=ax.transAxes, fontsize=8, va="top",
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
    oof = load_ablation_oof()
    results = load_ablation_results()
    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred = oof["v3_dualspace_age_piecewise_acdcb"].to_numpy(dtype=float)
    metrics = variant_metrics(results, "v3_dualspace_age_piecewise_acdcb")
    rmse_val = metrics["RMSE_mean"]
    r2_val = metrics["R2_mean"]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, c="#54A24B", edgecolor="none")
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1.2, linestyle="--", label=r"$y=x$")
    grid = np.linspace(lo, hi, 100)
    ax.fill_between(grid, grid - rmse_val, grid + rmse_val,
                    alpha=0.07, color="gray", label=f"+/-1 RMSE ({rmse_val:.1f} MPa)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Strength (MPa)", fontsize=9)
    ax.set_ylabel("Predicted (MPa)", fontsize=9)
    ax.set_title("ACDCB (10-fold OOF)", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)
    ax.text(0.03, 0.97, f"R$^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.2f} MPa",
            transform=ax.transAxes, fontsize=8, va="top",
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
    results = load_ablation_results()
    variant_keys = [
        "paper1_adaboost",
        "v1_primary_global_no_anchor",
        "v2_dualspace_global",
        "v3_dualspace_age_piecewise_acdcb",
        "v4_raw_age_piecewise",
    ]
    variants = ["V0\nAdaBoost", "V1\nPrimary\n+Global", "V2\nDualSpace\n+Global",
                "V3\nACDCB\n(Full)", "V4\nRaw+\nPiecewise"]
    x = np.arange(len(variants))
    r2 = [variant_metrics(results, key)["R2_mean"] for key in variant_keys]
    rmse = [variant_metrics(results, key)["RMSE_mean"] for key in variant_keys]
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(IEEE_WIDTH, 2.8))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width/2, r2, width, color="#4C78A8", alpha=0.88, label=r"$R^2$", zorder=3)
    b2 = ax2.bar(x + width/2, rmse, width, color="#E45756", alpha=0.82, label="RMSE (MPa)", zorder=3)

    # Highlight V3
    b1[3].set_edgecolor("#333333")
    b1[3].set_linewidth(1.5)
    b1[3].set_zorder(5)
    b2[3].set_edgecolor("#333333")
    b2[3].set_linewidth(1.5)
    b2[3].set_zorder(5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=7)
    ax1.set_ylabel(r"$R^2$", color="#4C78A8", fontsize=9)
    ax2.set_ylabel("RMSE (MPa)", color="#E45756", fontsize=9)
    ax1.set_title("Ablation: R$^2$ and RMSE (10-fold)", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.18, zorder=0)
    # Merged legend from both axes — force ax1 above ax2 so legend renders on top
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=7)
    legend.set_zorder(20)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "fig_ablation_r2_rmse.pdf", **SAVE_KWARGS)
    fig.savefig(OUT_DIR / "fig_ablation_r2_rmse.png", **SAVE_KWARGS)
    plt.close(fig)
    print("[OK] fig_ablation_r2_rmse")


# ============================================================
# Figure: Ablation Radar Chart
# ============================================================
def fig_ablation_radar():
    results = load_ablation_results()
    variant_keys = [
        "paper1_adaboost",
        "v1_primary_global_no_anchor",
        "v2_dualspace_global",
        "v3_dualspace_age_piecewise_acdcb",
        "v4_raw_age_piecewise",
    ]
    metrics = [variant_metrics(results, key) for key in variant_keys]
    r2_vals = [m["R2_mean"] for m in metrics]
    rmse_vals = [m["RMSE_mean"] for m in metrics]
    mae_vals = [m["MAE_mean"] for m in metrics]
    mape_vals = [m["MAPE_mean"] for m in metrics]

    # Normalize: all to [0,1] where 1 is best
    def normalize_high(values):
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo + 1e-12) for v in values]

    def normalize_low(values):
        lo, hi = min(values), max(values)
        return [(hi - v) / (hi - lo + 1e-12) for v in values]

    r2_n = normalize_high(r2_vals)
    rmse_n = normalize_low(rmse_vals)
    mae_n = normalize_low(mae_vals)
    mape_n = normalize_low(mape_vals)

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
    importance_rows = load_shap_importance()
    features = [row["display_name"] for row in importance_rows]
    importance = [row["relative_importance"] for row in importance_rows]
    idx_sorted = np.argsort(importance)
    feats_sorted = [features[i] for i in idx_sorted]
    imp_sorted = [importance[i] for i in idx_sorted]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, 3.0))
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(imp_sorted)))
    bars = ax.barh(range(len(feats_sorted)), imp_sorted, color=colors, edgecolor="#333333",
                    linewidth=0.4, height=0.7)
    top_n = min(3, len(bars))
    for i in range(1, top_n + 1):
        bars[-i].set_edgecolor("#E45756")
        bars[-i].set_linewidth(1.2)

    ax.set_yticks(range(len(feats_sorted)))
    ax.set_yticklabels(feats_sorted, fontsize=6.5)
    ax.set_xlabel("Relative mean |SHAP|", fontsize=9)
    ax.set_title("Raw XGBoost SHAP Importance", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    for bar, val in zip(bars, imp_sorted):
        if val >= 0.05:
            ax.text(val + 0.003, bar.get_y() + 0.35, f"{val:.3f}", fontsize=6)

    ax.set_xlim(0, max(imp_sorted) * 1.12)

    ax.text(0.95, 0.04,
            f"Red border: Top-{top_n}",
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
    y = np.arange(len(models))
    height = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_WIDTH, 2.5))

    b1 = ax1.barh(y, r2, height, color="#4C78A8", alpha=0.85, label=r"$R^2$", zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=6)
    ax1.set_xlabel(r"$R^2$", fontsize=7)
    ax1.set_title("$R^2$", fontsize=7, fontweight="bold")
    ax1.tick_params(axis='x', labelsize=6)
    ax1.grid(axis="x", alpha=0.18)
    ax1.legend(loc="lower right", fontsize=5.5).set_zorder(10)

    b2 = ax2.barh(y, rmse, height, color="#E45756", alpha=0.82, label="RMSE (MPa)", zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(models, fontsize=6)
    ax2.set_xlabel("RMSE (MPa)", fontsize=7)
    ax2.set_title("RMSE", fontsize=7, fontweight="bold")
    ax2.tick_params(axis='x', labelsize=6)
    ax2.grid(axis="x", alpha=0.18)
    ax2.legend(loc="lower right", fontsize=5.5).set_zorder(10)

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
    y = np.arange(len(models))
    height = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_WIDTH, 2.5))

    b1 = ax1.barh(y, mae, height, color="#F58518", alpha=0.85, label="MAE (MPa)", zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=6)
    ax1.set_xlabel("MAE (MPa)", fontsize=7)
    ax1.set_title("MAE", fontsize=7, fontweight="bold")
    ax1.tick_params(axis='x', labelsize=6)
    ax1.grid(axis="x", alpha=0.18, zorder=0)
    ax1.legend(loc="lower right", fontsize=5.5).set_zorder(10)

    b2 = ax2.barh(y, mape, height, color="#54A24B", alpha=0.82, label="MAPE (%)", zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(models, fontsize=6)
    ax2.set_xlabel("MAPE (%)", fontsize=7)
    ax2.set_title("MAPE", fontsize=7, fontweight="bold")
    ax2.tick_params(axis='x', labelsize=6)
    ax2.grid(axis="x", alpha=0.18, zorder=0)
    ax2.legend(loc="lower right", fontsize=5.5).set_zorder(10)

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
