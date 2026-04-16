from __future__ import annotations

"""ACDCB 论文图表生成脚本（高可读性自适应版）。

输入：
- data/Concrete_Data.xls
- doc/baseline_results.json
- doc/ablation_results_acdcb.json
- doc/ablation_oof_predictions.csv

输出：
- figures/*.png
- figures/figure_index.json（记录绘图决策与数据分布摘要）
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_figure_dir() -> Path:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def robust_limits(values: np.ndarray, low_q: float = 0.5, high_q: float = 99.5, pad_ratio: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    lo = float(np.percentile(v, low_q))
    hi = float(np.percentile(v, high_q))
    if hi <= lo:
        lo = float(np.min(v))
        hi = float(np.max(v))
    span = max(hi - lo, 1e-6)
    lo -= pad_ratio * span
    hi += pad_ratio * span
    return lo, hi


def analyze_distribution(df: pd.DataFrame) -> dict[str, Any]:
    strength = pd.to_numeric(df["strength"], errors="coerce").dropna().to_numpy()
    age = pd.to_numeric(df["age"], errors="coerce").dropna().to_numpy()

    strength_std = float(np.std(strength))
    age_std = float(np.std(age))

    strength_skew = float(pd.Series(strength).skew())
    age_skew = float(pd.Series(age).skew())

    use_log_age = bool(age_skew > 1.0 or (np.max(age) / max(np.min(age), 1e-6) > 50))

    return {
        "n_samples": int(len(df)),
        "strength": {
            "min": float(np.min(strength)),
            "max": float(np.max(strength)),
            "mean": float(np.mean(strength)),
            "std": strength_std,
            "skew": strength_skew,
        },
        "age": {
            "min": float(np.min(age)),
            "max": float(np.max(age)),
            "mean": float(np.mean(age)),
            "std": age_std,
            "skew": age_skew,
            "use_log_scale": use_log_age,
        },
    }


def plot_data_distribution(df: pd.DataFrame, out_dir: Path, dist_info: dict[str, Any]) -> Path:
    strength = df["strength"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 强度分布
    axes[0].hist(strength, bins=35, color="#4C78A8", edgecolor="white", alpha=0.95)
    axes[0].set_title("(a) Strength Distribution")
    axes[0].set_xlabel("Compressive Strength (MPa)")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25)

    # 龄期分布（自适应 log）
    if dist_info["age"]["use_log_scale"]:
        bins = np.unique(np.logspace(np.log10(max(age.min(), 1.0)), np.log10(age.max()), 30).astype(float))
        axes[1].hist(age, bins=bins, color="#F58518", edgecolor="white", alpha=0.95)
        axes[1].set_xscale("log")
        axes[1].set_title("(b) Age Distribution (log-scale x)")
    else:
        axes[1].hist(age, bins=30, color="#F58518", edgecolor="white", alpha=0.95)
        axes[1].set_title("(b) Age Distribution")
    axes[1].set_xlabel("Age (day)")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = out_dir / "01_data_distribution.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _density_or_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str) -> dict[str, Any]:
    n = len(x)
    unique_ratio = float(np.unique(np.round(y, 4)).size / max(n, 1))
    dense = bool(n >= 400 or unique_ratio < 0.8)

    xlo, xhi = robust_limits(x)
    ylo, yhi = robust_limits(y)
    lo = min(xlo, ylo)
    hi = max(xhi, yhi)

    if dense:
        hb = ax.hexbin(x, y, gridsize=42, bins="log", cmap="viridis", mincnt=1)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
        mode = "hexbin_log_density"
    else:
        ax.scatter(x, y, s=18, alpha=0.65, c="#4C78A8", edgecolor="none")
        mode = "scatter"

    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1.8, linestyle="--", label="Ideal: y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left")

    return {
        "mode": mode,
        "xlim": [float(lo), float(hi)],
        "ylim": [float(lo), float(hi)],
        "n_points": int(n),
    }


def plot_true_vs_pred(oof_df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    y_true = oof_df["y_true"].to_numpy(dtype=float)
    y_p1 = oof_df["paper1_adaboost"].to_numpy(dtype=float)
    y_acdcb = oof_df["v3_dualspace_age_piecewise_acdcb"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))

    d1 = _density_or_scatter(axes[0], y_true, y_p1, "(a) Paper1 AdaBoost (10-fold OOF)")
    d2 = _density_or_scatter(axes[1], y_true, y_acdcb, "(b) ACDCB (10-fold OOF)")

    fig.tight_layout()
    out = out_dir / "02_true_vs_pred_acdcb_vs_paper1.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out, {"paper1": d1, "acdcb": d2}


def plot_ablation_r2_rmse(ablation: dict[str, Any], out_dir: Path) -> Path:
    variants_order = [
        "paper1_adaboost",
        "v1_primary_global_no_anchor",
        "v2_dualspace_global",
        "v3_dualspace_age_piecewise_acdcb",
        "v4_raw_age_piecewise_no_feature_engineering",
    ]
    labels = [
        "Paper1\nAdaBoost",
        "V1\nPrimary+Global",
        "V2\nDualSpace+Global",
        "V3\nACDCB",
        "V4\nRaw+Piecewise",
    ]

    r2 = [ablation["variants"][v]["metrics"]["R2_mean"] for v in variants_order]
    rmse = [ablation["variants"][v]["metrics"]["RMSE_mean"] for v in variants_order]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(12.5, 5.4))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, r2, width=width, color="#4C78A8", alpha=0.9, label="R² mean")
    bars2 = ax2.bar(x + width / 2, rmse, width=width, color="#E45756", alpha=0.88, label="RMSE mean")

    # 动态缩放：R² 数值接近时放大差异
    r2_lo = min(r2)
    r2_hi = max(r2)
    span = max(r2_hi - r2_lo, 0.002)
    ax1.set_ylim(r2_lo - span * 0.30, r2_hi + span * 0.30)

    ax1.set_ylabel("R² mean", color="#1f4f7a")
    ax2.set_ylabel("RMSE mean (MPa)", color="#7a1f26")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Ablation Results: R² and RMSE (10-fold)")
    ax1.grid(axis="y", alpha=0.20)

    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = out_dir / "03_ablation_r2_rmse.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_optimizer_convergence(ablation: dict[str, Any], out_dir: Path) -> Path:
    conv = ablation["optimizer_convergence"]["v3_dualspace_age_piecewise_acdcb"]
    traces = {
        "Global (V2)": ablation["optimizer_convergence"]["v2_dualspace_global"].get("global", []),
        "Early (<=28d)": conv.get("early", []),
        "Late (>28d)": conv.get("late", []),
    }

    fig, ax = plt.subplots(figsize=(10.8, 5.0))

    colors = {
        "Global (V2)": "#4C78A8",
        "Early (<=28d)": "#54A24B",
        "Late (>28d)": "#E45756",
    }

    for name, vals in traces.items():
        if not vals:
            continue
        x = np.arange(len(vals))
        ax.plot(x, vals, marker="o", markersize=3.0, linewidth=1.8, color=colors[name], label=name)

    ax.set_title("Weight Optimization Convergence (SLSQP Objective: RMSE)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value (RMSE)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "04_optimizer_convergence.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fold_r2_boxplot(ablation: dict[str, Any], out_dir: Path) -> Path:
    variants_order = [
        "paper1_adaboost",
        "v1_primary_global_no_anchor",
        "v2_dualspace_global",
        "v3_dualspace_age_piecewise_acdcb",
        "v4_raw_age_piecewise_no_feature_engineering",
    ]
    labels = ["Paper1", "V1", "V2", "V3(ACDCB)", "V4"]

    data = [ablation["variants"][v]["fold_metrics"]["R2"] for v in variants_order]

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    palette = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756"]

    for patch, c in zip(bp["boxes"], palette):
        patch.set(facecolor=c, alpha=0.45)

    ax.set_title("Fold-wise R² Distribution Across Ablation Variants")
    ax.set_ylabel("R² by fold")
    ax.grid(axis="y", alpha=0.22)

    fig.tight_layout()
    out = out_dir / "05_fold_r2_boxplot.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ann_svm_comparison(baseline: dict[str, Any], ablation: dict[str, Any], out_dir: Path) -> Path:
    models = ["ANN", "SVM"]
    test_r2 = [baseline["models"][m]["test_metrics"]["R2"] for m in models]
    test_rmse = [baseline["models"][m]["test_metrics"]["RMSE"] for m in models]

    acdcb_r2 = ablation["variants"]["v3_dualspace_age_piecewise_acdcb"]["metrics"]["R2_mean"]
    acdcb_rmse = ablation["variants"]["v3_dualspace_age_piecewise_acdcb"]["metrics"]["RMSE_mean"]

    labels = ["ANN (test)", "SVM (test)", "ACDCB (10-fold)"]
    r2_vals = [test_r2[0], test_r2[1], acdcb_r2]
    rmse_vals = [test_rmse[0], test_rmse[1], acdcb_rmse]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(11.5, 5.0))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, r2_vals, width=width, color="#4C78A8", label="R²")
    ax2.bar(x + width / 2, rmse_vals, width=width, color="#E45756", label="RMSE")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("R²")
    ax2.set_ylabel("RMSE (MPa)")
    ax1.set_title("Comparison with ANN/SVM")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = out_dir / "06_ann_svm_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    out_dir = ensure_figure_dir()

    data_df = pd.read_excel(ROOT / "data" / "Concrete_Data.xls", engine="xlrd")
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
    data_df = data_df.rename(columns=rename_map)

    baseline = load_json(ROOT / "doc" / "baseline_results.json")
    ablation = load_json(ROOT / "doc" / "ablation_results_acdcb.json")
    oof_df = pd.read_csv(ROOT / "doc" / "ablation_oof_predictions.csv")

    dist_info = analyze_distribution(data_df)

    fig_paths = {}
    fig_paths["data_distribution"] = str(plot_data_distribution(data_df, out_dir, dist_info))
    p2, scatter_meta = plot_true_vs_pred(oof_df, out_dir)
    fig_paths["true_vs_pred"] = str(p2)
    fig_paths["ablation_r2_rmse"] = str(plot_ablation_r2_rmse(ablation, out_dir))
    fig_paths["optimizer_convergence"] = str(plot_optimizer_convergence(ablation, out_dir))
    fig_paths["fold_r2_boxplot"] = str(plot_fold_r2_boxplot(ablation, out_dir))
    fig_paths["ann_svm_comparison"] = str(plot_ann_svm_comparison(baseline, ablation, out_dir))

    index_payload = {
        "distribution_analysis": dist_info,
        "plot_adaptations": {
            "true_vs_pred": scatter_meta,
            "age_histogram_log_scale": dist_info["age"]["use_log_scale"],
            "ablation_r2_zoomed_axis": True,
            "convergence_plot": "SLSQP objective traces",
        },
        "figures": fig_paths,
    }

    with open(out_dir / "figure_index.json", "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    print("Generated figures:")
    for k, v in fig_paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
