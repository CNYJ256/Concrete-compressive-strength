from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_v9_strategy_comparison(v9: dict, out_dir: Path) -> Path:
    labels = ["Global Blend", "Age Piecewise"]
    r2_vals = [
        v9["global_blend"]["cv_10fold"]["R2_mean"],
        v9["age_piecewise_blend"]["cv_10fold"]["R2_mean"],
    ]
    rmse_vals = [
        v9["global_blend"]["cv_10fold"]["RMSE_mean"],
        v9["age_piecewise_blend"]["cv_10fold"]["RMSE_mean"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(labels, r2_vals, color=["#4C78A8", "#F58518"])
    axes[0].set_title("v9 Fusion Strategy Comparison (R²)")
    axes[0].set_ylabel("R² mean")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, rmse_vals, color=["#4C78A8", "#F58518"])
    axes[1].set_title("v9 Fusion Strategy Comparison (RMSE)")
    axes[1].set_ylabel("RMSE mean")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig_v9_strategy_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_v9_piecewise_weights(v9: dict, out_dir: Path) -> Path:
    early = v9["age_piecewise_blend"]["early_weights"]
    late = v9["age_piecewise_blend"]["late_weights"]
    global_w = v9["global_blend"]["weights"]

    models = list(early.keys())
    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width, [global_w[m] for m in models], width=width, label="Global", color="#4C78A8")
    ax.bar(x, [early[m] for m in models], width=width, label="Early (<=28d)", color="#54A24B")
    ax.bar(x + width, [late[m] for m in models], width=width, label="Late (>28d)", color="#E45756")

    ax.set_title("v9 Weight Distribution by Age Segment")
    ax.set_ylabel("Weight")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "fig_v9_piecewise_weights.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_version_comparison(v8: dict, v9: dict, out_dir: Path) -> Path:
    v7_r2 = v8["baseline_v7"]["R2_mean"]
    v7_rmse = v8["baseline_v7"]["RMSE_mean"]
    v8_r2 = v8["best_model"]["cv_10fold"]["R2_mean"]
    v8_rmse = v8["best_model"]["cv_10fold"]["RMSE_mean"]
    v9_r2 = v9["best_model"]["cv_10fold"]["R2_mean"]
    v9_rmse = v9["best_model"]["cv_10fold"]["RMSE_mean"]

    versions = ["v7", "v8", "v9"]
    r2_vals = [v7_r2, v8_r2, v9_r2]
    rmse_vals = [v7_rmse, v8_rmse, v9_rmse]
    x = np.arange(len(versions))

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()

    ax1.plot(x, r2_vals, marker="o", linewidth=2.2, color="#4C78A8", label="R²")
    ax2.plot(x, rmse_vals, marker="s", linewidth=2.2, color="#E45756", label="RMSE")

    ax1.set_xticks(x)
    ax1.set_xticklabels(versions)
    ax1.set_ylabel("R² mean", color="#4C78A8")
    ax2.set_ylabel("RMSE mean", color="#E45756")
    ax1.set_title("Performance Evolution from v7 to v9")
    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    out = out_dir / "fig_version_performance_evolution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_reproduction_summary(baseline: dict, paper2: dict, out_dir: Path) -> Path:
    p1_models = ["AdaBoost", "ANN", "SVM"]
    p1_test_r2 = [baseline["models"][m]["test_metrics"]["R2"] for m in p1_models]

    p2_models = ["ANN", "Reg_w_b", "Reg_w_c"]
    p2_mean_r2 = [paper2["source_like_summary"][m]["test_R2_mean"] for m in p2_models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(p1_models, p1_test_r2, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Paper1 Reproduction: Test R²")
    axes[0].set_ylabel("R²")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(["ANN", "Reg(w/b)", "Reg(w/c)"], p2_mean_r2, color=["#4C78A8", "#F58518", "#E45756"])
    axes[1].set_title("Paper2 Reproduction: Source-like Mean Test R²")
    axes[1].set_ylabel("R²")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig_reproduction_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = ensure_output_dir()

    v9 = load_json(root / "v9" / "metrics.json")
    v8 = load_json(root / "oldversion" / "v8" / "metrics.json")

    # 复现实验脚本当前将结果输出到根目录 doc/。
    baseline = load_json(root / "doc" / "baseline_results.json")
    paper2 = load_json(root / "doc" / "paper2_reproduction_results.json")

    generated = [
        plot_v9_strategy_comparison(v9, out_dir),
        plot_v9_piecewise_weights(v9, out_dir),
        plot_version_comparison(v8, v9, out_dir),
        plot_reproduction_summary(baseline, paper2, out_dir),
    ]

    print("Generated figures:")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
