from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


MODEL_DISPLAY_NAME = {
    "XGBoost_v8": "XGBoost",
    "LightGBM_v8": "LightGBM",
    "HGB_v8": "HGB",
    "HGB_v7_baseline": "HGB-Anchor",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HGB": "HGB",
    "HGB_Anchor": "HGB-Anchor",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "figures"
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
    axes[0].set_title("ACDCB Fusion Strategy Comparison (R²)")
    axes[0].set_ylabel("R² mean")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, rmse_vals, color=["#4C78A8", "#F58518"])
    axes[1].set_title("ACDCB Fusion Strategy Comparison (RMSE)")
    axes[1].set_ylabel("RMSE mean")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig_acdcb_strategy_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_v9_piecewise_weights(v9: dict, out_dir: Path) -> Path:
    early = v9["age_piecewise_blend"]["early_weights"]
    late = v9["age_piecewise_blend"]["late_weights"]
    global_w = v9["global_blend"]["weights"]

    models = list(early.keys())
    labels = [MODEL_DISPLAY_NAME.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width, [global_w[m] for m in models], width=width, label="Global", color="#4C78A8")
    ax.bar(x, [early[m] for m in models], width=width, label="Early (<=28d)", color="#54A24B")
    ax.bar(x + width, [late[m] for m in models], width=width, label="Late (>28d)", color="#E45756")

    ax.set_title("ACDCB Weight Distribution by Age Segment")
    ax.set_ylabel("Weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "fig_acdcb_piecewise_weights.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_v9_base_model_oof(v9: dict, out_dir: Path) -> Path:
    per_model = v9["per_model_oof"]
    model_ids = [item["model_id"] for item in per_model]
    labels = [MODEL_DISPLAY_NAME.get(mid, mid) for mid in model_ids]
    r2_vals = [item["cv_10fold"]["R2_mean"] for item in per_model]
    rmse_vals = [item["cv_10fold"]["RMSE_mean"] for item in per_model]

    x = np.arange(len(model_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x - width / 2, r2_vals, width=width, label="R² mean", color="#4C78A8")
    ax.bar(x + width / 2, rmse_vals, width=width, label="RMSE mean", color="#E45756")

    ax.set_title("ACDCB Base Learners OOF Performance (10-fold)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "fig_acdcb_base_models_oof.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_v9_vs_paper1_baselines(baseline: dict, v9: dict, out_dir: Path) -> Path:
    p1_models = ["AdaBoost", "ANN", "SVM"]
    p1_test_r2 = [baseline["models"][m]["test_metrics"]["R2"] for m in p1_models]

    adaboost_cv_r2 = baseline["adaboost_cv_10fold"]["R2_mean"]
    v9_cv_r2 = v9["best_model"]["cv_10fold"]["R2_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(p1_models, p1_test_r2, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Paper1 Baselines (Single Split Test R²)")
    axes[0].set_ylabel("R²")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(["Paper1 AdaBoost (10-fold)", "ACDCB (10-fold)"], [adaboost_cv_r2, v9_cv_r2], color=["#4C78A8", "#E45756"])
    axes[1].set_title("Core Benchmark: Paper1 vs ACDCB (10-fold R²)")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", labelrotation=12)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig_acdcb_vs_paper1_baselines.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = ensure_output_dir()

    v9 = load_json(root / "results" / "metrics" / "acdcb_metrics.json")
    baseline = load_json(root / "results" / "metrics" / "baseline_results.json")

    generated = [
        plot_v9_strategy_comparison(v9, out_dir),
        plot_v9_piecewise_weights(v9, out_dir),
        plot_v9_base_model_oof(v9, out_dir),
        plot_v9_vs_paper1_baselines(baseline, v9, out_dir),
    ]

    print("Generated figures:")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
