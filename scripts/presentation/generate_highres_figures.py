from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "presentation_work" / "presentation_metrics.json"
OUT_DIR = ROOT / "figures" / "presentation_highres"

PALETTE = {
    "navy": "#1E3A5F",
    "blue": "#2F5D8A",
    "steel": "#5E7FA6",
    "gray": "#8A8F99",
    "light_gray": "#DCE3EC",
    "accent": "#1B84F3",
    "green": "#2E8B57",
    "red": "#C23B3B",
}

NAME_MAP = {
    "paper1 AdaBoost (10-fold)": "paper1 AdaBoost（10折）",
    "ACDCB V3 (10-fold)": "ACDCB V3（10折）",
    "ANN (9:1 test)": "ANN（9:1划分）",
    "SVM (9:1 test)": "SVM（9:1划分）",
    "V1 Primary+Global": "V1 主空间+全局融合",
    "V2 Dual-space+Global": "V2 双空间+全局融合",
    "V3 ACDCB (Piecewise)": "V3 ACDCB（分段融合）",
    "V4 Raw+Piecewise": "V4 原始特征+分段融合",
}


def cn_name(name: str) -> str:
    return NAME_MAP.get(name, name)


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def load_data() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_margin(values: np.ndarray, ratio: float = 0.15, floor: float = 1e-4) -> float:
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    span = max(v_max - v_min, floor)
    return span * ratio


def plot_main_metrics(data: dict) -> Path:
    rows = data["main_protocol_results"]
    df = pd.DataFrame(rows)
    methods = [cn_name(m) for m in df["display_name"].tolist()]

    metrics = ["R2", "RMSE", "MAE", "MAPE"]
    labels = ["$R^2$（越高越好）", "RMSE（MPa）", "MAE（MPa）", "MAPE（%）"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8))
    axes = axes.ravel()

    bar_colors = [PALETTE["gray"], PALETTE["blue"]]

    for ax, metric, ylabel in zip(axes, metrics, labels):
        values = df[metric].values.astype(float)
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=bar_colors, width=0.58, edgecolor="black", linewidth=0.4)

        for i, bar in enumerate(bars):
            v = values[i]
            offset = 0.01 if metric == "R2" else max(0.04, safe_margin(values, ratio=0.2))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + offset,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#111111",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=10, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"统一协议对比：{metric}")

        margin = safe_margin(values, ratio=0.25)
        if metric == "R2":
            ax.set_ylim(max(0.0, values.min() - margin), min(1.0, values.max() + margin))
        else:
            ax.set_ylim(max(0.0, values.min() - margin), values.max() + margin)

    # Delta annotation (ACDCB vs paper1)
    baseline = df.iloc[0]
    improved = df.iloc[1]
    deltas = {
        "Δ$R^2$": improved["R2"] - baseline["R2"],
        "ΔRMSE": improved["RMSE"] - baseline["RMSE"],
        "ΔMAE": improved["MAE"] - baseline["MAE"],
        "ΔMAPE": improved["MAPE"] - baseline["MAPE"],
    }
    delta_text = "  |  ".join([f"{k}: {v:+.4f}" for k, v in deltas.items()])
    fig.suptitle("统一10折协议下 ACDCB 与 paper1 对比", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, delta_text, ha="center", fontsize=11, color=PALETTE["navy"])

    out_path = OUT_DIR / "01_main_metrics_2x2.png"
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_ablation_zoomed(data: dict) -> Path:
    df = pd.DataFrame(data["ablation_variants"])
    labels = [cn_name(m) for m in df["display_name"].tolist()]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # R2 (very close values, use zoomed y-range)
    r2_vals = df["R2"].astype(float).values
    axes[0].plot(x, r2_vals, marker="o", color=PALETTE["blue"], linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_title("消融实验（放大视图）：$R^2$")
    r2_margin = safe_margin(r2_vals, ratio=0.30, floor=5e-5)
    axes[0].set_ylim(r2_vals.min() - r2_margin, r2_vals.max() + r2_margin)

    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + r2_margin * 0.15, f"{v:.6f}", ha="center", fontsize=9)

    # RMSE (close values, zoom for readability)
    rmse_vals = df["RMSE"].astype(float).values
    axes[1].plot(x, rmse_vals, marker="s", color=PALETTE["navy"], linewidth=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel("RMSE（MPa）")
    axes[1].set_title("消融实验（放大视图）：RMSE")
    rmse_margin = safe_margin(rmse_vals, ratio=0.30, floor=0.002)
    axes[1].set_ylim(rmse_vals.min() - rmse_margin, rmse_vals.max() + rmse_margin)

    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + rmse_margin * 0.15, f"{v:.6f}", ha="center", fontsize=9)

    fig.suptitle("模块级消融：近值可读性优化", fontsize=16, fontweight="bold")
    out_path = OUT_DIR / "02_ablation_zoomed.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_age_weights(data: dict) -> Path:
    df = pd.DataFrame(data["age_segment_weights"])
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    b1 = ax.bar(x - width / 2, df["early"], width=width, color=PALETTE["blue"], label="早龄期（age ≤ 28 d）")
    b2 = ax.bar(x + width / 2, df["late"], width=width, color=PALETTE["gray"], label="晚龄期（age > 28 d）")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=0)
    ax.set_ylabel("融合权重")
    ax.set_title("龄期分段约束融合权重")
    ax.legend(frameon=True)
    ax.set_ylim(0, 0.75)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    out_path = OUT_DIR / "03_age_segment_weights.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_model_scatter(data: dict) -> Path:
    points = []

    for item in data["main_protocol_results"]:
        points.append(
            {
                "name": cn_name(item["display_name"]),
                "R2": float(item["R2"]),
                "RMSE": float(item["RMSE"]),
                "group": "10折交叉验证",
            }
        )

    for item in data["single_split_baselines"]:
        points.append(
            {
                "name": cn_name(item["display_name"]),
                "R2": float(item["R2"]),
                "RMSE": float(item["RMSE"]),
                "group": "单次划分",
            }
        )

    df = pd.DataFrame(points)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    colors = {"10折交叉验证": PALETTE["blue"], "单次划分": PALETTE["gray"]}

    for group, gdf in df.groupby("group"):
        ax.scatter(gdf["RMSE"], gdf["R2"], s=130, c=colors[group], label=group, edgecolors="black", linewidths=0.5)

    for _, row in df.iterrows():
        ax.annotate(
            row["name"],
            (row["RMSE"], row["R2"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlabel("RMSE（MPa）↓")
    ax.set_ylabel("$R^2$ ↑")
    ax.set_title("模型家族在精度-误差空间的对比")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()

    r2_margin = safe_margin(df["R2"].values, ratio=0.20, floor=0.005)
    rmse_margin = safe_margin(df["RMSE"].values, ratio=0.20, floor=0.2)
    ax.set_ylim(max(0.0, df["R2"].min() - r2_margin), min(1.0, df["R2"].max() + r2_margin))
    ax.set_xlim(max(0.0, df["RMSE"].min() - rmse_margin), df["RMSE"].max() + rmse_margin)

    out_path = OUT_DIR / "04_model_family_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_data()

    outputs = [
        plot_main_metrics(data),
        plot_ablation_zoomed(data),
        plot_age_weights(data),
        plot_model_scatter(data),
    ]

    print("已生成中文高分辨率图表：")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
