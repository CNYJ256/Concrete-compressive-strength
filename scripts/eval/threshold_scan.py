from __future__ import annotations

"""年龄阈值扫描脚本 (P0-2)。

扫描 τ ∈ {3, 7, 14, 21, 28, 56, 90, 180} 天，
对每个阈值运行 ACDCB piecewise blending，
收集指标并绘制 τ-metrics 曲线，
验证 28 天阈值的经验依据。
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import (
    BASE_FEATURES,
    BASE_MODEL_PARAMS,
    ANCHOR_MODEL_PARAMS,
    RANDOM_STATE,
    TARGET_COL,
    build_hgb,
    build_lgbm,
    build_xgb,
    feature_engineering,
    feature_engineering_anchor,
    load_data,
)

THRESHOLDS = [3, 7, 14, 21, 28, 56, 90, 180]
EPS = 1e-8


def get_logger() -> logging.Logger:
    logger = logging.getLogger("threshold_scan")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape_percent(y_true, y_pred),
    }


def optimize_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_models = P.shape[1]
    init = np.full(n_models, 1.0 / n_models)

    def obj(w: np.ndarray) -> float:
        return rmse(y, P @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        return init
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() <= 0:
        return init
    return w / w.sum()


def main() -> None:
    logger = get_logger()
    t0 = time.perf_counter()

    # ---- 数据准备 ----
    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_base)
    X_anchor = feature_engineering_anchor(X_base)
    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()
    logger.info("数据加载完成: N=%d", len(y))

    # ---- OOF 预测（共享，所有 τ 使用相同 P 矩阵）----
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    model_specs = [
        ("XGBoost", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary),
        ("LightGBM", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary),
        ("HGB", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary),
        ("HGB_Anchor", build_hgb(ANCHOR_MODEL_PARAMS), X_anchor),
    ]

    oof_list: list[np.ndarray] = []
    for name, est, X_used in model_specs:
        t1 = time.perf_counter()
        pred = cross_val_predict(est, X_used, y, cv=cv, n_jobs=-1, method="predict")
        single_metrics = compute_metrics(y, pred)
        logger.info("%s OOF: R2=%.6f RMSE=%.4f (%.1fs)", name, single_metrics["R2"], single_metrics["RMSE"], time.perf_counter() - t1)
        oof_list.append(pred)

    P = np.column_stack(oof_list)
    model_names = [m[0] for m in model_specs]

    # ---- 全局融合基线 ----
    w_global = optimize_weights(P, y)
    pred_global = P @ w_global
    metrics_global = compute_metrics(y, pred_global)
    logger.info("Global: R2=%.6f RMSE=%.4f", metrics_global["R2"], metrics_global["RMSE"])

    # ---- 阈值扫描 ----
    scan_results: list[dict[str, Any]] = []
    for tau in THRESHOLDS:
        early = age <= tau
        late = ~early
        n_early = int(np.sum(early))
        n_late = int(np.sum(late))

        if n_early < 4 or n_late < 4:
            logger.warning("τ=%d 分段样本不足 (早期=%d, 晚期=%d)，跳过", tau, n_early, n_late)
            continue

        w_e = optimize_weights(P[early], y[early])
        w_l = optimize_weights(P[late], y[late])

        pred = np.empty_like(y)
        pred[early] = P[early] @ w_e
        pred[late] = P[late] @ w_l
        m = compute_metrics(y, pred)

        scan_results.append({
            "tau": tau,
            "n_early": n_early,
            "n_late": n_late,
            "weights_early": {model_names[i]: float(w_e[i]) for i in range(len(model_names))},
            "weights_late": {model_names[i]: float(w_l[i]) for i in range(len(model_names))},
            "metrics": m,
        })
        logger.info("τ=%3d | 早期=%3d 晚期=%3d | R2=%.6f RMSE=%.4f MAE=%.4f MAPE=%.3f",
                    tau, n_early, n_late, m["R2"], m["RMSE"], m["MAE"], m["MAPE"])

    # ---- 保存 JSON ----
    output = {
        "meta": {
            "study": "Age Threshold Scan for ACDCB Piecewise Blending",
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 10, "shuffle": True},
            "n_samples": int(len(y)),
            "thresholds_scanned": [r["tau"] for r in scan_results],
        },
        "global_baseline": {
            "weights": {model_names[i]: float(w_global[i]) for i in range(len(model_names))},
            "metrics": metrics_global,
        },
        "scan_results": scan_results,
    }

    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "threshold_scan.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("扫描结果已保存: %s", json_path)

    # ---- 绘图 ----
    taus = [r["tau"] for r in scan_results]
    r2s = [r["metrics"]["R2"] for r in scan_results]
    rmses = [r["metrics"]["RMSE"] for r in scan_results]
    maes = [r["metrics"]["MAE"] for r in scan_results]
    mapes = [r["metrics"]["MAPE"] for r in scan_results]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ["#2166ac", "#b2182b", "#4daf4a", "#ff7f00"]
    metric_pairs = [
        (axes[0, 0], r2s, "R²", colors[0], metrics_global["R2"]),
        (axes[0, 1], rmses, "RMSE (MPa)", colors[1], metrics_global["RMSE"]),
        (axes[1, 0], maes, "MAE (MPa)", colors[2], metrics_global["MAE"]),
        (axes[1, 1], mapes, "MAPE (%)", colors[3], metrics_global["MAPE"]),
    ]

    for ax, vals, label, color, global_val in metric_pairs:
        ax.plot(taus, vals, "o-", color=color, linewidth=1.5, markersize=6, label="Piecewise (τ)")
        ax.axhline(y=global_val, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Global")
        ax.axvline(x=28, color="red", linestyle=":", linewidth=1, alpha=0.6, label="τ=28 (standard)")
        ax.set_xlabel("Age Threshold τ (days)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, alpha=0.3)

    fig.suptitle("Age Threshold Scan — ACDCB Piecewise Blending Performance", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "threshold_scan.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "threshold_scan.png", dpi=300, bbox_inches="tight")
    logger.info("图表已保存: %s", fig_path)
    plt.close(fig)

    # ---- 判定 ----
    best_idx_r2 = int(np.argmax(r2s))
    best_idx_rmse = int(np.argmin(rmses))
    logger.info("最优R²: τ=%d (R²=%.6f)", taus[best_idx_r2], r2s[best_idx_r2])
    logger.info("最优RMSE: τ=%d (RMSE=%.4f)", taus[best_idx_rmse], rmses[best_idx_rmse])
    logger.info("28天: R²=%.6f RMSE=%.4f", r2s[taus.index(28)], rmses[taus.index(28)])
    logger.info("阈值扫描完成，总耗时 %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
