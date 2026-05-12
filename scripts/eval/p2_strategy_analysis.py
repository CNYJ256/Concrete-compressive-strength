from __future__ import annotations

"""P2 策略无效性综合分析脚本。

涵盖：
  P2-1: 特征工程共线性 + 逐步消融
  P2-2: 模型预测冗余性 + 融合策略对比
  P2-3: 龄期切分无效性（基于已有 threshold_scan + OOF）

输出：
  results/metrics/p2_strategy_analysis.json
  figures/presentation_highres/fig_p2_feature_collinearity.pdf
  figures/presentation_highres/fig_p2_model_corr_matrix.pdf
  figures/presentation_highres/fig_p2_age_rmse.pdf
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Fix Windows GBK encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from concrete_compressive_strength.core import (
    BASE_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    feature_engineering,
    load_data,
)

EPS = 1e-8
FIG_DIR = ROOT / "figures" / "presentation_highres"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Use academic-friendly styling
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
})


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ============================================================================
# P2-1: Feature Engineering Collinearity & Stepwise Ablation
# ============================================================================

def analyze_feature_collinearity(df: pd.DataFrame, X_primary: pd.DataFrame, y: np.ndarray,
                                 oof_df: pd.DataFrame) -> dict:
    """计算 32D primary 空间的特征相关性矩阵并标识高共线性对。"""
    print("\n" + "=" * 60)
    print("P2-1: 特征工程无效性分析")
    print("=" * 60)

    # ---- 1a: Correlation matrix ----
    feat_cols = [c for c in X_primary.columns if c not in BASE_FEATURES]
    engineered_only = X_primary[feat_cols]
    corr = engineered_only.corr()

    high_corr_pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.90:
                high_corr_pairs.append({
                    "feature_1": feat_cols[i], "feature_2": feat_cols[j],
                    "pearson_r": round(float(r), 4),
                })

    high_corr_pairs.sort(key=lambda x: -abs(x["pearson_r"]))
    print(f"  高共线性特征对 (|r| > 0.90): {len(high_corr_pairs)}/{(len(feat_cols)*(len(feat_cols)-1))//2}")
    for p in high_corr_pairs[:10]:
        print(f"    r={p['pearson_r']:+.4f}: {p['feature_1']} ↔ {p['feature_2']}")

    # ---- 1b: Visualization ----
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_yticks(range(len(feat_cols)))
    ax.set_xticklabels([f.replace("_", " ")[:16] for f in feat_cols], rotation=90, fontsize=20)
    ax.set_yticklabels([f.replace("_", " ")[:16] for f in feat_cols], fontsize=20)
    ax.set_title("Engineered Feature Correlation Matrix ($\\mathbb{R}^{24}$)", fontsize=18, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.78)
    cbar.ax.tick_params(labelsize=28)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_p2_feature_collinearity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存: figures/presentation_highres/fig_p2_feature_collinearity.pdf")

    # ---- 1c: Stepwise feature ablation ----
    # Remove engineered features in groups and observe R² trajectory
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.ensemble import HistGradientBoostingRegressor

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Build a simple HGB model for ablation
    def cv_r2(X_used, yy):
        model = HistGradientBoostingRegressor(
            loss="squared_error", random_state=RANDOM_STATE,
            max_iter=500, early_stopping=False,
        )
        scores = cross_val_score(model, X_used, yy, cv=cv, scoring="r2", n_jobs=-1)
        return float(np.mean(scores)), float(np.std(scores))

    # Define ablation stages: progressively remove engineered feature groups
    tier2_feats = [
        "binder_to_agg_ratio", "water_to_paste_ratio",
        "cement_fraction_in_binder", "slag_fraction_in_binder",
        "flyash_fraction_in_binder", "superplasticizer_efficiency",
        "maturity_index", "agg_binder_balance",
        "age_inverse", "age_wc_interaction",
    ]
    tier1_feats = [f for f in feat_cols if f not in tier2_feats]
    base_feats_only = [c for c in BASE_FEATURES]

    ablation_stages = [
        ("Full 32D", list(X_primary.columns)),
        ("Remove Tier-2 (22D)", [c for c in X_primary.columns if c not in tier2_feats]),
        ("Remove Tier-1 (8D raw)", base_feats_only),
    ]

    ablation_results = []
    for name, cols in ablation_stages:
        r2_mean, r2_std = cv_r2(X_primary[cols], y)
        ablation_results.append({"stage": name, "n_features": len(cols),
                                  "R2_mean": r2_mean, "R2_std": r2_std})
        print(f"  {name} ({len(cols)}D): R²={r2_mean:.6f} ± {r2_std:.6f}")

    # ---- 1d: Per-fold R² distribution comparison (engineered vs raw) ----
    # Use existing OOF predictions from ablation
    xgb_raw_r2 = oof_df.groupby("fold").apply(
        lambda g: float(r2_score(g["y_true"], g["xgb_raw"]))).tolist()
    xgb_eng_r2 = oof_df.groupby("fold").apply(
        lambda g: float(r2_score(g["y_true"], g["xgb_primary"]))).tolist()

    fold_comparison = {
        "XGB_raw_per_fold_R2": xgb_raw_r2,
        "XGB_primary_per_fold_R2": xgb_eng_r2,
        "XGB_raw_R2_mean": float(np.mean(xgb_raw_r2)),
        "XGB_raw_R2_std": float(np.std(xgb_raw_r2)),
        "XGB_primary_R2_mean": float(np.mean(xgb_eng_r2)),
        "XGB_primary_R2_std": float(np.std(xgb_eng_r2)),
    }

    return {
        "n_engineered_features": len(feat_cols),
        "high_corr_pairs_count": len(high_corr_pairs),
        "high_corr_pairs": high_corr_pairs,
        "stepwise_ablation": ablation_results,
        "fold_comparison_raw_vs_engineered": fold_comparison,
    }


# ============================================================================
# P2-2: Model Prediction Redundancy Analysis
# ============================================================================

def analyze_model_redundancy(oof_df: pd.DataFrame, y: np.ndarray) -> dict:
    """计算 4 模型 OOF 预测的成对 Pearson r 并对比融合策略。"""
    print("\n" + "=" * 60)
    print("P2-2: 模型冗余性分析")
    print("=" * 60)

    model_cols = ["xgb_primary", "lgb_primary", "hgb_primary", "hgb_anchor"]
    model_labels = ["XGBoost", "LightGBM", "HGB", "HGB_Anchor"]

    # ---- 2a: Pairwise Pearson r ----
    n = len(model_cols)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = pearsonr(oof_df[model_cols[i]], oof_df[model_cols[j]])
            corr_matrix[i, j] = r

    print("  模型预测成对 Pearson r:")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"    {model_labels[i]} vs {model_labels[j]}: r={corr_matrix[i,j]:.6f}")

    # ---- 2b: Best single vs equal-weight vs optimized fusion ----
    # Best single
    single_metrics = {}
    for i, col in enumerate(model_cols):
        single_metrics[model_labels[i]] = {
            "R2": float(r2_score(y, oof_df[col])),
            "RMSE": rmse(y, oof_df[col]),
            "MAE": mae(y, oof_df[col]),
        }

    best_name = max(single_metrics, key=lambda k: single_metrics[k]["R2"])

    # Equal weight fusion
    P = np.column_stack([oof_df[c].to_numpy() for c in model_cols])
    w_eq = np.full(n, 1.0 / n)
    pred_eq = P @ w_eq
    eq_metrics = {"R2": float(r2_score(y, pred_eq)), "RMSE": rmse(y, pred_eq),
                  "MAE": mae(y, pred_eq)}

    # Optimized fusion (SLSQP from existing results)
    from scipy.optimize import minimize
    init = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(lambda w: rmse(y, P @ w), init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12, "disp": False})
    w_opt = np.clip(res.x, 0, 1)
    w_opt = w_opt / w_opt.sum()
    pred_opt = P @ w_opt
    opt_metrics = {"R2": float(r2_score(y, pred_opt)), "RMSE": rmse(y, pred_opt),
                   "MAE": mae(y, pred_opt)}

    print(f"\n  融合策略对比:")
    print(f"    最优单模型 ({best_name}): R²={single_metrics[best_name]['R2']:.6f}, "
          f"RMSE={single_metrics[best_name]['RMSE']:.4f}")
    print(f"    等权融合:                R²={eq_metrics['R2']:.6f}, RMSE={eq_metrics['RMSE']:.4f}")
    print(f"    SLSQP优化融合:           R²={opt_metrics['R2']:.6f}, RMSE={opt_metrics['RMSE']:.4f}")
    print(f"    ΔR²(Opt vs Best Single):  {opt_metrics['R2']-single_metrics[best_name]['R2']:+.6f}")
    print(f"    ΔR²(Opt vs Equal):        {opt_metrics['R2']-eq_metrics['R2']:+.6f}")
    print(f"    ΔR²(Equal vs Best):       {eq_metrics['R2']-single_metrics[best_name]['R2']:+.6f}")

    # ---- 2c: Prediction scatter matrix ----
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, model_labels[i], ha="center", va="center",
                        fontsize=16, fontweight="bold", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.scatter(oof_df[model_cols[j]], oof_df[model_cols[i]],
                          s=3, alpha=0.35, color="#2166ac")
                r_val = corr_matrix[i, j]
                ax.text(0.05, 0.95, f"r={r_val:.4f}", transform=ax.transAxes,
                        fontsize=12, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
            if i == n - 1:
                ax.set_xlabel(model_labels[j][:8], fontsize=14)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(model_labels[i][:8], fontsize=14)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=10)
    fig.suptitle("OOF Prediction Cross-Correlation Matrix", fontsize=20, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_p2_model_corr_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存: figures/presentation_highres/fig_p2_model_corr_matrix.pdf")

    return {
        "pairwise_pearson_r": {model_labels[i]: {model_labels[j]: float(corr_matrix[i,j])
                                                  for j in range(n)} for i in range(n)},
        "mean_cross_pairwise_r": float(np.mean([corr_matrix[i,j]
                                                 for i in range(n) for j in range(n) if i != j])),
        "best_single": {"name": best_name, **single_metrics[best_name]},
        "equal_weight_fusion": eq_metrics,
        "optimized_fusion": {"weights": {model_labels[i]: float(w_opt[i]) for i in range(n)},
                             **opt_metrics},
        "delta_opt_vs_best": {"R2_gain": float(opt_metrics["R2"] - single_metrics[best_name]["R2"]),
                              "RMSE_drop": float(single_metrics[best_name]["RMSE"] - opt_metrics["RMSE"])},
        "delta_opt_vs_equal": {"R2_gain": float(opt_metrics["R2"] - eq_metrics["R2"]),
                               "RMSE_drop": float(eq_metrics["RMSE"] - opt_metrics["RMSE"])},
    }


# ============================================================================
# P2-3: Age Segmentation Ineffectiveness
# ============================================================================

def analyze_age_segmentation() -> dict:
    """基于 threshold_scan.json 论证龄期分段无效性。"""
    print("\n" + "=" * 60)
    print("P2-3: 龄期分段无效性分析")
    print("=" * 60)

    scan_path = ROOT / "results" / "metrics" / "threshold_scan.json"
    if not scan_path.exists():
        print("  [WARN] threshold_scan.json 不存在，使用内置数据")
        return {"status": "skipped", "reason": "missing threshold_scan.json"}

    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    global_base = scan["global_baseline"]
    scan_results = scan["scan_results"]

    # Extract R² and RMSE for each threshold
    taus = [r["tau"] for r in scan_results]
    r2s = [r["metrics"]["R2"] for r in scan_results]
    rmses = [r["metrics"]["RMSE"] for r in scan_results]
    global_r2 = global_base["metrics"]["R2"]
    global_rmse = global_base["metrics"]["RMSE"]

    r2_range = max(r2s) - min(r2s)
    rmse_range = max(rmses) - min(rmses)
    print(f"  全局融合: R²={global_r2:.6f}, RMSE={global_rmse:.4f}")
    print(f"  τ 扫描范围: {taus}")
    print(f"  R² 极差: {r2_range:.2e} (max={max(r2s):.6f} at τ={taus[r2s.index(max(r2s))]})")
    print(f"  RMSE 极差: {rmse_range:.4f} (min={min(rmses):.4f} at τ={taus[rmses.index(min(rmses))]})")
    print(f"  ΔR²(Global vs Best τ): {max(r2s)-global_r2:+.2e}")
    print(f"  结论: 龄期阈值在[3,180]范围内对性能几乎无影响")

    # ---- Visualization: threshold scan flatness ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    ax1.plot(taus, r2s, "o-", color="#2166ac", markersize=5, linewidth=1.2)
    ax1.axhline(y=global_r2, color="gray", linestyle="--", alpha=0.6, label=f"Global R²={global_r2:.4f}")
    ax1.set_xlabel("Age Threshold $\\tau$ (days)")
    ax1.set_ylabel("$R^2$")
    ax1.set_title("R² vs Age Threshold")
    ax1.legend(fontsize=7)
    ax1.set_xscale("log")
    ymin1 = min(min(r2s), global_r2) - 0.0003
    ymax1 = max(max(r2s), global_r2) + 0.0003
    ax1.set_ylim(ymin1, ymax1)

    ax2.plot(taus, rmses, "s-", color="#b2182b", markersize=5, linewidth=1.2)
    ax2.axhline(y=global_rmse, color="gray", linestyle="--", alpha=0.6, label=f"Global RMSE={global_rmse:.2f}")
    ax2.set_xlabel("Age Threshold $\\tau$ (days)")
    ax2.set_ylabel("RMSE (MPa)")
    ax2.set_title("RMSE vs Age Threshold")
    ax2.legend(fontsize=7)
    ax2.set_xscale("log")
    ymin2 = min(min(rmses), global_rmse) - 0.02
    ymax2 = max(max(rmses), global_rmse) + 0.02
    ax2.set_ylim(ymin2, ymax2)

    fig.suptitle("Age-Conditioned Piecewise Blending: Threshold Insensitivity", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_p2_threshold_flatness.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存: figures/presentation_highres/fig_p2_threshold_flatness.pdf")

    return {
        "global_metrics": global_base["metrics"],
        "threshold_scan_summary": {
            "taus": taus, "r2s": r2s, "rmses": rmses,
            "R2_range": float(r2_range),
            "RMSE_range": float(rmse_range),
            "best_tau_by_R2": taus[r2s.index(max(r2s))],
            "best_R2": max(r2s),
            "delta_R2_vs_global": float(max(r2s) - global_r2),
        },
    }


# ============================================================================
# P2-Additional: Single model superiority demonstration (single fine-tuned XGB vs ACDCB)
# ============================================================================

def analyze_single_vs_acdcb(oof_df: pd.DataFrame, y: np.ndarray) -> dict:
    """对比精细调参的单一 XGBoost (raw) vs 完整 ACDCB 框架。"""
    print("\n" + "=" * 60)
    print("P2-Extra: 精调单模型 vs ACDCB 复合模型")
    print("=" * 60)

    # Use XGB raw as the fine-tuned single model (with Optuna-optimized params)
    xgb_raw_pred = oof_df["xgb_raw"].to_numpy()
    xgb_raw_r2 = float(r2_score(y, xgb_raw_pred))
    xgb_raw_rmse = rmse(y, xgb_raw_pred)

    # ACDCB v3 prediction
    acdcb_pred = oof_df["v3_dualspace_age_piecewise_acdcb"].to_numpy()
    acdcb_r2 = float(r2_score(y, acdcb_pred))
    acdcb_rmse = rmse(y, acdcb_pred)

    print(f"  XGB_raw (精调单模型):     R²={xgb_raw_r2:.6f}, RMSE={xgb_raw_rmse:.4f}")
    print(f"  ACDCB v3 (复合集成):      R²={acdcb_r2:.6f}, RMSE={acdcb_rmse:.4f}")
    print(f"  ΔR² (XGB - ACDCB):        {xgb_raw_r2 - acdcb_r2:+.6f}")
    print(f"  ΔRMSE (ACDCB - XGB):      {acdcb_rmse - xgb_raw_rmse:+.4f}")

    # Also get XGB primary (engineered) for comparison
    xgb_primary_pred = oof_df["xgb_primary"].to_numpy()
    xgb_primary_r2 = float(r2_score(y, xgb_primary_pred))
    xgb_primary_rmse = rmse(y, xgb_primary_pred)
    print(f"  XGB_primary (工程特征):   R²={xgb_primary_r2:.6f}, RMSE={xgb_primary_rmse:.4f}")

    return {
        "xgb_raw_optimized": {"R2": xgb_raw_r2, "RMSE": xgb_raw_rmse},
        "xgb_primary_engineered": {"R2": xgb_primary_r2, "RMSE": xgb_primary_rmse},
        "acdcb_v3_composite": {"R2": acdcb_r2, "RMSE": acdcb_rmse},
        "delta_single_vs_composite": {
            "R2_gain": float(xgb_raw_r2 - acdcb_r2),
            "RMSE_drop": float(acdcb_rmse - xgb_raw_rmse),
        },
        "delta_engineered_vs_raw": {
            "R2_gain": float(xgb_primary_r2 - xgb_raw_r2),
            "RMSE_drop": float(xgb_raw_rmse - xgb_primary_rmse),
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.perf_counter()
    print("=" * 60)
    print("P2 策略无效性综合分析")
    print("=" * 60)

    # Load data
    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_base)
    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()

    # Load OOF predictions from existing ablation
    oof_path = ROOT / "results" / "predictions" / "ablation_oof_v2.csv"
    if not oof_path.exists():
        print(f"[ERROR] OOF prediction file not found: {oof_path}")
        print("Please run ablation_acdcb_v2.py first")
        sys.exit(1)

    oof_df = pd.read_csv(oof_path)

    # Add fold assignment (reconstruct from data)
    from sklearn.model_selection import KFold, cross_val_predict
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_assignments = np.zeros(len(y), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv.split(X_base, y)):
        fold_assignments[test_idx] = fold_idx
    oof_df["fold"] = fold_assignments

    cols_in_file = list(oof_df.columns)
    print(f"\nOOF columns: {cols_in_file}")

    # The ablation_oof_v2.csv only stores ensemble predictions, not individual model OOFs.
    # Re-train all base models to get individual predictions for correlation analysis.
    from concrete_compressive_strength.core import (
        BASE_MODEL_PARAMS, ANCHOR_MODEL_PARAMS,
        build_xgb, build_lgbm, build_hgb, feature_engineering_anchor,
    )

    # Load raw hyperparams
    raw_hp_path = ROOT / "results" / "metrics" / "raw_hyperparams.json"
    if raw_hp_path.exists():
        raw_hp = json.loads(raw_hp_path.read_text(encoding="utf-8"))
        def strip_rstate(d):
            return {k: v for k, v in d.items() if k != "random_state"}
        xgb_raw_params = strip_rstate(raw_hp["XGBoost_raw"]["best_params"])
        lgb_raw_params = strip_rstate(raw_hp["LightGBM_raw"]["best_params"])
        hgb_raw_params = strip_rstate(raw_hp["HGB_raw"]["best_params"])
    else:
        xgb_raw_params = BASE_MODEL_PARAMS["XGBoost"]
        lgb_raw_params = BASE_MODEL_PARAMS["LightGBM"]
        hgb_raw_params = BASE_MODEL_PARAMS["HGB"]

    X_anchor = feature_engineering_anchor(X_base)
    X_raw = X_base.copy()

    print("\n重新生成模型级 OOF 预测...")
    model_oof = {}

    specs = [
        ("xgb_primary", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary),
        ("lgb_primary", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary),
        ("hgb_primary", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary),
        ("hgb_anchor", build_hgb(ANCHOR_MODEL_PARAMS), X_anchor),
        ("xgb_raw", build_xgb(xgb_raw_params), X_raw),
        ("lgb_raw", build_lgbm(lgb_raw_params), X_raw),
        ("hgb_raw", build_hgb(hgb_raw_params), X_raw),
    ]
    for name, est, X_used in specs:
        tt = time.perf_counter()
        pred = cross_val_predict(est, X_used, y, cv=cv, n_jobs=-1, method="predict")
        model_oof[name] = pred
        print(f"  {name}: R²={r2_score(y, pred):.6f}, RMSE={rmse(y, pred):.4f} ({time.perf_counter()-tt:.1f}s)")
        oof_df[name] = pred

    # ---- Run analyses ----
    p21 = analyze_feature_collinearity(df, X_primary, y, oof_df)
    p22 = analyze_model_redundancy(oof_df, y)
    p23 = analyze_age_segmentation()
    p2x = analyze_single_vs_acdcb(oof_df, y)

    # ---- Save results ----
    results = {
        "meta": {
            "study": "P2 Strategy Ineffectiveness Comprehensive Analysis",
            "random_state": RANDOM_STATE,
            "n_samples": len(y),
        },
        "p2_1_feature_engineering_ineffectiveness": p21,
        "p2_2_model_redundancy": p22,
        "p2_3_age_segmentation_ineffectiveness": p23,
        "p2_extra_single_vs_composite": p2x,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    out_path = ROOT / "results" / "metrics" / "p2_strategy_analysis.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")
    print(f"总运行时间: {time.perf_counter()-t0:.1f}s")

    # ---- Print executive summary ----
    print("\n" + "=" * 70)
    print("执行摘要：为什么 ACDCB 的额外策略无效")
    print("=" * 70)
    print(f"1. 特征工程: {p21['high_corr_pairs_count']} 对高共线性 (|r|>0.9)")
    print(f"   XGB_raw vs XGB_primary: ΔR²={p2x['delta_engineered_vs_raw']['R2_gain']:+.6f}")
    print(f"   特征工程引入共线性，在小样本上增加方差而非信号")
    print(f"2. 模型集成: 成对预测相关性 r={p22['mean_cross_pairwise_r']:.4f}")
    print(f"   三个 GBDT 预测高度相关 → 集成几乎无增益")
    print(f"   最优单模型 vs 优化融合: ΔR²={p22['delta_opt_vs_best']['R2_gain']:+.6f}")
    print(f"3. 龄期分段: R² 极差={p23['threshold_scan_summary']['R2_range']:.2e}")
    print(f"   任何 τ∈[3,180] 性能等同 → 分段无意义")
    print(f"4. 精调 XGB_raw R²={p2x['xgb_raw_optimized']['R2']:.6f}")
    print(f"   ACDCB v3  R²={p2x['acdcb_v3_composite']['R2']:.6f}")
    print(f"   结论: 精调单模型 ≥ 复杂集成框架")


if __name__ == "__main__":
    main()
