from __future__ import annotations

"""ACDCB 消融实验 V2（扩展版）。

在原版基础上新增：
1. OLS 无约束融合变体 — 验证约束优化的必要性 (P1-1)
2. 可选加载 raw 特征独立超参数 — V4 公平对比 (P0-1)

用法：
  python scripts/eval/ablation_acdcb_v2.py [raw_hyperparams.json]
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from concrete_compressive_strength.core import (
    AGE_SPLIT_DAY,
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
from model_factory import build_adaboost_model

EPS = 1e-8


@dataclass
class VariantResult:
    name: str
    metrics: dict[str, float]
    fold_metrics: dict[str, list[float]]
    weights: dict[str, float] | None
    weights_piecewise: dict[str, dict[str, float]] | None
    convergence_trace: dict[str, list[float]]
    optimizer_info: dict[str, Any]
    pred: np.ndarray


def get_logger() -> logging.Logger:
    logger = logging.getLogger("acdcb_ablation_v2")
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


def fold_metrics_from_cv(cv: KFold, X_ref: np.ndarray, y: np.ndarray, pred: np.ndarray):
    r2s, rmses, maes, mapes = [], [], [], []
    for _, test_idx in cv.split(X_ref, y):
        yt, yp = y[test_idx], pred[test_idx]
        r2s.append(float(r2_score(yt, yp)))
        rmses.append(rmse(yt, yp))
        maes.append(mae(yt, yp))
        mapes.append(mape_percent(yt, yp))
    summary = {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
        "MAPE_mean": float(np.mean(mapes)), "MAPE_std": float(np.std(mapes)),
    }
    return summary, {"R2": r2s, "RMSE": rmses, "MAE": maes, "MAPE": mapes}


def optimize_weights_with_trace(P: np.ndarray, y: np.ndarray, maxiter: int = 500):
    n_models = P.shape[1]
    init = np.full(n_models, 1.0 / n_models, dtype=float)

    def obj(w): return rmse(y, P @ w)
    trace = [float(obj(init))]

    def callback(wk): trace.append(float(obj(wk)))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints,
                   callback=callback, options={"maxiter": maxiter, "ftol": 1e-12, "disp": False})

    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init

    final_obj = float(obj(w))
    if not trace or abs(trace[-1] - final_obj) > 1e-12:
        trace.append(final_obj)
    info = {"success": bool(res.success), "status": int(res.status),
            "message": str(res.message), "nit": int(getattr(res, "nit", -1)),
            "fun": float(getattr(res, "fun", final_obj))}
    return w, trace, info


def run_global_blend(name, model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    w, trace, info = optimize_weights_with_trace(P, y)
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=weights, weights_piecewise=None,
                         convergence_trace={"global": trace},
                         optimizer_info={"global": info}, pred=pred)


def run_piecewise_blend(name, model_order, pred_cache, cv, X_ref, y, age, split_day):
    P = np.column_stack([pred_cache[m] for m in model_order])
    early_mask = age <= split_day
    late_mask = ~early_mask
    w_e, tr_e, info_e = optimize_weights_with_trace(P[early_mask], y[early_mask])
    w_l, tr_l, info_l = optimize_weights_with_trace(P[late_mask], y[late_mask])
    pred = np.empty_like(y)
    pred[early_mask] = P[early_mask] @ w_e
    pred[late_mask] = P[late_mask] @ w_l
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    wp = {"early": {model_order[i]: float(w_e[i]) for i in range(len(model_order))},
          "late": {model_order[i]: float(w_l[i]) for i in range(len(model_order))}}
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=None, weights_piecewise=wp,
                         convergence_trace={"early": tr_e, "late": tr_l},
                         optimizer_info={"early": info_e, "late": info_l}, pred=pred)


def run_ols_unconstrained(name, model_order, pred_cache, cv, X_ref, y):
    """OLS 无约束线性回归 stacking 变体 (P1-1)。"""
    P = np.column_stack([pred_cache[m] for m in model_order])
    ols = LinearRegression(fit_intercept=False)
    ols.fit(P, y)
    w = ols.coef_
    # 归一化仅用于权重比较，不影响预测
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    negative_count = int(np.sum(w < 0))
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=weights, weights_piecewise=None,
                         convergence_trace={}, optimizer_info={
                             "method": "OLS (unconstrained linear regression)",
                             "negative_weights": negative_count,
                             "weight_sum": float(np.sum(w)),
                         }, pred=pred)


def main():
    logger = get_logger()
    t0 = time.perf_counter()

    # ---- 加载 raw 超参数（如有）----
    raw_params_override = None
    if len(sys.argv) >= 2:
        raw_hp_path = Path(sys.argv[1])
        if raw_hp_path.exists():
            raw_params_override = json.loads(raw_hp_path.read_text(encoding="utf-8"))
            logger.info("已加载 raw 独立超参数: %s", raw_hp_path)
        else:
            logger.warning("raw 超参数文件不存在，将复用 engineered 超参数: %s", raw_hp_path)
    else:
        logger.info("未指定 raw 超参数文件，V4 将复用 engineered 超参数（不公平对比）")

    # ---- 数据准备 ----
    metrics_dir = ROOT / "results" / "metrics"
    pred_dir = ROOT / "results" / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_base)
    X_anchor = feature_engineering_anchor(X_base)
    X_raw = X_base.copy()
    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # ---- 确定 V4 raw 模型超参数 ----
    if raw_params_override:
        def _strip_rstate(d): return {k: v for k, v in d.items() if k != "random_state"}
        xgb_raw_params = _strip_rstate(raw_params_override["XGBoost_raw"]["best_params"])
        lgb_raw_params = _strip_rstate(raw_params_override["LightGBM_raw"]["best_params"])
        hgb_raw_params = _strip_rstate(raw_params_override["HGB_raw"]["best_params"])
        hgb_anchor_raw_params = _strip_rstate(raw_params_override["HGB_Anchor_raw"]["best_params"])
    else:
        xgb_raw_params = BASE_MODEL_PARAMS["XGBoost"]
        lgb_raw_params = BASE_MODEL_PARAMS["LightGBM"]
        hgb_raw_params = BASE_MODEL_PARAMS["HGB"]
        hgb_anchor_raw_params = ANCHOR_MODEL_PARAMS

    # ---- OOF 预测缓存 ----
    logger.info("生成 OOF 预测缓存...")
    pred_cache: dict[str, np.ndarray] = {}

    model_specs = [
        ("XGB_primary", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary),
        ("LGB_primary", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary),
        ("HGB_primary", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary),
        ("HGB_anchor", build_hgb(ANCHOR_MODEL_PARAMS), X_anchor),
        ("XGB_raw", build_xgb(xgb_raw_params), X_raw),
        ("LGB_raw", build_lgbm(lgb_raw_params), X_raw),
        ("HGB_raw", build_hgb(hgb_raw_params), X_raw),
        ("HGB_anchor_raw", build_hgb(hgb_anchor_raw_params), X_raw),
    ]

    for model_id, estimator, X_used in model_specs:
        tt = time.perf_counter()
        pred_cache[model_id] = cross_val_predict(estimator, X_used, y, cv=cv, n_jobs=-1, method="predict")
        sm = {"R2": float(r2_score(y, pred_cache[model_id])),
              "RMSE": rmse(y, pred_cache[model_id])}
        logger.info("  %s OOF: R2=%.6f RMSE=%.4f (%.1fs)", model_id, sm["R2"], sm["RMSE"], time.perf_counter() - tt)

    # ---- Paper1 AdaBoost 基线 ----
    logger.info("计算 AdaBoost 基线...")
    ada = build_adaboost_model()
    ada_pred = cross_val_predict(ada, X_raw, y, cv=cv, n_jobs=-1, method="predict")
    ada_metrics, ada_fold = fold_metrics_from_cv(cv, X_raw.to_numpy(), y, ada_pred)

    # ---- 消融变体 ----
    logger.info("执行消融变体...")

    v1 = run_global_blend("v1_primary_global_no_anchor",
                          ["XGB_primary", "LGB_primary", "HGB_primary"],
                          pred_cache, cv, X_primary.to_numpy(), y)

    v2 = run_global_blend("v2_dualspace_global",
                          ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                          pred_cache, cv, X_primary.to_numpy(), y)

    v3 = run_piecewise_blend("v3_dualspace_age_piecewise_acdcb",
                             ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                             pred_cache, cv, X_primary.to_numpy(), y, age, AGE_SPLIT_DAY)

    v4 = run_piecewise_blend("v4_raw_age_piecewise",
                             ["XGB_raw", "LGB_raw", "HGB_raw", "HGB_anchor_raw"],
                             pred_cache, cv, X_raw.to_numpy(), y, age, AGE_SPLIT_DAY)

    # --- P1-1: OLS 无约束 stacking ---
    v5_ols = run_ols_unconstrained("v5_ols_unconstrained_global",
                                   ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                                   pred_cache, cv, X_primary.to_numpy(), y)

    variant_objs = [v1, v2, v3, v4, v5_ols]

    variant_desc = {
        "v1_primary_global_no_anchor": "V1: 仅主空间(primary)3模型全局融合，无锚点模型",
        "v2_dualspace_global": "V2: 双空间(primary+anchor)4模型全局融合",
        "v3_dualspace_age_piecewise_acdcb": "V3: 完整ACDCB（双空间+龄期分段融合）",
        "v4_raw_age_piecewise": "V4: Raw特征+分段融合（独立超参数优化后）",
        "v5_ols_unconstrained_global": "V5: OLS无约束线性回归stacking（无simplex约束，P1-1消融）",
    }

    def delta(curr, ref):
        return {"R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
                "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
                "MAE_drop": float(ref["MAE_mean"] - curr["MAE_mean"]),
                "MAPE_drop": float(ref["MAPE_mean"] - curr["MAPE_mean"])}

    # ---- 构建结果 ----
    results: dict[str, Any] = {
        "meta": {
            "study_name": "ACDCB Ablation Study V2",
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 10, "shuffle": True},
            "age_split_day": AGE_SPLIT_DAY,
            "n_samples": int(len(df)),
            "raw_hyperparams_source": str(sys.argv[1]) if len(sys.argv) >= 2 else "reused_engineered_params",
        },
        "variants": {
            "paper1_adaboost": {
                "description": "V0: 论文1基线 AdaBoost",
                "metrics": ada_metrics, "fold_metrics": ada_fold,
                "weights": None, "weights_piecewise": None,
                "convergence_trace": {}, "optimizer_info": {},
            }
        },
    }

    for obj in variant_objs:
        results["variants"][obj.name] = {
            "description": variant_desc[obj.name],
            "metrics": obj.metrics, "fold_metrics": obj.fold_metrics,
            "weights": obj.weights, "weights_piecewise": obj.weights_piecewise,
            "convergence_trace": obj.convergence_trace,
            "optimizer_info": obj.optimizer_info,
        }

    m_p1 = results["variants"]["paper1_adaboost"]["metrics"]
    m_v1 = results["variants"]["v1_primary_global_no_anchor"]["metrics"]
    m_v2 = results["variants"]["v2_dualspace_global"]["metrics"]
    m_v3 = results["variants"]["v3_dualspace_age_piecewise_acdcb"]["metrics"]
    m_v4 = results["variants"]["v4_raw_age_piecewise"]["metrics"]
    m_v5 = results["variants"]["v5_ols_unconstrained_global"]["metrics"]

    results["comparisons"] = {
        "full_vs_paper1": delta(m_v3, m_p1),
        "dualspace_gain_over_primary_only": delta(m_v2, m_v1),
        "age_piecewise_gain_over_global": delta(m_v3, m_v2),
        "feature_engineering_gain_over_raw": delta(m_v3, m_v4),
        "constrained_vs_ols_unconstrained": {
            "constrained_R2_gain": float(m_v2["R2_mean"] - m_v5["R2_mean"]),
            "constrained_RMSE_drop": float(m_v5["RMSE_mean"] - m_v2["RMSE_mean"]),
            "ols_negative_weights": results["variants"]["v5_ols_unconstrained_global"]["optimizer_info"].get("negative_weights", "N/A"),
            "ols_weight_sum": results["variants"]["v5_ols_unconstrained_global"]["optimizer_info"].get("weight_sum", "N/A"),
        },
    }

    # ---- 保存 ----
    oof_df = pd.DataFrame({
        "y_true": y,
        "paper1_adaboost": ada_pred,
        v1.name: v1.pred, v2.name: v2.pred, v3.name: v3.pred,
        v4.name: v4.pred, v5_ols.name: v5_ols.pred, "age": age,
    })
    oof_df.to_csv(pred_dir / "ablation_oof_v2.csv", index=False, encoding="utf-8-sig")

    results["runtime_sec"] = float(time.perf_counter() - t0)

    json_path = metrics_dir / "ablation_results_acdcb_v2.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("消融结果已保存: %s", json_path)

    # ---- 输出汇总 ----
    print("\n" + "=" * 70)
    print("消融 V2 — 关键对比汇总")
    print("=" * 70)
    print(f"V0 (AdaBoost):          R2={m_p1['R2_mean']:.4f}, RMSE={m_p1['RMSE_mean']:.4f}")
    print(f"V1 (Primary+Global):    R2={m_v1['R2_mean']:.4f}, RMSE={m_v1['RMSE_mean']:.4f}")
    print(f"V2 (DualSpace+Global):  R2={m_v2['R2_mean']:.4f}, RMSE={m_v2['RMSE_mean']:.4f}")
    print(f"V3 (ACDCB Full):        R2={m_v3['R2_mean']:.4f}, RMSE={m_v3['RMSE_mean']:.4f}")
    print(f"V4 (Raw+Piecewise):     R2={m_v4['R2_mean']:.4f}, RMSE={m_v4['RMSE_mean']:.4f}")
    print(f"V5 (OLS Unconstrained): R2={m_v5['R2_mean']:.4f}, RMSE={m_v5['RMSE_mean']:.4f}")
    print()
    print(f"Feature Eng gain (V3-V4): dR2={m_v3['R2_mean']-m_v4['R2_mean']:+.6f}")
    print(f"Constrained vs OLS (V2-V5): dR2={m_v2['R2_mean']-m_v5['R2_mean']:+.6f}")
    ols_info = results["variants"]["v5_ols_unconstrained_global"]["optimizer_info"]
    print(f"OLS 负权重数: {ols_info['negative_weights']}")
    print(f"OLS 权重: {results['variants']['v5_ols_unconstrained_global']['weights']}")
    print(f"SLSQP 权重: {results['variants']['v2_dualspace_global']['weights']}")


if __name__ == "__main__":
    main()
