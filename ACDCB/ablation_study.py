from __future__ import annotations

"""ACDCB 消融实验脚本。

目标：
1) 围绕 ACDCB 的三类关键创新（特征工程、双空间锚点、龄期分段融合）执行可复现消融；
2) 输出 10 折 CV 指标（R2/RMSE/MAE/MAPE）与每折序列；
3) 输出融合优化收敛轨迹（SLSQP 目标函数迭代）；
4) 产出 OOF 预测文件，供后续论文绘图与对比分析。
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ACDCB.core import (  # noqa: E402
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
from model_factory import build_adaboost_model  # noqa: E402


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
    logger = logging.getLogger("acdcb_ablation")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def resolve_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    doc_dir = root / "doc"
    return {
        "data": root / "data" / "Concrete_Data.xls",
        "doc_dir": doc_dir,
        "ablation_json": doc_dir / "ablation_results_acdcb.json",
        "ablation_oof_csv": doc_dir / "ablation_oof_predictions.csv",
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def fold_metrics_from_cv(cv: KFold, X_ref: np.ndarray, y: np.ndarray, pred: np.ndarray) -> tuple[dict[str, float], dict[str, list[float]]]:
    r2s: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    mapes: list[float] = []

    for _, test_idx in cv.split(X_ref, y):
        yt = y[test_idx]
        yp = pred[test_idx]
        r2s.append(float(r2_score(yt, yp)))
        rmses.append(rmse(yt, yp))
        maes.append(mae(yt, yp))
        mapes.append(mape_percent(yt, yp))

    summary = {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "MAPE_mean": float(np.mean(mapes)),
        "MAPE_std": float(np.std(mapes)),
    }
    fold_payload = {
        "R2": r2s,
        "RMSE": rmses,
        "MAE": maes,
        "MAPE": mapes,
    }
    return summary, fold_payload


def optimize_weights_with_trace(P: np.ndarray, y: np.ndarray, maxiter: int = 500) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    n_models = P.shape[1]
    init = np.full(n_models, 1.0 / n_models, dtype=float)

    def obj(w: np.ndarray) -> float:
        return rmse(y, P @ w)

    trace: list[float] = [float(obj(init))]

    def callback(wk: np.ndarray) -> None:
        trace.append(float(obj(wk)))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(
        obj,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        if w.sum() <= 0:
            w = init
        else:
            w = w / w.sum()

    final_obj = float(obj(w))
    if not trace or abs(trace[-1] - final_obj) > 1e-12:
        trace.append(final_obj)

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", -1)),
        "fun": float(getattr(res, "fun", final_obj)),
    }
    return w, trace, info


def run_global_blend_variant(
    name: str,
    model_order: list[str],
    pred_cache: dict[str, np.ndarray],
    cv: KFold,
    X_ref: np.ndarray,
    y: np.ndarray,
) -> VariantResult:
    P = np.column_stack([pred_cache[m] for m in model_order])
    w, trace, info = optimize_weights_with_trace(P, y)
    pred = P @ w

    metrics, fold_payload = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}

    return VariantResult(
        name=name,
        metrics=metrics,
        fold_metrics=fold_payload,
        weights=weights,
        weights_piecewise=None,
        convergence_trace={"global": trace},
        optimizer_info={"global": info},
        pred=pred,
    )


def run_piecewise_blend_variant(
    name: str,
    model_order: list[str],
    pred_cache: dict[str, np.ndarray],
    cv: KFold,
    X_ref: np.ndarray,
    y: np.ndarray,
    age: np.ndarray,
    split_day: float,
) -> VariantResult:
    P = np.column_stack([pred_cache[m] for m in model_order])

    early_mask = age <= split_day
    late_mask = ~early_mask

    w_early, trace_early, info_early = optimize_weights_with_trace(P[early_mask], y[early_mask])
    w_late, trace_late, info_late = optimize_weights_with_trace(P[late_mask], y[late_mask])

    pred = np.empty_like(y, dtype=float)
    pred[early_mask] = P[early_mask] @ w_early
    pred[late_mask] = P[late_mask] @ w_late

    metrics, fold_payload = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights_piecewise = {
        "early": {model_order[i]: float(w_early[i]) for i in range(len(model_order))},
        "late": {model_order[i]: float(w_late[i]) for i in range(len(model_order))},
    }

    return VariantResult(
        name=name,
        metrics=metrics,
        fold_metrics=fold_payload,
        weights=None,
        weights_piecewise=weights_piecewise,
        convergence_trace={"early": trace_early, "late": trace_late},
        optimizer_info={"early": info_early, "late": info_late},
        pred=pred,
    )


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()
    paths["doc_dir"].mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    logger.info("===== ACDCB 消融实验开始 =====")

    # 1) 数据与特征空间
    df = load_data(paths["data"])
    X_base = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_base)
    X_anchor = feature_engineering_anchor(X_base)
    X_raw = X_base.copy()

    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # 2) 统一生成可复用 OOF 预测缓存（避免重复训练）
    logger.info("生成 OOF 预测缓存...")
    pred_cache: dict[str, np.ndarray] = {}

    model_specs = [
        ("XGB_primary", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary),
        ("LGB_primary", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary),
        ("HGB_primary", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary),
        ("HGB_anchor", build_hgb(ANCHOR_MODEL_PARAMS), X_anchor),
        ("XGB_raw", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_raw),
        ("LGB_raw", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_raw),
        ("HGB_raw", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_raw),
        ("HGB_anchor_raw", build_hgb(ANCHOR_MODEL_PARAMS), X_raw),
    ]

    for model_id, estimator, X_used in model_specs:
        tt = time.perf_counter()
        pred_cache[model_id] = cross_val_predict(estimator, X_used, y, cv=cv, n_jobs=-1, method="predict")
        logger.info("%s OOF 完成，用时 %.2fs", model_id, time.perf_counter() - tt)

    # 3) paper1 基线（AdaBoost）
    logger.info("计算 paper1 AdaBoost 基线（10折 OOF）...")
    ada = build_adaboost_model()
    ada_pred = cross_val_predict(ada, X_raw, y, cv=cv, n_jobs=-1, method="predict")
    ada_metrics, ada_fold = fold_metrics_from_cv(cv, X_raw.to_numpy(), y, ada_pred)

    results: dict[str, Any] = {
        "meta": {
            "study_name": "ACDCB Ablation Study",
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 10, "shuffle": True, "random_state": RANDOM_STATE},
            "age_split_day": AGE_SPLIT_DAY,
            "n_samples": int(len(df)),
            "feature_spaces": {
                "raw": list(X_raw.columns),
                "primary": list(X_primary.columns),
                "anchor": list(X_anchor.columns),
            },
        },
        "variants": {
            "paper1_adaboost": {
                "description": "论文1基线 AdaBoost（无融合、无双空间、无龄期分段）",
                "metrics": ada_metrics,
                "fold_metrics": ada_fold,
                "weights": None,
                "weights_piecewise": None,
                "convergence_trace": {},
                "optimizer_info": {},
            }
        },
        "comparisons": {},
        "optimizer_convergence": {},
        "runtime_sec": None,
    }

    # 4) 消融变体
    logger.info("执行 ACDCB 模块消融...")

    v1 = run_global_blend_variant(
        name="v1_primary_global_no_anchor",
        model_order=["XGB_primary", "LGB_primary", "HGB_primary"],
        pred_cache=pred_cache,
        cv=cv,
        X_ref=X_primary.to_numpy(),
        y=y,
    )

    v2 = run_global_blend_variant(
        name="v2_dualspace_global",
        model_order=["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
        pred_cache=pred_cache,
        cv=cv,
        X_ref=X_primary.to_numpy(),
        y=y,
    )

    v3 = run_piecewise_blend_variant(
        name="v3_dualspace_age_piecewise_acdcb",
        model_order=["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
        pred_cache=pred_cache,
        cv=cv,
        X_ref=X_primary.to_numpy(),
        y=y,
        age=age,
        split_day=AGE_SPLIT_DAY,
    )

    v4 = run_piecewise_blend_variant(
        name="v4_raw_age_piecewise_no_feature_engineering",
        model_order=["XGB_raw", "LGB_raw", "HGB_raw", "HGB_anchor_raw"],
        pred_cache=pred_cache,
        cv=cv,
        X_ref=X_raw.to_numpy(),
        y=y,
        age=age,
        split_day=AGE_SPLIT_DAY,
    )

    variant_objs = [v1, v2, v3, v4]

    variant_desc = {
        "v1_primary_global_no_anchor": "仅主空间(primary)融合：验证锚点模型缺失时表现",
        "v2_dualspace_global": "双空间(primary+anchor)全局融合：验证锚点模型与双空间贡献",
        "v3_dualspace_age_piecewise_acdcb": "完整ACDCB：双空间 + 龄期分段融合",
        "v4_raw_age_piecewise_no_feature_engineering": "去除特征工程，仅保留raw特征 + 分段融合：验证特征工程贡献",
    }

    # 5) 结果入库与对比增益
    def delta(curr: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
        return {
            "R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
            "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
            "MAE_drop": float(ref["MAE_mean"] - curr["MAE_mean"]),
            "MAPE_drop": float(ref["MAPE_mean"] - curr["MAPE_mean"]),
        }

    for obj in variant_objs:
        results["variants"][obj.name] = {
            "description": variant_desc[obj.name],
            "metrics": obj.metrics,
            "fold_metrics": obj.fold_metrics,
            "weights": obj.weights,
            "weights_piecewise": obj.weights_piecewise,
            "convergence_trace": obj.convergence_trace,
            "optimizer_info": obj.optimizer_info,
        }
        results["optimizer_convergence"][obj.name] = obj.convergence_trace

    # 针对关键创新做逐级对比
    m_p1 = results["variants"]["paper1_adaboost"]["metrics"]
    m_v1 = results["variants"]["v1_primary_global_no_anchor"]["metrics"]
    m_v2 = results["variants"]["v2_dualspace_global"]["metrics"]
    m_v3 = results["variants"]["v3_dualspace_age_piecewise_acdcb"]["metrics"]
    m_v4 = results["variants"]["v4_raw_age_piecewise_no_feature_engineering"]["metrics"]

    results["comparisons"] = {
        "full_vs_paper1": delta(m_v3, m_p1),
        "dualspace_gain_over_primary_only": delta(m_v2, m_v1),
        "age_piecewise_gain_over_global": delta(m_v3, m_v2),
        "feature_engineering_gain_over_raw": delta(m_v3, m_v4),
    }

    # 6) OOF 预测落盘（供后续绘图）
    oof_df = pd.DataFrame(
        {
            "y_true": y,
            "paper1_adaboost": ada_pred,
            v1.name: v1.pred,
            v2.name: v2.pred,
            v3.name: v3.pred,
            v4.name: v4.pred,
            "age": age,
        }
    )
    oof_df.to_csv(paths["ablation_oof_csv"], index=False, encoding="utf-8-sig")

    results["runtime_sec"] = float(time.perf_counter() - t0)

    with open(paths["ablation_json"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("消融 JSON 已保存: %s", paths["ablation_json"])
    logger.info("消融 OOF 已保存: %s", paths["ablation_oof_csv"])
    logger.info("完整 ACDCB 与 paper1 对比（R2增益）: %.6f", results["comparisons"]["full_vs_paper1"]["R2_gain"])
    logger.info("===== ACDCB 消融实验完成 =====")


if __name__ == "__main__":
    main()
