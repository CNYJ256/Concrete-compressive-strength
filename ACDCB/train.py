from __future__ import annotations

"""ACDCB 训练脚本：龄期条件化双空间约束融合。

目标：
1) 在 ACDCB 目录内完成数据加载、特征工程、模型训练与融合；
2) 不依赖 oldversion 目录；
3) 输出包含 R2 / RMSE / MAE / MAPE 的 10 折指标。
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ACDCB.core import (  # noqa: E402
    AGE_SPLIT_DAY,
    BASE_FEATURES,
    BASE_MODEL_PARAMS,
    METHOD_NAME_EN,
    METHOD_NAME_ZH,
    RANDOM_STATE,
    TARGET_COL,
    ANCHOR_MODEL_PARAMS,
    build_hgb,
    build_lgbm,
    build_xgb,
    feature_engineering,
    feature_engineering_anchor,
    is_better,
    load_data,
)


def get_logger() -> logging.Logger:
    """构建并返回训练日志器（幂等）。"""
    logger = logging.getLogger("acdcb_train")
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
    """统一解析训练阶段路径。"""
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "Concrete_Data.xls",
        "paper1_baseline": root / "doc" / "baseline_results.json",
        "model": root / "ACDCB" / "model.joblib",
        "metrics": root / "ACDCB" / "metrics.json",
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAE。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """计算百分比 MAPE（单位：%）。"""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def fold_metrics(cv: KFold, X_ref: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """按给定 CV 切分还原每折指标，并输出均值/标准差。"""
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

    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "MAPE_mean": float(np.mean(mapes)),
        "MAPE_std": float(np.std(mapes)),
    }


def optimize_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在“非负 + 和为1”约束下优化融合权重（目标：最小化 RMSE）。"""
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


def compare_to_ref(curr: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    """计算与参考方案的增益。"""
    payload = {
        "R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
        "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
    }
    if "MAE_mean" in ref:
        payload["MAE_drop"] = float(ref["MAE_mean"] - curr["MAE_mean"])
    if "MAPE_mean" in ref:
        payload["MAPE_drop"] = float(ref["MAPE_mean"] - curr["MAPE_mean"])
    return payload


def main() -> None:
    """ACDCB 训练主入口。"""
    logger = get_logger()
    paths = resolve_paths()

    total_start = time.perf_counter()

    try:
        logger.info("===== ACDCB 训练开始：%s =====", METHOD_NAME_ZH)

        # 1) 加载原始数据，构造主特征空间与锚点特征空间。
        df = load_data(paths["data"])
        X_base = df[BASE_FEATURES].copy()
        X_primary = feature_engineering(X_base)
        X_anchor = feature_engineering_anchor(X_base)
        y = df[TARGET_COL].to_numpy()
        age = X_base["age"].to_numpy()

        # 2) 与评估协议保持一致的 10 折 KFold。
        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # 3) 候选模型池：三类主模型 + 一个锚点模型。
        model_defs = [
            {
                "model_id": "XGBoost",
                "model_family": "XGBoost",
                "feature_space": "primary",
                "best_params": BASE_MODEL_PARAMS["XGBoost"],
                "estimator": build_xgb(BASE_MODEL_PARAMS["XGBoost"]),
                "X": X_primary,
            },
            {
                "model_id": "LightGBM",
                "model_family": "LightGBM",
                "feature_space": "primary",
                "best_params": BASE_MODEL_PARAMS["LightGBM"],
                "estimator": build_lgbm(BASE_MODEL_PARAMS["LightGBM"]),
                "X": X_primary,
            },
            {
                "model_id": "HGB",
                "model_family": "HGB",
                "feature_space": "primary",
                "best_params": BASE_MODEL_PARAMS["HGB"],
                "estimator": build_hgb(BASE_MODEL_PARAMS["HGB"]),
                "X": X_primary,
            },
            {
                "model_id": "HGB_Anchor",
                "model_family": "HGB",
                "feature_space": "anchor",
                "best_params": dict(ANCHOR_MODEL_PARAMS),
                "estimator": build_hgb(ANCHOR_MODEL_PARAMS),
                "X": X_anchor,
            },
        ]

        logger.info("生成 OOF 预测用于融合权重优化...")
        # 4) 对每个候选模型生成 OOF 预测，并记录单模型 CV 统计。
        oof_list: list[np.ndarray] = []
        per_model_cv: list[dict[str, Any]] = []

        for md in model_defs:
            t0 = time.perf_counter()
            pred = cross_val_predict(md["estimator"], md["X"], y, cv=cv, n_jobs=-1, method="predict")
            elapsed = time.perf_counter() - t0

            metrics = fold_metrics(cv, X_primary.to_numpy(), y, pred)
            metrics["oof_time_sec"] = float(elapsed)
            oof_list.append(pred)
            per_model_cv.append(
                {
                    "model_id": md["model_id"],
                    "model_family": md["model_family"],
                    "feature_space": md["feature_space"],
                    "best_params": md["best_params"],
                    "cv_10fold": metrics,
                }
            )
            logger.info(
                "%s OOF: R2=%.6f, RMSE=%.6f, MAE=%.6f, MAPE=%.4f%%",
                md["model_id"],
                metrics["R2_mean"],
                metrics["RMSE_mean"],
                metrics["MAE_mean"],
                metrics["MAPE_mean"],
            )

        P = np.column_stack(oof_list)
        names = [m["model_id"] for m in model_defs]

        # 5) 全局单权重融合。
        w_global = optimize_weights(P, y)
        pred_global = P @ w_global
        m_global = fold_metrics(cv, X_primary.to_numpy(), y, pred_global)

        # 6) 龄期分段融合：早龄期 / 后龄期分别优化权重。
        early_mask = age <= AGE_SPLIT_DAY
        late_mask = ~early_mask

        w_early = optimize_weights(P[early_mask], y[early_mask])
        w_late = optimize_weights(P[late_mask], y[late_mask])

        pred_piece = np.empty_like(y, dtype=float)
        pred_piece[early_mask] = P[early_mask] @ w_early
        pred_piece[late_mask] = P[late_mask] @ w_late
        m_piece = fold_metrics(cv, X_primary.to_numpy(), y, pred_piece)

        logger.info(
            "全局融合: R2=%.6f, RMSE=%.6f, MAE=%.6f, MAPE=%.4f%%",
            m_global["R2_mean"],
            m_global["RMSE_mean"],
            m_global["MAE_mean"],
            m_global["MAPE_mean"],
        )
        logger.info(
            "龄期分段融合: R2=%.6f, RMSE=%.6f, MAE=%.6f, MAPE=%.4f%%",
            m_piece["R2_mean"],
            m_piece["RMSE_mean"],
            m_piece["MAE_mean"],
            m_piece["MAPE_mean"],
        )

        # 7) 自动选择最优融合策略。
        best_strategy = "age_piecewise" if is_better(m_piece, m_global) else "global"
        best_metrics = m_piece if best_strategy == "age_piecewise" else m_global

        logger.info("最优策略: %s", best_strategy)

        # 8) 在全量数据上拟合候选基模型，用于后续推理阶段。
        fitted_models: dict[str, Any] = {}
        model_spaces: dict[str, str] = {}
        for md in model_defs:
            est = md["estimator"]
            est.fit(md["X"], y)
            fitted_models[md["model_id"]] = est
            model_spaces[md["model_id"]] = md["feature_space"]

        # 9) 组织可部署模型包：包含特征列、权重与策略元数据。
        model_bundle = {
            "version": "ACDCB",
            "method_name_zh": METHOD_NAME_ZH,
            "method_name_en": METHOD_NAME_EN,
            "model_type": "age_aware_weighted_ensemble",
            "models": fitted_models,
            "model_spaces": model_spaces,
            "feature_columns_primary": list(X_primary.columns),
            "feature_columns_anchor": list(X_anchor.columns),
            "base_features": BASE_FEATURES,
            "model_order": names,
            "weights_global": {names[i]: float(w_global[i]) for i in range(len(names))},
            "weights_age_piecewise": {
                "age_split_day": AGE_SPLIT_DAY,
                "early": {names[i]: float(w_early[i]) for i in range(len(names))},
                "late": {names[i]: float(w_late[i]) for i in range(len(names))},
            },
            "selected_strategy": best_strategy,
        }
        joblib.dump(model_bundle, paths["model"])

        # 10) 与 paper1 AdaBoost 10 折基准做对照（若存在）。
        compare_to_paper1: dict[str, Any] | None = None
        if paths["paper1_baseline"].exists():
            baseline_payload = json.loads(paths["paper1_baseline"].read_text(encoding="utf-8"))
            ada_cv = baseline_payload["adaboost_cv_10fold"]
            ref_metrics = {
                "R2_mean": float(ada_cv["R2_mean"]),
                "RMSE_mean": float(ada_cv["RMSE_mean"]),
                "MAE_mean": float(ada_cv["MAE_mean"]),
                "MAPE_mean": float(ada_cv["MAPE_mean"]),
            }
            delta = compare_to_ref(best_metrics, ref_metrics)
            compare_to_paper1 = {
                "paper1_adaboost_cv_10fold": ref_metrics,
                "delta": delta,
                "is_better": bool(delta["R2_gain"] > 0 and delta["RMSE_drop"] > 0),
            }

        # 11) 保存完整指标，便于报告撰写与对比。
        metrics_payload: dict[str, Any] = {
            "version": "ACDCB",
            "method_name_zh": METHOD_NAME_ZH,
            "method_name_en": METHOD_NAME_EN,
            "cv_protocol": {
                "type": "KFold",
                "n_splits": 10,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "age_split_day": AGE_SPLIT_DAY,
            "per_model_oof": per_model_cv,
            "global_blend": {
                "weights": {names[i]: float(w_global[i]) for i in range(len(names))},
                "cv_10fold": m_global,
            },
            "age_piecewise_blend": {
                "early_weights": {names[i]: float(w_early[i]) for i in range(len(names))},
                "late_weights": {names[i]: float(w_late[i]) for i in range(len(names))},
                "cv_10fold": m_piece,
            },
            "best_model": {
                "strategy_variant": best_strategy,
                "cv_10fold": best_metrics,
            },
            "training_time_sec": float(time.perf_counter() - total_start),
        }
        if compare_to_paper1 is not None:
            metrics_payload["compare_to_paper1"] = compare_to_paper1

        paths["metrics"].write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "ACDCB 最终10折CV: R2=%.6f±%.6f, RMSE=%.6f±%.6f, MAE=%.6f±%.6f, MAPE=%.4f%%±%.4f%%",
            best_metrics["R2_mean"],
            best_metrics["R2_std"],
            best_metrics["RMSE_mean"],
            best_metrics["RMSE_std"],
            best_metrics["MAE_mean"],
            best_metrics["MAE_std"],
            best_metrics["MAPE_mean"],
            best_metrics["MAPE_std"],
        )
        logger.info("模型已保存: %s", paths["model"])
        logger.info("指标已保存: %s", paths["metrics"])
        logger.info("===== ACDCB 训练完成 =====")

    except Exception as exc:  # noqa: BLE001
        logger.error("ACDCB 训练失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
