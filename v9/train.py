from __future__ import annotations

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

from v8.train import (
    BASE_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    V7_BASELINE_HGB_PARAMS,
    build_hgb,
    build_lgbm,
    build_xgb,
    feature_engineering,
    feature_engineering_v7,
    is_better,
    load_data,
)


AGE_SPLIT_DAY = 28


def get_logger() -> logging.Logger:
    logger = logging.getLogger("v9_train")
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
    return {
        "root": root,
        "data": root / "data" / "Concrete_Data.xls",
        "prev_v8": root / "v8" / "metrics.json",
        "model": root / "v9" / "model.joblib",
        "metrics": root / "v9" / "metrics.json",
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fold_metrics(cv: KFold, X_ref: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r2s: list[float] = []
    rmses: list[float] = []
    for _, test_idx in cv.split(X_ref, y):
        yt = y[test_idx]
        yp = pred[test_idx]
        r2s.append(float(r2_score(yt, yp)))
        rmses.append(rmse(yt, yp))

    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
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


def compare_to_ref(curr: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    return {
        "R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
        "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
    }


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()

    total_start = time.perf_counter()

    try:
        logger.info("===== v9 训练开始：龄期分段自适应融合 =====")

        prev_v8_payload = json.loads(paths["prev_v8"].read_text(encoding="utf-8"))
        iter_map = {it["iteration"]: it for it in prev_v8_payload["iteration_results"]}

        p_hgb_v8 = iter_map["Iter-1"]["best_params"]
        p_xgb_v8 = iter_map["Iter-2"]["best_params"]
        p_lgb_v8 = iter_map["Iter-3"]["best_params"]

        df = load_data(paths["data"])
        X_base = df[BASE_FEATURES].copy()
        X_v8 = feature_engineering(X_base)
        X_v7 = feature_engineering_v7(X_base)
        y = df[TARGET_COL].to_numpy()
        age = X_base["age"].to_numpy()

        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        model_defs = [
            {
                "model_id": "XGBoost_v8",
                "model_family": "XGBoost",
                "feature_space": "v8",
                "best_params": p_xgb_v8,
                "estimator": build_xgb(p_xgb_v8),
                "X": X_v8,
            },
            {
                "model_id": "LightGBM_v8",
                "model_family": "LightGBM",
                "feature_space": "v8",
                "best_params": p_lgb_v8,
                "estimator": build_lgbm(p_lgb_v8),
                "X": X_v8,
            },
            {
                "model_id": "HGB_v8",
                "model_family": "HGB",
                "feature_space": "v8",
                "best_params": p_hgb_v8,
                "estimator": build_hgb(p_hgb_v8),
                "X": X_v8,
            },
            {
                "model_id": "HGB_v7_baseline",
                "model_family": "HGB",
                "feature_space": "v7",
                "best_params": dict(V7_BASELINE_HGB_PARAMS),
                "estimator": build_hgb(V7_BASELINE_HGB_PARAMS),
                "X": X_v7,
            },
        ]

        logger.info("生成 OOF 预测用于融合权重优化...")
        oof_list: list[np.ndarray] = []
        per_model_cv: list[dict[str, Any]] = []

        for md in model_defs:
            t0 = time.perf_counter()
            pred = cross_val_predict(md["estimator"], md["X"], y, cv=cv, n_jobs=-1, method="predict")
            elapsed = time.perf_counter() - t0
            metrics = fold_metrics(cv, X_v8.to_numpy(), y, pred)
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
                "%s OOF: R2=%.6f, RMSE=%.6f",
                md["model_id"],
                metrics["R2_mean"],
                metrics["RMSE_mean"],
            )

        P = np.column_stack(oof_list)
        names = [m["model_id"] for m in model_defs]

        # 全局融合
        w_global = optimize_weights(P, y)
        pred_global = P @ w_global
        m_global = fold_metrics(cv, X_v8.to_numpy(), y, pred_global)

        # 龄期分段融合
        early_mask = age <= AGE_SPLIT_DAY
        late_mask = ~early_mask

        w_early = optimize_weights(P[early_mask], y[early_mask])
        w_late = optimize_weights(P[late_mask], y[late_mask])

        pred_piece = np.empty_like(y, dtype=float)
        pred_piece[early_mask] = P[early_mask] @ w_early
        pred_piece[late_mask] = P[late_mask] @ w_late
        m_piece = fold_metrics(cv, X_v8.to_numpy(), y, pred_piece)

        logger.info(
            "全局融合: R2=%.6f, RMSE=%.6f",
            m_global["R2_mean"],
            m_global["RMSE_mean"],
        )
        logger.info(
            "龄期分段融合: R2=%.6f, RMSE=%.6f",
            m_piece["R2_mean"],
            m_piece["RMSE_mean"],
        )

        best_strategy = "age_piecewise" if is_better(m_piece, m_global) else "global"
        best_metrics = m_piece if best_strategy == "age_piecewise" else m_global

        logger.info("最优策略: %s", best_strategy)

        # 全量拟合模型
        fitted_models: dict[str, Any] = {}
        model_spaces: dict[str, str] = {}
        for md in model_defs:
            est = md["estimator"]
            est.fit(md["X"], y)
            fitted_models[md["model_id"]] = est
            model_spaces[md["model_id"]] = md["feature_space"]

        model_bundle = {
            "version": "v9",
            "strategy": "Age-aware piecewise weighted blending with v7 anchor",
            "model_type": "age_aware_weighted_ensemble",
            "models": fitted_models,
            "model_spaces": model_spaces,
            "feature_columns": list(X_v8.columns),
            "feature_columns_v8": list(X_v8.columns),
            "feature_columns_v7": list(X_v7.columns),
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

        v8_best = prev_v8_payload["best_model"]["cv_10fold"]
        v7_base = prev_v8_payload["baseline_v7"]

        metrics_payload = {
            "version": "v9",
            "strategy": "Age-aware piecewise weighted blending with v7 anchor",
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
            "compare_to_v8": {
                "v8": v8_best,
                "delta": compare_to_ref(best_metrics, v8_best),
                "is_better": bool(compare_to_ref(best_metrics, v8_best)["R2_gain"] > 0 and compare_to_ref(best_metrics, v8_best)["RMSE_drop"] > 0),
            },
            "compare_to_v7": {
                "v7": v7_base,
                "delta": compare_to_ref(best_metrics, v7_base),
                "is_better": bool(compare_to_ref(best_metrics, v7_base)["R2_gain"] > 0 and compare_to_ref(best_metrics, v7_base)["RMSE_drop"] > 0),
            },
            "training_time_sec": float(time.perf_counter() - total_start),
        }

        paths["metrics"].write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "v9 最终10折CV: R2=%.6f±%.6f, RMSE=%.6f±%.6f",
            best_metrics["R2_mean"],
            best_metrics["R2_std"],
            best_metrics["RMSE_mean"],
            best_metrics["RMSE_std"],
        )
        logger.info("模型已保存: %s", paths["model"])
        logger.info("指标已保存: %s", paths["metrics"])
        logger.info("===== v9 训练完成 =====")

    except Exception as exc:  # noqa: BLE001
        logger.error("v9 训练失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
