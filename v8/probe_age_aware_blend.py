from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from train import BASE_FEATURES, TARGET_COL, V7_BASELINE_HGB_PARAMS, build_hgb, build_lgbm, build_xgb, feature_engineering, feature_engineering_v7, load_data


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def optimize_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = P.shape[1]
    init = np.full(m, 1.0 / m)

    def obj(w: np.ndarray) -> float:
        pred = P @ w
        return rmse(y, pred)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(m)]
    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return init
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() <= 0:
        return init
    return w / w.sum()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "v8" / "metrics.json").read_text(encoding="utf-8"))
    iter_map = {it["iteration"]: it for it in metrics["iteration_results"]}

    p_hgb = iter_map["Iter-1"]["best_params"]
    p_xgb = iter_map["Iter-2"]["best_params"]
    p_lgb = iter_map["Iter-3"]["best_params"]

    df = load_data(root / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    X_v8 = feature_engineering(X_base)
    X_v7 = feature_engineering_v7(X_base)
    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    model_defs = [
        ("XGBoost_v8", build_xgb(p_xgb), X_v8),
        ("LightGBM_v8", build_lgbm(p_lgb), X_v8),
        ("HGB_v8", build_hgb(p_hgb), X_v8),
        (
            "HGB_v7_baseline",
            HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=42,
                early_stopping=False,
                **V7_BASELINE_HGB_PARAMS,
            ),
            X_v7,
        ),
    ]

    pred_cols = []
    names = []
    for name, est, X_used in model_defs:
        p = cross_val_predict(est, X_used, y, cv=cv, n_jobs=-1, method="predict")
        pred_cols.append(p)
        names.append(name)

    P = np.column_stack(pred_cols)

    # 全局权重
    w_global = optimize_weights(P, y)
    pred_global = P @ w_global

    # 龄期分段权重（<=28天 与 >28天）
    mask_early = age <= 28
    mask_late = ~mask_early

    w_early = optimize_weights(P[mask_early], y[mask_early])
    w_late = optimize_weights(P[mask_late], y[mask_late])

    pred_piece = np.empty_like(y, dtype=float)
    pred_piece[mask_early] = P[mask_early] @ w_early
    pred_piece[mask_late] = P[mask_late] @ w_late

    def fold_stats(pred: np.ndarray) -> dict[str, float]:
        r2s, rmses = [], []
        for _, test_idx in cv.split(X_v8, y):
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

    print("global_weights", {names[i]: float(w_global[i]) for i in range(len(names))})
    print("global_metrics", fold_stats(pred_global))

    print("early_weights", {names[i]: float(w_early[i]) for i in range(len(names))})
    print("late_weights", {names[i]: float(w_late[i]) for i in range(len(names))})
    print("piecewise_metrics", fold_stats(pred_piece))


if __name__ == "__main__":
    main()
