from __future__ import annotations

"""Raw特征超参数搜索脚本 (P0-1)。

用 Optuna 对 XGBoost / LightGBM / HGB / HGB_Anchor
在 raw 8维特征上独立做贝叶斯超参数搜索，
目标是最小化 10-fold CV RMSE。
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import (
    BASE_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    load_data,
)
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

N_TRIALS = 200
N_FOLDS = 10
N_JOBS = -1

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_cv() -> KFold:
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def neg_rmse(y_true, y_pred):
    return -float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---- XGBoost search ----
def objective_xgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = XGBRegressor(objective="reg:squarederror", **params)
    cv = make_cv()
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
    return float(np.mean(scores))


# ---- LightGBM search ----
def objective_lgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 63),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 0.1),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMRegressor(objective="regression", **params)
    cv = make_cv()
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
    return float(np.mean(scores))


# ---- HGB search ----
def objective_hgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 31),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 0.1, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
        "random_state": RANDOM_STATE,
        "early_stopping": False,
    }
    model = HistGradientBoostingRegressor(loss="squared_error", **params)
    cv = make_cv()
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
    return float(np.mean(scores))


def run_search(name: str, objective_fn, X: np.ndarray, y: np.ndarray, n_trials: int = N_TRIALS):
    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    t0 = time.perf_counter()
    study.optimize(lambda trial: objective_fn(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    elapsed = time.perf_counter() - t0

    best = study.best_params
    best["random_state"] = RANDOM_STATE
    print(f"\n[{name}] {n_trials} trials, {elapsed:.0f}s, best CV RMSE={-study.best_value:.4f}")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return {"name": name, "best_params": best, "best_cv_neg_rmse": study.best_value,
            "n_trials": n_trials, "elapsed_sec": elapsed, "study": study}


def main():
    print(f"=== Raw特征 Hyperparam Search ===")
    print(f"每个模型 {N_TRIALS} trials, {N_FOLDS}-fold CV")
    print()

    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_raw = df[BASE_FEATURES].to_numpy()
    y = df[TARGET_COL].to_numpy()
    print(f"数据: N={len(y)}, D={X_raw.shape[1]}")

    all_results: dict[str, Any] = {}

    # 1) XGBoost on raw
    print("\n--- XGBoost (raw) ---")
    res_xgb = run_search("XGBoost_raw", objective_xgb, X_raw, y)
    all_results["XGBoost_raw"] = {k: v for k, v in res_xgb.items() if k != "study"}

    # 2) LightGBM on raw
    print("\n--- LightGBM (raw) ---")
    res_lgb = run_search("LightGBM_raw", objective_lgb, X_raw, y)
    all_results["LightGBM_raw"] = {k: v for k, v in res_lgb.items() if k != "study"}

    # 3) HGB on raw (primary equivalent)
    print("\n--- HGB (raw) ---")
    res_hgb = run_search("HGB_raw", objective_hgb, X_raw, y)
    all_results["HGB_raw"] = {k: v for k, v in res_hgb.items() if k != "study"}

    # 4) HGB_Anchor on raw (same as HGB but optimized separately)
    print("\n--- HGB_Anchor (raw) ---")
    res_anchor = run_search("HGB_Anchor_raw", objective_hgb, X_raw, y)
    all_results["HGB_Anchor_raw"] = {k: v for k, v in res_anchor.items() if k != "study"}

    # Save
    out_path = ROOT / "results" / "metrics" / "raw_hyperparams.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # Print summary
    print("\n=== 搜索完成 ===")
    print(f"XGBoost_raw  best_neg_rmse = {all_results['XGBoost_raw']['best_cv_neg_rmse']:.4f}")
    print(f"LightGBM_raw best_neg_rmse = {all_results['LightGBM_raw']['best_cv_neg_rmse']:.4f}")
    print(f"HGB_raw      best_neg_rmse = {all_results['HGB_raw']['best_cv_neg_rmse']:.4f}")
    print(f"HGB_Anchor_raw best_neg_rmse = {all_results['HGB_Anchor_raw']['best_cv_neg_rmse']:.4f}")


if __name__ == "__main__":
    main()
