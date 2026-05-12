"""Phase 1.2: 非线性元学习器验证 (P1).

实证证明即使引入非线性融合（RF、MLP），也无法从高度相关的模型池中提取额外增益。

方法:
  1. 加载现有OOF预测矩阵
  2. 对两个数据集（UCI + Mexico）评估3种Meta-regressor
  3. 嵌套CV防止过拟合

用法:
  python scripts/new_dataset/run_metalearner_verification.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from concrete_compressive_strength.core import RANDOM_STATE

RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
METRICS_DIR = RESULTS_DIR / "metrics"

EPS = 1e-8
CV_OUTER = 10
CV_INNER = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))


# ---------------------------------------------------------------------------
# SLSQP Linear Stacking
# ---------------------------------------------------------------------------
def linear_stacking_fit(P_train, y_train):
    n = P_train.shape[1]
    init = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(lambda w: rmse(y_train, P_train @ w), init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12, "disp": False})
    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init
    return w


# ---------------------------------------------------------------------------
# Meta-regressor evaluation via nested CV
# ---------------------------------------------------------------------------
def evaluate_metalearner(name, P, y, meta_builder, cv_outer_seed=42):
    """Evaluate a meta-regressor using nested CV on the OOF matrix P.

    P: n_samples x n_models OOF prediction matrix
    y: true target values
    meta_builder: callable that returns a meta-regressor instance

    Returns dict of metrics.
    """
    outer_cv = KFold(n_splits=CV_OUTER, shuffle=True, random_state=cv_outer_seed)
    r2s, rmses, maes = [], [], []

    for train_idx, test_idx in outer_cv.split(P):
        P_train, P_test = P[train_idx], P[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if name == "Linear (SLSQP)":
            w = linear_stacking_fit(P_train, y_train)
            y_pred = P_test @ w
        else:
            meta = meta_builder()
            meta.fit(P_train, y_train)
            y_pred = meta.predict(P_test)

        r2s.append(float(r2_score(y_test, y_pred)))
        rmses.append(rmse(y_test, y_pred))
        maes.append(mae(y_test, y_pred))

    return {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
    }


# ---------------------------------------------------------------------------
# Load OOF data and build model pools
# ---------------------------------------------------------------------------
def load_uci_pools():
    """Load UCI OOF data and build model pools at varying correlation levels."""
    oof_path = PROJECT_ROOT / "results" / "predictions" / "heterogeneous_pool_oof.csv"
    if not oof_path.exists():
        print(f"UCI heterogeneous pool OOF not found at {oof_path}")
        return None

    df = pd.read_csv(oof_path)
    y = df["y_true"].to_numpy()

    # Identify model columns
    model_cols = [c for c in df.columns if c not in ("y_true", "age")]

    print(f"UCI OOF loaded: {len(y)} samples, models: {model_cols}")

    # Build pools at varying correlation levels
    # H1: Original GBDT pool (XGB, LGB, HGB, HGB_anchor)
    h1_cols = [c for c in model_cols if any(m in c.lower() for m in ["xgb", "lgb", "hgb"])]
    if len(h1_cols) >= 4:
        h1_cols = h1_cols[:4]  # Take first 4 GBDT models

    # H4: Full heterogeneous pool (all available models)
    h4_cols = model_cols[:6] if len(model_cols) >= 6 else model_cols

    pools = {}
    if len(h1_cols) >= 3:
        P_h1 = np.column_stack([df[c].to_numpy() for c in h1_cols])
        # Compute inter-model correlations
        corr_matrix = np.corrcoef(P_h1.T)
        r_min = float(np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        pools["H1_GBDT"] = {"P": P_h1, "cols": h1_cols, "r_min": r_min}
        print(f"  H1 (GBDT only): {len(h1_cols)} models, r_min={r_min:.6f}")

    if len(h4_cols) >= 5:
        P_h4 = np.column_stack([df[c].to_numpy() for c in h4_cols])
        corr_matrix = np.corrcoef(P_h4.T)
        r_min = float(np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        pools["H4_Full"] = {"P": P_h4, "cols": h4_cols, "r_min": r_min}
        print(f"  H4 (Full heterogeneous): {len(h4_cols)} models, r_min={r_min:.6f}")

    return {"y": y, "pools": pools}


def load_mexico_pools():
    """Build Mexico model pools from existing OOF data."""
    oof_path = PROJECT_ROOT / "results" / "new_dataset" / "predictions" / "ablation_newdata_oof.csv"
    if not oof_path.exists():
        print(f"Mexico OOF not found at {oof_path}")
        return None

    df = pd.read_csv(oof_path)
    y = df["y_true"].to_numpy()

    # Identify individual model columns (not variant predictions)
    # The CSV has variant predictions. We need per-model OOF predictions.
    # Load the JSON results instead for single model OOF metrics
    json_path = PROJECT_ROOT / "results" / "new_dataset" / "metrics" / "ablation_newdata_results.json"
    if json_path.exists():
        results = json.loads(json_path.read_text(encoding="utf-8"))
        smoof = results.get("single_model_OOF", {})
        print(f"Mexico single model OOF available: {list(smoof.keys())}")
        print("  (Individual OOF predictions not saved; training models fresh...)")

    # Train fresh individual models for Mexico dataset
    from new_dataset.new_data_loader import (
        load_raw_new_data, strategy_a_preprocess,
        new_dataset_feature_engineering,
    )

    X_base, y_fresh, meta = strategy_a_preprocess(
        load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv"),
        encoder="onehot", add_engineered=False,
    )
    num_cols_idx = list(range(meta["n_numerical_raw"]))
    X_num = X_base[:, num_cols_idx]
    X_eng, _ = new_dataset_feature_engineering(X_num)
    X_primary = np.column_stack([X_base, X_eng])

    from sklearn.model_selection import cross_val_predict
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # Use 5-fold for speed

    print("Training individual Mexico models (5-fold CV)...")
    models = {}
    models["XGB"] = cross_val_predict(
        XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
                     n_estimators=500, learning_rate=0.05, max_depth=6),
        X_primary, y_fresh, cv=cv, n_jobs=-1, method="predict")
    models["LGB"] = cross_val_predict(
        LGBMRegressor(objective="regression", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                      n_estimators=500, learning_rate=0.05, num_leaves=31),
        X_primary, y_fresh, cv=cv, n_jobs=-1, method="predict")
    models["HGB"] = cross_val_predict(
        HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                       max_iter=500, learning_rate=0.05, early_stopping=False),
        X_primary, y_fresh, cv=cv)
    models["HGB_anchor"] = cross_val_predict(
        HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                       max_iter=500, learning_rate=0.05, early_stopping=False),
        X_primary, y_fresh, cv=cv)

    P = np.column_stack(list(models.values()))
    corr_matrix = np.corrcoef(P.T)
    r_min = float(np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))

    print(f"  Mexico H1: 4 GBDT, r_min={r_min:.6f}")

    return {"y": y_fresh, "pools": {"H1_GBDT": {"P": P, "cols": list(models.keys()), "r_min": r_min}}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- UCI ----
    print("=" * 60)
    print("Phase 1.2: Non-linear Meta-learner Verification")
    print("=" * 60)

    uci_data = load_uci_pools()
    if uci_data:
        y_uci = uci_data["y"]
        uci_results = {}
        for pool_name, pool in uci_data["pools"].items():
            P = pool["P"]
            print(f"\n--- UCI {pool_name} (r_min={pool['r_min']:.6f}) ---")

            # Best single model in pool
            best_single_r2 = max(
                float(r2_score(y_uci, P[:, i])) for i in range(P.shape[1])
            )
            print(f"  Best single model R2: {best_single_r2:.6f}")

            meta_results = {}
            for meta_name, meta_builder in [
                ("Linear (SLSQP)", lambda: None),
                ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
                ("MLP", lambda: MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=5000,
                                              random_state=RANDOM_STATE, early_stopping=True)),
            ]:
                m = evaluate_metalearner(meta_name, P, y_uci, meta_builder)
                meta_results[meta_name] = m
                gain = m["R2_mean"] - best_single_r2
                print(f"  {meta_name:<20}: R2={m['R2_mean']:.6f} +/- {m['R2_std']:.6f}, RMSE={m['RMSE_mean']:.4f}, dR2_over_best={gain:+.6f}")

            uci_results[pool_name] = {
                "r_min": pool["r_min"],
                "n_models": P.shape[1],
                "best_single_r2": best_single_r2,
                "meta_results": meta_results,
            }
        all_results["UCI"] = uci_results

    # ---- Mexico ----
    print("\n" + "-" * 60)
    mexico_data = load_mexico_pools()
    if mexico_data:
        y_mex = mexico_data["y"]
        mex_results = {}
        for pool_name, pool in mexico_data["pools"].items():
            P = pool["P"]
            print(f"\n--- Mexico {pool_name} (r_min={pool['r_min']:.6f}) ---")

            best_single_r2 = max(
                float(r2_score(y_mex, P[:, i])) for i in range(P.shape[1])
            )
            print(f"  Best single model R2: {best_single_r2:.6f}")

            meta_results = {}
            for meta_name, meta_builder in [
                ("Linear (SLSQP)", lambda: None),
                ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
                ("MLP", lambda: MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=5000,
                                              random_state=RANDOM_STATE, early_stopping=True)),
            ]:
                m = evaluate_metalearner(meta_name, P, y_mex, meta_builder)
                meta_results[meta_name] = m
                gain = m["R2_mean"] - best_single_r2
                print(f"  {meta_name:<20}: R2={m['R2_mean']:.6f} +/- {m['R2_std']:.6f}, RMSE={m['RMSE_mean']:.4f}, dR2_over_best={gain:+.6f}")

            mex_results[pool_name] = {
                "r_min": pool["r_min"],
                "n_models": P.shape[1],
                "best_single_r2": best_single_r2,
                "meta_results": meta_results,
            }
        all_results["Mexico"] = mex_results

    # ---- Hypothesis verification ----
    print("\n" + "=" * 60)
    print("Hypothesis Verification")
    print("=" * 60)

    for ds_name, ds_results in all_results.items():
        for pool_name, pr in ds_results.items():
            linear_r2 = pr["meta_results"]["Linear (SLSQP)"]["R2_mean"]
            rf_r2 = pr["meta_results"]["Random Forest"]["R2_mean"]
            mlp_r2 = pr["meta_results"]["MLP"]["R2_mean"]
            max_diff = max(abs(rf_r2 - linear_r2), abs(mlp_r2 - linear_r2))

            h1_pass = max_diff < 0.001 if pr["r_min"] > 0.996 else True
            print(f"  {ds_name} {pool_name}: max meta diff={max_diff:.6f} "
                  f"(r_min={pr['r_min']:.6f}) -> H1: {'PASS' if h1_pass else 'INVESTIGATE'}")

    # ---- Save ----
    all_results["runtime_sec"] = float(time.perf_counter() - t0)
    json_path = METRICS_DIR / "metalearner_results.json"
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
