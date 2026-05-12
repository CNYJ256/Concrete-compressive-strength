"""Phase 3.1: SLSQP安全机制验证 (P1).

实证证明约束优化在基模型质量参差不齐时确实起到防止性能崩塌的作用。

方法:
  1. 构建4组弱模型库变体（欠拟合/过拟合/噪声/混合）
  2. 对比OLS无约束 vs SLSQP约束的权重和性能
  3. 与强模型库（当前ACDCB）对比

用法:
  python scripts/new_dataset/run_slsqp_safeguard.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from concrete_compressive_strength.core import (
    BASE_FEATURES, TARGET_COL, RANDOM_STATE, load_data,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
METRICS_DIR = RESULTS_DIR / "metrics"

EPS = 1e-8
CV_FOLDS = 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def optimize_slsqp(P, y):
    """SLSQP constrained optimization (simplex)."""
    n = P.shape[1]
    init = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(lambda w: rmse(y, P @ w), init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12, "disp": False})
    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init
    return w, res.success


def optimize_ols(P, y):
    """OLS unconstrained optimization."""
    ols = LinearRegression(fit_intercept=False)
    ols.fit(P, y)
    return ols.coef_


# ---------------------------------------------------------------------------
# Build model pools with varying quality
# ---------------------------------------------------------------------------
def build_pools(X, y, cv):
    """Build 5 model pools: Strong, Underfit, Overfit, Noisy, Mixed."""
    print("Building model pools...")

    # --- Strong pool (normal models) ---
    xgb_normal = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                               n_jobs=-1, tree_method="hist",
                               n_estimators=500, learning_rate=0.05, max_depth=6)
    lgb_normal = LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                                n_jobs=-1, verbosity=-1,
                                n_estimators=500, learning_rate=0.05, num_leaves=31)
    hgb_normal = HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                                max_iter=500, learning_rate=0.05, early_stopping=False)
    hgb_normal2 = HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE + 1,
                                                 max_iter=500, learning_rate=0.05, early_stopping=False)

    p_strong = {
        "XGB": cross_val_predict(xgb_normal, X, y, cv=cv, n_jobs=-1, method="predict"),
        "LGB": cross_val_predict(lgb_normal, X, y, cv=cv, n_jobs=-1, method="predict"),
        "HGB": cross_val_predict(hgb_normal, X, y, cv=cv),
        "HGB_2": cross_val_predict(hgb_normal2, X, y, cv=cv),
    }

    # --- Underfit pool ---
    xgb_under = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                              n_jobs=-1, tree_method="hist",
                              n_estimators=5, max_depth=2, learning_rate=0.3)
    lgb_under = LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                               n_jobs=-1, verbosity=-1,
                               n_estimators=5, num_leaves=4, learning_rate=0.3)
    hgb_under = HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                               max_iter=10, learning_rate=0.3, early_stopping=False)
    hgb_under2 = HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE + 1,
                                                max_iter=10, learning_rate=0.3, early_stopping=False)

    p_underfit = {
        "XGB_under": cross_val_predict(xgb_under, X, y, cv=cv, n_jobs=-1, method="predict"),
        "LGB_under": cross_val_predict(lgb_under, X, y, cv=cv, n_jobs=-1, method="predict"),
        "HGB_under": cross_val_predict(hgb_under, X, y, cv=cv),
        "HGB_under2": cross_val_predict(hgb_under2, X, y, cv=cv),
    }

    # --- Overfit pool ---
    xgb_over = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                             n_jobs=-1, tree_method="hist",
                             n_estimators=500, max_depth=20, min_child_weight=0,
                             learning_rate=0.05, subsample=1.0, reg_alpha=0, reg_lambda=0)
    lgb_over = LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                              n_jobs=-1, verbosity=-1,
                              n_estimators=500, num_leaves=256, min_child_samples=1,
                              learning_rate=0.05, subsample=1.0, reg_alpha=0, reg_lambda=0)
    hgb_over = HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                              max_iter=500, max_depth=None, max_leaf_nodes=255,
                                              min_samples_leaf=1, l2_regularization=0,
                                              early_stopping=False)

    p_overfit = {
        "XGB_over": cross_val_predict(xgb_over, X, y, cv=cv, n_jobs=-1, method="predict"),
        "LGB_over": cross_val_predict(lgb_over, X, y, cv=cv, n_jobs=-1, method="predict"),
        "HGB_over": cross_val_predict(hgb_over, X, y, cv=cv),
    }

    # --- Noisy pool (normal predictions + Gaussian noise) ---
    rng = np.random.RandomState(RANDOM_STATE)
    p_noisy = {}
    for name, pred in p_strong.items():
        p_noisy[f"{name}_noisy"] = pred + rng.normal(0, 5.0, size=len(pred))

    # --- Mixed quality pool (2 normal + 2 underfit) ---
    p_mixed = {
        "XGB_normal": p_strong["XGB"],
        "LGB_normal": p_strong["LGB"],
        "HGB_under": p_underfit["HGB_under"],
        "HGB_under2": p_underfit["HGB_under2"],
    }

    pools = {
        "Strong": p_strong,
        "Underfit": p_underfit,
        "Overfit": p_overfit,
        "Noisy": p_noisy,
        "Mixed": p_mixed,
    }

    # Print quality
    for name, pool in pools.items():
        P = np.column_stack(list(pool.values()))
        r2s = [float(r2_score(y, P[:, i])) for i in range(P.shape[1])]
        print(f"  {name}: R2 range [{min(r2s):.4f}, {max(r2s):.4f}], mean={np.mean(r2s):.4f}")

    return pools


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3.1: SLSQP Safeguard Mechanism Verification")
    print("=" * 60)

    # Load UCI data
    print("\nLoading UCI dataset...")
    df = load_data(PROJECT_ROOT / "data" / "Concrete_Data.xls")
    X = df[BASE_FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()
    print(f"  N={len(y)}, features={X.shape[1]}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Build pools
    pools = build_pools(X, y, cv)

    # For each pool, compare OLS vs SLSQP
    print("\n" + "=" * 90)
    print(f"{'Pool':<12} {'Method':<10} {'R2':>10} {'RMSE':>10} {'#NegW':>8} {'Weights'}")
    print("-" * 90)

    all_results = {}

    for pool_name, pool in pools.items():
        P = np.column_stack(list(pool.values()))
        model_names = list(pool.keys())

        # OLS
        w_ols = optimize_ols(P, y)
        pred_ols = P @ w_ols
        r2_ols = float(r2_score(y, pred_ols))
        rmse_ols = rmse(y, pred_ols)
        n_neg_ols = int(np.sum(w_ols < 0))

        # SLSQP
        w_slsqp, success = optimize_slsqp(P, y)
        pred_slsqp = P @ w_slsqp
        r2_slsqp = float(r2_score(y, pred_slsqp))
        rmse_slsqp = rmse(y, pred_slsqp)
        n_neg_slsqp = int(np.sum(w_slsqp < 0))

        # Format weights
        ols_w_str = ", ".join([f"{model_names[i]}={w_ols[i]:.3f}" for i in range(len(model_names))])
        slsqp_w_str = ", ".join([f"{model_names[i]}={w_slsqp[i]:.3f}" for i in range(len(model_names))])

        print(f"{pool_name:<12} {'OLS':<10} {r2_ols:>10.4f} {rmse_ols:>10.4f} {n_neg_ols:>8}  {ols_w_str}")
        print(f"{'':<12} {'SLSQP':<10} {r2_slsqp:>10.4f} {rmse_slsqp:>10.4f} {n_neg_slsqp:>8}  {slsqp_w_str}")

        all_results[pool_name] = {
            "n_models": P.shape[1],
            "OLS": {
                "R2": r2_ols, "RMSE": rmse_ols,
                "weights": {model_names[i]: float(w_ols[i]) for i in range(len(model_names))},
                "negative_weights": n_neg_ols,
            },
            "SLSQP": {
                "R2": r2_slsqp, "RMSE": rmse_slsqp,
                "weights": {model_names[i]: float(w_slsqp[i]) for i in range(len(model_names))},
                "negative_weights": n_neg_slsqp,
                "success": success,
            },
            "SLSQP_vs_OLS": {
                "R2_gain": r2_slsqp - r2_ols,
                "RMSE_drop": rmse_ols - rmse_slsqp,
            },
        }

    # ---- Hypothesis verification ----
    print("\n" + "=" * 60)
    print("Hypothesis Verification")
    print("=" * 60)

    strong_ols = all_results["Strong"]["OLS"]
    strong_slsqp = all_results["Strong"]["SLSQP"]

    # H1: Strong pool -> OLS has no negative weights (or very few)
    h1_pass = strong_ols["negative_weights"] == 0
    print(f"H0 (strong pool): OLS neg weights = {strong_ols['negative_weights']} (expect 0) -> {'PASS' if h1_pass else 'NOTE: unexpected'}")

    # H2: Weak pools -> OLS produces negative weights
    for pool_name in ["Underfit", "Overfit", "Noisy"]:
        n_neg = all_results[pool_name]["OLS"]["negative_weights"]
        h2_pass = n_neg > 0
        print(f"H1 ({pool_name} pool): OLS neg weights = {n_neg} (expect >0) -> {'PASS' if h2_pass else 'INVESTIGATE'}")

    # H3: Weak pools -> SLSQP >= OLS
    for pool_name in all_results:
        gain = all_results[pool_name]["SLSQP_vs_OLS"]["R2_gain"]
        h3_pass = gain >= -0.001  # Allow tiny numerical noise
        print(f"H2 ({pool_name}): SLSQP vs OLS dR2={gain:+.6f} (expect >=0) -> {'PASS' if h3_pass else 'INVESTIGATE'}")

    # H4: Mixed pool -> SLSQP implicitly selects good models
    mixed_slsqp = all_results["Mixed"]["SLSQP"]
    mixed_ols = all_results["Mixed"]["OLS"]
    slsqp_has_zero = any(abs(v) < 1e-6 for v in mixed_slsqp["weights"].values())
    print(f"H3 (Mixed pool): SLSQP clips weak models to 0? {'Yes' if slsqp_has_zero else 'No'}")

    # ---- Save ----
    output = {
        "meta": {
            "study": "SLSQP Safeguard Mechanism Verification (Phase 3.1)",
            "dataset": "UCI Concrete (N=1,030)",
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
        "pools": all_results,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    json_path = METRICS_DIR / "slsqp_safeguard_results.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
