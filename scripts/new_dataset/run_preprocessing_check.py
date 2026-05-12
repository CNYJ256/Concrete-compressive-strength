"""Phase 2.2: 数据预处理优化验证 (P2).

对比"中位填充"与"物理边界约束"两种预处理策略对消融结果的影响。
预期差异极小（< 0.0005 R2），但需诚实报告。

用法:
  python scripts/new_dataset/run_preprocessing_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    BASE_FEATURES, TARGET_COL, RANDOM_STATE, BASE_MODEL_PARAMS,
    build_xgb, build_lgbm, build_hgb, load_data,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
METRICS_DIR = RESULTS_DIR / "metrics"

EPS = 1e-8
PHYSICAL_EPS = 1e-6


# ---------------------------------------------------------------------------
# Two preprocessing strategies for feature engineering
# ---------------------------------------------------------------------------
def feature_engineering_median_fill(X: pd.DataFrame) -> np.ndarray:
    """Current strategy: replace Inf/NaN with median."""
    eps = 1e-6
    out = X.copy()

    binder = out["cement"] + out["slag"] + out["fly_ash"]
    total_agg = out["coarse_agg"] + out["fine_agg"]
    age_log = np.log1p(np.maximum(out["age"], 0.0))

    out["binder"] = binder
    out["water_cement_ratio"] = out["water"] / (out["cement"] + eps)
    out["water_binder_ratio"] = out["water"] / (binder + eps)
    out["sp_binder_ratio"] = out["superplasticizer"] / (binder + eps)
    out["scm_ratio"] = (out["slag"] + out["fly_ash"]) / (binder + eps)
    out["fine_ratio_in_agg"] = out["fine_agg"] / (total_agg + eps)

    out["age_log1p"] = age_log
    out["age_sqrt"] = np.sqrt(np.maximum(out["age"], 0.0))
    out["age_pow_0_25"] = np.power(np.maximum(out["age"], 0.0), 0.25)

    out["abrams_index"] = age_log / (out["water_binder_ratio"] + eps)
    out["cement_age_interaction"] = out["cement"] * age_log
    out["binder_age_interaction"] = binder * age_log
    out["wb_age_interaction"] = out["water_binder_ratio"] * age_log

    out["paste_index"] = (out["cement"] + out["slag"] + out["fly_ash"] +
                          out["water"] + out["superplasticizer"]) / (total_agg + eps)
    out["binder_to_agg_ratio"] = binder / (total_agg + eps)
    out["water_to_paste_ratio"] = out["water"] / (out["water"] + binder + out["superplasticizer"] + eps)
    out["cement_fraction_in_binder"] = out["cement"] / (binder + eps)
    out["slag_fraction_in_binder"] = out["slag"] / (binder + eps)
    out["flyash_fraction_in_binder"] = out["fly_ash"] / (binder + eps)
    out["superplasticizer_efficiency"] = out["superplasticizer"] / (out["water"] + eps)
    out["maturity_index"] = age_log * (binder / (out["water"] + eps))
    out["agg_binder_balance"] = total_agg / (binder + eps)
    out["age_inverse"] = 1.0 / (out["age"] + 1.0)
    out["age_wc_interaction"] = age_log * out["water_cement_ratio"]

    out = out.replace([np.inf, -np.inf], np.nan)
    if out.isna().any(axis=1).any():
        out = out.fillna(out.median(numeric_only=True))

    return out.to_numpy(dtype=float)


def feature_engineering_physical_fill(X: pd.DataFrame) -> np.ndarray:
    """Physical boundary strategy: use physically meaningful bounds instead of median.

    For ratio features:
      - When denominator = 0, use the feature's physical upper bound (e.g., 10.0 for w/c)
    For log features:
      - When input <= 0, use log(PHYSICAL_EPS)
    """
    arr = X.to_numpy(dtype=float)
    cement = arr[:, 0]
    slag = arr[:, 1]
    fly_ash = arr[:, 2]
    water = arr[:, 3]
    sp = arr[:, 4]
    coarse = arr[:, 5]
    fine = arr[:, 6]
    age = arr[:, 7]

    binder = cement + slag + fly_ash
    total_agg = coarse + fine
    age_clipped = np.maximum(age, 0.0)
    age_log = np.log1p(age_clipped)

    features = {}

    # Binder
    features["binder"] = binder

    # Ratio features with physical bounds
    # w/c: can't exceed ~10 in practice (extreme case with very low cement)
    wc_denom = np.maximum(cement, PHYSICAL_EPS)
    features["water_cement_ratio"] = np.where(cement > 0, water / wc_denom, 10.0)

    wb_denom = np.maximum(binder, PHYSICAL_EPS)
    features["water_binder_ratio"] = np.where(binder > 0, water / wb_denom, 10.0)

    spb_denom = np.maximum(binder, PHYSICAL_EPS)
    features["sp_binder_ratio"] = np.where(binder > 0, sp / spb_denom, 0.5)

    features["scm_ratio"] = np.where(binder > 0, (slag + fly_ash) / wb_denom, 1.0)

    ta_denom = np.maximum(total_agg, PHYSICAL_EPS)
    features["fine_ratio_in_agg"] = np.where(total_agg > 0, fine / ta_denom, 0.5)

    # Age transforms
    features["age_log1p"] = age_log
    features["age_sqrt"] = np.sqrt(age_clipped)
    features["age_pow_0_25"] = np.power(age_clipped, 0.25)

    # Interaction features
    features["abrams_index"] = np.where(features["water_binder_ratio"] > PHYSICAL_EPS,
                                         age_log / np.maximum(np.abs(features["water_binder_ratio"]), PHYSICAL_EPS),
                                         100.0)
    features["cement_age_interaction"] = cement * age_log
    features["binder_age_interaction"] = binder * age_log
    features["wb_age_interaction"] = features["water_binder_ratio"] * age_log

    features["paste_index"] = np.where(total_agg > 0,
                                        (cement + slag + fly_ash + water + sp) / ta_denom,
                                        10.0)
    features["binder_to_agg_ratio"] = np.where(total_agg > 0, binder / ta_denom, 10.0)

    wp_denom = np.maximum(water + binder + sp, PHYSICAL_EPS)
    features["water_to_paste_ratio"] = water / wp_denom

    features["cement_fraction_in_binder"] = np.where(binder > 0, cement / wb_denom, 1.0)
    features["slag_fraction_in_binder"] = np.where(binder > 0, slag / wb_denom, 0.0)
    features["flyash_fraction_in_binder"] = np.where(binder > 0, fly_ash / wb_denom, 0.0)

    w_denom = np.maximum(water, PHYSICAL_EPS)
    features["superplasticizer_efficiency"] = sp / w_denom

    features["maturity_index"] = age_log * (binder / np.maximum(water, PHYSICAL_EPS))
    features["agg_binder_balance"] = np.where(binder > 0, total_agg / wb_denom, 100.0)
    features["age_inverse"] = 1.0 / (age + 1.0)
    features["age_wc_interaction"] = age_log * features["water_cement_ratio"]

    eng_arr = np.column_stack(list(features.values()))
    eng_arr = np.nan_to_num(eng_arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Prepend base 8 features for consistent dimensionality
    return np.column_stack([arr, eng_arr])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2.2: Data Preprocessing Sanity Check")
    print("=" * 60)

    # Load UCI data
    print("\nLoading UCI dataset...")
    df = load_data(PROJECT_ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    y = df[TARGET_COL].to_numpy()

    # Generate features with both strategies
    print("Generating features with both preprocessing strategies...")
    X_median = feature_engineering_median_fill(X_base)
    X_physical = feature_engineering_physical_fill(X_base)

    print(f"  Median fill: {X_median.shape}, NaN count: {np.sum(np.isnan(X_median))}")
    print(f"  Physical fill: {X_physical.shape}, NaN count: {np.sum(np.isnan(X_physical))}")

    # Check feature-wise differences
    diff = np.abs(X_median - X_physical)
    max_diffs = np.max(diff, axis=0)
    problematic_cols = np.where(max_diffs > 1e-6)[0]
    print(f"  Features with non-trivial differences: {len(problematic_cols)}/{X_median.shape[1]}")
    for idx in problematic_cols[:5]:
        print(f"    Feature {idx}: max diff = {max_diffs[idx]:.6f}")

    # Run simple V1 ablation on both
    print("\nRunning V1 (Primary+Global) with both strategies...")

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    xgb_p = BASE_MODEL_PARAMS["XGBoost"]
    lgb_p = BASE_MODEL_PARAMS["LightGBM"]
    hgb_p = BASE_MODEL_PARAMS["HGB"]

    def run_v1(X_features, label):
        xgb = build_xgb(xgb_p)
        lgb = build_lgbm(lgb_p)
        hgb = build_hgb(hgb_p)

        p_xgb = cross_val_predict(xgb, X_features, y, cv=cv, n_jobs=-1, method="predict")
        p_lgb = cross_val_predict(lgb, X_features, y, cv=cv, n_jobs=-1, method="predict")
        p_hgb = cross_val_predict(hgb, X_features, y, cv=cv)

        P = np.column_stack([p_xgb, p_lgb, p_hgb])

        # SLSQP blend
        n = P.shape[1]
        init = np.full(n, 1.0 / n)
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0) for _ in range(n)]
        res = minimize(lambda w: rmse(y, P @ w), init, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-12, "disp": False})
        w = res.x
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init

        pred = P @ w
        r2 = float(r2_score(y, pred))
        rmse_val = rmse(y, pred)
        print(f"  {label}: R2={r2:.6f}, RMSE={rmse_val:.4f}")
        return r2, rmse_val

    r2_median, rmse_median = run_v1(X_median, "Median fill")
    r2_physical, rmse_physical = run_v1(X_physical, "Physical fill")

    dr2 = r2_physical - r2_median
    print(f"\n  Delta R2 (Physical - Median): {dr2:+.8f}")
    print(f"  Delta RMSE: {rmse_median - rmse_physical:+.8f}")

    significance = "NEGLIGIBLE" if abs(dr2) < 0.0005 else "NOTABLE"
    print(f"  Assessment: {significance} (threshold: 0.0005 R2)")

    # ---- Save ----
    results = {
        "meta": {
            "study": "Data Preprocessing Sanity Check (Phase 2.2)",
            "dataset": "UCI Concrete (N=1,030)",
        },
        "median_fill": {"R2": r2_median, "RMSE": rmse_median},
        "physical_fill": {"R2": r2_physical, "RMSE": rmse_physical},
        "delta": {"R2": dr2, "RMSE": rmse_median - rmse_physical},
        "assessment": significance,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    json_path = METRICS_DIR / "preprocessing_check.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
