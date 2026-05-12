"""P1-2: Soft Weighting (Sigmoid) vs Hard Threshold Experiment.

实现 Sigmoid 平滑权重函数替代 if/else 硬阈值分流：
  alpha(t) = sigma(-kappa * (t - tau))
  w(t) = alpha(t) * w_early + (1-alpha(t)) * w_late

在 UCI 和新数据集上测试不同 kappa 值，对比硬阈值 vs 软加权。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

EPS = 1e-8


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def optimize_weights(P, y, maxiter=500):
    n = P.shape[1]
    init = np.full(n, 1.0 / n)

    def obj(w):
        return rmse(y, P @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": maxiter, "ftol": 1e-12, "disp": False})
    if not res.success:
        return init
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else init


def soft_blend_predict(P, age, tau, kappa, y_true=None):
    """Soft sigmoid blending with age-dependent weights.

    alpha(t) = sigmoid(-kappa * (t - tau))
    w(t) = alpha(t) * w_early + (1 - alpha(t)) * w_late

    Weights w_early and w_late are optimized on early/late subsets defined
    by the sigmoid transition.
    """
    n_models = P.shape[1]

    # Optimize w_early on samples where alpha > 0.5 (mostly early)
    alpha_vec = sigmoid(-kappa * (age - tau))
    early_mask = alpha_vec >= 0.5
    late_mask = alpha_vec < 0.5

    if early_mask.sum() < n_models or late_mask.sum() < n_models:
        # Not enough samples for piecewise — fall back to global blend
        w_global = optimize_weights(P, y_true) if y_true is not None else np.full(n_models, 1.0 / n_models)
        pred = P @ w_global
        return pred, {"method": "global_fallback", "weights_global": w_global.tolist()}

    w_early = optimize_weights(P[early_mask], y_true[early_mask])
    w_late = optimize_weights(P[late_mask], y_true[late_mask])

    # Compute sample-wise blended weights
    n_samples = P.shape[0]
    weights_per_sample = np.zeros((n_samples, n_models))
    pred = np.zeros(n_samples)
    for i in range(n_samples):
        a = sigmoid(-kappa * (age[i] - tau))
        w = a * w_early + (1 - a) * w_late
        weights_per_sample[i] = w
        pred[i] = P[i] @ w

    info = {
        "method": "sigmoid_soft_blend",
        "tau": tau, "kappa": kappa,
        "w_early": w_early.tolist(),
        "w_late": w_late.tolist(),
        "alpha_stats": {"mean": float(np.mean(alpha_vec)), "std": float(np.std(alpha_vec)),
                        "min": float(np.min(alpha_vec)), "max": float(np.max(alpha_vec))},
    }
    return pred, info


def hard_blend_predict(P, age, tau, y_true):
    """Hard threshold piecewise blending (ACDCB original)."""
    n_models = P.shape[1]
    early_mask = age <= tau
    late_mask = ~early_mask

    if early_mask.sum() < n_models or late_mask.sum() < n_models:
        w_g = optimize_weights(P, y_true)
        pred = P @ w_g
        return pred, {"method": "hard_fallback"}

    w_early = optimize_weights(P[early_mask], y_true[early_mask])
    w_late = optimize_weights(P[late_mask], y_true[late_mask])

    pred = np.empty(len(y_true))
    pred[early_mask] = P[early_mask] @ w_early
    pred[late_mask] = P[late_mask] @ w_late

    return pred, {"method": "hard_threshold", "tau": tau,
                   "w_early": w_early.tolist(), "w_late": w_late.tolist()}


def run_soft_weighting_experiment(dataset_name, pred_cache_path, age, y_true, model_order):
    """Run soft vs hard weighting comparison on one dataset."""
    print(f"\n{'='*60}")
    print(f"Soft Weighting Experiment: {dataset_name}")
    print(f"{'='*60}")

    P = np.column_stack([pred_cache_path[m] for m in model_order])

    # Baseline: global blend (no age conditioning)
    w_global = optimize_weights(P, y_true)
    pred_global = P @ w_global
    r2_global = float(r2_score(y_true, pred_global))
    rmse_global = rmse(y_true, pred_global)
    print(f"  Global blend: R2={r2_global:.6f}, RMSE={rmse_global:.4f}")

    # Hard threshold at tau=28
    tau = 28
    pred_hard, info_hard = hard_blend_predict(P, age, tau, y_true)
    r2_hard = float(r2_score(y_true, pred_hard))
    rmse_hard = rmse(y_true, pred_hard)
    print(f"  Hard threshold (tau={tau}): R2={r2_hard:.6f}, RMSE={rmse_hard:.4f}")

    # Soft weighting with varying kappa
    kappas = [0.1, 0.5, 1.0, 2.0, 5.0]
    soft_results = []
    print(f"\n  {'Kappa':<8} {'R2':>10} {'RMSE':>10} {'vs Global':>12} {'vs Hard':>12}")
    print(f"  {'-'*52}")
    for kappa in kappas:
        pred_soft, info = soft_blend_predict(P, age, tau, kappa, y_true)
        r2_s = float(r2_score(y_true, pred_soft))
        rmse_s = rmse(y_true, pred_soft)
        d_global = r2_s - r2_global
        d_hard = r2_s - r2_hard
        soft_results.append({
            "kappa": kappa, "R2": r2_s, "RMSE": rmse_s,
            "delta_vs_global": d_global, "delta_vs_hard": d_hard,
            "alpha_mean": info.get("alpha_stats", {}).get("mean", float("nan")),
        })
        print(f"  {kappa:<8} {r2_s:>10.6f} {rmse_s:>10.4f} {d_global:>+12.6f} {d_hard:>+12.6f}")

    return {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "n_models": len(model_order),
        "model_order": model_order,
        "global_blend": {"R2": r2_global, "RMSE": rmse_global},
        "hard_threshold": {"R2": r2_hard, "RMSE": rmse_hard, "tau": tau},
        "soft_weighting": soft_results,
    }


def load_pred_cache_from_oof(oof_path, model_cols):
    """Load model predictions from OOF CSV file."""
    df = pd.read_csv(oof_path)
    y_true = df["y_true"].to_numpy()
    pred_cache = {}
    for col in model_cols:
        if col in df.columns:
            pred_cache[col] = df[col].to_numpy()
        else:
            print(f"  WARNING: column {col} not found in {oof_path}")
    return pred_cache, y_true


def main():
    results = {}

    # ---- UCI Dataset ----
    uci_oof_path = PROJECT_ROOT / "results" / "predictions" / "ablation_oof_v2.csv"
    if uci_oof_path.exists():
        # The ablation_oof_v2.csv doesn't contain individual model predictions
        # We need to run from scratch or use the SLSQP blending results
        # For UCI, we need the individual model OOF predictions
        # Let's load from the existing ablation results
        print("UCI: Loading individual model OOF from ablation results...")
        # Since the OOF CSV only has blended predictions, we need to compute individual ones
        # Let's run a lightweight version
        SRC_DIR = PROJECT_ROOT / "src"
        sys.path.insert(0, str(SRC_DIR))
        from concrete_compressive_strength.core import (
            load_data, BASE_MODEL_PARAMS, ANCHOR_MODEL_PARAMS,
            BASE_FEATURES, TARGET_COL, AGE_SPLIT_DAY, RANDOM_STATE,
            feature_engineering, feature_engineering_anchor,
            build_xgb, build_lgbm, build_hgb,
        )
        from sklearn.model_selection import KFold, cross_val_predict

        df = load_data(PROJECT_ROOT / "data" / "Concrete_Data.xls")
        X_base = df[BASE_FEATURES].copy()
        X_primary = feature_engineering(X_base)
        X_anchor = feature_engineering_anchor(X_base)
        y = df[TARGET_COL].to_numpy()
        age = X_base["age"].to_numpy()

        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # Generate individual model OOF
        models = {
            "XGB_primary": (build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary),
            "LGB_primary": (build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary),
            "HGB_primary": (build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary),
            "HGB_anchor": (build_hgb(ANCHOR_MODEL_PARAMS), X_anchor),
        }

        pred_cache = {}
        for name, (est, X_used) in models.items():
            pred_cache[name] = cross_val_predict(est, X_used, y, cv=cv, n_jobs=-1, method="predict")
            print(f"  {name}: R2={r2_score(y, pred_cache[name]):.6f}")

        model_order = ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"]
        uci_result = run_soft_weighting_experiment(
            "UCI (N=1,030)", pred_cache, age, y, model_order)
        results["uci"] = uci_result

    # ---- New Dataset ----
    new_oof_path = PROJECT_ROOT / "results" / "new_dataset" / "predictions" / "ablation_newdata_oof.csv"
    if new_oof_path.exists():
        print("\nNew Dataset: Individual model OOF not in CSV, running from scratch...")
        from new_dataset.new_data_loader import (
            load_raw_new_data, strategy_a_preprocess, get_age_array,
            new_dataset_feature_engineering,
        )
        from new_dataset.run_ablation_new_data import (
            build_xgb, build_lgb, build_hgb, anchor_feature_engineering,
        )
        from sklearn.model_selection import KFold, cross_val_predict

        # Load HPO results
        hpo_path = PROJECT_ROOT / "results" / "new_dataset" / "hyperparams" / "newdata_hpo_strategy_A.json"
        if hpo_path.exists():
            hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
            xgb_p = hpo["primary_space"]["XGBoost"]["best_params"]
            lgb_p = hpo["primary_space"]["LightGBM"]["best_params"]
            hgb_p = hpo["primary_space"]["HGB"]["best_params"]
        else:
            print("  WARNING: No HPO results, using defaults")

        df_new = load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv")
        age_new = get_age_array(df_new)
        X_base, y_new, meta = strategy_a_preprocess(df_new, encoder="onehot")

        num_idx = list(range(meta["n_numerical_raw"]))
        X_num = X_base[:, num_idx]

        X_primary_new = np.column_stack([X_base, new_dataset_feature_engineering(X_num)[0]])
        X_anchor_new = np.column_stack([X_base, anchor_feature_engineering(X_num)[0]])

        cv_new = KFold(n_splits=10, shuffle=True, random_state=42)

        models_new = {
            "XGB_primary": (build_xgb(xgb_p), X_primary_new),
            "LGB_primary": (build_lgb(lgb_p), X_primary_new),
            "HGB_primary": (build_hgb(hgb_p), X_primary_new),
            "HGB_anchor": (build_hgb({k: v for k, v in hgb_p.items()}), X_anchor_new),
        }

        pred_cache_new = {}
        for name, (est, X_used) in models_new.items():
            pred_cache_new[name] = cross_val_predict(est, X_used, y_new, cv=cv_new, n_jobs=-1, method="predict")
            print(f"  {name}: R2={r2_score(y_new, pred_cache_new[name]):.6f}")

        model_order_new = ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"]
        new_result = run_soft_weighting_experiment(
            "New Data (N=4,420)", pred_cache_new, age_new, y_new, model_order_new)
        results["new_data"] = new_result

    # ---- Save ----
    out_path = PROJECT_ROOT / "results" / "new_dataset" / "metrics" / "p12_soft_weighting.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("P1-2 Summary: Soft vs Hard Weighting")
    print(f"{'='*60}")
    for ds_name, r in results.items():
        print(f"\n{ds_name}:")
        print(f"  Global:    R2={r['global_blend']['R2']:.6f}")
        print(f"  Hard (28d): R2={r['hard_threshold']['R2']:.6f}")
        best_soft = max(r["soft_weighting"], key=lambda x: x["R2"])
        print(f"  Best soft (kappa={best_soft['kappa']}): R2={best_soft['R2']:.6f}")
        print(f"  Best soft vs Hard: dR2={best_soft['delta_vs_hard']:+.6f}")
        print(f"  Best soft vs Global: dR2={best_soft['delta_vs_global']:+.6f}")


if __name__ == "__main__":
    main()
