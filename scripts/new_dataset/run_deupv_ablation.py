"""Phase 1.1: 去UPV消融实验 (P0).

在墨西哥数据集上构建3个特征子集，运行完整V0-V5消融：
  Subset 1: Full (w/ UPV) — 复用已有结果
  Subset 2: No-UPV — 移除UPV及UPV衍生特征
  Subset 3: No-NDT — 移除所有NDT（UPV, Er, Ts, Fs）

用法:
  python scripts/new_dataset/run_deupv_ablation.py [--skip-hpo] [--trials 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from new_dataset.new_data_loader import (
    load_raw_new_data,
    strategy_a_preprocess,
    CATEGORICAL_COLS,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
HPO_DIR = RESULTS_DIR / "hyperparams"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"

RANDOM_STATE = 42
CV_FOLDS = 10
EPS = 1e-8


# ---------------------------------------------------------------------------
# Feature engineering variants for different subsets
# ---------------------------------------------------------------------------
def get_num_cols_for_subset(subset: str) -> list[str]:
    """Return numerical column names for each subset."""
    if subset == "Full":
        return ["Design_F'c (Mpa)", "Curing_age_(days)", "Er_(ohm-cm)", "UPV_(m/s)"]
    elif subset == "No-UPV":
        return ["Design_F'c (Mpa)", "Curing_age_(days)", "Er_(ohm-cm)"]
    elif subset == "No-NDT":
        return ["Design_F'c (Mpa)", "Curing_age_(days)"]
    else:
        raise ValueError(f"Unknown subset: {subset}")


def feature_engineering_for_subset(X_arr: np.ndarray, subset: str):
    """Feature engineering adapted for each feature subset.

    X_arr columns: [Design_F'c, Age, Er, UPV] for Full
                   [Design_F'c, Age, Er] for No-UPV
                   [Design_F'c, Age] for No-NDT
    """
    eps = 1e-6
    design_fc = X_arr[:, 0]
    age = X_arr[:, 1]
    age_clipped = np.maximum(age, 0.0)

    features = {}
    # Age transforms (always included)
    features["age_log1p"] = np.log1p(age_clipped)
    features["age_sqrt"] = np.sqrt(age_clipped)
    features["age_pow_0_25"] = np.power(age_clipped, 0.25)
    features["age_inverse"] = 1.0 / (age_clipped + 1.0)
    features["design_fc_age_log"] = design_fc * np.log1p(age_clipped)
    features["design_fc_age_sqrt"] = design_fc * np.sqrt(age_clipped)
    features["maturity_proxy"] = age_clipped * design_fc

    if subset in ("Full", "No-UPV"):
        # Er-dependent features
        er = X_arr[:, 2]
        features["er_log"] = np.log(np.maximum(er, eps))
        features["er_inv"] = 1.0 / (np.maximum(er, eps))
        features["er_per_age"] = er / (age_clipped + eps)

    if subset == "Full":
        # UPV-dependent features
        upv = X_arr[:, 3]
        features["upv_squared"] = upv ** 2
        features["upv_per_age"] = upv / (age_clipped + eps)
        features["upv_times_age_log"] = upv * np.log1p(age_clipped)
        features["upv_times_design"] = upv * design_fc
        features["design_fc_to_upv"] = design_fc / (upv + eps)
        features["er_to_upv"] = er / (upv + eps)
        features["quality_index"] = upv / (er + eps)

    out = np.column_stack(list(features.values()))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, list(features.keys())


def anchor_feature_engineering_for_subset(X_arr: np.ndarray, subset: str):
    """Reduced anchor space feature engineering per subset."""
    eps = 1e-6
    design_fc = X_arr[:, 0]
    age = X_arr[:, 1]
    age_clipped = np.maximum(age, 0.0)

    features = {}
    features["age_log1p"] = np.log1p(age_clipped)
    features["age_sqrt"] = np.sqrt(age_clipped)
    features["age_pow_0_25"] = np.power(age_clipped, 0.25)
    features["design_fc_age_log"] = design_fc * np.log1p(age_clipped)

    if subset in ("Full", "No-UPV"):
        er = X_arr[:, 2]
        features["er_log"] = np.log(np.maximum(er, eps))

    if subset == "Full":
        upv = X_arr[:, 3]
        features["upv_squared"] = upv ** 2
        features["upv_per_age"] = upv / (age_clipped + eps)
        features["upv_times_design"] = upv * design_fc
        features["maturity_proxy"] = age_clipped * design_fc

    out = np.column_stack(list(features.values()))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, list(features.keys())


# ---------------------------------------------------------------------------
# Data preprocessing per subset
# ---------------------------------------------------------------------------
def preprocess_subset(df: pd.DataFrame, subset: str):
    """Preprocess data for a given feature subset.

    Returns: X_base (with categorical encoding), y, age_array, numerical_cols
    """
    y = df["Cs_(Mpa)"].copy().to_numpy()
    age = df["Curing_age_(days)"].to_numpy()

    from sklearn.preprocessing import OneHotEncoder

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
    X_cat = cat_encoder.fit_transform(df[categorical_cols])
    cat_names = cat_encoder.get_feature_names_out(categorical_cols)

    num_cols = get_num_cols_for_subset(subset)
    X_num = df[num_cols].copy().to_numpy(dtype=float)

    X_base = np.column_stack([X_num, X_cat])
    return X_base, y, age, num_cols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))


def mape_percent(yt, yp):
    return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), EPS))) * 100.0)


def fold_metrics_from_cv(cv, X_ref, y, pred):
    r2s, rmses, maes, mapes = [], [], [], []
    for _, test_idx in cv.split(X_ref, y):
        yt, yp = y[test_idx], pred[test_idx]
        r2s.append(float(r2_score(yt, yp)))
        rmses.append(rmse(yt, yp))
        maes.append(mae(yt, yp))
        mapes.append(mape_percent(yt, yp))
    return {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
        "MAPE_mean": float(np.mean(mapes)), "MAPE_std": float(np.std(mapes)),
    }, {"R2": r2s, "RMSE": rmses, "MAE": maes, "MAPE": mapes}


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def optimize_weights(P, y, maxiter=500):
    n = P.shape[1]
    init = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(lambda w: rmse(y, P @ w), init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": maxiter, "ftol": 1e-12, "disp": False})
    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init
    return w


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_xgb(params):
    return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                        n_jobs=-1, tree_method="hist", **params)


def build_lgb(params):
    return LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                         n_jobs=-1, verbosity=-1, **params)


def build_hgb(params):
    return HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                         early_stopping=False, **params)


def build_adaboost():
    base = DecisionTreeRegressor(max_depth=15, min_samples_split=5, min_samples_leaf=1,
                                  random_state=RANDOM_STATE)
    return AdaBoostRegressor(estimator=base, n_estimators=250, learning_rate=0.05,
                              loss="square", random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------
def _objective_xgb(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    model = build_xgb(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method="predict")
    return float(r2_score(y, pred))


def _objective_lgb(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 127),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 0.1, log=True),
    }
    model = build_lgb(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method="predict")
    return float(r2_score(y, pred))


def _objective_hgb(trial, X, y, cv):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 3000),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10, 12, 15]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 32, 255),
    }
    model = build_hgb(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method="predict")
    return float(r2_score(y, pred))


def run_hpo(X, y, n_trials=100, seed=42):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
    results = {}

    study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_xgb.optimize(lambda t: _objective_xgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["XGBoost"] = {"best_params": study_xgb.best_params, "best_r2": study_xgb.best_value}

    study_lgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_lgb.optimize(lambda t: _objective_lgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["LightGBM"] = {"best_params": study_lgb.best_params, "best_r2": study_lgb.best_value}

    study_hgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_hgb.optimize(lambda t: _objective_hgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["HGB"] = {"best_params": study_hgb.best_params, "best_r2": study_hgb.best_value}

    return results


# ---------------------------------------------------------------------------
# Blending functions
# ---------------------------------------------------------------------------
def run_global_blend(model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    w = optimize_weights(P, y)
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    return metrics, folds, weights, pred


def run_piecewise_blend(model_order, pred_cache, cv, X_ref, y, age, split_day=28):
    P = np.column_stack([pred_cache[m] for m in model_order])
    early = age <= split_day
    late = ~early
    w_e = optimize_weights(P[early], y[early])
    w_l = optimize_weights(P[late], y[late])
    pred = np.empty_like(y)
    pred[early] = P[early] @ w_e
    pred[late] = P[late] @ w_l
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    wp = {"early": {model_order[i]: float(w_e[i]) for i in range(len(model_order))},
          "late": {model_order[i]: float(w_l[i]) for i in range(len(model_order))}}
    return metrics, folds, wp, pred


def run_ols_unconstrained(model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    ols = LinearRegression(fit_intercept=False)
    ols.fit(P, y)
    w = ols.coef_
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    negative_count = int(np.sum(w < 0))
    return metrics, folds, weights, pred, negative_count


# ---------------------------------------------------------------------------
# Run ablation for one subset
# ---------------------------------------------------------------------------
def run_ablation_for_subset(subset: str, df: pd.DataFrame, n_trials: int = 100):
    """Run full V0-V5 ablation for a single feature subset."""
    print(f"\n{'='*60}")
    print(f"Running ablation for subset: {subset}")
    print(f"{'='*60}")

    X_base, y, age, num_cols = preprocess_subset(df, subset)
    print(f"  X_base shape: {X_base.shape}, y: {len(y)}")

    # Feature engineering
    X_num_only = X_base[:, :len(num_cols)]
    X_eng, eng_names = feature_engineering_for_subset(X_num_only, subset)
    X_anchor_eng, anchor_eng_names = anchor_feature_engineering_for_subset(X_num_only, subset)

    X_primary = np.column_stack([X_base, X_eng])
    X_anchor = np.column_stack([X_base, X_anchor_eng])
    X_raw = X_base.copy()

    print(f"  Primary: {X_primary.shape[1]}d, Anchor: {X_anchor.shape[1]}d, Raw: {X_raw.shape[1]}d")
    print(f"  Engineered features: {eng_names}")
    print(f"  Anchor features: {anchor_eng_names}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # HPO
    hpo_path = HPO_DIR / f"deupv_hpo_{subset}.json"
    hpo_path.parent.mkdir(parents=True, exist_ok=True)
    if hpo_path.exists():
        print(f"  Loading saved HPO from {hpo_path}")
        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
    else:
        print(f"  Running HPO on primary space ({n_trials} trials)...")
        hpo_primary = run_hpo(X_primary, y, n_trials=n_trials)
        print(f"  Running HPO on raw space ({n_trials} trials)...")
        hpo_raw = run_hpo(X_raw, y, n_trials=n_trials)
        hpo = {"primary_space": hpo_primary, "raw_space": hpo_raw}
        hpo_path.write_text(json.dumps(hpo, ensure_ascii=False, indent=2), encoding="utf-8")

    xgb_p = hpo["primary_space"]["XGBoost"]["best_params"]
    lgb_p = hpo["primary_space"]["LightGBM"]["best_params"]
    hgb_p = hpo["primary_space"]["HGB"]["best_params"]
    xgb_rp = hpo["raw_space"]["XGBoost"]["best_params"]
    lgb_rp = hpo["raw_space"]["LightGBM"]["best_params"]
    hgb_rp = hpo["raw_space"]["HGB"]["best_params"]

    # OOF Predictions
    print("  Generating OOF predictions...")
    pred_cache = {}
    specs = [
        ("XGB_primary", build_xgb(xgb_p), X_primary),
        ("LGB_primary", build_lgb(lgb_p), X_primary),
        ("HGB_primary", build_hgb(hgb_p), X_primary),
        ("HGB_anchor", build_hgb(hgb_p), X_anchor),
        ("XGB_raw", build_xgb(xgb_rp), X_raw),
        ("LGB_raw", build_lgb(lgb_rp), X_raw),
        ("HGB_raw", build_hgb(hgb_rp), X_raw),
        ("HGB_anchor_raw", build_hgb(hgb_rp), X_raw),
    ]
    for model_id, est, X_used in specs:
        tt = time.perf_counter()
        pred_cache[model_id] = cross_val_predict(est, X_used, y, cv=cv, n_jobs=-1, method="predict")
        r2 = float(r2_score(y, pred_cache[model_id]))
        print(f"    {model_id}: R2={r2:.6f} ({time.perf_counter()-tt:.1f}s)")

    # V0: AdaBoost
    ada = build_adaboost()
    ada_pred = cross_val_predict(ada, X_raw, y, cv=cv, n_jobs=-1, method="predict")
    ada_metrics, ada_folds = fold_metrics_from_cv(cv, X_raw, y, ada_pred)

    # V1-V5
    v1_m, v1_f, v1_w, v1_p = run_global_blend(
        ["XGB_primary", "LGB_primary", "HGB_primary"], pred_cache, cv, X_primary, y)
    v2_m, v2_f, v2_w, v2_p = run_global_blend(
        ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"], pred_cache, cv, X_primary, y)
    v3_m, v3_f, v3_wp, v3_p = run_piecewise_blend(
        ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"], pred_cache, cv, X_primary, y, age)
    v4_m, v4_f, v4_wp, v4_p = run_piecewise_blend(
        ["XGB_raw", "LGB_raw", "HGB_raw", "HGB_anchor_raw"], pred_cache, cv, X_raw, y, age)
    v5_m, v5_f, v5_w, v5_p, v5_neg = run_ols_unconstrained(
        ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"], pred_cache, cv, X_primary, y)

    # Best single model
    best_single = max(
        [(mid, float(r2_score(y, pred_cache[mid]))) for mid in pred_cache],
        key=lambda x: x[1]
    )

    return {
        "subset": subset,
        "n_features_raw": X_raw.shape[1],
        "n_features_primary": X_primary.shape[1],
        "n_features_anchor": X_anchor.shape[1],
        "engineered_features": eng_names,
        "v0_adaboost": {"metrics": ada_metrics, "fold_metrics": ada_folds},
        "v1_primary_global": {"metrics": v1_m, "fold_metrics": v1_f, "weights": v1_w},
        "v2_dualspace_global": {"metrics": v2_m, "fold_metrics": v2_f, "weights": v2_w},
        "v3_acdcb_full": {"metrics": v3_m, "fold_metrics": v3_f, "weights_piecewise": v3_wp},
        "v4_raw_piecewise": {"metrics": v4_m, "fold_metrics": v4_f, "weights_piecewise": v4_wp},
        "v5_ols_unconstrained": {"metrics": v5_m, "fold_metrics": v5_f, "weights": v5_w, "negative_weights": v5_neg},
        "best_single_model": {"name": best_single[0], "r2": best_single[1]},
        "pred_cache": pred_cache,
        "ada_pred": ada_pred,
        "y": y,
        "age": age,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--subsets", nargs="+", default=["Full", "No-UPV", "No-NDT"],
                        help="Which subsets to run (default: all three)")
    args = parser.parse_args()

    for d in [RESULTS_DIR, HPO_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load data
    print("Loading Mexico dataset...")
    df = load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv")

    # Load existing Full results if available
    existing_full_path = METRICS_DIR / "ablation_newdata_results.json"
    existing_full = None
    if existing_full_path.exists():
        existing_full = json.loads(existing_full_path.read_text(encoding="utf-8"))
        print(f"Loaded existing Full-subset results from {existing_full_path}")

    all_subset_results = {}
    subset_summaries = {}

    for subset in args.subsets:
        if subset == "Full" and existing_full is not None:
            # Reuse existing full results
            print(f"\nReusing existing Full-subset results...")
            ev = existing_full["variants"]
            es = existing_full["single_model_OOF"]
            all_subset_results["Full"] = {
                "subset": "Full",
                "source": "existing",
                "n_features_raw": existing_full["meta"]["n_features_raw"],
                "n_features_primary": existing_full["meta"]["n_features_primary"],
                "n_features_anchor": existing_full["meta"]["n_features_anchor"],
                "v0_adaboost": ev["v0_adaboost"],
                "v1_primary_global": ev["v1_primary_global"],
                "v2_dualspace_global": ev["v2_dualspace_global"],
                "v3_acdcb_full": ev["v3_dualspace_piecewise_acdcb"],
                "v4_raw_piecewise": ev["v4_raw_piecewise"],
                "v5_ols_unconstrained": ev["v5_ols_unconstrained"],
                "best_single_model": {"name": max(es, key=lambda x: es[x]["R2"]),
                                       "r2": es[max(es, key=lambda x: es[x]["R2"])]["R2"]},
            }
        else:
            result = run_ablation_for_subset(subset, df, n_trials=args.trials)
            all_subset_results[subset] = result

        # Extract summary
        r = all_subset_results[subset]
        v0 = r["v0_adaboost"]["metrics"]
        v1 = r["v1_primary_global"]["metrics"]
        v2 = r["v2_dualspace_global"]["metrics"]
        v3 = r["v3_acdcb_full"]["metrics"]
        v4 = r["v4_raw_piecewise"]["metrics"]
        v5 = r["v5_ols_unconstrained"]["metrics"]
        subset_summaries[subset] = {
            "V0_AdaBoost": {"R2": v0["R2_mean"], "RMSE": v0["RMSE_mean"]},
            "V1_PrimaryGlobal": {"R2": v1["R2_mean"], "RMSE": v1["RMSE_mean"]},
            "V2_DualGlobal": {"R2": v2["R2_mean"], "RMSE": v2["RMSE_mean"]},
            "V3_ACDCB_Full": {"R2": v3["R2_mean"], "RMSE": v3["RMSE_mean"]},
            "V4_RawPiecewise": {"R2": v4["R2_mean"], "RMSE": v4["RMSE_mean"]},
            "V5_OLS_Unconstrained": {"R2": v5["R2_mean"], "RMSE": v5["RMSE_mean"]},
            "BestSingle": r["best_single_model"],
            "feature_dim": r.get("n_features_raw", "N/A"),
        }

    # ---- Save results ----
    output = {
        "meta": {
            "study": "De-UPV Ablation (Phase 1.1)",
            "subsets": args.subsets,
            "n_trials_hpo": args.trials,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "dataset": "data/Data.csv",
            "n_samples": int(len(df)),
        },
        "subset_results": all_subset_results,
        "summaries": subset_summaries,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    json_path = METRICS_DIR / "deupv_ablation_results.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to {json_path}")

    # ---- Print comparison table ----
    print("\n" + "=" * 90)
    print("Phase 1.1: 去UPV消融实验 — 三列对比汇总")
    print("=" * 90)

    header = f"{'Variant':<28}"
    for s in args.subsets:
        header += f" {s+' R2':>10} {s+' RMSE':>10}"
    print(header)
    print("-" * 90)

    variants = ["V0_AdaBoost", "V1_PrimaryGlobal", "V2_DualGlobal", "V3_ACDCB_Full",
                "V4_RawPiecewise", "V5_OLS_Unconstrained"]
    vlabels = ["V0: AdaBoost", "V1: Primary+Global", "V2: Dual+Global",
               "V3: ACDCB Full", "V4: Raw+Piecewise", "V5: OLS Unconstrained"]

    for vkey, vlabel in zip(variants, vlabels):
        row = f"{vlabel:<28}"
        for s in args.subsets:
            m = subset_summaries[s][vkey]
            row += f" {m['R2']:>10.4f} {m['RMSE']:>10.4f}"
        print(row)

    # Best single model row
    row = f"{'Best Single Model':<28}"
    for s in args.subsets:
        bsm = subset_summaries[s]["BestSingle"]
        row += f" {bsm['r2']:>10.4f} {'--':>10}"
    print(row)

    print("\nKey comparisons:")
    for s in args.subsets:
        sm = subset_summaries[s]
        d_model = sm["V1_PrimaryGlobal"]["R2"] - sm["V0_AdaBoost"]["R2"]
        d_eng = sm["V3_ACDCB_Full"]["R2"] - sm["V4_RawPiecewise"]["R2"]
        best_vs_acdcb = sm["BestSingle"]["r2"] - sm["V3_ACDCB_Full"]["R2"]
        print(f"  {s}: Model upgrade dR2={d_model:+.6f}, FeatureEng dR2={d_eng:+.6f}, BestSingle vs ACDCB dR2={best_vs_acdcb:+.6f}")

    print(f"\nTotal runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
