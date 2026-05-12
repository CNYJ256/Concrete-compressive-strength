"""P0-1: 新数据集全消融实验复现 (V0-V5)。

在 data/Data.csv (N=4,420) 上完整复现 ACDCB 消融管线：
- V0: AdaBoost 基线
- V1: Primary Space + 3 GBDT + Global Blend
- V2: Dual Space (Primary + Anchor) + Global Blend
- V3: ACDCB Full (Dual Space + Piecewise)
- V4: Raw Features + Piecewise (无特征工程)
- V5: OLS Unconstrained stacking

包含 Optuna 超参数搜索和 10 折 CV 评估。

用法：
  python scripts/new_dataset/run_ablation_new_data.py [--skip-hpo] [--strategy B]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
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
RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
HPO_DIR = RESULTS_DIR / "hyperparams"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"

sys.path.insert(0, str(SCRIPTS_ROOT))
from new_dataset.new_data_loader import (
    load_raw_new_data,
    strategy_a_preprocess,
    strategy_b_preprocess,
    get_age_array,
    CATEGORICAL_COLS,
)

RANDOM_STATE = 42
CV_FOLDS = 10
EPS = 1e-8


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger() -> logging.Logger:
    logger = logging.getLogger("newdata_ablation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger


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
    trace = []

    def obj(w):
        v = rmse(y, P @ w)
        trace.append(v)
        return v

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": maxiter, "ftol": 1e-12, "disp": False})
    if not res.success:
        w = init
    else:
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else init
    return w, trace, res


# ---------------------------------------------------------------------------
# Blending functions
# ---------------------------------------------------------------------------
@dataclass
class VariantResult:
    name: str
    metrics: dict
    fold_metrics: dict
    weights: dict | None
    weights_piecewise: dict | None
    pred: np.ndarray


def run_global_blend(name, model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    w, _, _ = optimize_weights(P, y)
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=weights, weights_piecewise=None, pred=pred)


def run_piecewise_blend(name, model_order, pred_cache, cv, X_ref, y, age, split_day):
    P = np.column_stack([pred_cache[m] for m in model_order])
    early = age <= split_day
    late = ~early
    w_e, _, _ = optimize_weights(P[early], y[early])
    w_l, _, _ = optimize_weights(P[late], y[late])
    pred = np.empty_like(y)
    pred[early] = P[early] @ w_e
    pred[late] = P[late] @ w_l
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    wp = {"early": {model_order[i]: float(w_e[i]) for i in range(len(model_order))},
          "late": {model_order[i]: float(w_l[i]) for i in range(len(model_order))}}
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=None, weights_piecewise=wp, pred=pred)


def run_ols_unconstrained(name, model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    ols = LinearRegression(fit_intercept=False)
    ols.fit(P, y)
    w = ols.coef_
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    weights = {model_order[i]: float(w[i]) for i in range(len(model_order))}
    return VariantResult(name=name, metrics=metrics, fold_metrics=folds,
                         weights=weights, weights_piecewise=None, pred=pred)


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
# Optuna hyperparameter search
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


def run_hyperparameter_search(X, y, n_trials=150, seed=42):
    """Run Optuna HPO for XGB, LGB, HGB on the given feature matrix."""
    logger = get_logger()
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)  # inner CV for HPO

    results = {}
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)

    logger.info("HPO XGBoost (%d trials)...", n_trials)
    study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_xgb.optimize(lambda t: _objective_xgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["XGBoost"] = {"best_params": study_xgb.best_params, "best_r2": study_xgb.best_value}

    logger.info("HPO LightGBM (%d trials)...", n_trials)
    study_lgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_lgb.optimize(lambda t: _objective_lgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["LightGBM"] = {"best_params": study_lgb.best_params, "best_r2": study_lgb.best_value}

    logger.info("HPO HGB (%d trials)...", n_trials)
    study_hgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_hgb.optimize(lambda t: _objective_hgb(t, X, y, cv), n_trials=n_trials,
                       show_progress_bar=False, n_jobs=1)
    results["HGB"] = {"best_params": study_hgb.best_params, "best_r2": study_hgb.best_value}

    return results


# ---------------------------------------------------------------------------
# Reduced feature engineering for Anchor space (new data)
# ---------------------------------------------------------------------------
def anchor_feature_engineering(X_arr):
    """Reduced feature engineering for anchor space on new dataset."""
    eps = 1e-6
    design_fc = X_arr[:, 0]
    age = X_arr[:, 1]
    er = X_arr[:, 2]
    upv = X_arr[:, 3]

    age_clipped = np.maximum(age, 0.0)
    features = {}
    features["age_log1p"] = np.log1p(age_clipped)
    features["age_sqrt"] = np.sqrt(age_clipped)
    features["age_pow_0_25"] = np.power(age_clipped, 0.25)
    features["design_fc_age_log"] = design_fc * np.log1p(age_clipped)
    features["upv_squared"] = upv ** 2
    features["upv_per_age"] = upv / (age_clipped + eps)
    features["upv_times_design"] = upv * design_fc
    features["er_log"] = np.log(np.maximum(er, eps))
    features["maturity_proxy"] = age_clipped * design_fc

    out = np.column_stack(list(features.values()))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, list(features.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hpo", action="store_true", help="Skip HPO, use saved params")
    parser.add_argument("--strategy", choices=["A", "B"], default="A",
                        help="Missing data strategy: A (drop Ts/Fs) or B (impute)")
    parser.add_argument("--trials", type=int, default=150, help="Optuna trials per model")
    parser.add_argument("--age-split", type=int, default=28, help="Age split day for piecewise")
    args = parser.parse_args()

    for d in [RESULTS_DIR, HPO_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---- Load and preprocess data ----
    logger.info("Loading new dataset...")
    df = load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv")
    age_raw = get_age_array(df)

    # Strategy A: drop Ts/Fs (complete features)
    logger.info("Preprocessing (Strategy %s)...", args.strategy)
    if args.strategy == "A":
        X_base, y, meta = strategy_a_preprocess(df, encoder="onehot", add_engineered=False)
    else:
        X_base, y, meta = strategy_b_preprocess(df, encoder="onehot")

    logger.info("Feature matrix: %s, target: %s", X_base.shape, y.shape)

    # ---- Feature engineering ----
    # Extract just the 4 numerical columns for feature engineering
    num_cols_idx = list(range(meta["n_numerical_raw"]))
    X_num_only = X_base[:, num_cols_idx]

    from new_dataset.new_data_loader import new_dataset_feature_engineering
    X_primary_eng, eng_names = new_dataset_feature_engineering(X_num_only)
    X_anchor_eng, anchor_eng_names = anchor_feature_engineering(X_num_only)

    # Primary space = raw features + onehot + full eng
    X_primary = np.column_stack([X_base, X_primary_eng])
    # Anchor space = raw features + onehot + reduced eng
    X_anchor = np.column_stack([X_base, X_anchor_eng])
    # Raw space = just base features (no feature engineering)
    X_raw = X_base.copy()

    logger.info("Primary space: %d features, Anchor: %d, Raw: %d",
                X_primary.shape[1], X_anchor.shape[1], X_raw.shape[1])

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Hyperparameter search ----
    hpo_path = HPO_DIR / f"newdata_hpo_strategy_{args.strategy}.json"
    if args.skip_hpo and hpo_path.exists():
        logger.info("Loading saved HPO results...")
        hpo_results = json.loads(hpo_path.read_text(encoding="utf-8"))
    else:
        logger.info("Running HPO on Primary space (%d trials each)...", args.trials)
        # Run HPO on primary space (with engineering) and raw space separately
        hpo_results = {
            "primary_space": run_hyperparameter_search(X_primary, y, n_trials=args.trials),
        }
        # Also run HPO on raw space for V4 fairness
        logger.info("Running HPO on Raw space...")
        hpo_results["raw_space"] = run_hyperparameter_search(X_raw, y, n_trials=args.trials)
        hpo_path.write_text(json.dumps(hpo_results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("HPO results saved to %s", hpo_path)

    # Extract best params
    xgb_params = hpo_results["primary_space"]["XGBoost"]["best_params"]
    lgb_params = hpo_results["primary_space"]["LightGBM"]["best_params"]
    hgb_params = hpo_results["primary_space"]["HGB"]["best_params"]

    xgb_raw_params = hpo_results["raw_space"]["XGBoost"]["best_params"]
    lgb_raw_params = hpo_results["raw_space"]["LightGBM"]["best_params"]
    hgb_raw_params = hpo_results["raw_space"]["HGB"]["best_params"]

    # For anchor space, use the same HGB params as primary
    hgb_anchor_params = {k: v for k, v in hgb_params.items()}
    hgb_anchor_raw_params = {k: v for k, v in hgb_raw_params.items()}

    # ---- OOF Prediction Cache ----
    logger.info("Generating OOF predictions...")
    pred_cache = {}

    model_specs = [
        # Primary space (with engineering)
        ("XGB_primary", build_xgb(xgb_params), X_primary),
        ("LGB_primary", build_lgb(lgb_params), X_primary),
        ("HGB_primary", build_hgb(hgb_params), X_primary),
        # Anchor space (reduced engineering)
        ("HGB_anchor", build_hgb(hgb_anchor_params), X_anchor),
        # Raw space (no engineering)
        ("XGB_raw", build_xgb(xgb_raw_params), X_raw),
        ("LGB_raw", build_lgb(lgb_raw_params), X_raw),
        ("HGB_raw", build_hgb(hgb_raw_params), X_raw),
        ("HGB_anchor_raw", build_hgb(hgb_anchor_raw_params), X_raw),
    ]

    for model_id, estimator, X_used in model_specs:
        tt = time.perf_counter()
        pred_cache[model_id] = cross_val_predict(estimator, X_used, y, cv=cv, n_jobs=-1, method="predict")
        sm = {"R2": float(r2_score(y, pred_cache[model_id])),
              "RMSE": rmse(y, pred_cache[model_id])}
        logger.info("  %s OOF: R2=%.6f RMSE=%.4f (%.1fs)", model_id, sm["R2"], sm["RMSE"], time.perf_counter() - tt)

    # ---- V0: AdaBoost Baseline ----
    logger.info("Computing AdaBoost baseline (V0)...")
    ada = build_adaboost()
    ada_pred = cross_val_predict(ada, X_raw, y, cv=cv, n_jobs=-1, method="predict")
    ada_metrics, ada_fold = fold_metrics_from_cv(cv, X_raw, y, ada_pred)
    logger.info("  V0 AdaBoost: R2=%.6f RMSE=%.4f", ada_metrics["R2_mean"], ada_metrics["RMSE_mean"])

    # ---- Ablation Variants ----
    logger.info("Running ablation variants...")

    v1 = run_global_blend("v1_primary_global",
                          ["XGB_primary", "LGB_primary", "HGB_primary"],
                          pred_cache, cv, X_primary, y)

    v2 = run_global_blend("v2_dualspace_global",
                          ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                          pred_cache, cv, X_primary, y)

    v3 = run_piecewise_blend("v3_dualspace_piecewise_acdcb",
                             ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                             pred_cache, cv, X_primary, y, age_raw, args.age_split)

    v4 = run_piecewise_blend("v4_raw_piecewise",
                             ["XGB_raw", "LGB_raw", "HGB_raw", "HGB_anchor_raw"],
                             pred_cache, cv, X_raw, y, age_raw, args.age_split)

    v5 = run_ols_unconstrained("v5_ols_unconstrained",
                               ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"],
                               pred_cache, cv, X_primary, y)

    # ---- Compile Results ----
    variant_desc = {
        "v1_primary_global": "V1: Primary Space + 3 GBDT + Global Blend",
        "v2_dualspace_global": "V2: Dual Space + 4 Models + Global Blend",
        "v3_dualspace_piecewise_acdcb": "V3: ACDCB Full (Dual Space + Piecewise)",
        "v4_raw_piecewise": "V4: Raw Features + Piecewise (no feature engineering)",
        "v5_ols_unconstrained": "V5: OLS Unconstrained stacking",
    }

    variants = [v1, v2, v3, v4, v5]

    def delta(curr, ref):
        return {
            "R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
            "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
            "MAE_drop": float(ref["MAE_mean"] - curr["MAE_mean"]),
            "MAPE_drop": float(ref["MAPE_mean"] - curr["MAPE_mean"]),
        }

    m_p1 = ada_metrics
    m_v1 = v1.metrics
    m_v2 = v2.metrics
    m_v3 = v3.metrics
    m_v4 = v4.metrics
    m_v5 = v5.metrics

    results = {
        "meta": {
            "study_name": "ACDCB Ablation on New Dataset (data.csv)",
            "dataset": "data/Data.csv",
            "n_samples": int(len(y)),
            "n_features_raw": int(X_raw.shape[1]),
            "n_features_primary": int(X_primary.shape[1]),
            "n_features_anchor": int(X_anchor.shape[1]),
            "strategy": args.strategy,
            "encoder": "onehot",
            "age_split_day": args.age_split,
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": CV_FOLDS, "shuffle": True},
            "hpo_trials": args.trials,
            "target_stats": {
                "mean": float(np.mean(y)), "std": float(np.std(y)),
                "skewness": float(pd.Series(y).skew()),
                "min": float(np.min(y)), "max": float(np.max(y)),
            },
        },
        "variants": {
            "v0_adaboost": {
                "description": "V0: AdaBoost baseline",
                "metrics": ada_metrics,
                "fold_metrics": ada_fold,
                "weights": None,
                "weights_piecewise": None,
            },
        },
        "comparisons": {},
    }

    for obj in variants:
        results["variants"][obj.name] = {
            "description": variant_desc[obj.name],
            "metrics": obj.metrics,
            "fold_metrics": obj.fold_metrics,
            "weights": obj.weights,
            "weights_piecewise": obj.weights_piecewise,
        }

    results["comparisons"] = {
        "full_vs_adaboost": delta(m_v3, m_p1),
        "dualspace_gain_over_primary": delta(m_v2, m_v1),
        "age_piecewise_gain_over_global": delta(m_v3, m_v2),
        "feature_engineering_gain_over_raw": delta(m_v3, m_v4),
        "constrained_vs_ols": {
            "constrained_R2_gain": float(m_v2["R2_mean"] - m_v5["R2_mean"]),
            "constrained_RMSE_drop": float(m_v5["RMSE_mean"] - m_v2["RMSE_mean"]),
            "ols_negative_weights": int(np.sum(np.array(list(v5.weights.values())) < 0)),
        },
    }

    results["single_model_OOF"] = {
        m_id: {"R2": float(r2_score(y, pred_cache[m_id])), "RMSE": rmse(y, pred_cache[m_id])}
        for m_id in pred_cache
    }

    # ---- Save ----
    results["runtime_sec"] = float(time.perf_counter() - t0)

    oof_df = pd.DataFrame({"y_true": y, "v0_adaboost": ada_pred, "age": age_raw})
    for obj in variants:
        oof_df[obj.name] = obj.pred
    oof_df.to_csv(PRED_DIR / "ablation_newdata_oof.csv", index=False, encoding="utf-8-sig")

    json_path = METRICS_DIR / "ablation_newdata_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", json_path)

    # ---- Summary ----
    print("\n" + "=" * 75)
    print("P0-1 新数据集消融实验 — 关键结果汇总")
    print("=" * 75)
    print(f"Data: N={len(y)}, features: raw={X_raw.shape[1]}, primary={X_primary.shape[1]}")
    print(f"Strategy: {args.strategy}, Age split: {args.age_split}d")
    print()
    print(f"{'Variant':<35} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
    print("-" * 75)
    print(f"{'V0: AdaBoost baseline':<35} {ada_metrics['R2_mean']:>8.4f} {ada_metrics['RMSE_mean']:>8.4f} {ada_metrics['MAE_mean']:>8.4f} {ada_metrics['MAPE_mean']:>8.2f}")
    print(f"{'V1: Primary+Global':<35} {m_v1['R2_mean']:>8.4f} {m_v1['RMSE_mean']:>8.4f} {m_v1['MAE_mean']:>8.4f} {m_v1['MAPE_mean']:>8.2f}")
    print(f"{'V2: DualSpace+Global':<35} {m_v2['R2_mean']:>8.4f} {m_v2['RMSE_mean']:>8.4f} {m_v2['MAE_mean']:>8.4f} {m_v2['MAPE_mean']:>8.2f}")
    print(f"{'V3: ACDCB Full (Dual+Piecewise)':<35} {m_v3['R2_mean']:>8.4f} {m_v3['RMSE_mean']:>8.4f} {m_v3['MAE_mean']:>8.4f} {m_v3['MAPE_mean']:>8.2f}")
    print(f"{'V4: Raw+Piecewise':<35} {m_v4['R2_mean']:>8.4f} {m_v4['RMSE_mean']:>8.4f} {m_v4['MAE_mean']:>8.4f} {m_v4['MAPE_mean']:>8.2f}")
    print(f"{'V5: OLS Unconstrained':<35} {m_v5['R2_mean']:>8.4f} {m_v5['RMSE_mean']:>8.4f} {m_v5['MAE_mean']:>8.4f} {m_v5['MAPE_mean']:>8.2f}")
    print()
    print("Key comparisons:")
    print(f"  Model upgrade (V1-V0):  dR2={m_v1['R2_mean']-ada_metrics['R2_mean']:+.6f}")
    print(f"  Dual Space (V2-V1):     dR2={m_v2['R2_mean']-m_v1['R2_mean']:+.6f}")
    print(f"  Age Piecewise (V3-V2):   dR2={m_v3['R2_mean']-m_v2['R2_mean']:+.6f}")
    print(f"  Feature Eng (V3-V4):     dR2={m_v3['R2_mean']-m_v4['R2_mean']:+.6f}")
    print(f"  Constrained vs OLS:      dR2={m_v2['R2_mean']-m_v5['R2_mean']:+.6f}")
    best_single = max(results["single_model_OOF"].items(), key=lambda x: x[1]["R2"])
    print(f"\n  Best single model OOF: {best_single[0]} R2={best_single[1]['R2']:.6f} RMSE={best_single[1]['RMSE']:.4f}")
    print(f"  V3 (ACDCB Full) OOF: R2={m_v3['R2_mean']:.6f} RMSE={m_v3['RMSE_mean']:.4f}")
    print(f"  Single best vs ACDCB: dR2={best_single[1]['R2']-m_v3['R2_mean']:+.6f}")


if __name__ == "__main__":
    main()
