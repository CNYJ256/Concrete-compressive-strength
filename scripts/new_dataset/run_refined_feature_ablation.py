"""Phase 2.1: A/B级精炼特征空间消融实验 (P0).

对UCI数据集的32维工程特征进行A/B/C三级分类，构建三个精炼特征空间，
验证"仅使用物理基础扎实的特征"结论不变。

特征分类:
  Level A (已建立物理模型, 4个): w/c, w/b, sqrt(age), log(age)
  Level B (力学/维度合理, 12个): sp/b, scm_ratio, fine_ratio_in_agg,
           paste_index, binder_to_agg_ratio, water_to_paste_ratio,
           cement_fraction_in_binder, slag_fraction_in_binder,
           flyash_fraction_in_binder, superplasticizer_efficiency,
           agg_binder_balance, binder
  Level C (纯假设驱动, 8个): age^0.25, abrams_index, cement_age_interaction,
           binder_age_interaction, wb_age_interaction, maturity_index,
           1/(age+1), age_wc_interaction

用法:
  python scripts/new_dataset/run_refined_feature_ablation.py [--skip-hpo] [--trials 50]
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from concrete_compressive_strength.core import (
    BASE_FEATURES, TARGET_COL, RANDOM_STATE, AGE_SPLIT_DAY,
    BASE_MODEL_PARAMS, ANCHOR_MODEL_PARAMS,
    build_xgb, build_lgbm, build_hgb,
    feature_engineering, feature_engineering_anchor, load_data,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
HPO_DIR = RESULTS_DIR / "hyperparams"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"

CV_FOLDS = 10
EPS = 1e-8


# ---------------------------------------------------------------------------
# Feature Level Classification
# ---------------------------------------------------------------------------
LEVEL_A_FEATURES = [
    "water_cement_ratio",    # Abrams' law
    "water_binder_ratio",    # Abrams' law extension
    "age_sqrt",              # Maturity method (Nurse-Saul)
    "age_log1p",             # Maturity method (log)
]

LEVEL_B_FEATURES = [
    "binder",
    "sp_binder_ratio",
    "scm_ratio",
    "fine_ratio_in_agg",
    "paste_index",
    "binder_to_agg_ratio",
    "water_to_paste_ratio",
    "cement_fraction_in_binder",
    "slag_fraction_in_binder",
    "flyash_fraction_in_binder",
    "superplasticizer_efficiency",
    "agg_binder_balance",
]

LEVEL_C_FEATURES = [
    "age_pow_0_25",
    "abrams_index",
    "cement_age_interaction",
    "binder_age_interaction",
    "wb_age_interaction",
    "maturity_index",
    "age_inverse",
    "age_wc_interaction",
]

ALL_ENG_FEATURES = LEVEL_A_FEATURES + LEVEL_B_FEATURES + LEVEL_C_FEATURES


def build_refined_feature_space(X_base: pd.DataFrame, eng_features: list[str]) -> np.ndarray:
    """Build a feature matrix with base 8 + selected engineered features."""
    X_eng_full = feature_engineering(X_base)
    selected = X_base[BASE_FEATURES].copy()
    for feat in eng_features:
        if feat in X_eng_full.columns:
            selected[feat] = X_eng_full[feat].values
    return selected.to_numpy(dtype=float)


def build_anchor_refined_space(X_base: pd.DataFrame, eng_features: list[str]) -> np.ndarray:
    """Build anchor feature matrix with base 8 + selected anchor features."""
    X_anchor_full = feature_engineering_anchor(X_base)
    selected = X_base[BASE_FEATURES].copy()
    for feat in eng_features:
        if feat in X_anchor_full.columns:
            selected[feat] = X_anchor_full[feat].values
    return selected.to_numpy(dtype=float)


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


def run_global_blend(model_order, pred_cache, cv, X_ref, y):
    P = np.column_stack([pred_cache[m] for m in model_order])
    w = optimize_weights(P, y)
    pred = P @ w
    metrics, folds = fold_metrics_from_cv(cv, X_ref, y, pred)
    return metrics, folds, {m: float(w[i]) for i, m in enumerate(model_order)}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_xgb_model(params):
    return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                        n_jobs=1, tree_method="hist", **params)


def build_lgb_model(params):
    return LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                         n_jobs=1, verbosity=-1, **params)


def build_hgb_model(params):
    return HistGradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE,
                                         early_stopping=False, **params)


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
    model = build_xgb_model(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1, method="predict")
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
    model = build_lgb_model(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1, method="predict")
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
    model = build_hgb_model(params)
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1, method="predict")
    return float(r2_score(y, pred))


def run_hpo(X, y, n_trials=50, seed=42):
    """Run Optuna HPO for XGB, LGB, HGB."""
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    for d in [RESULTS_DIR, HPO_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---- Load data ----
    print("Loading UCI dataset...")
    df = load_data(PROJECT_ROOT / "data" / "Concrete_Data.xls")
    X_base = df[BASE_FEATURES].copy()
    y = df[TARGET_COL].to_numpy()
    age = X_base["age"].to_numpy()
    X_raw_arr = X_base.to_numpy(dtype=float)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Build refined feature spaces ----
    print("Building refined feature spaces...")
    X_refined_A = build_refined_feature_space(X_base, LEVEL_A_FEATURES)
    X_refined_AB = build_refined_feature_space(X_base, LEVEL_A_FEATURES + LEVEL_B_FEATURES)
    X_refined_All = build_refined_feature_space(X_base, ALL_ENG_FEATURES)
    X_anchor_A = build_anchor_refined_space(X_base, LEVEL_A_FEATURES)
    X_anchor_AB = build_anchor_refined_space(X_base, LEVEL_A_FEATURES + LEVEL_B_FEATURES)
    X_anchor_All = build_anchor_refined_space(X_base, ALL_ENG_FEATURES)

    spaces = {
        "Refined-A": {"primary": X_refined_A, "anchor": X_anchor_A,
                      "desc": f"8 raw + {len(LEVEL_A_FEATURES)} Level A = {X_refined_A.shape[1]} dims"},
        "Refined-AB": {"primary": X_refined_AB, "anchor": X_anchor_AB,
                       "desc": f"8 raw + {len(LEVEL_A_FEATURES) + len(LEVEL_B_FEATURES)} Level A+B = {X_refined_AB.shape[1]} dims"},
        "Refined-All": {"primary": X_refined_All, "anchor": X_anchor_All,
                        "desc": f"8 raw + {len(ALL_ENG_FEATURES)} All engineered = {X_refined_All.shape[1]} dims"},
    }

    print("Feature space summary:")
    for name, sp in spaces.items():
        print(f"  {name}: {sp['desc']}")

    # ---- HPO or load ----
    hpo_path = HPO_DIR / "refined_feature_hpo.json"
    if args.skip_hpo and hpo_path.exists():
        print("Loading saved HPO results...")
        all_hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
    else:
        print(f"Running HPO for each feature space ({args.trials} trials each)...")
        all_hpo = {}
        for name, sp in spaces.items():
            print(f"  HPO for {name}...")
            all_hpo[name] = run_hpo(sp["primary"], y, n_trials=args.trials)
        hpo_path.write_text(json.dumps(all_hpo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"HPO saved to {hpo_path}")

    # ---- OOF Predictions ----
    print("Generating OOF predictions...")
    pred_cache = {}

    for space_name, sp in spaces.items():
        hpo = all_hpo[space_name]
        xgb_p = {k: v for k, v in hpo["XGBoost"]["best_params"].items()}
        lgb_p = {k: v for k, v in hpo["LightGBM"]["best_params"].items()}
        hgb_p = {k: v for k, v in hpo["HGB"]["best_params"].items()}

        specs = [
            (f"{space_name}_XGB", build_xgb_model(xgb_p), sp["primary"]),
            (f"{space_name}_LGB", build_lgb_model(lgb_p), sp["primary"]),
            (f"{space_name}_HGB", build_hgb_model(hgb_p), sp["primary"]),
            (f"{space_name}_HGB_anchor", build_hgb_model(hgb_p), sp["anchor"]),
        ]

        for model_id, est, X_used in specs:
            tt = time.perf_counter()
            pred_cache[model_id] = cross_val_predict(est, X_used, y, cv=cv, n_jobs=1, method="predict")
            r2 = float(r2_score(y, pred_cache[model_id]))
            print(f"  {model_id}: R2={r2:.6f} ({time.perf_counter()-tt:.1f}s)")

    # ---- Also run raw space with existing hyperparams ----
    print("Generating Raw space OOF (control)...")
    raw_xgb = build_xgb_model(BASE_MODEL_PARAMS["XGBoost"])
    raw_lgb = build_lgb_model(BASE_MODEL_PARAMS["LightGBM"])
    raw_hgb = build_hgb_model(BASE_MODEL_PARAMS["HGB"])
    raw_hgb_a = build_hgb_model(ANCHOR_MODEL_PARAMS)

    pred_cache["Raw_XGB"] = cross_val_predict(raw_xgb, X_raw_arr, y, cv=cv, n_jobs=1, method="predict")
    pred_cache["Raw_LGB"] = cross_val_predict(raw_lgb, X_raw_arr, y, cv=cv, n_jobs=1, method="predict")
    pred_cache["Raw_HGB"] = cross_val_predict(raw_hgb, X_raw_arr, y, cv=cv, n_jobs=1, method="predict")
    pred_cache["Raw_HGB_anchor"] = cross_val_predict(raw_hgb_a, X_raw_arr, y, cv=cv, n_jobs=1, method="predict")

    for mid in ["Raw_XGB", "Raw_LGB", "Raw_HGB", "Raw_HGB_anchor"]:
        print(f"  {mid}: R2={float(r2_score(y, pred_cache[mid])):.6f}")

    # ---- V0: AdaBoost baseline ----
    print("Computing AdaBoost baseline...")
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=15, min_samples_split=5,
                                         min_samples_leaf=1, random_state=RANDOM_STATE),
        n_estimators=250, learning_rate=0.05, loss="square", random_state=RANDOM_STATE,
    )
    ada_pred = cross_val_predict(ada, X_raw_arr, y, cv=cv, n_jobs=1, method="predict")
    ada_metrics, ada_folds = fold_metrics_from_cv(cv, X_raw_arr, y, ada_pred)

    # ---- Ablation for each space ----
    all_results = {
        "meta": {
            "study": "Refined Feature Space Ablation (Phase 2.1)",
            "feature_levels": {
                "A": {"count": len(LEVEL_A_FEATURES), "features": LEVEL_A_FEATURES},
                "B": {"count": len(LEVEL_B_FEATURES), "features": LEVEL_B_FEATURES},
                "C": {"count": len(LEVEL_C_FEATURES), "features": LEVEL_C_FEATURES},
            },
            "spaces": {name: sp["desc"] for name, sp in spaces.items()},
            "n_trials_hpo": args.trials,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "v0_adaboost": {"metrics": ada_metrics, "fold_metrics": ada_folds},
        "spaces": {},
        "raw_control": {},
    }

    print("\nRunning V1 (Primary+Global) for each refined space...")
    for space_name in spaces:
        model_order = [f"{space_name}_XGB", f"{space_name}_LGB", f"{space_name}_HGB"]
        metrics, folds, weights = run_global_blend(model_order, pred_cache, cv,
                                                    spaces[space_name]["primary"], y)
        all_results["spaces"][f"{space_name}_V1"] = {
            "description": f"V1: {space_name} Primary+Global Blend",
            "metrics": metrics, "fold_metrics": folds, "weights": weights,
            "feature_dim": spaces[space_name]["primary"].shape[1],
        }
        print(f"  {space_name} V1: R2={metrics['R2_mean']:.6f} RMSE={metrics['RMSE_mean']:.4f}")

    # Raw V4: Raw+Piecewise (no feature engineering)
    print("\nRunning V4 (Raw+Piecewise) as control...")
    raw_model_order = ["Raw_XGB", "Raw_LGB", "Raw_HGB", "Raw_HGB_anchor"]
    P_raw = np.column_stack([pred_cache[m] for m in raw_model_order])
    early = age <= AGE_SPLIT_DAY
    late = ~early
    w_e = optimize_weights(P_raw[early], y[early])
    w_l = optimize_weights(P_raw[late], y[late])
    pred_raw_pw = np.empty_like(y)
    pred_raw_pw[early] = P_raw[early] @ w_e
    pred_raw_pw[late] = P_raw[late] @ w_l
    raw_v4_metrics, raw_v4_folds = fold_metrics_from_cv(cv, X_raw_arr, y, pred_raw_pw)
    all_results["raw_control"] = {
        "description": "V4: Raw 8D + Piecewise (no feature engineering)",
        "metrics": raw_v4_metrics, "fold_metrics": raw_v4_folds,
        "weights_early": {raw_model_order[i]: float(w_e[i]) for i in range(len(raw_model_order))},
        "weights_late": {raw_model_order[i]: float(w_l[i]) for i in range(len(raw_model_order))},
    }
    print(f"  Raw V4: R2={raw_v4_metrics['R2_mean']:.6f} RMSE={raw_v4_metrics['RMSE_mean']:.4f}")

    # ---- Load existing full ablation for comparison ----
    existing_path = PROJECT_ROOT / "results" / "metrics" / "ablation_results_acdcb_v2.json"
    existing_v4_r2 = None
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_v4_r2 = existing["variants"]["v4_raw_age_piecewise"]["metrics"]["R2_mean"]
        print(f"\nExisting V4 (full pipeline) R2={existing_v4_r2:.6f} (for reference)")

    # ---- Save ----
    json_path = METRICS_DIR / "refined_feature_ablation.json"
    all_results["runtime_sec"] = float(time.perf_counter() - t0)
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {json_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Phase 2.1: 精炼特征空间消融 — 关键结果汇总")
    print("=" * 70)
    print(f"{'Variant':<30} {'Dim':>5} {'R2':>8} {'RMSE':>8} {'vs Raw dR2':>12}")
    print("-" * 70)
    print(f"{'V0 AdaBoost (Raw 8D)':<30} {8:>5} {ada_metrics['R2_mean']:>8.4f} {ada_metrics['RMSE_mean']:>8.4f} {'--':>12}")
    print(f"{'V4 Raw+Piecewise (8D)':<30} {8:>5} {raw_v4_metrics['R2_mean']:>8.4f} {raw_v4_metrics['RMSE_mean']:>8.4f} {'(baseline)':>12}")
    for space_name in spaces:
        m = all_results["spaces"][f"{space_name}_V1"]["metrics"]
        d = all_results["spaces"][f"{space_name}_V1"]["feature_dim"]
        dr2 = m['R2_mean'] - raw_v4_metrics['R2_mean']
        print(f"{space_name + ' V1':<30} {d:>5} {m['R2_mean']:>8.4f} {m['RMSE_mean']:>8.4f} {dr2:>+12.6f}")

    print()
    print("Hypothesis check:")
    for space_name in spaces:
        m = all_results["spaces"][f"{space_name}_V1"]["metrics"]
        dr2 = m['R2_mean'] - raw_v4_metrics['R2_mean']
        status = "PASS" if dr2 <= 0.001 else "FAIL (feature eng shows gain)"
        print(f"  {space_name}: R2 <= Raw? dR2={dr2:+.6f} [{status}]")


if __name__ == "__main__":
    main()
