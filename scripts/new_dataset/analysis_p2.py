"""P2-1 + P2-2 分析脚本。

P2-1: LASSO 特征选择 + 受限联合超参搜索 (50 trials Optuna)
P2-2: Split Conformal Prediction 不确定性量化对比 (manual implementation)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SRC_ROOT))

RANDOM_STATE = 42
EPS = 1e-8


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ============================================================================
# P2-1: LASSO Feature Selection + Constrained Joint HPO
# ============================================================================
def run_p21_lasso_joint_search():
    print("=" * 70)
    print("P2-1: LASSO Feature Selection + Constrained Joint HPO (50 trials)")
    print("=" * 70)

    from new_dataset.new_data_loader import (
        load_raw_new_data, strategy_a_preprocess, new_dataset_feature_engineering,
    )

    df = load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv")
    X_base, y, meta = strategy_a_preprocess(df, encoder="onehot")

    num_idx = list(range(meta["n_numerical_raw"]))
    X_num = X_base[:, num_idx]
    X_eng, eng_names = new_dataset_feature_engineering(X_num)
    X_full = np.column_stack([X_base, X_eng])
    full_feature_names = meta["feature_names"] + eng_names
    print(f"Full feature matrix: {X_full.shape}")

    # LASSO feature selection
    print(f"\nRunning LASSO CV for feature selection...")
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000, n_alphas=100, eps=1e-4, n_jobs=-1)
    lasso.fit(X_full, y)
    print(f"LASSO alpha: {lasso.alpha_:.6f}")

    coef_abs = np.abs(lasso.coef_)
    top_n = 12
    top_indices = np.argsort(coef_abs)[-top_n:][::-1]
    top_features = [full_feature_names[i] for i in top_indices]
    top_coefs = [float(lasso.coef_[i]) for i in top_indices]
    nz = int(np.sum(coef_abs > 1e-10))

    print(f"\nLASSO Top-{top_n} features (by |coef|):")
    for i, (name, coef) in enumerate(zip(top_features, top_coefs)):
        print(f"  {i+1:2d}. {name:<40s} coef={coef:+.6f}")
    print(f"\nNon-zero coef count: {nz}/{len(lasso.coef_)}")

    X_top = X_full[:, top_indices]
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective_xgb(trial, X):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
        }
        model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                             n_jobs=-1, tree_method="hist", **params)
        pred = cross_val_predict(model, X, y, cv=cv_inner, n_jobs=-1, method="predict")
        return float(r2_score(y, pred))

    n_trials = 50
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    # 1. Top-12 features
    print(f"\nHPO on LASSO Top-{top_n} features ({n_trials} trials)...")
    study_top = optuna.create_study(direction="maximize", pruner=pruner)
    study_top.optimize(lambda t: objective_xgb(t, X_top), n_trials=n_trials,
                        show_progress_bar=False, n_jobs=1)

    # 2. Full features
    print(f"HPO on full features ({n_trials} trials)...")
    study_full = optuna.create_study(direction="maximize", pruner=pruner)
    study_full.optimize(lambda t: objective_xgb(t, X_full), n_trials=n_trials,
                         show_progress_bar=False, n_jobs=1)

    # 3. Raw features
    print(f"HPO on raw features ({n_trials} trials)...")
    study_raw = optuna.create_study(direction="maximize", pruner=pruner)
    study_raw.optimize(lambda t: objective_xgb(t, X_base), n_trials=n_trials,
                        show_progress_bar=False, n_jobs=1)

    top_r2, full_r2, raw_r2 = study_top.best_value, study_full.best_value, study_raw.best_value
    print(f"\n{'='*50}")
    print(f"P2-1 Results:")
    print(f"  LASSO Top-{top_n} + HPO: R2={top_r2:.6f}")
    print(f"  Full features + HPO:  R2={full_r2:.6f}")
    print(f"  Raw features + HPO:   R2={raw_r2:.6f}")
    print(f"  Top12 vs Full: dR2={top_r2-full_r2:+.6f}")
    print(f"  Full vs Raw:   dR2={full_r2-raw_r2:+.6f}")
    print(f"  Top12 vs Raw:  dR2={top_r2-raw_r2:+.6f}")

    results = {
        "lasso": {"best_alpha": float(lasso.alpha_), "n_nonzero_coef": nz,
                   "n_total_features": len(lasso.coef_), "top_n": top_n,
                   "top_features": top_features, "top_coefs": top_coefs},
        "hpo_comparison": {
            "top12_features_r2": top_r2, "top12_best_params": study_top.best_params,
            "full_features_r2": full_r2, "full_best_params": study_full.best_params,
            "raw_features_r2": raw_r2, "raw_best_params": study_raw.best_params,
            "n_trials": n_trials,
        },
    }
    out_path = RESULTS_DIR / "metrics" / "p21_lasso_joint_hpo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")
    return results


# ============================================================================
# P2-2: Split Conformal Prediction (manual implementation)
# ============================================================================
def run_p22_conformal_prediction():
    print("\n" + "=" * 70)
    print("P2-2: Split Conformal Prediction — Single XGB vs ACDCB Ensemble")
    print("=" * 70)

    from new_dataset.new_data_loader import (
        load_raw_new_data, strategy_a_preprocess, get_age_array, new_dataset_feature_engineering,
    )
    from new_dataset.run_ablation_new_data import (
        build_xgb, build_lgb, build_hgb, anchor_feature_engineering,
    )

    # Load data
    df = load_raw_new_data(PROJECT_ROOT / "data" / "Data.csv")
    X_base, y, meta = strategy_a_preprocess(df, encoder="onehot")
    age = get_age_array(df)

    num_idx = list(range(meta["n_numerical_raw"]))
    X_num = X_base[:, num_idx]
    X_primary = np.column_stack([X_base, new_dataset_feature_engineering(X_num)[0]])
    X_anchor = np.column_stack([X_base, anchor_feature_engineering(X_num)[0]])

    # Load HPO params
    hpo_path = PROJECT_ROOT / "results" / "new_dataset" / "hyperparams" / "newdata_hpo_strategy_A.json"
    if hpo_path.exists():
        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
        xgb_p = hpo["primary_space"]["XGBoost"]["best_params"]
        lgb_p = hpo["primary_space"]["LightGBM"]["best_params"]
        hgb_p = hpo["primary_space"]["HGB"]["best_params"]
        print("Loaded HPO params from saved results.")
    else:
        print("WARNING: No HPO params, using defaults")
        xgb_p = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3,
                 "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0, "reg_alpha": 0.1, "reg_lambda": 1.0}
        lgb_p = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                 "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                 "reg_alpha": 0.1, "reg_lambda": 1.0, "min_split_gain": 0.01}
        hgb_p = {"learning_rate": 0.05, "max_iter": 500, "max_depth": None, "max_leaf_nodes": 31,
                 "min_samples_leaf": 20, "l2_regularization": 0.1, "max_bins": 255}

    # Split: train(70%) / calibrate(15%) / test(15%)
    X_tr_cal, X_test, y_tr_cal, y_test = train_test_split(
        X_primary, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tr_cal, y_tr_cal, test_size=0.15/0.85, random_state=RANDOM_STATE)

    print(f"\nSplit: train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")

    # Train models (avoid tree_method="hist" for XGB — crashes on this version)
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                             n_jobs=1, **{k: v for k, v in xgb_p.items()
                                          if k not in ("tree_method", "n_jobs")})
    print(f"\nTraining models on train split...")
    xgb_model.fit(X_train, y_train)

    lgb_model = build_lgb(lgb_p)
    lgb_model.fit(X_train, y_train)

    hgb_model = build_hgb(hgb_p)
    hgb_model.fit(X_train, y_train)

    # HGB anchor on anchor features
    train_end = len(X_train)
    cal_end = train_end + len(X_cal)
    X_anchor_train = X_anchor[:train_end]
    X_anchor_cal = X_anchor[train_end:cal_end]
    X_anchor_test = X_anchor[cal_end:]

    hgb_anchor_model = build_hgb(hgb_p)
    hgb_anchor_model.fit(X_anchor_train, y_train)

    # Ensemble predictions on calibrate
    P_cal = np.column_stack([
        xgb_model.predict(X_cal),
        lgb_model.predict(X_cal),
        hgb_model.predict(X_cal),
        hgb_anchor_model.predict(X_anchor_cal),
    ])

    # Optimize ensemble weights on calibration set
    from scipy.optimize import minimize
    def optimize_weights(P, y_true):
        n = P.shape[1]
        init = np.full(n, 1.0 / n)
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0) for _ in range(n)]
        res = minimize(lambda w: rmse(y_true, P @ w), init, method="SLSQP",
                       bounds=bnds, constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "disp": False})
        w = np.clip(res.x if res.success else init, 0.0, 1.0)
        return w / w.sum() if w.sum() > 0 else init

    w_ens = optimize_weights(P_cal, y_cal)
    model_names = ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_anchor"]
    print(f"Ensemble weights: {dict(zip(model_names, [float(x) for x in w_ens]))}")

    # Predictions on test set
    P_test = np.column_stack([
        xgb_model.predict(X_test),
        lgb_model.predict(X_test),
        hgb_model.predict(X_test),
        hgb_anchor_model.predict(X_anchor_test),
    ])
    ensemble_test_pred = P_test @ w_ens
    xgb_test_pred = xgb_model.predict(X_test)

    # Calibration residuals
    xgb_cal_pred = xgb_model.predict(X_cal)
    ensemble_cal_pred = P_cal @ w_ens
    residuals_xgb = np.abs(y_cal - xgb_cal_pred)
    residuals_ens = np.abs(y_cal - ensemble_cal_pred)

    print(f"\nXGB Single: test R2={r2_score(y_test, xgb_test_pred):.6f}, RMSE={rmse(y_test, xgb_test_pred):.4f}")
    print(f"ACDCB Ensemble: test R2={r2_score(y_test, ensemble_test_pred):.6f}, RMSE={rmse(y_test, ensemble_test_pred):.4f}")

    # Split Conformal Prediction
    print(f"\n{'='*60}")
    print(f"Split Conformal Prediction Results")
    print(f"{'='*60}")

    cp_results = {}
    for alpha, label in [(0.10, "90%"), (0.05, "95%")]:
        print(f"\n  {label} confidence (alpha={alpha}):")
        cp_results[label] = {}
        n_cal = len(y_cal)
        q_idx = int(np.ceil((1 - alpha) * (n_cal + 1))) - 1
        q_idx = min(max(q_idx, 0), n_cal - 1)

        for m_name, residuals, test_pred in [
            ("XGB Single", residuals_xgb, xgb_test_pred),
            ("ACDCB Ensemble", residuals_ens, ensemble_test_pred),
        ]:
            q_hat = float(np.sort(residuals)[q_idx])
            lower = test_pred - q_hat
            upper = test_pred + q_hat
            picp = float(np.mean((y_test >= lower) & (y_test <= upper)))
            mpiw = float(np.mean(upper - lower))

            print(f"    {m_name:<20s}: PICP={picp:.4f} (target={1-alpha:.2f}), MPIW={mpiw:.4f} MPa, Q={q_hat:.4f}")
            cp_results[label][m_name] = {"PICP": picp, "MPIW": mpiw, "Q_hat": q_hat}

    # Save
    out_path = RESULTS_DIR / "metrics" / "p22_conformal_prediction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "split": {"train": len(X_train), "calibrate": len(X_cal), "test": len(X_test)},
        "ensemble_weights": {n: float(w) for n, w in zip(model_names, w_ens)},
        "test_metrics": {
            "xgb_single": {"R2": float(r2_score(y_test, xgb_test_pred)),
                           "RMSE": rmse(y_test, xgb_test_pred)},
            "acdcb_ensemble": {"R2": float(r2_score(y_test, ensemble_test_pred)),
                               "RMSE": rmse(y_test, ensemble_test_pred)},
        },
        "conformal": cp_results,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"P2-2 Summary:")
    for label, data in cp_results.items():
        xgb_picp = data["XGB Single"]["PICP"]
        ens_picp = data["ACDCB Ensemble"]["PICP"]
        xgb_mpiw = data["XGB Single"]["MPIW"]
        ens_mpiw = data["ACDCB Ensemble"]["MPIW"]
        print(f"  {label}: XGB PICP={xgb_picp:.4f} MPIW={xgb_mpiw:.4f} | Ensemble PICP={ens_picp:.4f} MPIW={ens_mpiw:.4f}")
        print(f"          dMPIW={ens_mpiw-xgb_mpiw:+.4f} MPa (ensemble has {'WIDER' if ens_mpiw>xgb_mpiw else 'NARROWER'} intervals)")

    return cp_results


if __name__ == "__main__":
    run_p21_lasso_joint_search()
    run_p22_conformal_prediction()
