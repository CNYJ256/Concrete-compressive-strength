from __future__ import annotations

"""Test whether heterogeneous model families rescue ensemble blending.

The evaluated pools are:
  H1: XGB + LGB + HGB + HGB_Anchor (original ACDCB pool)
  H2: XGB + MLP + HGB + HGB_Anchor
  H3: XGB + GPR + HGB + HGB_Anchor
  H4: XGB + MLP + GPR + HGB_Anchor

Outputs:
  results/metrics/heterogeneous_pool_test.json
  results/predictions/heterogeneous_pool_oof.csv
  figures/presentation_highres/fig_heterogeneous_pool_comparison.pdf
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from concrete_compressive_strength.core import (  # noqa: E402
    ANCHOR_MODEL_PARAMS,
    BASE_FEATURES,
    BASE_MODEL_PARAMS,
    RANDOM_STATE,
    TARGET_COL,
    build_hgb,
    build_lgbm,
    build_xgb,
    feature_engineering,
    feature_engineering_anchor,
    load_data,
)

FIG_DIR = ROOT / "figures" / "presentation_highres"
METRICS_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight", pad_inches=0.02)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def make_cv() -> KFold:
    return KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def strip_params(params: dict[str, Any]) -> dict[str, Any]:
    blocked = {"random_state", "n_jobs", "objective", "tree_method", "verbosity", "early_stopping"}
    return {k: v for k, v in params.items() if k not in blocked}


def build_mlp(params: dict[str, Any] | None = None) -> Pipeline:
    if params is None:
        params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 1e-3,
            "learning_rate_init": 1e-3,
            "max_iter": 2500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 80,
        }
    model = MLPRegressor(random_state=RANDOM_STATE, **params)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def build_gpr() -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(len(BASE_FEATURES)), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 1e2))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def tune_mlp(X: np.ndarray, y: np.ndarray, n_trials: int, timeout: int | None) -> dict[str, Any]:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        width1 = trial.suggest_categorical("width1", [64, 96, 128, 192])
        width2 = trial.suggest_categorical("width2", [0, 32, 64, 96])
        hidden = (width1,) if width2 == 0 else (width1, width2)
        params = {
            "hidden_layer_sizes": hidden,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "max_iter": 2500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 80,
        }
        scores = cross_val_score(build_mlp(params), X, y, cv=make_cv(), scoring="neg_root_mean_squared_error", n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = dict(study.best_params)
    width1 = int(best.pop("width1"))
    width2 = int(best.pop("width2"))
    best["hidden_layer_sizes"] = (width1,) if width2 == 0 else (width1, width2)
    best.update({"max_iter": 2500, "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 80})
    return best


def optimize_weights(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    n_models = P.shape[1]
    init = np.full(n_models, 1.0 / n_models)

    def obj(w: np.ndarray) -> float:
        return rmse(y, P @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    res = minimize(
        obj,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )
    if res.success:
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / np.sum(weights)
    else:
        weights = init
    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "fun": float(obj(weights)),
    }
    return weights, info


def fold_metrics(y: np.ndarray, pred: np.ndarray, X_ref: np.ndarray) -> dict[str, Any]:
    r2s, rmses, maes = [], [], []
    for _, test_idx in make_cv().split(X_ref, y):
        yt, yp = y[test_idx], pred[test_idx]
        r2s.append(float(r2_score(yt, yp)))
        rmses.append(rmse(yt, yp))
        maes.append(mae(yt, yp))
    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "R2_oof": float(r2_score(y, pred)),
        "RMSE_oof": rmse(y, pred),
        "MAE_oof": mae(y, pred),
        "folds": {"R2": r2s, "RMSE": rmses, "MAE": maes},
    }


def evaluate_pool(pool_id: str, model_names: list[str], pred_cache: dict[str, np.ndarray], y: np.ndarray, X_ref: np.ndarray) -> dict[str, Any]:
    P = np.column_stack([pred_cache[name] for name in model_names])
    corr = np.corrcoef(P, rowvar=False)
    weights, opt_info = optimize_weights(P, y)
    pred = P @ weights

    singles = {
        name: {
            "R2_oof": float(r2_score(y, pred_cache[name])),
            "RMSE_oof": rmse(y, pred_cache[name]),
            "MAE_oof": mae(y, pred_cache[name]),
        }
        for name in model_names
    }
    best_name = max(model_names, key=lambda n: singles[n]["R2_oof"])
    metrics = fold_metrics(y, pred, X_ref)
    cross_corr = corr[np.triu_indices_from(corr, k=1)]

    return {
        "pool_id": pool_id,
        "model_names": model_names,
        "pairwise_pearson_r": {
            model_names[i]: {model_names[j]: float(corr[i, j]) for j in range(len(model_names))}
            for i in range(len(model_names))
        },
        "mean_cross_pairwise_r": float(np.mean(cross_corr)),
        "min_cross_pairwise_r": float(np.min(cross_corr)),
        "max_cross_pairwise_r": float(np.max(cross_corr)),
        "single_models": singles,
        "best_single": {"name": best_name, **singles[best_name]},
        "optimized_fusion": {
            "weights": {name: float(weights[i]) for i, name in enumerate(model_names)},
            "optimizer_info": opt_info,
            "metrics": metrics,
        },
        "delta_opt_vs_best_single": {
            "R2_gain_oof": float(metrics["R2_oof"] - singles[best_name]["R2_oof"]),
            "RMSE_drop_oof": float(singles[best_name]["RMSE_oof"] - metrics["RMSE_oof"]),
        },
    }


def plot_pool_comparison(results: list[dict[str, Any]]) -> None:
    labels = [r["pool_id"] for r in results]
    r2 = [r["optimized_fusion"]["metrics"]["R2_oof"] for r in results]
    delta = [r["delta_opt_vs_best_single"]["R2_gain_oof"] for r in results]
    min_corr = [r["min_cross_pairwise_r"] for r in results]
    x = np.arange(len(results))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(3.5, 2.85))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width / 2, r2, width, color="#4C78A8", alpha=0.88, label=r"Fusion $R^2$")
    b2 = ax2.bar(x + width / 2, delta, width, color="#E45756", alpha=0.82, label=r"$\Delta R^2$ vs best")
    ax1.plot(x, min_corr, color="#54A24B", marker="o", linewidth=1.1, markersize=3.5, label="Min pairwise r")

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00025, f"{bar.get_height():.4f}",
                 ha="center", va="bottom", fontsize=6, rotation=90)
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00004, f"{bar.get_height():+.4f}",
                 ha="center", va="bottom", fontsize=6, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"Fusion $R^2$", color="#4C78A8")
    ax2.set_ylabel(r"$\Delta R^2$ vs best single", color="#E45756")
    ax1.set_title("Heterogeneous Pool Test", fontweight="bold", fontsize=9)
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(loc="lower left", fontsize=6.5)
    ax2.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig_heterogeneous_pool_comparison.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / "fig_heterogeneous_pool_comparison.png", **SAVE_KWARGS)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-trials", type=int, default=0, help="Optional Optuna trials for MLP tuning.")
    parser.add_argument("--mlp-timeout", type=int, default=None, help="Optional MLP tuning timeout in seconds.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for CV wrappers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_raw = df[BASE_FEATURES].copy()
    X_primary = feature_engineering(X_raw)
    X_anchor = feature_engineering_anchor(X_raw)
    y = df[TARGET_COL].to_numpy(dtype=float)
    cv = make_cv()
    X_ref = X_raw.to_numpy(dtype=float)

    mlp_params = None
    if args.mlp_trials > 0:
        print(f"[INFO] Tuning MLP for {args.mlp_trials} Optuna trials...")
        mlp_params = tune_mlp(X_ref, y, args.mlp_trials, args.mlp_timeout)

    model_specs: list[tuple[str, Any, pd.DataFrame | np.ndarray, int]] = [
        ("XGB_primary", build_xgb(BASE_MODEL_PARAMS["XGBoost"]), X_primary, args.n_jobs),
        ("LGB_primary", build_lgbm(BASE_MODEL_PARAMS["LightGBM"]), X_primary, args.n_jobs),
        ("HGB_primary", build_hgb(BASE_MODEL_PARAMS["HGB"]), X_primary, args.n_jobs),
        ("HGB_Anchor", build_hgb(ANCHOR_MODEL_PARAMS), X_anchor, args.n_jobs),
        ("MLP_raw", build_mlp(mlp_params), X_raw, args.n_jobs),
        ("GPR_raw", build_gpr(), X_raw, 1),
    ]

    pred_cache: dict[str, np.ndarray] = {}
    oof_df = pd.DataFrame({"y_true": y})
    for name, estimator, X_used, n_jobs in model_specs:
        print(f"[INFO] Generating OOF predictions: {name}")
        pred = cross_val_predict(clone(estimator), X_used, y, cv=cv, n_jobs=n_jobs, method="predict")
        pred_cache[name] = pred
        oof_df[name] = pred
        print(f"       R2={r2_score(y, pred):.6f}, RMSE={rmse(y, pred):.4f}")

    pools = [
        ("H1_original", ["XGB_primary", "LGB_primary", "HGB_primary", "HGB_Anchor"]),
        ("H2_mlp", ["XGB_primary", "MLP_raw", "HGB_primary", "HGB_Anchor"]),
        ("H3_gpr", ["XGB_primary", "GPR_raw", "HGB_primary", "HGB_Anchor"]),
        ("H4_mlp_gpr", ["XGB_primary", "MLP_raw", "GPR_raw", "HGB_Anchor"]),
    ]
    pool_results = [evaluate_pool(pool_id, names, pred_cache, y, X_ref) for pool_id, names in pools]
    for result in pool_results:
        m = result["optimized_fusion"]["metrics"]
        d = result["delta_opt_vs_best_single"]
        print(
            f"[POOL] {result['pool_id']}: R2={m['R2_oof']:.6f}, RMSE={m['RMSE_oof']:.4f}, "
            f"min r={result['min_cross_pairwise_r']:.4f}, dR2={d['R2_gain_oof']:+.6f}"
        )

    plot_pool_comparison(pool_results)
    oof_df.to_csv(PRED_DIR / "heterogeneous_pool_oof.csv", index=False, encoding="utf-8-sig")

    payload = {
        "meta": {
            "study": "Heterogeneous model pool diagnostic",
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 10, "shuffle": True},
            "n_samples": int(len(y)),
            "mlp_tuning": {"trials": int(args.mlp_trials), "params": mlp_params},
        },
        "pools": {r["pool_id"]: r for r in pool_results},
        "runtime_sec": float(time.perf_counter() - t0),
    }
    (METRICS_DIR / "heterogeneous_pool_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[OK] Saved metrics: results/metrics/heterogeneous_pool_test.json")
    print("[OK] Saved figure: fig_heterogeneous_pool_comparison")


if __name__ == "__main__":
    main()
