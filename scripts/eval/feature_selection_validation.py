from __future__ import annotations

"""Validate whether collinearity mitigation rescues engineered features.

The script evaluates:
  G1: raw 8-dimensional inputs
  G2: full 32-dimensional engineered feature space
  G3: VIF-filtered engineered space with VIF < 10
  G4: VIF-filtered engineered space with VIF < 5
  G5: LASSO-ranked top-K subsets (K = 8, 12, 16, 20 by default)

Outputs:
  results/metrics/feature_selection_validation.json
  results/predictions/feature_selection_validation_oof.csv
  figures/presentation_highres/fig_vif_bar.pdf
  figures/presentation_highres/fig_lasso_path.pdf
  figures/presentation_highres/fig_feature_selection_comparison.pdf
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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LinearRegression, lasso_path
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore", category=OptimizeWarning)

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import (  # noqa: E402
    BASE_FEATURES,
    BASE_MODEL_PARAMS,
    RANDOM_STATE,
    TARGET_COL,
    feature_engineering,
    load_data,
)

FIG_DIR = ROOT / "figures" / "presentation_highres"
METRICS_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight", pad_inches=0.02)

DISPLAY_NAMES = {
    "cement": "Cement",
    "slag": "Slag",
    "fly_ash": "Fly ash",
    "water": "Water",
    "superplasticizer": "SP",
    "coarse_agg": "Coarse agg.",
    "fine_agg": "Fine agg.",
    "age": "Age",
    "binder": "Binder",
    "water_cement_ratio": "W/C",
    "water_binder_ratio": "W/B",
    "sp_binder_ratio": "SP/B",
    "scm_ratio": "SCM/B",
    "fine_ratio_in_agg": "Fine/agg.",
    "age_log1p": "log(1+t)",
    "age_sqrt": "sqrt(t)",
    "age_pow_0_25": "t^0.25",
    "abrams_index": "Abrams index",
    "cement_age_interaction": "Cement*log(t)",
    "binder_age_interaction": "Binder*log(t)",
    "wb_age_interaction": "W/B*log(t)",
    "paste_index": "Paste index",
    "binder_to_agg_ratio": "Binder/agg.",
    "water_to_paste_ratio": "Water/paste",
    "cement_fraction_in_binder": "Cement/binder",
    "slag_fraction_in_binder": "Slag/binder",
    "flyash_fraction_in_binder": "Fly ash/binder",
    "superplasticizer_efficiency": "SP/water",
    "maturity_index": "Maturity index",
    "agg_binder_balance": "Agg./binder",
    "age_inverse": "1/(t+1)",
    "age_wc_interaction": "W/C*log(t)",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def make_cv() -> KFold:
    return KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def strip_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    blocked = {"random_state", "n_jobs", "objective", "tree_method", "verbosity"}
    return {k: v for k, v in params.items() if k not in blocked}


def load_xgb_params() -> tuple[dict[str, Any], str]:
    raw_hp = METRICS_DIR / "raw_hyperparams.json"
    if raw_hp.exists():
        data = json.loads(raw_hp.read_text(encoding="utf-8"))
        return strip_xgb_params(data["XGBoost_raw"]["best_params"]), str(raw_hp)
    return strip_xgb_params(BASE_MODEL_PARAMS["XGBoost"]), "BASE_MODEL_PARAMS['XGBoost']"


def build_xgb(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        **params,
    )


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    rows: list[dict[str, Any]] = []

    for idx, name in enumerate(X.columns):
        y_col = Xs[:, idx]
        X_other = np.delete(Xs, idx, axis=1)
        if np.nanstd(y_col) < 1e-12:
            r2 = 1.0
        else:
            model = LinearRegression()
            model.fit(X_other, y_col)
            r2 = float(r2_score(y_col, model.predict(X_other)))
        vif = np.inf if r2 >= 0.999999 else 1.0 / max(1.0 - r2, 1e-12)
        rows.append(
            {
                "feature": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "r2_against_others": r2,
                "vif": float(vif) if np.isfinite(vif) else "inf",
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False, key=lambda s: s.replace("inf", np.inf))


def iterative_vif_filter(X: pd.DataFrame, threshold: float) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame]:
    remaining = list(X.columns)
    removed: list[dict[str, Any]] = []

    while len(remaining) > 2:
        table = compute_vif_table(X[remaining])
        top = table.iloc[0]
        top_vif = np.inf if top["vif"] == "inf" else float(top["vif"])
        if top_vif < threshold:
            break
        feature = str(top["feature"])
        removed.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES.get(feature, feature),
                "vif_at_removal": "inf" if not np.isfinite(top_vif) else float(top_vif),
                "remaining_before_removal": len(remaining),
            }
        )
        remaining.remove(feature)

    final_table = compute_vif_table(X[remaining])
    return remaining, removed, final_table


def fit_lasso_ranker(X: pd.DataFrame, y: np.ndarray, n_jobs: int) -> dict[str, Any]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    y_centered = y - np.mean(y)

    alphas = np.logspace(-4, 2, 120)
    lasso = LassoCV(alphas=alphas, cv=make_cv(), random_state=RANDOM_STATE, max_iter=20000, n_jobs=n_jobs)
    lasso.fit(Xs, y)

    abs_coef = np.abs(lasso.coef_)
    fallback_rank = np.abs(np.corrcoef(Xs, y_centered, rowvar=False)[-1, :-1])
    rank_score = np.where(abs_coef > 0, abs_coef, fallback_rank * 1e-6)
    order = np.argsort(rank_score)[::-1]
    ranked_features = [str(X.columns[i]) for i in order]

    path_alphas = np.logspace(-4, 2, 120)
    _, coefs, _ = lasso_path(Xs, y_centered, alphas=path_alphas, max_iter=20000)

    return {
        "alpha": float(lasso.alpha_),
        "coef": {str(col): float(coef) for col, coef in zip(X.columns, lasso.coef_)},
        "ranked_features": ranked_features,
        "path_alphas": path_alphas,
        "path_coefs": coefs,
    }


def tune_xgb_params(X: pd.DataFrame, y: np.ndarray, n_trials: int, timeout: int | None) -> dict[str, Any]:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        }
        model = build_xgb(params)
        scores = cross_val_score(
            model,
            X.to_numpy(dtype=float),
            y,
            cv=make_cv(),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return strip_xgb_params(study.best_params)


def evaluate_group(
    group_id: str,
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    base_params: dict[str, Any],
    n_jobs: int,
    optuna_trials: int,
    optuna_timeout: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    params = base_params
    params_source = "shared_xgb_params"
    if optuna_trials > 0:
        params = tune_xgb_params(X, y, optuna_trials, optuna_timeout)
        params_source = f"optuna_{optuna_trials}_trials"

    model = build_xgb(params)
    cv = make_cv()
    X_arr = X.to_numpy(dtype=float)
    pred = cross_val_predict(clone(model), X_arr, y, cv=cv, n_jobs=n_jobs, method="predict")

    fold_r2, fold_rmse, fold_mae = [], [], []
    for _, test_idx in cv.split(X_arr, y):
        yt, yp = y[test_idx], pred[test_idx]
        fold_r2.append(float(r2_score(yt, yp)))
        fold_rmse.append(rmse(yt, yp))
        fold_mae.append(mae(yt, yp))

    result = {
        "group_id": group_id,
        "name": name,
        "n_features": int(X.shape[1]),
        "features": list(map(str, X.columns)),
        "params_source": params_source,
        "params": params,
        "metrics": {
            "R2_mean": float(np.mean(fold_r2)),
            "R2_std": float(np.std(fold_r2)),
            "RMSE_mean": float(np.mean(fold_rmse)),
            "RMSE_std": float(np.std(fold_rmse)),
            "MAE_mean": float(np.mean(fold_mae)),
            "MAE_std": float(np.std(fold_mae)),
            "R2_oof": float(r2_score(y, pred)),
            "RMSE_oof": rmse(y, pred),
            "MAE_oof": mae(y, pred),
        },
        "fold_metrics": {"R2": fold_r2, "RMSE": fold_rmse, "MAE": fold_mae},
    }
    return result, pred


def plot_vif_bar(vif_table: pd.DataFrame) -> None:
    plot_df = vif_table.copy()
    numeric_vif = plot_df["vif"].replace("inf", np.inf).astype(float)
    finite_max = numeric_vif[np.isfinite(numeric_vif)].max()
    cap = max(100.0, finite_max * 1.05)
    plot_df["vif_plot"] = np.where(np.isfinite(numeric_vif), numeric_vif, cap)
    plot_df = plot_df.sort_values("vif_plot", ascending=True).tail(24)

    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    colors = np.where(plot_df["vif_plot"].to_numpy(dtype=float) >= 10, "#E45756", "#4C78A8")
    ax.barh(range(len(plot_df)), plot_df["vif_plot"], color=colors, edgecolor="#333333", linewidth=0.35)
    ax.axvline(5, color="#F58518", linestyle="--", linewidth=0.9, label="VIF = 5")
    ax.axvline(10, color="#E45756", linestyle="--", linewidth=0.9, label="VIF = 10")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["display_name"], fontsize=6.3)
    ax.set_xscale("log")
    ax.set_xlabel("Variance Inflation Factor (log scale)")
    ax.set_title("Collinearity in Engineered Features", fontweight="bold", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig_vif_bar.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / "fig_vif_bar.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_lasso_path(lasso_info: dict[str, Any], feature_names: list[str]) -> None:
    alphas = lasso_info["path_alphas"]
    coefs = lasso_info["path_coefs"]
    alpha_cv = lasso_info["alpha"]
    coef_abs = np.abs(np.asarray(list(lasso_info["coef"].values())))
    top_idx = set(np.argsort(coef_abs)[::-1][:10])

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, feature in enumerate(feature_names):
        color = None
        alpha = 0.9 if i in top_idx else 0.15
        lw = 1.1 if i in top_idx else 0.5
        ax.plot(alphas, coefs[i, :], linewidth=lw, alpha=alpha, color=color)
    ax.axvline(alpha_cv, color="#E45756", linestyle="--", linewidth=1.0, label=fr"$\alpha_{{CV}}$={alpha_cv:.3g}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("LASSO penalty alpha")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("LASSO Coefficient Path", fontweight="bold", fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig_lasso_path.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / "fig_lasso_path.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_comparison(results: list[dict[str, Any]]) -> None:
    labels = [r["group_id"].replace("_", "\n") for r in results]
    r2 = [r["metrics"]["R2_mean"] for r in results]
    rmse_vals = [r["metrics"]["RMSE_mean"] for r in results]
    x = np.arange(len(results))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - width / 2, r2, width, color="#4C78A8", alpha=0.88, label=r"$R^2$", zorder=3)
    b2 = ax2.bar(x + width / 2, rmse_vals, width, color="#E45756", alpha=0.82, label="RMSE", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.2)
    ax1.set_ylabel(r"$R^2$", color="#4C78A8")
    ax2.set_ylabel("RMSE (MPa)", color="#E45756")
    ax1.set_title("Feature Selection Validation", fontweight="bold", fontsize=9)
    ax1.grid(axis="y", alpha=0.18)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=6.5)
    legend.set_zorder(20)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig_feature_selection_comparison.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / "fig_feature_selection_comparison.png", **SAVE_KWARGS)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lasso-k", default="8,12,16,20", help="Comma-separated top-K feature counts.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Trials per group. Default uses shared XGB params.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout per group in seconds.")
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
    X_engineered = feature_engineering(X_raw)
    y = df[TARGET_COL].to_numpy(dtype=float)
    base_params, params_source = load_xgb_params()

    print(f"[INFO] Data: N={len(y)}, raw D={X_raw.shape[1]}, engineered D={X_engineered.shape[1]}")
    print(f"[INFO] XGBoost parameter source: {params_source}")

    vif_table = compute_vif_table(X_engineered)
    selected_vif10, removed_vif10, final_vif10 = iterative_vif_filter(X_engineered, threshold=10.0)
    selected_vif5, removed_vif5, final_vif5 = iterative_vif_filter(X_engineered, threshold=5.0)

    lasso_info = fit_lasso_ranker(X_engineered, y, args.n_jobs)
    lasso_k = [int(k.strip()) for k in args.lasso_k.split(",") if k.strip()]

    group_defs: list[tuple[str, str, pd.DataFrame]] = [
        ("G1_raw8", "Raw 8D baseline", X_raw),
        ("G2_full32", "Full engineered 32D", X_engineered),
        ("G3_vif10", "VIF-filtered engineered subset (VIF < 10)", X_engineered[selected_vif10]),
        ("G4_vif5", "VIF-filtered engineered subset (VIF < 5)", X_engineered[selected_vif5]),
    ]
    for k in lasso_k:
        feats = lasso_info["ranked_features"][:k]
        group_defs.append((f"G5_lasso{k}", f"LASSO top-{k} subset", X_engineered[feats]))

    results: list[dict[str, Any]] = []
    pred_out = pd.DataFrame({"y_true": y})
    for group_id, name, X_group in group_defs:
        print(f"[INFO] Evaluating {group_id}: {name} (D={X_group.shape[1]})")
        result, pred = evaluate_group(
            group_id=group_id,
            name=name,
            X=X_group,
            y=y,
            base_params=base_params,
            n_jobs=args.n_jobs,
            optuna_trials=args.optuna_trials,
            optuna_timeout=args.optuna_timeout,
        )
        results.append(result)
        pred_out[group_id] = pred
        m = result["metrics"]
        print(f"       R2={m['R2_mean']:.6f}±{m['R2_std']:.6f}, RMSE={m['RMSE_mean']:.4f}±{m['RMSE_std']:.4f}")

    plot_vif_bar(vif_table)
    plot_lasso_path(lasso_info, list(X_engineered.columns))
    plot_comparison(results)

    pred_out.to_csv(PRED_DIR / "feature_selection_validation_oof.csv", index=False, encoding="utf-8-sig")

    payload = {
        "meta": {
            "study": "Feature selection and collinearity validation",
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 10, "shuffle": True},
            "n_samples": int(len(y)),
            "xgb_params_source": params_source,
            "optuna_trials_per_group": int(args.optuna_trials),
        },
        "vif": {
            "full_table": vif_table.replace({np.inf: "inf"}).to_dict(orient="records"),
            "threshold_10": {
                "selected_features": selected_vif10,
                "n_selected": len(selected_vif10),
                "removed_features": removed_vif10,
                "final_vif": final_vif10.replace({np.inf: "inf"}).to_dict(orient="records"),
            },
            "threshold_5": {
                "selected_features": selected_vif5,
                "n_selected": len(selected_vif5),
                "removed_features": removed_vif5,
                "final_vif": final_vif5.replace({np.inf: "inf"}).to_dict(orient="records"),
            },
        },
        "lasso": {
            "alpha_cv": lasso_info["alpha"],
            "coefficients": lasso_info["coef"],
            "ranked_features": lasso_info["ranked_features"],
            "top_k": {str(k): lasso_info["ranked_features"][:k] for k in lasso_k},
        },
        "groups": results,
        "runtime_sec": float(time.perf_counter() - t0),
    }
    (METRICS_DIR / "feature_selection_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[OK] Saved metrics: results/metrics/feature_selection_validation.json")
    print("[OK] Saved figures: fig_vif_bar, fig_lasso_path, fig_feature_selection_comparison")


if __name__ == "__main__":
    main()
