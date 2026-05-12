from __future__ import annotations

"""Compute real TreeSHAP explanations for the tuned raw-feature XGBoost model.

Outputs:
  results/metrics/shap_analysis.json
  results/predictions/shap_values_raw_xgb.csv
  figures/presentation_highres/fig_shap_summary.pdf
  figures/presentation_highres/fig_shap_dependence_age.pdf
  figures/presentation_highres/fig_shap_dependence_water_binder_ratio.pdf
  figures/presentation_highres/fig_shap_dependence_cement.pdf
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import (  # noqa: E402
    BASE_FEATURES,
    BASE_MODEL_PARAMS,
    RANDOM_STATE,
    TARGET_COL,
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
    "superplasticizer": "Superplasticizer",
    "coarse_agg": "Coarse aggregate",
    "fine_agg": "Fine aggregate",
    "age": "Age",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def strip_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    blocked = {"random_state", "n_jobs", "objective", "tree_method", "verbosity"}
    return {k: v for k, v in params.items() if k not in blocked}


def load_xgb_params() -> tuple[dict[str, Any], str]:
    raw_hp = METRICS_DIR / "raw_hyperparams.json"
    if raw_hp.exists():
        data = json.loads(raw_hp.read_text(encoding="utf-8"))
        return strip_xgb_params(data["XGBoost_raw"]["best_params"]), str(raw_hp)
    return strip_xgb_params(BASE_MODEL_PARAMS["XGBoost"]), "BASE_MODEL_PARAMS['XGBoost']"


def build_xgb(params: dict[str, Any], n_jobs: int = -1) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
        **params,
    )


def make_display_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.rename(columns=DISPLAY_NAMES)


def binned_median(x: np.ndarray, y: np.ndarray, bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    xs, ys = [], []
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        mask = (x >= lo) & (x <= hi)
        if np.sum(mask) < 3:
            continue
        xs.append(float(np.median(x[mask])))
        ys.append(float(np.median(y[mask])))
    return np.asarray(xs), np.asarray(ys)


def dependence_plot(
    x: np.ndarray,
    shap_y: np.ndarray,
    color_by: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    sc = ax.scatter(x, shap_y, c=color_by, cmap="viridis", s=10, alpha=0.58, edgecolor="none")
    bx, by = binned_median(x, shap_y)
    if len(bx) > 1:
        ax.plot(bx, by, color="#E45756", linewidth=1.5, label="Binned median")
        ax.legend(fontsize=7, loc="best")
    ax.axhline(0, color="#333333", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.grid(alpha=0.18)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Age (days)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / f"{filename}.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / f"{filename}.png", **SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(ROOT / "data" / "Concrete_Data.xls")
    X_raw = df[BASE_FEATURES].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    params, params_source = load_xgb_params()

    print(f"[INFO] Data: N={len(y)}, D={X_raw.shape[1]}")
    print(f"[INFO] XGBoost parameter source: {params_source}")

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = cross_val_predict(build_xgb(params, n_jobs=1), X_raw.to_numpy(dtype=float), y, cv=cv, n_jobs=-1)
    cv_metrics = {
        "R2_oof": float(r2_score(y, oof_pred)),
        "RMSE_oof": rmse(y, oof_pred),
    }
    print(f"[INFO] Tuned raw XGB OOF: R2={cv_metrics['R2_oof']:.6f}, RMSE={cv_metrics['RMSE_oof']:.4f}")

    model = build_xgb(params)
    model.fit(X_raw, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_raw)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    rel = mean_abs / np.sum(mean_abs)
    order = np.argsort(mean_abs)[::-1]

    X_display = make_display_frame(X_raw)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_display,
        max_display=len(BASE_FEATURES),
        show=False,
        plot_size=(3.5, 3.1),
        color_bar=True,
    )
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP Summary: Tuned Raw XGBoost", fontweight="bold", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig_shap_summary.pdf", **SAVE_KWARGS)
    fig.savefig(FIG_DIR / "fig_shap_summary.png", **SAVE_KWARGS)
    plt.close(fig)

    shap_by_feature = {feature: shap_values[:, i] for i, feature in enumerate(BASE_FEATURES)}
    age = X_raw["age"].to_numpy(dtype=float)
    cement = X_raw["cement"].to_numpy(dtype=float)
    binder = (
        X_raw["cement"].to_numpy(dtype=float)
        + X_raw["slag"].to_numpy(dtype=float)
        + X_raw["fly_ash"].to_numpy(dtype=float)
    )
    water_binder = X_raw["water"].to_numpy(dtype=float) / np.maximum(binder, 1e-6)
    wb_component_shap = (
        shap_by_feature["water"]
        + shap_by_feature["cement"]
        + shap_by_feature["slag"]
        + shap_by_feature["fly_ash"]
    )

    dependence_plot(
        x=age,
        shap_y=shap_by_feature["age"],
        color_by=age,
        xlabel="Age (days)",
        ylabel="SHAP value for age (MPa)",
        title="Age Dependence",
        filename="fig_shap_dependence_age",
    )
    dependence_plot(
        x=water_binder,
        shap_y=wb_component_shap,
        color_by=age,
        xlabel="Water-binder ratio",
        ylabel="Aggregate SHAP: water + binder inputs (MPa)",
        title="Water-Binder Dependence",
        filename="fig_shap_dependence_water_binder_ratio",
    )
    dependence_plot(
        x=cement,
        shap_y=shap_by_feature["cement"],
        color_by=age,
        xlabel="Cement content (kg/m$^3$)",
        ylabel="SHAP value for cement (MPa)",
        title="Cement Dependence",
        filename="fig_shap_dependence_cement",
    )

    shap_df = pd.DataFrame({"y_true": y, "xgb_raw_oof": oof_pred})
    for i, feature in enumerate(BASE_FEATURES):
        shap_df[f"shap_{feature}"] = shap_values[:, i]
    shap_df["water_binder_ratio"] = water_binder
    shap_df["shap_water_binder_aggregate"] = wb_component_shap
    shap_df.to_csv(PRED_DIR / "shap_values_raw_xgb.csv", index=False, encoding="utf-8-sig")

    importance = [
        {
            "feature": str(BASE_FEATURES[i]),
            "display_name": DISPLAY_NAMES[str(BASE_FEATURES[i])],
            "mean_abs_shap": float(mean_abs[i]),
            "relative_importance": float(rel[i]),
        }
        for i in order
    ]
    payload = {
        "meta": {
            "study": "Raw XGBoost TreeSHAP analysis",
            "random_state": RANDOM_STATE,
            "n_samples": int(len(y)),
            "n_features": int(X_raw.shape[1]),
            "params_source": params_source,
        },
        "model": {
            "name": "XGBoost_raw_tuned",
            "params": params,
            "cv_metrics": cv_metrics,
            "expected_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        },
        "mean_abs_shap": importance,
        "dependence_definitions": {
            "age": "x = raw age, y = SHAP value of raw age feature",
            "water_binder_ratio": (
                "x = water/(cement+slag+fly_ash), y = summed SHAP contribution of "
                "water, cement, slag, and fly_ash raw inputs; W/B is not a separate raw-model input"
            ),
            "cement": "x = cement content, y = SHAP value of raw cement feature",
        },
    }
    (METRICS_DIR / "shap_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[OK] Saved metrics: results/metrics/shap_analysis.json")
    print("[OK] Saved figures: fig_shap_summary and three dependence plots")


if __name__ == "__main__":
    main()
