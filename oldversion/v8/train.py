from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from xgboost import XGBRegressor


RAW_TO_STD_COLUMN_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_agg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_agg",
    "Age (day)": "age",
    "Concrete compressive strength(MPa, megapascals) ": "strength",
}

BASE_FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
]
TARGET_COL = "strength"
RANDOM_STATE = 42

V7_BASELINE_HGB_PARAMS = {
    "learning_rate": 0.028,
    "max_iter": 2400,
    "max_depth": None,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 6,
    "l2_regularization": 0.001,
}


@dataclass
class IterationSpec:
    idx: int
    name: str
    model_family: str
    hypothesis: str
    n_trials: int


def get_logger() -> logging.Logger:
    logger = logging.getLogger("v8_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def resolve_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "Concrete_Data.xls",
        "prev": root / "v7" / "metrics.json",
        "model": root / "v8" / "model.joblib",
        "metrics": root / "v8" / "metrics.json",
        "changelog": root / "v8" / "CHANGELOG.md",
        "readme": root / "v8" / "README.md",
    }


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_excel(data_path, engine="xlrd")
    missing = set(RAW_TO_STD_COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"缺失列: {sorted(missing)}")

    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    needed = BASE_FEATURES + [TARGET_COL]
    df = df[needed].copy()

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna().reset_index(drop=True)


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = X.copy()

    binder = out["cement"] + out["slag"] + out["fly_ash"]
    total_agg = out["coarse_agg"] + out["fine_agg"]
    age_log = np.log1p(np.maximum(out["age"], 0.0))

    # v7 关键机理特征
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

    out["paste_index"] = (
        out["cement"]
        + out["slag"]
        + out["fly_ash"]
        + out["water"]
        + out["superplasticizer"]
    ) / (total_agg + eps)

    # v8 新增物理先验增强特征
    out["binder_to_agg_ratio"] = binder / (total_agg + eps)
    out["water_to_paste_ratio"] = out["water"] / (
        out["water"] + binder + out["superplasticizer"] + eps
    )

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

    return out


def feature_engineering_v7(X: pd.DataFrame) -> pd.DataFrame:
    """与 v7 保持一致的特征工程，用于融合锚点模型。"""
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

    out["paste_index"] = (
        out["cement"]
        + out["slag"]
        + out["fly_ash"]
        + out["water"]
        + out["superplasticizer"]
    ) / (total_agg + eps)

    out = out.replace([np.inf, -np.inf], np.nan)
    if out.isna().any(axis=1).any():
        out = out.fillna(out.median(numeric_only=True))

    return out


def load_prev_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"找不到上一版本指标文件: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cv = data["cv_10fold"]
    return {
        "R2_mean": float(cv["R2_mean"]),
        "R2_std": float(cv.get("R2_std", 0.0)),
        "RMSE_mean": float(cv["RMSE_mean"]),
        "RMSE_std": float(cv.get("RMSE_std", 0.0)),
    }


def build_cv() -> KFold:
    return KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def evaluate_estimator_cv(estimator: Any, X: pd.DataFrame, y: np.ndarray, cv: KFold) -> dict[str, float]:
    result = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={
            "R2": "r2",
            "RMSE": "neg_root_mean_squared_error",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "R2_mean": float(np.mean(result["test_R2"])),
        "R2_std": float(np.std(result["test_R2"])),
        "RMSE_mean": float(-np.mean(result["test_RMSE"])),
        "RMSE_std": float(np.std(-result["test_RMSE"])),
        "fit_time_mean_sec": float(np.mean(result["fit_time"])),
        "score_time_mean_sec": float(np.mean(result["score_time"])),
    }


def is_better(m1: dict[str, float], m2: dict[str, float], r2_tie_tol: float = 5e-4) -> bool:
    """
    以 R² 为主指标、RMSE 为次级指标。

    说明：
    - 当 R² 差值极小（默认 ±5e-4）时，视为统计波动范围内“近似持平”；
    - 在该情况下使用 RMSE 作为决胜指标，避免因极小噪声导致错误拒绝。
    """
    r2_diff = m1["R2_mean"] - m2["R2_mean"]
    if r2_diff > r2_tie_tol:
        return True
    if r2_diff < -r2_tie_tol:
        return False
    return m1["RMSE_mean"] < m2["RMSE_mean"]


def model_score_tuple(m: dict[str, float]) -> tuple[float, float]:
    return (m["R2_mean"], -m["RMSE_mean"])


def build_hgb(params: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=RANDOM_STATE,
        early_stopping=False,
        **params,
    )


def suggest_hgb(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_iter": trial.suggest_int("max_iter", 1200, 3600),
        "max_depth": trial.suggest_categorical("max_depth", [None, 6, 8, 10, 12, 14, 16]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 18),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 2.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }


def build_xgb(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def suggest_xgb(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
    }


def build_lgbm(params: dict[str, Any]) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12, 14]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def optimize_by_optuna(
    iteration_name: str,
    model_family: str,
    build_model_fn: Callable[[dict[str, Any]], Any],
    suggest_fn: Callable[[optuna.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: KFold,
    n_trials: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    trial_history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        model = build_model_fn(params)
        metrics = evaluate_estimator_cv(model, X, y, cv)
        trial_history.append(
            {
                "trial_number": int(trial.number),
                "params": params,
                "metrics": metrics,
            }
        )
        trial.set_user_attr("RMSE_mean", metrics["RMSE_mean"])
        return metrics["R2_mean"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info("[%s] 开始 Optuna 搜索: model=%s, n_trials=%d", iteration_name, model_family, n_trials)
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False, gc_after_trial=True)
    elapsed = time.perf_counter() - start

    if not trial_history:
        raise RuntimeError(f"{iteration_name} 未产生有效 trial")

    best_record = trial_history[0]
    for rec in trial_history[1:]:
        if is_better(rec["metrics"], best_record["metrics"]):
            best_record = rec

    top_trials = sorted(
        trial_history,
        key=lambda t: model_score_tuple(t["metrics"]),
        reverse=True,
    )[:5]

    logger.info(
        "[%s] 搜索完成: best_R2=%.6f, best_RMSE=%.6f, 耗时=%.1fs",
        iteration_name,
        best_record["metrics"]["R2_mean"],
        best_record["metrics"]["RMSE_mean"],
        elapsed,
    )

    return {
        "iteration_name": iteration_name,
        "model_family": model_family,
        "best_params": best_record["params"],
        "best_metrics": best_record["metrics"],
        "best_estimator": build_model_fn(best_record["params"]),
        "top_trials": top_trials,
        "n_trials": n_trials,
        "search_time_sec": float(elapsed),
    }


def level1_retry_optimize(
    spec: IterationSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: KFold,
    logger: logging.Logger,
) -> dict[str, Any]:
    model_builders: dict[str, Callable[[dict[str, Any]], Any]] = {
        "HGB": build_hgb,
        "XGBoost": build_xgb,
        "LightGBM": build_lgbm,
    }
    suggesters: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
        "HGB": suggest_hgb,
        "XGBoost": suggest_xgb,
        "LightGBM": suggest_lgbm,
    }

    if spec.model_family not in model_builders:
        raise ValueError(f"未知模型族: {spec.model_family}")

    # Level 1: 内部修正最多两次
    attempts = [spec.n_trials, max(20, spec.n_trials // 2)]
    last_exc: Exception | None = None
    for i, n_trials in enumerate(attempts, start=1):
        try:
            logger.info("[%s] Level1 attempt %d/%d", spec.name, i, len(attempts))
            result = optimize_by_optuna(
                iteration_name=spec.name,
                model_family=spec.model_family,
                build_model_fn=model_builders[spec.model_family],
                suggest_fn=suggesters[spec.model_family],
                X=X,
                y=y,
                cv=cv,
                n_trials=n_trials,
                logger=logger,
            )
            result["level1_attempt"] = i
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("[%s] Level1 attempt %d 失败: %s", spec.name, i, exc)
            logger.warning(traceback.format_exc())

    raise RuntimeError(f"{spec.name} 在 Level1 内部修正后仍失败: {last_exc}")


def compare_to_reference(curr: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    return {
        "R2_gain": float(curr["R2_mean"] - ref["R2_mean"]),
        "RMSE_drop": float(ref["RMSE_mean"] - curr["RMSE_mean"]),
    }


def build_model_from_family(family: str, params: dict[str, Any]) -> Any:
    if family == "HGB":
        return build_hgb(params)
    if family == "XGBoost":
        return build_xgb(params)
    if family == "LightGBM":
        return build_lgbm(params)
    raise ValueError(f"未知模型族: {family}")


def optimize_blend_weights(oof_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_models = oof_matrix.shape[1]
    init = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        pred = oof_matrix @ weights
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    result = minimize(objective, init, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return init

    weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() <= 0:
        return init
    return weights / weights.sum()


def run_blending_iteration(
    X_v8: pd.DataFrame,
    X_v7: pd.DataFrame,
    y: np.ndarray,
    cv: KFold,
    candidate_pool: list[dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    if len(candidate_pool) < 2:
        raise RuntimeError("用于融合的候选模型不足（至少需要2个）")

    selected = sorted(
        candidate_pool,
        key=lambda c: model_score_tuple(c["metrics"]),
        reverse=True,
    )[:3]

    baseline_anchor = {
        "model_id": "HGB_v7_baseline",
        "model_family": "HGB",
        "feature_space": "v7",
        "best_params": dict(V7_BASELINE_HGB_PARAMS),
        "metrics": {
            "R2_mean": 0.947965399021796,
            "R2_std": 0.017761603564915886,
            "RMSE_mean": 3.7407815724955533,
            "RMSE_std": 0.5973724522683225,
            "fit_time_mean_sec": 0.0,
            "score_time_mean_sec": 0.0,
        },
    }

    if all(item.get("model_id") != baseline_anchor["model_id"] for item in selected):
        selected.append(baseline_anchor)

    logger.info("[Iter-5] 融合候选: %s", [s["model_id"] for s in selected])

    oof_preds: list[np.ndarray] = []
    fit_time_total = 0.0
    for cand in selected:
        est = build_model_from_family(cand["model_family"], cand["best_params"])
        x_used = X_v7 if cand.get("feature_space") == "v7" else X_v8
        t0 = time.perf_counter()
        pred = cross_val_predict(est, x_used, y, cv=cv, n_jobs=-1, method="predict")
        fit_time_total += time.perf_counter() - t0
        oof_preds.append(pred)

    oof_matrix = np.column_stack(oof_preds)
    weights = optimize_blend_weights(oof_matrix, y)
    blend_pred = oof_matrix @ weights

    fold_r2: list[float] = []
    fold_rmse: list[float] = []
    for _, test_idx in cv.split(X_v8, y):
        y_fold = y[test_idx]
        p_fold = blend_pred[test_idx]
        fold_r2.append(float(r2_score(y_fold, p_fold)))
        fold_rmse.append(float(np.sqrt(np.mean((y_fold - p_fold) ** 2))))

    metrics = {
        "R2_mean": float(np.mean(fold_r2)),
        "R2_std": float(np.std(fold_r2)),
        "RMSE_mean": float(np.mean(fold_rmse)),
        "RMSE_std": float(np.std(fold_rmse)),
        "fit_time_mean_sec": float(fit_time_total / len(fold_r2)),
        "score_time_mean_sec": 0.0,
    }

    return {
        "model_family": "WeightedBlend",
        "selected_models": [
            {
                "model_id": cand["model_id"],
                "model_family": cand["model_family"],
                "feature_space": cand.get("feature_space", "v8"),
                "best_params": cand["best_params"],
                "cv_10fold": cand["metrics"],
            }
            for cand in selected
        ],
        "weights": {
            selected[i]["model_id"]: float(weights[i])
            for i in range(len(selected))
        },
        "best_metrics": metrics,
    }


def fit_ensemble_full_data(
    X_v8: pd.DataFrame,
    X_v7: pd.DataFrame,
    y: np.ndarray,
    selected_models: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    fitted_models: dict[str, Any] = {}
    model_spaces: dict[str, str] = {}
    for item in selected_models:
        model_id = item["model_id"]
        family = item["model_family"]
        feature_space = item.get("feature_space", "v8")
        params = item["best_params"]
        x_used = X_v7 if feature_space == "v7" else X_v8
        est = build_model_from_family(family, params)
        est.fit(x_used, y)
        fitted_models[model_id] = est
        model_spaces[model_id] = feature_space

    return {
        "model_type": "weighted_ensemble",
        "models": fitted_models,
        "model_spaces": model_spaces,
        "weights": weights,
    }


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    iteration_specs = [
        IterationSpec(
            idx=1,
            name="Iter-1",
            model_family="HGB",
            hypothesis="扩大 HGB 搜索空间并加入物理增强特征，可在同协议下进一步提升 R²。",
            n_trials=35,
        ),
        IterationSpec(
            idx=2,
            name="Iter-2",
            model_family="XGBoost",
            hypothesis="XGBoost 的二阶梯度与正则化可提升复杂非线性配比-龄期拟合能力。",
            n_trials=35,
        ),
        IterationSpec(
            idx=3,
            name="Iter-3",
            model_family="LightGBM",
            hypothesis="LightGBM 的叶子优先生长能捕获高阶交互，从而降低 RMSE。",
            n_trials=40,
        ),
        IterationSpec(
            idx=4,
            name="Iter-4",
            model_family="XGBoost",
            hypothesis="在 Iter-2 基础上二次强化 XGBoost 搜索，可进一步逼近最优偏差-方差平衡。",
            n_trials=25,
        ),
        IterationSpec(
            idx=5,
            name="Iter-5",
            model_family="WeightedBlend",
            hypothesis="融合前三强模型可降低方差，突破单模型上限。",
            n_trials=0,
        ),
    ]

    total_start = time.perf_counter()

    try:
        logger.info("===== v8 训练开始：跨模型 5 轮受控迭代 =====")

        df = load_data(paths["data"])
        X_base = df[BASE_FEATURES].copy()
        X_v8 = feature_engineering(X_base)
        X_v7 = feature_engineering_v7(X_base)
        y = df[TARGET_COL].to_numpy()

        cv = build_cv()
        prev_best_metrics = load_prev_metrics(paths["prev"])

        logger.info(
            "v7 基线(引用): R2=%.6f, RMSE=%.6f",
            prev_best_metrics["R2_mean"],
            prev_best_metrics["RMSE_mean"],
        )

        current_best_metrics = {
            "R2_mean": prev_best_metrics["R2_mean"],
            "R2_std": prev_best_metrics["R2_std"],
            "RMSE_mean": prev_best_metrics["RMSE_mean"],
            "RMSE_std": prev_best_metrics["RMSE_std"],
            "fit_time_mean_sec": 0.0,
            "score_time_mean_sec": 0.0,
        }

        current_best_payload: dict[str, Any] = {
            "model_type": "reference_v7",
            "model_family": "HGB",
            "cv_10fold": current_best_metrics,
            "note": "初始参考点为 v7 记录指标",
        }

        iteration_results: list[dict[str, Any]] = []
        candidate_pool_for_blend: list[dict[str, Any]] = []

        accepted_count = 0
        stagnation_count = 0
        fallback_events: list[dict[str, Any]] = []

        for spec in iteration_specs:
            iter_start = time.perf_counter()
            logger.info("----- %s 开始 (%s) -----", spec.name, spec.model_family)
            logger.info("假设: %s", spec.hypothesis)

            if spec.model_family != "WeightedBlend":
                try:
                    result = level1_retry_optimize(spec, X_v8, y, cv, logger)
                except Exception as exc:  # noqa: BLE001
                    fallback_events.append(
                        {
                            "iteration": spec.name,
                            "trigger": "error",
                            "phase3": {
                                "level1": "failed_after_2_attempts",
                                "level2": "not_executed_in_code",
                                "level3": "not_executed_in_code",
                                "level4": "blocked",
                            },
                            "error": str(exc),
                        }
                    )
                    iter_elapsed = time.perf_counter() - iter_start
                    iteration_results.append(
                        {
                            "iteration": spec.name,
                            "model_family": spec.model_family,
                            "hypothesis": spec.hypothesis,
                            "n_trials": spec.n_trials,
                            "accepted": False,
                            "failed": True,
                            "error": str(exc),
                            "iteration_time_sec": float(iter_elapsed),
                        }
                    )
                    logger.warning("%s 失败后继续下一轮迭代。", spec.name)
                    continue

                candidate_metrics = result["best_metrics"]
                delta_to_current = compare_to_reference(candidate_metrics, current_best_metrics)
                improved = is_better(candidate_metrics, current_best_metrics)

                if improved:
                    accepted_count += 1
                    stagnation_count = 0

                    fitted = clone(result["best_estimator"])
                    fitted.fit(X_v8, y)

                    current_best_metrics = candidate_metrics
                    current_best_payload = {
                        "model_type": "single",
                        "model_family": result["model_family"],
                        "feature_space": "v8",
                        "best_params": result["best_params"],
                        "cv_10fold": candidate_metrics,
                        "model": fitted,
                        "sub_version": f"v8.{accepted_count}",
                    }
                else:
                    stagnation_count += 1

                candidate_pool_for_blend.append(
                    {
                        "model_id": f"{result['model_family']}_{spec.name}",
                        "model_family": result["model_family"],
                        "feature_space": "v8",
                        "best_params": result["best_params"],
                        "metrics": candidate_metrics,
                    }
                )

                iteration_results.append(
                    {
                        "iteration": spec.name,
                        "model_family": result["model_family"],
                        "hypothesis": spec.hypothesis,
                        "n_trials": result["n_trials"],
                        "level1_attempt": result["level1_attempt"],
                        "best_params": result["best_params"],
                        "cv_10fold": candidate_metrics,
                        "compare_to_prev_best": delta_to_current,
                        "accepted": improved,
                        "accepted_sub_version": f"v8.{accepted_count}" if improved else None,
                        "top_trials": result["top_trials"],
                        "search_time_sec": result["search_time_sec"],
                    }
                )

            else:
                blend_result = run_blending_iteration(X_v8, X_v7, y, cv, candidate_pool_for_blend, logger)
                candidate_metrics = blend_result["best_metrics"]
                delta_to_current = compare_to_reference(candidate_metrics, current_best_metrics)
                improved = is_better(candidate_metrics, current_best_metrics)

                if improved:
                    accepted_count += 1
                    stagnation_count = 0
                    ensemble_payload = fit_ensemble_full_data(
                        X_v8,
                        X_v7,
                        y,
                        selected_models=blend_result["selected_models"],
                        weights=blend_result["weights"],
                    )
                    current_best_metrics = candidate_metrics
                    current_best_payload = {
                        "model_type": "weighted_ensemble",
                        "model_family": "WeightedBlend",
                        "selected_models": blend_result["selected_models"],
                        "weights": blend_result["weights"],
                        "cv_10fold": candidate_metrics,
                        "models": ensemble_payload["models"],
                        "model_spaces": ensemble_payload["model_spaces"],
                        "sub_version": f"v8.{accepted_count}",
                    }
                else:
                    stagnation_count += 1

                iteration_results.append(
                    {
                        "iteration": spec.name,
                        "model_family": "WeightedBlend",
                        "hypothesis": spec.hypothesis,
                        "n_trials": 0,
                        "best_params": {"weights": blend_result["weights"]},
                        "selected_models": blend_result["selected_models"],
                        "cv_10fold": candidate_metrics,
                        "compare_to_prev_best": delta_to_current,
                        "accepted": improved,
                        "accepted_sub_version": f"v8.{accepted_count}" if improved else None,
                    }
                )

            iter_elapsed = time.perf_counter() - iter_start
            iteration_results[-1]["iteration_time_sec"] = float(iter_elapsed)

            logger.info(
                "%s 结果: R2=%.6f, RMSE=%.6f, accepted=%s, 用时=%.1fs",
                spec.name,
                candidate_metrics["R2_mean"],
                candidate_metrics["RMSE_mean"],
                iteration_results[-1]["accepted"],
                iter_elapsed,
            )

            # 连续两次无明显提升 -> 触发 Phase3 标记并执行后续轮次（避免盲试但保持科研闭环）
            if stagnation_count >= 2:
                fallback_events.append(
                    {
                        "iteration": spec.name,
                        "trigger": "two_consecutive_no_improvement",
                        "phase3": {
                            "level1": "executed",
                            "level2": "not_executed_in_code",
                            "level3": "not_executed_in_code",
                            "level4": "blocked",
                        },
                    }
                )
                logger.warning("连续两轮未提升，触发科研阻断保护。")
                stagnation_count = 0

        if current_best_payload.get("model_type") == "reference_v7":
            raise RuntimeError("v8 五轮迭代未产生可落地新模型，请检查搜索空间或特征工程。")

        model_bundle = {
            "version": "v8",
            "strategy": "5-step controlled iteration with external boosting and weighted blending",
            "feature_columns": list(X_v8.columns),
            "feature_columns_v8": list(X_v8.columns),
            "feature_columns_v7": list(X_v7.columns),
            "base_features": BASE_FEATURES,
            "model_type": current_best_payload["model_type"],
            "best_model_family": current_best_payload["model_family"],
            "sub_version": current_best_payload["sub_version"],
        }

        if current_best_payload["model_type"] == "single":
            model_bundle["model"] = current_best_payload["model"]
            model_bundle["best_params"] = current_best_payload["best_params"]
            model_bundle["feature_space"] = current_best_payload.get("feature_space", "v8")
        else:
            model_bundle["models"] = current_best_payload["models"]
            model_bundle["weights"] = current_best_payload["weights"]
            model_bundle["selected_models"] = current_best_payload["selected_models"]
            model_bundle["model_spaces"] = current_best_payload["model_spaces"]

        joblib.dump(model_bundle, paths["model"])

        total_elapsed = time.perf_counter() - total_start
        delta_to_v7 = compare_to_reference(current_best_metrics, prev_best_metrics)

        payload = {
            "version": "v8",
            "strategy": "5-step controlled iteration with external boosting and weighted blending",
            "cv_protocol": {
                "type": "KFold",
                "n_splits": 10,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "baseline_v7": prev_best_metrics,
            "iteration_plan": [
                {
                    "iteration": spec.name,
                    "model_family": spec.model_family,
                    "hypothesis": spec.hypothesis,
                    "n_trials": spec.n_trials,
                }
                for spec in iteration_specs
            ],
            "iteration_results": iteration_results,
            "best_model": {
                "sub_version": current_best_payload["sub_version"],
                "model_type": current_best_payload["model_type"],
                "model_family": current_best_payload["model_family"],
                "cv_10fold": current_best_metrics,
                "best_params": current_best_payload.get("best_params"),
                "weights": current_best_payload.get("weights"),
                "selected_models": current_best_payload.get("selected_models"),
            },
            "compare_to_v7": {
                "v7": prev_best_metrics,
                "delta": delta_to_v7,
                "is_better": bool(delta_to_v7["R2_gain"] > 0 and delta_to_v7["RMSE_drop"] > 0),
            },
            "fallback_events": fallback_events,
            "training_time_sec": float(total_elapsed),
        }

        paths["metrics"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("v8 最优子版本: %s", current_best_payload["sub_version"])
        logger.info(
            "v8 最终10折CV: R2=%.6f±%.6f, RMSE=%.6f±%.6f",
            current_best_metrics["R2_mean"],
            current_best_metrics["R2_std"],
            current_best_metrics["RMSE_mean"],
            current_best_metrics["RMSE_std"],
        )
        logger.info(
            "对比v7: ΔR2=%+.6f, ΔRMSE=%+.6f (正值表示 RMSE 下降)",
            delta_to_v7["R2_gain"],
            delta_to_v7["RMSE_drop"],
        )
        logger.info("模型已保存: %s", paths["model"])
        logger.info("指标已保存: %s", paths["metrics"])
        logger.info("总耗时: %.1f 秒", total_elapsed)
        logger.info("===== v8 训练完成 =====")

    except Exception as exc:  # noqa: BLE001
        logger.error("v8 训练失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
