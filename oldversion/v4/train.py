from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_validate


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


def get_logger() -> logging.Logger:
    logger = logging.getLogger("v2_train")
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
        "prev": root / "v1" / "metrics.json",
        "model": root / "v2" / "model.joblib",
        "metrics": root / "v2" / "metrics.json",
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
    age_log = np.log1p(out["age"])

    out["binder"] = binder
    out["water_cement_ratio"] = out["water"] / (out["cement"] + eps)
    out["water_binder_ratio"] = out["water"] / (binder + eps)
    out["sp_binder_ratio"] = out["superplasticizer"] / (binder + eps)
    out["scm_ratio"] = (out["slag"] + out["fly_ash"]) / (binder + eps)
    out["fine_ratio_in_agg"] = out["fine_agg"] / (total_agg + eps)
    out["age_log1p"] = age_log
    out["age_sqrt"] = np.sqrt(np.maximum(out["age"], 0.0))
    out["binder_age_interaction"] = binder * age_log
    out["wb_age_interaction"] = out["water_binder_ratio"] * age_log
    out["paste_index"] = (out["cement"] + out["slag"] + out["fly_ash"] + out["water"] + out["superplasticizer"]) / (total_agg + eps)

    return out


def build_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_cv(model, X, y) -> dict[str, float]:
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "R2": "r2",
        "RMSE": "neg_root_mean_squared_error",
    }
    result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
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


def load_prev_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"找不到上一版本指标文件: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cv = data["cv_10fold"]
    return {
        "R2_mean": float(cv["R2_mean"]),
        "RMSE_mean": float(cv["RMSE_mean"]),
    }


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()

    try:
        logger.info("===== v2 训练开始：增强交互特征 + ExtraTrees =====")

        df = load_data(paths["data"])
        X = feature_engineering(df[BASE_FEATURES])
        y = df[TARGET_COL].to_numpy()

        model = build_model()
        cv_metrics = evaluate_cv(model, X, y)

        logger.info(
            "v2 10折CV: R2=%.4f±%.4f, RMSE=%.4f±%.4f",
            cv_metrics["R2_mean"],
            cv_metrics["R2_std"],
            cv_metrics["RMSE_mean"],
            cv_metrics["RMSE_std"],
        )

        model.fit(X, y)

        model_bundle = {
            "model": model,
            "feature_columns": list(X.columns),
            "base_features": BASE_FEATURES,
            "version": "v2",
            "strategy": "Enhanced interaction features + ExtraTrees",
        }
        joblib.dump(model_bundle, paths["model"])

        prev = load_prev_metrics(paths["prev"])
        delta = {
            "R2_gain": cv_metrics["R2_mean"] - prev["R2_mean"],
            "RMSE_drop": prev["RMSE_mean"] - cv_metrics["RMSE_mean"],
        }

        payload = {
            "version": "v2",
            "strategy": "Enhanced interaction features + ExtraTrees",
            "cv_10fold": cv_metrics,
            "compare_to_v1": {
                "v1": prev,
                "delta": delta,
                "is_better": bool(delta["R2_gain"] > 0 and delta["RMSE_drop"] > 0),
            },
        }

        paths["metrics"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "对比v1: ΔR2=%+.4f, ΔRMSE=%+.4f (正值代表RMSE下降)",
            delta["R2_gain"],
            delta["RMSE_drop"],
        )
        logger.info("模型已保存: %s", paths["model"])
        logger.info("指标已保存: %s", paths["metrics"])
        logger.info("===== v2 训练完成 =====")

    except Exception as exc:
        logger.error("v2 训练失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
