from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RAW_TO_STD_COLUMN_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_agg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_agg",
    "Age (day)": "age",
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


def get_logger() -> logging.Logger:
    logger = logging.getLogger("v8_predict")
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
        "data": root / "data" / "Concrete_Data.xls",
        "model": root / "v8" / "model.joblib",
        "output": root / "v8" / "predictions.csv",
    }


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if set(BASE_FEATURES).issubset(cols):
        out = df.copy()
    elif set(RAW_TO_STD_COLUMN_MAP).issubset(cols):
        out = df.rename(columns=RAW_TO_STD_COLUMN_MAP).copy()
    else:
        raise ValueError("输入列不匹配，请提供标准列或 UCI 原始列")

    out = out[BASE_FEATURES].copy()
    for c in BASE_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if out.isna().any(axis=1).any():
        raise ValueError("输入数据存在无法解析的缺失值")
    return out


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = X.copy()

    binder = out["cement"] + out["slag"] + out["fly_ash"]
    total_agg = out["coarse_agg"] + out["fine_agg"]
    age_log = np.log1p(np.maximum(out["age"], 0.0))

    # v7 机理特征
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

    # v8 物理增强特征
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


def build_demo_input(data_path: Path, n: int = 5) -> pd.DataFrame:
    df = pd.read_excel(data_path, engine="xlrd")
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    return df[BASE_FEATURES].head(n).copy()


def predict_with_bundle(bundle: dict, features_v8: pd.DataFrame, features_v7: pd.DataFrame | None = None) -> np.ndarray:
    model_type = bundle.get("model_type", "single")
    if model_type == "single":
        model = bundle["model"]
        feature_space = bundle.get("feature_space", "v8")
        if feature_space == "v7" and features_v7 is not None:
            return np.asarray(model.predict(features_v7), dtype=float)
        return np.asarray(model.predict(features_v8), dtype=float)

    if model_type == "weighted_ensemble":
        models: dict = bundle["models"]
        weights: dict = bundle["weights"]
        model_spaces: dict = bundle.get("model_spaces", {})

        pred = np.zeros(len(features_v8), dtype=float)
        for model_id, w in weights.items():
            model = models[model_id]
            feature_space = model_spaces.get(model_id, "v8")
            x_used = features_v7 if (feature_space == "v7" and features_v7 is not None) else features_v8
            pred += float(w) * np.asarray(model.predict(x_used), dtype=float)
        return pred

    raise ValueError(f"未知模型类型: {model_type}")


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()

    try:
        if not paths["model"].exists():
            raise FileNotFoundError(f"未找到模型文件: {paths['model']}，请先运行 train.py")

        bundle = joblib.load(paths["model"])
        expected_cols_v8 = bundle.get("feature_columns_v8", bundle["feature_columns"])
        expected_cols_v7 = bundle.get("feature_columns_v7")

        input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else paths["output"]

        if input_path is not None:
            logger.info("读取外部输入: %s", input_path)
            raw = pd.read_csv(input_path)
            base = normalize_input_columns(raw)
        else:
            logger.info("未提供输入文件，默认使用数据集前5行")
            base = build_demo_input(paths["data"], n=5)

        fe_v8 = feature_engineering(base)
        fe_v8 = fe_v8.reindex(columns=expected_cols_v8)

        fe_v7 = None
        if expected_cols_v7 is not None:
            fe_v7 = feature_engineering_v7(base)
            fe_v7 = fe_v7.reindex(columns=expected_cols_v7)

        pred = predict_with_bundle(bundle, fe_v8, fe_v7)
        out = base.copy()
        out["predicted_strength_mpa"] = pred

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info("预测完成，共 %d 条", len(out))
        logger.info("输出文件: %s", output_path)

    except Exception as exc:  # noqa: BLE001
        logger.error("v8 推理失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
