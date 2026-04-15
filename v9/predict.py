from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v8.train import BASE_FEATURES, feature_engineering, feature_engineering_v7


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


def get_logger() -> logging.Logger:
    logger = logging.getLogger("v9_predict")
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
        "model": root / "v9" / "model.joblib",
        "output": root / "v9" / "predictions.csv",
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


def build_demo_input(data_path: Path, n: int = 5) -> pd.DataFrame:
    df = pd.read_excel(data_path, engine="xlrd")
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    return df[BASE_FEATURES].head(n).copy()


def predict_with_bundle(bundle: dict, base: pd.DataFrame) -> np.ndarray:
    model_type = bundle.get("model_type")
    if model_type != "age_aware_weighted_ensemble":
        raise ValueError(f"当前 v9 仅支持 age_aware_weighted_ensemble，收到: {model_type}")

    models: dict = bundle["models"]
    model_spaces: dict = bundle["model_spaces"]

    fe_v8 = feature_engineering(base).reindex(columns=bundle["feature_columns_v8"])
    fe_v7 = feature_engineering_v7(base).reindex(columns=bundle["feature_columns_v7"])

    per_model_pred: dict[str, np.ndarray] = {}
    for model_id, model in models.items():
        fs = model_spaces.get(model_id, "v8")
        x_used = fe_v7 if fs == "v7" else fe_v8
        per_model_pred[model_id] = np.asarray(model.predict(x_used), dtype=float)

    pred = np.zeros(len(base), dtype=float)
    strategy = bundle.get("selected_strategy", "global")

    if strategy == "global":
        weights = bundle["weights_global"]
        for model_id, w in weights.items():
            pred += float(w) * per_model_pred[model_id]
        return pred

    if strategy == "age_piecewise":
        cfg = bundle["weights_age_piecewise"]
        split_day = float(cfg["age_split_day"])
        early_weights = cfg["early"]
        late_weights = cfg["late"]

        age = base["age"].to_numpy()
        early_mask = age <= split_day
        late_mask = ~early_mask

        for model_id, w in early_weights.items():
            pred[early_mask] += float(w) * per_model_pred[model_id][early_mask]
        for model_id, w in late_weights.items():
            pred[late_mask] += float(w) * per_model_pred[model_id][late_mask]
        return pred

    raise ValueError(f"未知策略: {strategy}")


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()

    try:
        if not paths["model"].exists():
            raise FileNotFoundError(f"未找到模型文件: {paths['model']}，请先运行 train.py")

        bundle = joblib.load(paths["model"])

        input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else paths["output"]

        if input_path is not None:
            logger.info("读取外部输入: %s", input_path)
            raw = pd.read_csv(input_path)
            base = normalize_input_columns(raw)
        else:
            logger.info("未提供输入文件，默认使用数据集前5行")
            base = build_demo_input(paths["data"], n=5)

        pred = predict_with_bundle(bundle, base)

        out = base.copy()
        out["predicted_strength_mpa"] = pred
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info("预测完成，共 %d 条", len(out))
        logger.info("输出文件: %s", output_path)

    except Exception as exc:  # noqa: BLE001
        logger.error("v9 推理失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
