from __future__ import annotations

"""ACDCB 推理脚本（重构版）。"""

import logging
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import (  # noqa: E402
    BASE_FEATURES,
    RAW_TO_STD_COLUMN_MAP,
    feature_engineering,
    feature_engineering_anchor,
)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("acdcb_predict")
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
    prediction_dir = ROOT / "results" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data": ROOT / "data" / "Concrete_Data.xls",
        "model": ROOT / "results" / "models" / "acdcb_model.joblib",
        "output": prediction_dir / "acdcb_predictions.csv",
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


def build_default_input(data_path: Path) -> pd.DataFrame:
    df = pd.read_excel(data_path, engine="xlrd")
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    return df[BASE_FEATURES].copy()


def predict_with_bundle(bundle: dict, base: pd.DataFrame) -> np.ndarray:
    model_type = bundle.get("model_type")
    if model_type != "age_aware_weighted_ensemble":
        raise ValueError(f"当前 ACDCB 仅支持 age_aware_weighted_ensemble，收到: {model_type}")

    models: dict = bundle["models"]
    model_spaces: dict = bundle["model_spaces"]

    primary_cols = bundle.get("feature_columns_primary")
    anchor_cols = bundle.get("feature_columns_anchor")
    if primary_cols is None or anchor_cols is None:
        raise ValueError("模型包缺少特征列定义（primary/anchor）")

    fe_primary = feature_engineering(base).reindex(columns=primary_cols)
    fe_anchor = feature_engineering_anchor(base).reindex(columns=anchor_cols)

    per_model_pred: dict[str, np.ndarray] = {}
    for model_id, model in models.items():
        fs = model_spaces.get(model_id, "primary")
        x_used = fe_anchor if fs == "anchor" else fe_primary
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
            raise FileNotFoundError(f"未找到模型文件: {paths['model']}，请先运行 train_acdcb.py")

        bundle = joblib.load(paths["model"])

        input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else paths["output"]

        if input_path is not None:
            logger.info("读取外部输入: %s", input_path)
            raw = pd.read_csv(input_path)
            base = normalize_input_columns(raw)
        else:
            logger.info("未提供输入文件，默认使用原始数据集全量样本: %s", paths["data"])
            base = build_default_input(paths["data"])

        pred = predict_with_bundle(bundle, base)

        out = base.copy()
        out["predicted_strength_mpa"] = pred
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info("预测完成，共 %d 条", len(out))
        logger.info("输出文件: %s", output_path)

    except Exception as exc:  # noqa: BLE001
        logger.error("ACDCB 推理失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
