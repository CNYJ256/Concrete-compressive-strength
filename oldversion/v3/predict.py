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
    logger = logging.getLogger("v1_predict")
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
        "model": root / "v1" / "model.joblib",
        "output": root / "v1" / "predictions.csv",
    }


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    if set(BASE_FEATURES).issubset(cols):
        out = df.copy()
    elif set(RAW_TO_STD_COLUMN_MAP).issubset(cols):
        out = df.rename(columns=RAW_TO_STD_COLUMN_MAP).copy()
    else:
        raise ValueError(
            "输入数据列不匹配。请提供标准列名(8特征)或原始 UCI 列名。"
        )

    out = out[BASE_FEATURES].copy()
    for c in BASE_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    na_rows = int(out.isna().any(axis=1).sum())
    if na_rows > 0:
        raise ValueError(f"输入存在无法解析的数值，共 {na_rows} 行")

    return out


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = X.copy()

    binder = out["cement"] + out["slag"] + out["fly_ash"]
    total_agg = out["coarse_agg"] + out["fine_agg"]

    out["binder"] = binder
    out["water_cement_ratio"] = out["water"] / (out["cement"] + eps)
    out["water_binder_ratio"] = out["water"] / (binder + eps)
    out["sp_binder_ratio"] = out["superplasticizer"] / (binder + eps)
    out["fine_ratio_in_agg"] = out["fine_agg"] / (total_agg + eps)
    out["age_log1p"] = np.log1p(out["age"])
    out["paste_index"] = (out["cement"] + out["slag"] + out["fly_ash"] + out["water"] + out["superplasticizer"]) / (total_agg + eps)

    return out


def build_demo_input(data_path: Path, n: int = 5) -> pd.DataFrame:
    df = pd.read_excel(data_path, engine="xlrd")
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    return df[BASE_FEATURES].head(n).copy()


def main() -> None:
    logger = get_logger()
    paths = resolve_paths()

    try:
        if not paths["model"].exists():
            raise FileNotFoundError(f"未找到模型文件: {paths['model']}，请先运行 train.py")

        bundle = joblib.load(paths["model"])
        model = bundle["model"]
        expected_cols = bundle["feature_columns"]

        input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else paths["output"]

        if input_path is not None:
            logger.info("读取外部输入文件: %s", input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            raw = pd.read_csv(input_path)
            base = normalize_input_columns(raw)
        else:
            logger.info("未提供输入文件，使用数据集前5行做示例预测")
            base = build_demo_input(paths["data"], n=5)

        fe = feature_engineering(base)
        fe = fe.reindex(columns=expected_cols)

        pred = model.predict(fe)

        out = base.copy()
        out["predicted_strength_mpa"] = pred
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info("预测完成，共 %d 条", len(out))
        logger.info("输出文件: %s", output_path)

    except Exception as exc:
        logger.error("v1 推理失败: %s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
