from __future__ import annotations

"""ACDCB 推理脚本：基于已训练模型包执行强度预测。

支持两种输入格式：
1) 标准列名（cement/slag/.../age）；
2) UCI 原始英文列名（自动映射到标准列）。

默认行为：
未传入输入文件时，直接使用 `data/Concrete_Data.xls` 的原始数据（全量样本）进行推理。
"""

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

from ACDCB.core import (  # noqa: E402
    BASE_FEATURES,
    RAW_TO_STD_COLUMN_MAP,
    feature_engineering,
    feature_engineering_anchor,
)


def get_logger() -> logging.Logger:
    """构建并返回推理日志器（幂等）。"""
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
    """统一解析推理阶段常用路径。"""
    root = Path(__file__).resolve().parents[1]
    return {
        "data": root / "data" / "Concrete_Data.xls",
        "model": root / "ACDCB" / "model.joblib",
        "output": root / "ACDCB" / "predictions.csv",
    }


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将输入数据标准化为训练所需字段与类型。

    规则：
    - 若已是标准列，则直接使用；
    - 若为 UCI 原始列，则映射后使用；
    - 否则抛出错误，避免静默列错位。
    """
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
    """构造默认输入（原始数据集全量样本）。"""
    df = pd.read_excel(data_path, engine="xlrd")
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)
    return df[BASE_FEATURES].copy()


def predict_with_bundle(bundle: dict, base: pd.DataFrame) -> np.ndarray:
    """按模型包元数据执行推理，并自动匹配融合策略。"""
    model_type = bundle.get("model_type")
    if model_type != "age_aware_weighted_ensemble":
        raise ValueError(f"当前 ACDCB 仅支持 age_aware_weighted_ensemble，收到: {model_type}")

    models: dict = bundle["models"]
    model_spaces: dict = bundle["model_spaces"]

    # 构造主特征空间与锚点特征空间（兼容新旧模型包字段）。
    primary_cols = bundle.get("feature_columns_primary", bundle.get("feature_columns_v8"))
    anchor_cols = bundle.get("feature_columns_anchor", bundle.get("feature_columns_v7"))
    if primary_cols is None or anchor_cols is None:
        raise ValueError("模型包缺少特征列定义（primary/anchor）")

    fe_primary = feature_engineering(base).reindex(columns=primary_cols)
    fe_anchor = feature_engineering_anchor(base).reindex(columns=anchor_cols)

    # 先分别得到每个子模型的预测结果，再进行加权融合。
    per_model_pred: dict[str, np.ndarray] = {}
    for model_id, model in models.items():
        fs = model_spaces.get(model_id, "primary")
        x_used = fe_anchor if fs in {"anchor", "v7"} else fe_primary
        per_model_pred[model_id] = np.asarray(model.predict(x_used), dtype=float)

    pred = np.zeros(len(base), dtype=float)
    strategy = bundle.get("selected_strategy", "global")

    # 全局权重：所有样本共享同一组权重。
    if strategy == "global":
        weights = bundle["weights_global"]
        for model_id, w in weights.items():
            pred += float(w) * per_model_pred[model_id]
        return pred

    # 分段权重：按龄期分别应用 early / late 两套权重。
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
    """ACDCB 推理主入口。"""
    logger = get_logger()
    paths = resolve_paths()

    try:
        # 1) 校验模型文件并加载模型包。
        if not paths["model"].exists():
            raise FileNotFoundError(f"未找到模型文件: {paths['model']}，请先运行 train.py")

        bundle = joblib.load(paths["model"])

        # 2) 解析输入/输出路径：支持命令行覆写。
        input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else paths["output"]

        # 3) 读取输入数据（外部 CSV 或默认原始数据）。
        if input_path is not None:
            logger.info("读取外部输入: %s", input_path)
            raw = pd.read_csv(input_path)
            base = normalize_input_columns(raw)
        else:
            logger.info("未提供输入文件，默认使用原始数据集全量样本: %s", paths["data"])
            base = build_default_input(paths["data"])

        # 4) 执行融合预测并输出结果 CSV。
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
