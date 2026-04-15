"""
数据加载与清洗模块。

功能：
1) 读取 .xls 数据；
2) 标准化列名；
3) 进行基础数据质量检查；
4) 输出建模所需的 X / y。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import DATA_PATH, FEATURE_COLUMNS, RAW_TO_STD_COLUMN_MAP, TARGET_COLUMN
from logger_utils import get_logger

logger = get_logger(__name__)


def load_concrete_data(file_path: Path | None = None) -> pd.DataFrame:
    """
    读取并清洗混凝土抗压强度数据。

    参数：
    - file_path: 可选路径，默认读取 config.py 中 DATA_PATH。

    返回：
    - 清洗后的 DataFrame（列名已标准化）。
    """
    data_path = Path(file_path) if file_path is not None else DATA_PATH
    logger.info("开始读取数据文件: %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    try:
        # 读取 .xls 需要 xlrd。
        df = pd.read_excel(data_path, engine="xlrd")
    except ImportError as exc:
        raise ImportError(
            "读取 .xls 失败：缺少 xlrd。请先执行 `python -m pip install xlrd`。"
        ) from exc

    logger.info("原始数据维度: %s", df.shape)

    # 检查列名是否完整
    raw_cols = set(df.columns)
    required_raw_cols = set(RAW_TO_STD_COLUMN_MAP.keys())
    missing_cols = required_raw_cols - raw_cols
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {sorted(missing_cols)}")

    # 标准化列名
    df = df.rename(columns=RAW_TO_STD_COLUMN_MAP)

    # 只保留我们关心的列，避免隐藏列干扰
    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[keep_cols].copy()

    # 转数值，若出现不可解析值会变为 NaN
    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除缺失行并记录日志
    na_count = int(df.isna().any(axis=1).sum())
    if na_count > 0:
        logger.warning("检测到 %d 行含缺失值，已自动删除。", na_count)
        df = df.dropna(axis=0).reset_index(drop=True)

    # 仅统计重复行，不删除。
    # 原始数据集论文通常按 1030 条样本使用，直接去重会改变实验设定。
    duplicated = int(df.duplicated().sum())
    if duplicated > 0:
        logger.warning("检测到 %d 行完全重复样本；为与原始论文设定一致，本流程保留这些样本。", duplicated)

    # 数据类型安全检查
    non_numeric_cols = [c for c in keep_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric_cols:
        raise TypeError(f"存在非数值列，无法建模: {non_numeric_cols}")

    logger.info("清洗后数据维度: %s", df.shape)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    将 DataFrame 拆分为特征矩阵 X 与目标向量 y。
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # 最基本的维度检查，防止后续训练时报错
    if len(X) != len(y):
        raise ValueError(f"X 与 y 样本数不一致: X={len(X)}, y={len(y)}")

    return X, y
