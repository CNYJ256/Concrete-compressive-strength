"""
评估指标模块。

统一计算：R2 / RMSE / MAPE / MAE
并提供简单格式化输出函数。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算稳定版 MAPE，避免真实值为 0 导致除零。

    返回值为百分比（例如 6.78 表示 6.78%）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    统一计算回归任务常用指标。
    """
    r2 = float(r2_score(y_true, y_pred))
    # 兼容旧版 sklearn：有些版本的 mean_squared_error 不支持 squared 参数。
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(safe_mape(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE": mae,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    将指标格式化为适合日志/报告输出的一行文本。
    """
    return (
        f"R2={metrics['R2']:.4f}, "
        f"RMSE={metrics['RMSE']:.4f}, "
        f"MAPE={metrics['MAPE']:.2f}%, "
        f"MAE={metrics['MAE']:.4f}"
    )
