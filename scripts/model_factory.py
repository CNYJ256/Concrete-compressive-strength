"""
模型构建模块。

说明：
- 提供基线模型（AdaBoost / ANN / SVM）
- 提供优化模型（改进版 AdaBoost）
- 统一封装，减少主流程脚本重复代码
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from config import (
    ADABOOST_PARAMS,
    ANN_PARAMS,
    OPT_ADABOOST_PARAMS,
    OPT_TREE_PARAMS,
    SVM_PARAMS,
    TREE_PARAMS,
)


def _build_adaboost_with_tree(tree_params: dict, adaboost_params: dict) -> AdaBoostRegressor:
    """
    构造 AdaBoost 回归器。

    兼容性处理：
    - 新版 sklearn 使用 estimator
    - 老版 sklearn 使用 base_estimator
    """
    tree = DecisionTreeRegressor(**tree_params)

    try:
        model = AdaBoostRegressor(estimator=tree, **adaboost_params)
    except TypeError:
        model = AdaBoostRegressor(base_estimator=tree, **adaboost_params)

    return model


def build_adaboost_model() -> AdaBoostRegressor:
    """
    论文基线 AdaBoost 模型（接近文中参数）。
    """
    return _build_adaboost_with_tree(TREE_PARAMS, ADABOOST_PARAMS)


def build_ann_model() -> Pipeline:
    """
    ANN 对比模型。

    注意：
    - 神经网络对特征尺度敏感，因此先做 StandardScaler。
    """
    ann = MLPRegressor(**ANN_PARAMS)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ann),
    ])


def build_svm_model() -> Pipeline:
    """
    SVM 对比模型。

    注意：
    - SVM 同样对特征尺度敏感，使用标准化流水线。
    """
    svm = SVR(**SVM_PARAMS)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", svm),
    ])


def build_baseline_models() -> Dict[str, object]:
    """
    返回 Step 3 复现所需的模型字典。
    """
    return {
        "AdaBoost": build_adaboost_model(),
        "ANN": build_ann_model(),
        "SVM": build_svm_model(),
    }


def build_optimized_adaboost_model() -> AdaBoostRegressor:
    """
    返回 Step 4 创新模型使用的 AdaBoost 基础结构。
    """
    return _build_adaboost_with_tree(OPT_TREE_PARAMS, OPT_ADABOOST_PARAMS)
