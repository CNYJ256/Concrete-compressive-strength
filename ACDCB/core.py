from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# v9 学术命名：龄期条件化双空间约束融合。
METHOD_NAME_ZH = "龄期条件化双空间约束融合（Age-Conditioned Dual-Space Constrained Blending, ACDCB）"
METHOD_NAME_EN = "Age-Conditioned Dual-Space Constrained Blending (ACDCB)"

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
AGE_SPLIT_DAY = 28

# 由 v9 训练流程内置的固定候选参数（不依赖 oldversion 文件）。
BASE_MODEL_PARAMS = {
    "XGBoost": {
        "n_estimators": 1482,
        "learning_rate": 0.044604576412178736,
        "max_depth": 4,
        "min_child_weight": 7.394621020486849,
        "subsample": 0.6642601673434315,
        "colsample_bytree": 0.6247108136598432,
        "gamma": 2.4343952947126315,
        "reg_alpha": 0.49332537524084563,
        "reg_lambda": 3.0210800926855668,
    },
    "LightGBM": {
        "n_estimators": 2127,
        "learning_rate": 0.031969519083032506,
        "num_leaves": 32,
        "max_depth": 4,
        "min_child_samples": 8,
        "subsample": 0.838331183259241,
        "colsample_bytree": 0.5905247550993065,
        "reg_alpha": 0.7113890443035059,
        "reg_lambda": 0.0002216693328024357,
        "min_split_gain": 0.006463967504229284,
    },
    "HGB": {
        "learning_rate": 0.05678535332386602,
        "max_iter": 1809,
        "max_depth": 12,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 14,
        "l2_regularization": 4.580924468859991e-05,
        "max_bins": 213,
    },
}

ANCHOR_MODEL_PARAMS = {
    "learning_rate": 0.028,
    "max_iter": 2400,
    "max_depth": None,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 6,
    "l2_regularization": 0.001,
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

    # 机理特征
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

    # 增强特征
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


def feature_engineering_anchor(X: pd.DataFrame) -> pd.DataFrame:
    """锚点特征空间：保持紧凑机理特征，增强稳健性。"""
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


def build_hgb(params: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=RANDOM_STATE,
        early_stopping=False,
        **params,
    )


def build_xgb(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def build_lgbm(params: dict[str, Any]) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def is_better(m1: dict[str, float], m2: dict[str, float], r2_tie_tol: float = 5e-4) -> bool:
    """以 R² 为主、RMSE 为次的双指标比较。"""
    r2_diff = m1["R2_mean"] - m2["R2_mean"]
    if r2_diff > r2_tie_tol:
        return True
    if r2_diff < -r2_tie_tol:
        return False
    return m1["RMSE_mean"] < m2["RMSE_mean"]
