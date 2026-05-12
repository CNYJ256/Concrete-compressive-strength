"""新数据集 (data.csv, N=4,420) 加载与预处理模块。

功能：
1. 读取 CSV 数据，处理分类变量编码
2. 缺失值处理策略（A: 无缺失子集 / B: 插补扩展）
3. 新数据集的特征工程（物理衍生特征）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Data.csv"

CATEGORICAL_COLS = [
    "Type_of_cement",
    "Brand",
    "Additives",
    "Type_of_aggregates",
]

NUMERICAL_COLS = [
    "Design_F'c (Mpa)",
    "Curing_age_(days)",
    "Ts_(Mpa)",
    "Fs_(Mpa)",
    "Er_(ohm-cm)",
    "UPV_(m/s)",
]

TARGET_COL = "Cs_(Mpa)"
AGE_COL = "Curing_age_(days)"


def load_raw_new_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw new dataset CSV."""
    fp = Path(file_path) if file_path is not None else DATA_PATH
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")
    df = pd.read_csv(fp, index_col=0)
    # Strip trailing/leading whitespace in categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def strategy_a_preprocess(
    df: pd.DataFrame,
    encoder: str = "onehot",
    add_engineered: bool = False,
    target_encoder_y: Optional[np.ndarray] = None,
    drop_extra_numerical: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Strategy A: only use complete numerical variables (drop Ts, Fs).

    Args:
        df: raw dataframe
        encoder: "onehot" or "target"
        add_engineered: whether to add physics-derived features
        target_encoder_y: target values for target encoding (required if encoder=="target")
        drop_extra_numerical: if True, drop Ts and Fs (21.7% and 60.2% missing)

    Returns:
        (X, y, metadata_dict)
    """
    y = df[TARGET_COL].copy().to_numpy()
    X = df.copy()

    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    numerical_cols_to_use = [
        "Design_F'c (Mpa)", "Curing_age_(days)", "Er_(ohm-cm)", "UPV_(m/s)"
    ]

    if not drop_extra_numerical:
        # Strategy B: keep Ts and Fs with imputation
        numerical_cols_to_use = NUMERICAL_COLS.copy()

    X_num = X[numerical_cols_to_use].copy()

    if not drop_extra_numerical:
        # Impute missing Ts and Fs with median
        for col in ["Ts_(Mpa)", "Fs_(Mpa)"]:
            if col in X_num.columns:
                median_val = X_num[col].median()
                X_num[col] = X_num[col].fillna(median_val)
                if X_num[col].isna().any():
                    X_num[col] = X_num[col].fillna(median_val)

    # Encode categorical
    if encoder == "onehot":
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        X_cat_encoded = cat_encoder.fit_transform(X[categorical_cols])
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    elif encoder == "target":
        if target_encoder_y is None:
            raise ValueError("target_encoder_y required for target encoding")
        cat_encoder = TargetEncoder()
        X_cat_encoded = cat_encoder.fit_transform(X[categorical_cols], target_encoder_y)
        cat_feature_names = categorical_cols
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    X_num_arr = X_num.values
    X_combined = np.column_stack([X_num_arr, X_cat_encoded])
    feature_names = numerical_cols_to_use + list(cat_feature_names)

    # Feature engineering on numerical part
    if add_engineered:
        X_eng, eng_names = new_dataset_feature_engineering(X_num)
        X_combined = np.column_stack([X_combined, X_eng])
        feature_names = feature_names + eng_names

    meta = {
        "n_samples": len(X_combined),
        "n_features": X_combined.shape[1],
        "feature_names": feature_names,
        "n_numerical_raw": len(numerical_cols_to_use),
        "n_categorical": len(categorical_cols),
        "n_engineered": len(eng_names) if add_engineered else 0,
        "encoder": encoder,
        "drop_extra_numerical": drop_extra_numerical,
    }

    return X_combined, y, meta


def strategy_b_preprocess(
    df: pd.DataFrame,
    encoder: str = "onehot",
    target_encoder_y: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Strategy B: impute Ts and Fs, include all numerical features."""
    return strategy_a_preprocess(
        df, encoder=encoder, add_engineered=False,
        target_encoder_y=target_encoder_y, drop_extra_numerical=False,
    )


def new_dataset_feature_engineering(
    X_num: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Physics-derived features for the new dataset.

    Key numerical features: Design_F'c, Curing_age, Er, UPV (and optionally Ts, Fs).

    The feature generation focuses on:
    - Age-related nonlinear transforms
    - Interaction terms between design strength and age
    - UPV-based derived features (elastic modulus proxy)
    - Er (electrical resistivity) based transforms
    """
    eps = 1e-6

    if isinstance(X_num, pd.DataFrame):
        cols = list(X_num.columns)
        arr = X_num.values
    else:
        cols = None
        arr = X_num

    # Identify column indices
    # "Design_F'c (Mpa)" at 0, "Curing_age_(days)" at 1, "Er_(ohm-cm)" at 2, "UPV_(m/s)" at 3
    design_fc = arr[:, 0]
    age = arr[:, 1]
    er = arr[:, 2]
    upv = arr[:, 3]

    features = {}
    age_clipped = np.maximum(age, 0.0)
    features["age_log1p"] = np.log1p(age_clipped)
    features["age_sqrt"] = np.sqrt(age_clipped)
    features["age_pow_0_25"] = np.power(age_clipped, 0.25)
    features["age_inverse"] = 1.0 / (age_clipped + 1.0)

    features["design_fc_age_log"] = design_fc * np.log1p(age_clipped)
    features["design_fc_age_sqrt"] = design_fc * np.sqrt(age_clipped)

    features["upv_squared"] = upv ** 2
    features["upv_per_age"] = upv / (age_clipped + eps)
    features["upv_times_age_log"] = upv * np.log1p(age_clipped)
    features["upv_times_design"] = upv * design_fc

    features["er_log"] = np.log(np.maximum(er, eps))
    features["er_inv"] = 1.0 / (np.maximum(er, eps))
    features["er_per_age"] = er / (age_clipped + eps)

    features["design_fc_to_upv"] = design_fc / (upv + eps)
    features["er_to_upv"] = er / (upv + eps)
    features["maturity_proxy"] = age_clipped * design_fc
    features["quality_index"] = upv / (er + eps)

    out = np.column_stack(list(features.values()))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out, list(features.keys())


def get_age_array(df: pd.DataFrame) -> np.ndarray:
    """Extract age (curing days) from dataframe."""
    return df[AGE_COL].to_numpy()
