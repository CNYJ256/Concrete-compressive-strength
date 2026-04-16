"""全局配置文件。

说明：
- 统一维护数据路径、随机种子与模型参数；
- 统一约束实验产物输出目录，避免结果散落在源码目录。
"""

from pathlib import Path

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Concrete_Data.xls"

DOC_DIR = PROJECT_ROOT / "docs"
DOC_REPORT_DIR = DOC_DIR / "reports"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_PREDICTIONS_DIR = RESULTS_DIR / "predictions"

BASELINE_RESULT_MD = DOC_REPORT_DIR / "Baseline_Reproduction_Report.md"
BASELINE_RESULT_JSON = RESULTS_METRICS_DIR / "baseline_results.json"

PAPER2_RESULT_MD = DOC_REPORT_DIR / "Paper2_Reproduction_Report.md"
PAPER2_RESULT_JSON = RESULTS_METRICS_DIR / "paper2_reproduction_results.json"

OPT_RESULT_MD = DOC_REPORT_DIR / "Optimization_Comparison_Report.md"
OPT_RESULT_JSON = RESULTS_METRICS_DIR / "optimization_results.json"

# =========================
# 实验配置
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.1
CV_FOLDS = 10

# =========================
# 原始列名 -> 标准列名
# =========================
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

FEATURE_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
]
TARGET_COLUMN = "strength"

# =========================
# 基线模型参数（参考论文）
# =========================
ADABOOST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.2,
    "loss": "linear",
    "random_state": RANDOM_STATE,
}

TREE_PARAMS = {
    "max_depth": 50,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "min_impurity_decrease": 1e-4,
    "random_state": RANDOM_STATE,
}

# 这里的 ANN/SVM 主要用于“论文对比复现”，不追求极致调参。
ANN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 3000,
    "random_state": RANDOM_STATE,
}

SVM_PARAMS = {
    "C": 50.0,
    "epsilon": 0.5,
    "kernel": "rbf",
    "gamma": "scale",
}

# =========================
# 创新模型参数（优化版本）
# =========================
OPT_ADABOOST_PARAMS = {
    "n_estimators": 320,
    "learning_rate": 0.08,
    "loss": "square",
    "random_state": RANDOM_STATE,
}

OPT_TREE_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "min_impurity_decrease": 5e-5,
    "random_state": RANDOM_STATE,
}
