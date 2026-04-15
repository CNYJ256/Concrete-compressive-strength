"""
Step 4：创新模型与对比脚本。

创新点（基于原论文 AdaBoost 逻辑）：
1) 引入“配比机理特征”增强（如水灰比、水胶比、龄期对数）；
2) 使用“两阶段难样本重加权”机制：
   - 阶段1先训练一个初始模型；
   - 根据训练残差构造样本难度权重；
   - 阶段2用新权重再次训练 AdaBoost。

输出：
- doc/Optimization_Comparison_Report.md
- doc/optimization_results.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 让脚本在任意 cwd 下都能正确导入当前目录与 scripts 公共模块
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import OPT_RESULT_JSON, OPT_RESULT_MD, RANDOM_STATE, TEST_SIZE
from data_loader import load_concrete_data, split_features_target
from logger_utils import get_logger
from metrics_utils import format_metrics, regression_metrics
from model_factory import build_adaboost_model, build_optimized_adaboost_model

logger = get_logger(__name__)


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：构造与材料机理相关的派生特征。

    说明：
    - 这些特征都不使用目标值，不会造成标签泄漏。
    - eps 用于防止除零。
    """
    eps = 1e-6
    out = X.copy()

    binder = out["cement"] + out["slag"] + out["fly_ash"]

    out["water_cement_ratio"] = out["water"] / (out["cement"] + eps)
    out["water_binder_ratio"] = out["water"] / (binder + eps)
    out["sp_binder_ratio"] = out["superplasticizer"] / (binder + eps)
    out["slag_ratio"] = out["slag"] / (binder + eps)
    out["flyash_ratio"] = out["fly_ash"] / (binder + eps)
    out["age_log1p"] = np.log1p(out["age"])

    return out


def fit_and_eval(model: Any, X_train, y_train, X_test, y_test, sample_weight=None) -> Dict[str, Any]:
    """
    统一训练与评估函数（支持可选样本权重）。
    """
    t0 = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = regression_metrics(y_train, pred_train)
    test_metrics = regression_metrics(y_test, pred_test)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_time_sec": fit_time,
        "train_pred": pred_train,
    }


def build_hard_sample_weights(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    根据残差构建“难样本权重”。

    思路：
    - 残差越大，说明样本越难；
    - 在第二阶段训练中，给难样本更高权重；
    - 为防止极端权重导致不稳定，做裁剪。
    """
    residual = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    norm = residual / (np.mean(residual) + 1e-6)

    # 基础权重 1.0，难样本可提升到 3.0
    weights = 1.0 + norm
    weights = np.clip(weights, 1.0, 3.0)
    return weights


def compare_metrics(baseline: Dict[str, float], optimized: Dict[str, float]) -> Dict[str, float]:
    """
    计算优化模型相对基线的变化百分比。

    对于 R2：越大越好，变化率 = (opt-base)/|base| * 100%
    对于误差类指标：越小越好，变化率 = (base-opt)/|base| * 100%
    """
    result = {}

    # R2：越大越好
    result["R2_change_pct"] = (optimized["R2"] - baseline["R2"]) / (abs(baseline["R2"]) + 1e-8) * 100.0

    # RMSE / MAPE / MAE：越小越好
    for key in ["RMSE", "MAPE", "MAE"]:
        result[f"{key}_improve_pct"] = (baseline[key] - optimized[key]) / (abs(baseline[key]) + 1e-8) * 100.0

    return result


def build_markdown_report(results: Dict[str, Any]) -> str:
    """
    构建优化对比 Markdown 报告。
    """
    b_train = results["baseline"]["train_metrics"]
    b_test = results["baseline"]["test_metrics"]
    b_time = results["baseline"]["fit_time_sec"]

    o_train = results["optimized"]["train_metrics"]
    o_test = results["optimized"]["test_metrics"]
    o_time = results["optimized"]["fit_time_sec"]

    delta = results["comparison"]

    lines = [
        "# 创新模型对比报告（Step 4）",
        "",
        "## 创新方法说明",
        "",
        "- 在 AdaBoost 框架下引入配比机理特征（如水灰比、水胶比、龄期对数）。",
        "- 使用两阶段难样本重加权机制：先训练初始模型，再依据残差提升难样本权重后重训。",
        "",
        "## 基线模型 vs 创新模型",
        "",
        "| 模型 | 数据集 | R2 | RMSE | MAPE(%) | MAE | 训练耗时(s) |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| 基线 AdaBoost | 训练集 | {b_train['R2']:.4f} | {b_train['RMSE']:.4f} | {b_train['MAPE']:.2f} | {b_train['MAE']:.4f} | {b_time:.4f} |",
        f"| 基线 AdaBoost | 测试集 | {b_test['R2']:.4f} | {b_test['RMSE']:.4f} | {b_test['MAPE']:.2f} | {b_test['MAE']:.4f} | {b_time:.4f} |",
        f"| 创新 AdaBoost | 训练集 | {o_train['R2']:.4f} | {o_train['RMSE']:.4f} | {o_train['MAPE']:.2f} | {o_train['MAE']:.4f} | {o_time:.4f} |",
        f"| 创新 AdaBoost | 测试集 | {o_test['R2']:.4f} | {o_test['RMSE']:.4f} | {o_test['MAPE']:.2f} | {o_test['MAE']:.4f} | {o_time:.4f} |",
        "",
        "## 量化变化（测试集）",
        "",
        f"- R2 变化：{delta['R2_change_pct']:+.2f}%（正值表示提升）",
        f"- RMSE 改善：{delta['RMSE_improve_pct']:+.2f}%（正值表示误差下降）",
        f"- MAPE 改善：{delta['MAPE_improve_pct']:+.2f}%（正值表示误差下降）",
        f"- MAE 改善：{delta['MAE_improve_pct']:+.2f}%（正值表示误差下降）",
        "",
        "## 结论",
        "",
        "- 已完成创新算法实现与同数据集对比评估。",
        "- 可将该结果用于课程汇报中“延伸新结果/改进方法”部分。",
    ]

    return "\n".join(lines)


def save_results(results: Dict[str, Any]) -> None:
    """
    保存优化结果 JSON + Markdown。
    """
    OPT_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OPT_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md = build_markdown_report(results)
    with open(OPT_RESULT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("已保存优化结果 JSON: %s", OPT_RESULT_JSON)
    logger.info("已保存优化对比报告: %s", OPT_RESULT_MD)


def main() -> None:
    """
    优化流程主入口。
    """
    try:
        logger.info("===== Step 4 创新模型开始 =====")

        # 1) 读取数据并拆分
        df = load_concrete_data()
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        # 2) 基线模型（原论文思路）
        baseline_model = build_adaboost_model()
        baseline_res = fit_and_eval(
            model=baseline_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        logger.info("基线测试集: %s", format_metrics(baseline_res["test_metrics"]))

        # 3) 创新模型：特征工程 + 两阶段难样本重加权
        X_train_fe = feature_engineering(X_train)
        X_test_fe = feature_engineering(X_test)

        # 阶段1：先训练一次，估计样本难度
        stage1_model = build_optimized_adaboost_model()
        stage1_res = fit_and_eval(
            model=stage1_model,
            X_train=X_train_fe,
            y_train=y_train,
            X_test=X_test_fe,
            y_test=y_test,
        )

        sample_weights = build_hard_sample_weights(y_train, stage1_res["train_pred"])

        # 阶段2：带权重重训练
        stage2_model = build_optimized_adaboost_model()
        optimized_res = fit_and_eval(
            model=stage2_model,
            X_train=X_train_fe,
            y_train=y_train,
            X_test=X_test_fe,
            y_test=y_test,
            sample_weight=sample_weights,
        )
        logger.info("创新模型测试集: %s", format_metrics(optimized_res["test_metrics"]))

        # 4) 对比分析
        comparison = compare_metrics(
            baseline=baseline_res["test_metrics"],
            optimized=optimized_res["test_metrics"],
        )

        results = {
            "baseline": {
                "train_metrics": baseline_res["train_metrics"],
                "test_metrics": baseline_res["test_metrics"],
                "fit_time_sec": baseline_res["fit_time_sec"],
            },
            "optimized": {
                "train_metrics": optimized_res["train_metrics"],
                "test_metrics": optimized_res["test_metrics"],
                "fit_time_sec": optimized_res["fit_time_sec"],
            },
            "comparison": comparison,
            "notes": {
                "innovation": "特征工程 + 两阶段难样本重加权 AdaBoost",
                "same_split": True,
                "random_state": RANDOM_STATE,
            },
        }

        save_results(results)
        logger.info("===== Step 4 创新模型完成 =====")

    except Exception as exc:
        logger.error("创新流程失败：%s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
