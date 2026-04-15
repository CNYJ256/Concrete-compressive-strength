"""
Step 3：论文复现主脚本（基线模型）。

执行内容：
1) 加载并清洗数据；
2) 训练 AdaBoost / ANN / SVM 三个模型；
3) 输出训练集与测试集指标（R2, RMSE, MAPE, MAE）；
4) 对 AdaBoost 做 10 折交叉验证；
5) 将结果保存到 doc 目录。

运行方式（在项目根目录）：
python scripts/reproduce_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

# 让脚本在任意 cwd 下都能正确导入项目公共模块（scripts/）。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
COMMON_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS_DIR))

from config import (
    BASELINE_RESULT_JSON,
    BASELINE_RESULT_MD,
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
)
from data_loader import load_concrete_data, split_features_target
from logger_utils import get_logger
from metrics_utils import format_metrics, regression_metrics
from model_factory import build_adaboost_model, build_baseline_models

logger = get_logger(__name__)


def evaluate_model(model_name: str, model: Any, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    训练并评估单个模型。

    返回：
    - 训练指标
    - 测试指标
    - 训练耗时
    - 推理耗时
    """
    logger.info("开始训练模型: %s", model_name)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    predict_time = time.perf_counter() - t1

    train_metrics = regression_metrics(y_train, pred_train)
    test_metrics = regression_metrics(y_test, pred_test)

    logger.info("%s 训练集: %s", model_name, format_metrics(train_metrics))
    logger.info("%s 测试集: %s", model_name, format_metrics(test_metrics))

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
    }


def run_adaboost_cross_validation(X, y) -> Dict[str, float]:
    """
    对 AdaBoost 执行 10 折交叉验证。
    """
    logger.info("开始 AdaBoost 的 %d 折交叉验证。", CV_FOLDS)

    model = build_adaboost_model()

    def rmse_func(y_true, y_pred):
        # 兼容旧版 sklearn：避免使用 squared=False 参数。
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    scoring = {
        "R2": make_scorer(r2_score),
        "RMSE": make_scorer(rmse_func, greater_is_better=False),
        "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    # cross_validate 对“误差型指标”返回的是负数（因为定义了 greater_is_better=False）
    r2_vals = cv_result["test_R2"]
    rmse_vals = -cv_result["test_RMSE"]
    mape_vals = -cv_result["test_MAPE"] * 100.0  # sklearn 返回比例，这里转成百分数
    mae_vals = -cv_result["test_MAE"]

    summary = {
        "R2_mean": float(np.mean(r2_vals)),
        "R2_std": float(np.std(r2_vals)),
        "RMSE_mean": float(np.mean(rmse_vals)),
        "RMSE_std": float(np.std(rmse_vals)),
        "MAPE_mean": float(np.mean(mape_vals)),
        "MAPE_std": float(np.std(mape_vals)),
        "MAE_mean": float(np.mean(mae_vals)),
        "MAE_std": float(np.std(mae_vals)),
    }

    logger.info(
        "AdaBoost 10折: R2=%.4f±%.4f, RMSE=%.4f±%.4f, MAPE=%.2f%%±%.2f%%, MAE=%.4f±%.4f",
        summary["R2_mean"],
        summary["R2_std"],
        summary["RMSE_mean"],
        summary["RMSE_std"],
        summary["MAPE_mean"],
        summary["MAPE_std"],
        summary["MAE_mean"],
        summary["MAE_std"],
    )

    return summary


def build_markdown_report(results: Dict[str, Any]) -> str:
    """
    将结果组织成 Markdown，便于直接用于课程汇报。
    """
    overview = results["dataset_overview"]
    split = results["split"]
    models = results["models"]
    cv = results["adaboost_cv_10fold"]

    lines = [
        "# 基线复现实验报告（Step 3）",
        "",
        "## 数据概况",
        "",
        f"- 样本数：{overview['n_samples']}",
        f"- 特征数：{overview['n_features']}",
        f"- 训练集样本数：{split['train_size']}",
        f"- 测试集样本数：{split['test_size']}",
        f"- 划分比例（训练:测试）：{int((1-TEST_SIZE)*10)}:{int(TEST_SIZE*10)}（约）",
        "",
        "## 单次划分结果（9:1）",
        "",
        "| 模型 | 集合 | R2 | RMSE | MAPE(%) | MAE | 训练耗时(s) | 推理耗时(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name, model_res in models.items():
        train_m = model_res["train_metrics"]
        test_m = model_res["test_metrics"]
        fit_t = model_res["fit_time_sec"]
        pred_t = model_res["predict_time_sec"]

        lines.append(
            f"| {model_name} | 训练集 | {train_m['R2']:.4f} | {train_m['RMSE']:.4f} | {train_m['MAPE']:.2f} | {train_m['MAE']:.4f} | {fit_t:.4f} | {pred_t:.4f} |"
        )
        lines.append(
            f"| {model_name} | 测试集 | {test_m['R2']:.4f} | {test_m['RMSE']:.4f} | {test_m['MAPE']:.2f} | {test_m['MAE']:.4f} | {fit_t:.4f} | {pred_t:.4f} |"
        )

    lines.extend([
        "",
        "## AdaBoost 10 折交叉验证",
        "",
        "| 指标 | 均值 | 标准差 |",
        "|---|---:|---:|",
        f"| R2 | {cv['R2_mean']:.4f} | {cv['R2_std']:.4f} |",
        f"| RMSE | {cv['RMSE_mean']:.4f} | {cv['RMSE_std']:.4f} |",
        f"| MAPE(%) | {cv['MAPE_mean']:.2f} | {cv['MAPE_std']:.2f} |",
        f"| MAE | {cv['MAE_mean']:.4f} | {cv['MAE_std']:.4f} |",
        "",
        "## 复现结论",
        "",
        "- 已完成 AdaBoost、ANN、SVM 三模型复现与对比。",
        "- 已完成 AdaBoost 10 折交叉验证，得到稳定性统计结果。",
        "- 结果可直接用于课程汇报中的“核心代码展示+结果对比分析”部分。",
    ])

    return "\n".join(lines)


def save_results(results: Dict[str, Any]) -> None:
    """
    保存 JSON + Markdown 两种结果文件。
    """
    BASELINE_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(BASELINE_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_text = build_markdown_report(results)
    with open(BASELINE_RESULT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)

    logger.info("已保存 JSON 结果: %s", BASELINE_RESULT_JSON)
    logger.info("已保存 Markdown 报告: %s", BASELINE_RESULT_MD)


def main() -> None:
    """
    主流程入口。
    """
    try:
        logger.info("===== Step 3 基线复现开始 =====")

        df = load_concrete_data()
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        models = build_baseline_models()
        model_results: Dict[str, Any] = {}

        for model_name, model in models.items():
            model_results[model_name] = evaluate_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        cv_summary = run_adaboost_cross_validation(X, y)

        results = {
            "dataset_overview": {
                "n_samples": int(len(df)),
                "n_features": int(X.shape[1]),
            },
            "split": {
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "random_state": RANDOM_STATE,
            },
            "models": model_results,
            "adaboost_cv_10fold": cv_summary,
        }

        save_results(results)
        logger.info("===== Step 3 基线复现完成 =====")

    except Exception as exc:
        logger.error("复现流程失败：%s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
