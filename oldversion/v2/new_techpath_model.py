"""
新技术路径实验脚本（在现有优化 AdaBoost 之外继续迭代）。

技术路径：Stacking 集成回归
- 基学习器：HistGradientBoosting + ExtraTrees + SVR
- 元学习器：Ridge
- 特点：融合树模型与核方法，提升泛化能力

输出：
- doc/new_techpath_results.json
- doc/New_TechPath_Comparison_Report.md
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 让脚本在任意 cwd 下都能正确导入当前目录与 scripts 公共模块
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import OPT_RESULT_JSON, RANDOM_STATE, TEST_SIZE
from data_loader import load_concrete_data, split_features_target
from logger_utils import get_logger
from metrics_utils import format_metrics, regression_metrics
from model_factory import build_optimized_adaboost_model

logger = get_logger(__name__)

OUTPUT_JSON = Path(__file__).resolve().parents[1] / "doc" / "new_techpath_results.json"
OUTPUT_MD = Path(__file__).resolve().parents[1] / "doc" / "New_TechPath_Comparison_Report.md"


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    与优化 AdaBoost 一致的机理特征工程，保证对比公平。
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


def build_hard_sample_weights(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    与 optimized_model.py 保持一致：按残差构造难样本权重。
    """
    residual = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    norm = residual / (np.mean(residual) + 1e-6)
    weights = np.clip(1.0 + norm, 1.0, 3.0)
    return weights


def fit_eval(model: Any, X_train, y_train, X_test, y_test, sample_weight=None) -> Dict[str, Any]:
    """
    统一训练与评估函数。
    """
    t0 = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "train_metrics": regression_metrics(y_train, pred_train),
        "test_metrics": regression_metrics(y_test, pred_test),
        "fit_time_sec": fit_time,
        "train_pred": pred_train,
    }


def build_stacking_model() -> StackingRegressor:
    """
    构建新技术路径模型：Stacking 集成。
    """
    hgb = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=8,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )

    etr = ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    svr = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=80.0, epsilon=0.2, gamma="scale", kernel="rbf")),
    ])

    model = StackingRegressor(
        estimators=[
            ("hgb", hgb),
            ("etr", etr),
            ("svr", svr),
        ],
        final_estimator=Ridge(alpha=0.8),
        passthrough=True,
        cv=5,
        n_jobs=-1,
    )
    return model


def compare_metrics(ref: Dict[str, float], newm: Dict[str, float]) -> Dict[str, float]:
    """
    对比新模型相对参考模型（这里参考=优化AdaBoost）。
    """
    return {
        "R2_change_pct": (newm["R2"] - ref["R2"]) / (abs(ref["R2"]) + 1e-8) * 100.0,
        "RMSE_improve_pct": (ref["RMSE"] - newm["RMSE"]) / (abs(ref["RMSE"]) + 1e-8) * 100.0,
        "MAPE_improve_pct": (ref["MAPE"] - newm["MAPE"]) / (abs(ref["MAPE"]) + 1e-8) * 100.0,
        "MAE_improve_pct": (ref["MAE"] - newm["MAE"]) / (abs(ref["MAE"]) + 1e-8) * 100.0,
    }


def build_md(result: Dict[str, Any]) -> str:
    ref = result["reference_optimized_adaboost"]
    newm = result["new_techpath_stacking"]
    comp = result["comparison"]

    lines = [
        "# 新技术路径对比报告（Step 4 深化版）",
        "",
        "## 方法说明",
        "",
        "- 参考方法：优化 AdaBoost（特征工程 + 两阶段难样本重加权）",
        "- 新技术路径：Stacking 集成（HGB + ExtraTrees + SVR -> Ridge）",
        "",
        "## 指标对比（测试集）",
        "",
        "| 模型 | R2 | RMSE | MAPE(%) | MAE | 训练耗时(s) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 优化 AdaBoost | {ref['test_metrics']['R2']:.4f} | {ref['test_metrics']['RMSE']:.4f} | {ref['test_metrics']['MAPE']:.2f} | {ref['test_metrics']['MAE']:.4f} | {ref['fit_time_sec']:.4f} |",
        f"| 新技术路径 Stacking | {newm['test_metrics']['R2']:.4f} | {newm['test_metrics']['RMSE']:.4f} | {newm['test_metrics']['MAPE']:.2f} | {newm['test_metrics']['MAE']:.4f} | {newm['fit_time_sec']:.4f} |",
        "",
        "## 相对变化（新技术路径 vs 优化 AdaBoost）",
        "",
        f"- R2 变化：{comp['R2_change_pct']:+.2f}%",
        f"- RMSE 改善：{comp['RMSE_improve_pct']:+.2f}%",
        f"- MAPE 改善：{comp['MAPE_improve_pct']:+.2f}%",
        f"- MAE 改善：{comp['MAE_improve_pct']:+.2f}%",
        "",
        "## 结论",
        "",
        "- 已引入新技术路径并完成同分割对比。",
        "- 该结果可用于最终汇报中的“展望落地迭代”补充页。",
    ]
    return "\n".join(lines)


def main() -> None:
    try:
        logger.info("===== 新技术路径实验开始 =====")

        # 读取数据
        df = load_concrete_data()
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        # 统一特征工程
        X_train_fe = feature_engineering(X_train)
        X_test_fe = feature_engineering(X_test)

        # 参考模型：优化 AdaBoost（两阶段）
        stage1 = build_optimized_adaboost_model()
        stage1_res = fit_eval(stage1, X_train_fe, y_train, X_test_fe, y_test)
        weights = build_hard_sample_weights(y_train, stage1_res["train_pred"])

        stage2 = build_optimized_adaboost_model()
        ref_res = fit_eval(stage2, X_train_fe, y_train, X_test_fe, y_test, sample_weight=weights)
        logger.info("参考模型（优化AdaBoost）测试集: %s", format_metrics(ref_res["test_metrics"]))

        # 新技术路径
        stack_model = build_stacking_model()
        new_res = fit_eval(stack_model, X_train_fe, y_train, X_test_fe, y_test)
        logger.info("新技术路径（Stacking）测试集: %s", format_metrics(new_res["test_metrics"]))

        comp = compare_metrics(ref_res["test_metrics"], new_res["test_metrics"])

        # 读取已有优化结果（用于校对）
        prev_opt = None
        if OPT_RESULT_JSON.exists():
            with open(OPT_RESULT_JSON, "r", encoding="utf-8") as f:
                prev_opt = json.load(f)

        result = {
            "reference_optimized_adaboost": {
                "train_metrics": ref_res["train_metrics"],
                "test_metrics": ref_res["test_metrics"],
                "fit_time_sec": ref_res["fit_time_sec"],
            },
            "new_techpath_stacking": {
                "train_metrics": new_res["train_metrics"],
                "test_metrics": new_res["test_metrics"],
                "fit_time_sec": new_res["fit_time_sec"],
            },
            "comparison": comp,
            "notes": {
                "tech_path": "Stacking(HGB + ExtraTrees + SVR -> Ridge)",
                "same_split": True,
                "random_state": RANDOM_STATE,
                "previous_optimization_file_loaded": prev_opt is not None,
            },
        }

        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write(build_md(result))

        logger.info("已保存 JSON：%s", OUTPUT_JSON)
        logger.info("已保存 Markdown：%s", OUTPUT_MD)
        logger.info("===== 新技术路径实验完成 =====")

    except Exception as exc:
        logger.error("新技术路径实验失败：%s", exc)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
