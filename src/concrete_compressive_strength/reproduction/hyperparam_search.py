"""
AdaBoost 超参数搜索脚本。

目标：寻找能使 10 折 CV 指标接近论文原始数据的参数组合。
论文报告：R²=0.952, RMSE=4.856, MAPE=11.39%, MAE=3.205
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # 修正: 原为 parents[3]
COMMON_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(COMMON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS_DIR))

from config import RANDOM_STATE
from data_loader import load_concrete_data, split_features_target
from logger_utils import get_logger

logger = get_logger(__name__)


def run_cv(model: AdaBoostRegressor, X, y, cv_folds: int = 10) -> Dict[str, float]:
    """执行 10 折 CV，返回各指标均值。"""
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    def rmse_func(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    scoring = {
        "R2": "r2",
        "RMSE": "neg_root_mean_squared_error",
        "MAPE": "neg_mean_absolute_percentage_error",
        "MAE": "neg_mean_absolute_error",
    }

    result = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    r2_mean = float(np.mean(result["test_R2"]))
    rmse_mean = float(np.mean(-result["test_RMSE"]))
    mape_mean = float(np.mean(-result["test_MAPE"]) * 100.0)
    mae_mean = float(np.mean(-result["test_MAE"]))

    return {"R2": r2_mean, "RMSE": rmse_mean, "MAPE": mape_mean, "MAE": mae_mean}


def score_to_paper(results: Dict[str, float]) -> float:
    """
    计算与论文数据的综合距离。
    各指标的相对误差加权求和：R² 和 MAPE 权重更高（论文更关注）。
    """
    paper = {"R2": 0.952, "RMSE": 4.856, "MAPE": 11.39, "MAE": 3.205}

    # 归一化权重：各指标相对偏差的加权和
    weights = {"R2": 2.0, "RMSE": 1.0, "MAPE": 1.5, "MAE": 1.0}

    total = 0.0
    for key in paper:
        rel_err = abs(results[key] - paper[key]) / paper[key]
        total += weights[key] * rel_err

    return total


# ---- 搜索空间 ----
SEARCH_SPACE = {
    "n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 500],
    "learning_rate": [0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0],
    "loss": ["linear", "square", "exponential"],
    "max_depth": [3, 5, 7, 8, 10, 12, 15, 20, 30, 40, 50, 60, 80],
    "min_samples_split": [2, 3, 5, 8, 10, 15, 20],
    "min_samples_leaf": [1, 2, 3, 5, 8, 10],
    "min_impurity_decrease": [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}


def search_grid(X, y) -> List[Dict[str, Any]]:
    """网格搜索，返回按综合得分排序的结果列表。"""
    import random as _random

    all_results = []

    # 核心参数组合
    core_combos = list(itertools.product(
        [50, 100, 150, 200, 250, 300, 350, 400, 500],
        [0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7],
        ["linear", "square", "exponential"],
        [3, 5, 7, 8, 10, 12, 15, 20, 30, 50, 80],
        [2, 3, 5, 8, 10, 15],
        [1, 2, 3, 5, 8],
        [1e-6, 1e-5, 5e-5, 1e-4, 5e-4],
    ))

    # 组合数过大，随机采样
    _random.seed(RANDOM_STATE)
    max_combos = 500
    if len(core_combos) > max_combos:
        core_combos = _random.sample(core_combos, max_combos)

    logger.info("开始搜索，共 %d 组参数组合", len(core_combos))

    for idx, (n_est, lr, loss, depth, min_split, min_leaf, min_imp) in enumerate(core_combos):
        try:
            tree = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                min_impurity_decrease=min_imp,
                random_state=RANDOM_STATE,
            )
            model = AdaBoostRegressor(
                estimator=tree,
                n_estimators=n_est,
                learning_rate=lr,
                loss=loss,
                random_state=RANDOM_STATE,
            )

            cv_results = run_cv(model, X, y)
            distance = score_to_paper(cv_results)

            all_results.append({
                "params": {
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "loss": loss,
                    "max_depth": depth,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf,
                    "min_impurity_decrease": min_imp,
                },
                "cv_results": cv_results,
                "paper_distance": distance,
            })

            if (idx + 1) % 50 == 0:
                logger.info("进度: %d/%d, 当前最优距离: %.4f",
                            idx + 1, len(core_combos),
                            min(r["paper_distance"] for r in all_results))
        except Exception as exc:
            logger.warning("参数组合失败: %s", exc)
            continue

    all_results.sort(key=lambda x: x["paper_distance"])
    return all_results


def main():
    logger.info("===== 超参数搜索开始 =====")

    df = load_concrete_data()
    X, y = split_features_target(df)

    t0 = time.perf_counter()
    results = search_grid(X, y)
    elapsed = time.perf_counter() - t0

    logger.info("搜索完成，耗时 %.1f 秒", elapsed)

    # 先保存结果
    output_path = PROJECT_ROOT / "results" / "metrics" / "hyperparam_search_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([
            {"rank": i+1, "params": r["params"],
             "cv_results": r["cv_results"],
             "paper_distance": r["paper_distance"]}
            for i, r in enumerate(results[:50])
        ], f, ensure_ascii=False, indent=2)
    logger.info("搜索结果已保存至: %s", output_path)

    # 输出 top 20
    print("\n" + "=" * 80)
    print("Top 20 parameter combinations (sorted by distance to paper)")
    print("=" * 80)

    for i, r in enumerate(results[:20]):
        p = r["params"]
        cv = r["cv_results"]
        print(f"\n--- Rank {i+1}, distance={r['paper_distance']:.4f} ---")
        print(f"  n_estimators={p['n_estimators']}, lr={p['learning_rate']}, "
              f"loss={p['loss']}, max_depth={p['max_depth']}, "
              f"min_split={p['min_samples_split']}, min_leaf={p['min_samples_leaf']}, "
              f"min_imp={p['min_impurity_decrease']}")
        print(f"  CV: R2={cv['R2']:.4f}, RMSE={cv['RMSE']:.4f}, "
              f"MAPE={cv['MAPE']:.2f}%, MAE={cv['MAE']:.4f}")

    # 结果已在上面保存

    # 输出最佳参数可直接复制到 config.py
    best = results[0]
    print("\n" + "=" * 80)
    print("推荐最佳参数（可复制到 config.py）:")
    print("=" * 80)
    print(f"ADABOOST_PARAMS = {{")
    print(f'    "n_estimators": {best["params"]["n_estimators"]},')
    print(f'    "learning_rate": {best["params"]["learning_rate"]},')
    print(f'    "loss": "{best["params"]["loss"]}",')
    print(f'    "random_state": RANDOM_STATE,')
    print(f"}}")
    print(f"TREE_PARAMS = {{")
    print(f'    "max_depth": {best["params"]["max_depth"]},')
    print(f'    "min_samples_split": {best["params"]["min_samples_split"]},')
    print(f'    "min_samples_leaf": {best["params"]["min_samples_leaf"]},')
    print(f'    "min_impurity_decrease": {best["params"]["min_impurity_decrease"]},')
    print(f'    "random_state": RANDOM_STATE,')
    print(f"}}")

    logger.info("===== 超参数搜索完成 =====")


if __name__ == "__main__":
    main()
