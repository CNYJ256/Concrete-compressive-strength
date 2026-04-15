# CHANGELOG - v8

## 版本定位

- 目标：围绕 v7 README 中的“单模型族局限”完成下一代算法迭代。
- 策略：执行 5 轮假设驱动实验，最终采用跨特征空间加权融合（含 v7 锚点模型）。

---

## 本版核心改动

1. 新增 `v8/train.py`
   - 统一 10 折 CV 协议。
   - 前四轮通过 Optuna 进行 HGB/XGBoost/LightGBM 受控搜索。
   - 第五轮执行 WeightedBlend，并引入 `HGB_v7_baseline` 作为融合锚点。
   - 记录完整迭代日志与 fallback 事件。

2. 新增 `v8/predict.py`
   - 支持 `single` 与 `weighted_ensemble` 两种模型类型。
   - 支持按子模型特征空间自动切换（`v8` / `v7`）。

3. 新增 `v8/README.md`
   - 完整写入 5 轮科研过程、踩坑与下一步方向。

---

## 5轮结果摘要

| Iter | 模型 | R²_mean | RMSE_mean | 结果 |
|---|---|---:|---:|---|
| 1 | HGB | 0.945497 | 3.808984 | 未接纳 |
| 2 | XGBoost | 0.945934 | 3.781660 | 未接纳 |
| 3 | LightGBM | 0.945652 | 3.805196 | 未接纳 |
| 4 | XGBoost(二次强化) | 0.945215 | 3.801014 | 未接纳 |
| 5 | WeightedBlend(+v7锚点) | **0.948725** | **3.700053** | ✅ 接纳（v8.1） |

---

## 对比 v7

- `ΔR² = +0.000759`
- `ΔRMSE = +0.040728`（正值表示 RMSE 下降）

判定：✅ v8 双指标优于 v7。

---

## 已产出文件

- `v8/model.joblib`
- `v8/metrics.json`
- `v8/predictions.csv`
- `v8/README.md`
- `v8/CHANGELOG.md`
