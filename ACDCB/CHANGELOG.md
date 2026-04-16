# CHANGELOG - v9

## 版本定位

- 基于 v8 的继续深度优化版本。
- 核心目标：在不重新重型搜索的情况下，利用融合层建模能力继续榨取性能上限。

---

## 核心改动

1. 新增 `v9/train.py`
   - 复用 v8 三个候选模型最优参数。
   - 引入 `HGB_v7_baseline` 锚点模型。
   - 实现两种融合：
     - 全局单权重融合
     - 龄期分段融合（<=28 天 / >28 天）
   - 自动比较并固化最优策略。

2. 新增 `v9/predict.py`
   - 支持 `age_aware_weighted_ensemble` 推理；
   - 按模型特征空间（v8/v7）自动路由特征；
   - 按 `selected_strategy` 自动选择全局或分段权重。

3. 新增文档
   - `v9/README.md`
   - `v9/CHANGELOG.md`

---

## 指标变化

### 相对 v8

- `R²_mean`: `+0.000030339`
- `RMSE_mean`: `-0.000482255`

### 相对 v7

- `R²_mean`: `+0.000789788`
- `RMSE_mean`: `-0.041210372`

判定：✅ v9 相对 v8 与 v7 均为双指标正向提升。

---

## 最终策略

- `selected_strategy = age_piecewise`
- 分段阈值：`age_split_day = 28`

---

## 输出文件

- `v9/model.joblib`
- `v9/metrics.json`
- `v9/predictions.csv`
