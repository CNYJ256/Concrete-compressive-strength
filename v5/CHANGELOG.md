# CHANGELOG - v3

## 版本定位
- 版本目标：在 `v2` 之上通过 Boosting 进一步压低误差，形成高质量强基线。
- 本版策略：`HistGradientBoosting with richer mechanism interactions`。

## 核心改动
- 继续扩展非线性龄期特征：`age_pow_0_25`、`cement_age_interaction`。
- 模型切换为 `HistGradientBoostingRegressor`，并采用较低学习率 + 更高迭代轮次（1600）进行细粒度残差拟合。
- 维持统一 10 折协议，避免评估口径漂移。

## 指标变化（10折CV）
| 指标 | v2 | v3 | 变化 |
|---|---:|---:|---:|
| R²_mean | 0.9254 | 0.9388 | +0.0134 |
| RMSE_mean | 4.4776 | 4.0231 | -0.4545 |

## 结果判定
- 判定：✅ **优于上一版（v2）**。
- 结论：Boosting 在该数据集上显著提升，v3 成为后续迭代参照系。

## 材料机理反思
- 水胶比与龄期的耦合项（`wb_age_interaction`）在 Boosting 中被反复利用，说明其是关键控制变量。
- 水泥含量与龄期交互提升了对早强/后强差异样本的区分能力。

## 下一版计划
- 在 v3 基础上进行受控超参数搜索（AutoML 风格），重点优化 `max_leaf_nodes`、`min_samples_leaf` 与学习率。
