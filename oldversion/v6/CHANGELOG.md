# CHANGELOG - v4

## 版本定位
- 版本目标：在 `v3` 基础上通过自动化搜索进一步提效，并保持工程可复现。
- 本版策略：`AutoML-style randomized hyperparameter search for HGB`。

## 核心改动
- 使用 `RandomizedSearchCV` + 10折 `KFold(shuffle=True, random_state=42)` 搜索 HGB 参数。
- 搜索空间聚焦：`learning_rate`、`max_iter`、`max_depth`、`max_leaf_nodes`、`min_samples_leaf`、`l2_regularization`。
- 最优参数：
  - `max_leaf_nodes=15`
  - `min_samples_leaf=6`
  - `max_iter=1800`
  - `learning_rate=0.025`
  - `max_depth=None`
  - `l2_regularization=0.0`

## 调试与自纠记录（瓶颈突破）
- 失败尝试1：Stacking 路线，显著退化。
- 失败尝试2：随机搜索但使用 `cv=10` 默认不打乱，性能异常下滑。
- 失败尝试3：加权投票，仍低于 v3。
- 根因修复：将搜索 CV 与全项目协议对齐为 `KFold(shuffle=True, random_state=42)`，性能恢复并反超。

## 指标变化（10折CV）
| 指标 | v3 | v4 | 变化 |
|---|---:|---:|---:|
| R²_mean | 0.9388 | 0.9471 | +0.0083 |
| RMSE_mean | 4.0231 | 3.7654 | -0.2577 |

## 结果判定
- 判定：✅ **优于上一版（v3）**。
- 结论：在统一协议下，受控超参搜索可稳定挖掘 HGB 潜力。

## 材料机理反思
- 更小叶节点数（15）+ 合理叶子样本阈值（6）有助于在复杂机理特征下控制方差。
- 说明当前样本规模下“中等复杂度树结构 + 充足迭代”是更优折中。

## 下一版计划
- 采用多随机种子强化搜索，增加候选覆盖率，争取在 v4 高基线上继续小幅突破。
