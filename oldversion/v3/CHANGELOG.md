# CHANGELOG - v1

## 版本定位
- 版本目标：在 `v0` AdaBoost 复现基线之上，先建立“机理特征 + 稳健树模型”的可复现增益版本。
- 本版策略：`Mechanism feature engineering + RandomForest`。

## 核心改动
- 新增机理特征：`binder`、`water_cement_ratio`、`water_binder_ratio`、`sp_binder_ratio`、`fine_ratio_in_agg`、`age_log1p`、`paste_index`。
- 将模型从单一 AdaBoost 路线切换为更稳健的 `RandomForestRegressor`（800棵树，受控随机种子）。
- 统一评估协议：10 折 `KFold(shuffle=True, random_state=42)`，指标为 `R²` 和 `RMSE`。

## 指标变化（10折CV）
| 指标 | v0 | v1 | 变化 |
|---|---:|---:|---:|
| R²_mean | 0.9090 | 0.9152 | +0.0062 |
| RMSE_mean | 4.9695 | 4.7991 | -0.1703 |

## 结果判定
- 判定：✅ **优于上一版（v0）**。
- 结论：机理约束特征对混凝土强度预测有效，且随机森林对特征尺度不敏感，作为迭代起点稳定。

## 材料机理反思
- `water-binder ratio` 与强度呈负相关趋势，加入比值后模型更容易学习配合比规律。
- `age_log1p` 能更贴近早期强度快速增长、后期趋缓的物理直觉。

## 下一版计划
- 引入更丰富交互项（龄期与胶凝材料耦合、矿物掺合料占比），并切换到更强的集成树（ExtraTrees）探索上限。
