# CHANGELOG - v2

## 版本定位
- 版本目标：在 `v1` 的基础上增强特征表达能力，并验证更高方差树集成是否带来泛化收益。
- 本版策略：`Enhanced interaction features + ExtraTrees`。

## 核心改动
- 新增/强化特征：`scm_ratio`、`age_sqrt`、`binder_age_interaction`、`wb_age_interaction`。
- 模型升级为 `ExtraTreesRegressor`（1200棵树，`max_features='sqrt'`），提升非线性交互捕捉能力。
- 保持与 v1 完全一致的 10 折评估协议，保证横向可比。

## 指标变化（10折CV）
| 指标 | v1 | v2 | 变化 |
|---|---:|---:|---:|
| R²_mean | 0.9152 | 0.9254 | +0.0102 |
| RMSE_mean | 4.7991 | 4.4776 | -0.3215 |

## 结果判定
- 判定：✅ **优于上一版（v1）**。
- 结论：交互特征与高随机性树集成协同有效，明显降低误差。

## 材料机理反思
- 掺合料占比（`scm_ratio`）帮助区分不同胶凝体系对中后期强度的影响。
- 龄期交互特征可显式表达“同配比在不同龄期强度演化速度不同”的规律。

## 下一版计划
- 进入 Boosting 路线，尝试更细粒度残差学习（HistGradientBoosting），继续提高拟合精度与稳定性。
