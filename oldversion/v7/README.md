# v7 算法详解（最新版）

## 1. 版本定位

`v7` 对应当前最新版强度预测训练脚本，核心思想是：

- 以 `HistGradientBoostingRegressor` 为主模型；
- 通过 **多随机种子 + 随机搜索** 扩展超参数探索覆盖率；
- 在统一 `10-fold KFold(shuffle=True, random_state=42)` 协议下选择最优参数组合；
- 以 `R²` 为主优化目标，`RMSE` 作为次级判据。

---

## 2. 数据与特征

### 2.1 原始输入

使用 8 个基础变量：

- `cement`
- `slag`
- `fly_ash`
- `water`
- `superplasticizer`
- `coarse_agg`
- `fine_agg`
- `age`

目标变量：`strength`。

### 2.2 机理增强特征

在基础变量上构造与材料机理相关的派生特征：

- 胶凝材料总量：`binder`
- 比值特征：`water_cement_ratio`、`water_binder_ratio`、`sp_binder_ratio`、`scm_ratio`、`fine_ratio_in_agg`
- 龄期非线性：`age_log1p`、`age_sqrt`、`age_pow_0_25`
- Abrams 相关：`abrams_index`
- 交互项：`cement_age_interaction`、`binder_age_interaction`、`wb_age_interaction`
- 浆体-骨料平衡：`paste_index`

这些特征用于提升模型对“配比-龄期-强度”耦合关系的表达能力。

---

## 3. 训练策略

### 3.1 模型主体

- 模型：`HistGradientBoostingRegressor`
- 损失：`squared_error`
- 早停：关闭（`early_stopping=False`），由固定迭代轮次控制收敛。

### 3.2 参数搜索空间

随机搜索参数包括：

- `learning_rate`
- `max_iter`
- `max_depth`
- `max_leaf_nodes`
- `min_samples_leaf`
- `l2_regularization`

每个 seed 采样 `n_iter=80` 组超参数。

### 3.3 多随机种子强化搜索

脚本对多个 seed（如 `11, 42, 77, 123`）分别执行随机搜索，记录每个 seed 的最优结果：

1. 每个 seed 独立进行 `RandomizedSearchCV`；
2. 提取该 seed 下的最佳参数与 10 折指标；
3. 全局比较不同 seed 的最优结果，选取最终模型。

> 这样做的目的是降低随机采样偏差，提升找到高质量参数组合的概率。

---

## 4. 评估与择优规则

### 4.1 统一评估协议

- 交叉验证：`KFold(n_splits=10, shuffle=True, random_state=42)`
- 指标：
  - 主指标：`R²`（越大越好）
  - 辅指标：`RMSE`（越小越好）

### 4.2 候选比较规则

候选模型比较顺序：

1. 先比较 `R²_mean`；
2. 若 `R²_mean` 持平，再比较 `RMSE_mean`（更小者更优）。

### 4.3 与上一版对比

最终会生成：

- `cv_10fold`：当前版本平均性能与方差；
- `compare_to_prev`：相对上一版本的 `R²` 增益与 `RMSE` 降幅。

---

## 5. 输出产物

训练完成后会保存：

- `model.joblib`：模型与特征列定义（可直接推理复用）
- `metrics.json`：详细指标、最优参数、各 seed 搜索结果
- `predictions.csv`（由推理脚本生成）：示例或外部输入预测结果

---

## 6. 方法优缺点

### 优点

- 与单次随机搜索相比，鲁棒性更强；
- 在高基线阶段仍可获得小幅稳定增益；
- 工程实现简单，易于复现与扩展。

### 局限

- 训练耗时随 seed 数与 `n_iter` 增加而上升；
- 当前仍依赖单一模型族（HGB），若继续冲刺上限可引入外部 boosting 库（LightGBM/XGBoost/CatBoost）对照。

---

## 7. 后续可扩展方向

- 使用 Repeated K-Fold 提升评估稳定性；
- 引入 SHAP 做机理可解释性分析；
- 与贝叶斯优化（Optuna）进行效率对比；
- 增加跨数据集迁移验证，评估泛化能力。
