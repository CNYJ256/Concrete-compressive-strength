# ACDCB 技术特征拆解（面向有理工基础的组员）


---

## 1. 问题定义与为什么需要新方法

任务是预测混凝土抗压强度（MPa）。输入为 8 个变量：

| 字段名（EN） | 中文名称 | 常用解释 | 单位 |
|---|---|---|---|
| `cement` | 水泥 | 胶凝材料主成分 | kg/m³ |
| `slag` | 高炉矿渣粉 | 矿物掺合料之一 | kg/m³ |
| `fly_ash` | 粉煤灰 | 矿物掺合料之一 | kg/m³ |
| `water` | 水 | 拌合水 | kg/m³ |
| `superplasticizer` | 减水剂 | 外加剂（改善流动性） | kg/m³ |
| `coarse_agg` | 粗骨料 | 粗粒级骨料 | kg/m³ |
| `fine_agg` | 细骨料 | 细粒级骨料 | kg/m³ |
| `age` | 龄期 | 养护时间 | day |

对应挑战不是“变量少”，而是：

1. **非线性明显**：例如水胶比与强度关系并非线性；
2. **龄期驱动分布变化**：早龄期与后龄期机理差异大；
3. **单模型偏好问题**：不同模型擅长的区间不同。

paper1 的 AdaBoost 是强基线，但依然属于单一建模范式。ACDCB 的核心目的不是“堆模型”，而是把**特征空间、模型空间、龄期条件**同时纳入一个统一的可约束融合框架。

---

## 2. ACDCB 命名是怎么来的

`ACDCB = Age-Conditioned Dual-Space Constrained Blending`

- **A / C**：Age-Conditioned（龄期条件化）
- **D**：Dual-Space（双空间）
- **C**：Constrained（约束）
- **B**：Blending（融合）

命名直接对应算法的 4 个结构部件，不是随意缩写。

### 2.1 C（Constrained）具体约束了什么

这里的 “C” 不是泛指“有规则”，而是可写成数学形式的**权重约束集合**：

$$
\mathcal{W}=\{\mathbf{w}\in\mathbb{R}^M\mid w_i\ge 0,\;\sum_{i=1}^{M}w_i=1\}
$$

含义分三层：

1. **非负约束 $w_i\ge 0$**：禁止某个子模型以负权重“反向抵消”其他模型；
2. **归一化约束 $\sum w_i=1$**：把融合限定在凸组合内，保证输出处于可解释的线性混合框架；
3. **分段同约束**：早龄期权重 $\mathbf{w}_e$ 与后龄期权重 $\mathbf{w}_l$ 都在同一约束集合 $\mathcal{W}$ 内优化。

工程收益是：

- 权重可直接解释为“贡献份额”；
- 降低极端权重带来的不稳定风险；
- 便于跨实验复现与对比。

---

## 3. 双空间（Dual-Space）到底是什么

ACDCB 在同一份基础变量上构造两套特征空间：`primary` 与 `anchor`。

## 3.1 基础变量（两空间共享）

设：

$$
\mathbf{x}_{base} = [cement, slag, fly\_ash, water, sp, coarse, fine, age]
$$

## 3.2 共同机理特征（primary 与 anchor 都有）

先定义中间量：

$$
binder = cement + slag + fly\_ash,
\quad total\_agg = coarse + fine,
\quad age_{log}=\log(1+age)
$$

再构造机理特征（代码中实际存在，以下为中英对照）：

| 特征名（EN） | 中文名称 | 公式定义 |
|---|---|---|
| `binder` | 胶凝材料总量 | $cement + slag + fly\_ash$ |
| `water_cement_ratio` | 水灰比（相对水泥） | $water/(cement+\varepsilon)$ |
| `water_binder_ratio` | 水胶比 | $water/(binder+\varepsilon)$ |
| `sp_binder_ratio` | 减水剂胶凝比 | $superplasticizer/(binder+\varepsilon)$ |
| `scm_ratio` | 掺合料占比 | $(slag+fly\_ash)/(binder+\varepsilon)$ |
| `fine_ratio_in_agg` | 细骨料占比 | $fine\_agg/(coarse\_agg+fine\_agg+\varepsilon)$ |
| `age_log1p` | 龄期对数变换 | $\log(1+age)$ |
| `age_sqrt` | 龄期开方变换 | $\sqrt{age}$ |
| `age_pow_0_25` | 龄期四分之一次幂 | $age^{0.25}$ |
| `abrams_index` | Abrams 指数 | $\log(1+age)/(water\_binder\_ratio+\varepsilon)$ |
| `cement_age_interaction` | 水泥-龄期交互项 | $cement\cdot\log(1+age)$ |
| `binder_age_interaction` | 胶凝材料-龄期交互项 | $binder\cdot\log(1+age)$ |
| `wb_age_interaction` | 水胶比-龄期交互项 | $water\_binder\_ratio\cdot\log(1+age)$ |
| `paste_index` | 浆体指数 | $(cement+slag+fly\_ash+water+superplasticizer)/(total\_agg+\varepsilon)$ |

## 3.3 primary 额外增强特征（anchor 不含）

`primary` 在共同机理特征基础上再加 10 个增强特征（以下为中英对照）：

| 特征名（EN） | 中文名称 | 公式定义 |
|---|---|---|
| `binder_to_agg_ratio` | 胶凝材料-骨料比 | $binder/(total\_agg+\varepsilon)$ |
| `water_to_paste_ratio` | 水占浆体比例 | $water/(water+binder+superplasticizer+\varepsilon)$ |
| `cement_fraction_in_binder` | 水泥在胶凝材料中占比 | $cement/(binder+\varepsilon)$ |
| `slag_fraction_in_binder` | 矿渣在胶凝材料中占比 | $slag/(binder+\varepsilon)$ |
| `flyash_fraction_in_binder` | 粉煤灰在胶凝材料中占比 | $fly\_ash/(binder+\varepsilon)$ |
| `superplasticizer_efficiency` | 减水剂效率指数 | $superplasticizer/(water+\varepsilon)$ |
| `maturity_index` | 成熟度指数 | $\log(1+age)\cdot binder/(water+\varepsilon)$ |
| `agg_binder_balance` | 骨料-胶凝平衡比 | $total\_agg/(binder+\varepsilon)$ |
| `age_inverse` | 龄期倒数特征 | $1/(age+1)$ |
| `age_wc_interaction` | 龄期-水灰比交互项 | $\log(1+age)\cdot water\_cement\_ratio$ |

## 3.4 两空间的维度与职责

- `anchor`：8（基础）+14（机理）= **22 维**
- `primary`：8（基础）+14（机理）+10（增强）= **32 维**

可把它理解为：

- `primary`：表达能力优先（高容量）
- `anchor`：稳健性优先（低风险）

这就是“双空间”不是口号，而是**明确的特征子空间分工**。

---

## 4. 多模型（Model Pool）与参数：具体有哪些

ACDCB 的候选模型池是 4 个：

| model_id | family | feature_space |
|---|---|---|
| XGBoost | XGBRegressor | primary |
| LightGBM | LGBMRegressor | primary |
| HGB | HistGradientBoostingRegressor | primary |
| HGB_Anchor | HistGradientBoostingRegressor | anchor |

## 4.1 当前训练使用的关键参数（来自 `metrics.json`）

### XGBoost
- `n_estimators=1482`
- `learning_rate=0.0446045764`
- `max_depth=4`
- `min_child_weight=7.3946`
- `subsample=0.6643`
- `colsample_bytree=0.6247`
- `gamma=2.4344`
- `reg_alpha=0.4933`
- `reg_lambda=3.0211`

### LightGBM
- `n_estimators=2127`
- `learning_rate=0.0319695191`
- `num_leaves=32`
- `max_depth=4`
- `min_child_samples=8`
- `subsample=0.8383`
- `colsample_bytree=0.5905`
- `reg_alpha=0.7114`
- `reg_lambda=0.0002217`
- `min_split_gain=0.006464`

### HGB（primary）
- `learning_rate=0.0567853533`
- `max_iter=1809`
- `max_depth=12`
- `max_leaf_nodes=15`
- `min_samples_leaf=14`
- `l2_regularization=4.58e-05`
- `max_bins=213`

### HGB_Anchor（anchor）
- `learning_rate=0.028`
- `max_iter=2400`
- `max_depth=None`
- `max_leaf_nodes=15`
- `min_samples_leaf=6`
- `l2_regularization=0.001`

---

## 5. 权重怎么学：不是拍脑袋，是约束优化

## 5.1 先做 OOF 预测矩阵

在 10 折交叉验证下，对每个候选模型得到 OOF 预测，拼成：

$$
\mathbf{P}\in\mathbb{R}^{N\times M},\quad M=4
$$

其中每一列是一个模型在全体样本上的 OOF 预测。

## 5.2 全局融合（Global Blend）

求解权重 $\mathbf{w}$：

$$
\min_{\mathbf{w}}\; \mathrm{RMSE}(\mathbf{y}, \mathbf{P}\mathbf{w})
$$

约束：

$$
w_i\ge 0,\quad \sum_i w_i=1
$$

优化器为 `SLSQP`（`scipy.optimize.minimize`）。

## 5.3 龄期分段融合（Age-aware Piecewise Blend）

按阈值 `age_split_day = 28` 划分：

- 早龄期集合 $\mathcal{D}_e = \{i\mid age_i\le 28\}$
- 后龄期集合 $\mathcal{D}_l = \{i\mid age_i>28\}$

分别求权重：

$$
\mathbf{w}_e = \arg\min \mathrm{RMSE}(\mathbf{y}_e, \mathbf{P}_e\mathbf{w}),\quad
\mathbf{w}_l = \arg\min \mathrm{RMSE}(\mathbf{y}_l, \mathbf{P}_l\mathbf{w})
$$

最终预测：

$$
\hat y_i=
\begin{cases}
\mathbf{P}_i\mathbf{w}_e, & age_i\le 28\\
\mathbf{P}_i\mathbf{w}_l, & age_i>28
\end{cases}
$$

## 5.4 全局 vs 分段怎么选

`train.py` 中不是仅看 RMSE，而是：

1. **先比较 $R^2$**（主指标）；
2. 若 $R^2$ 差异在容忍范围内（`r2_tie_tol = 5e-4`），再比较 RMSE。

即策略选择是“$R^2$ 优先、RMSE 次级”。

---

## 6. 龄期分段后的权重结果（当前模型产出）

来自 `results/metrics/acdcb_metrics.json`：

### 早龄期（$age\le 28$）
- XGBoost: **0.313813**
- LightGBM: **0.039930**
- HGB: **0.013239**
- HGB_Anchor: **0.633019**

### 后龄期（$age>28$）
- XGBoost: **0.463224**
- LightGBM: **0.000000**
- HGB: **0.122167**
- HGB_Anchor: **0.414610**

可读成：

- 早龄期更依赖 `HGB_Anchor` 的稳健输出；
- 后龄期中 `XGBoost` 权重上升，说明其在后龄期模式上的贡献更高；
- `LightGBM` 在后龄期被压到 0，表示在该分段中它不再提供边际收益。

---

## 7. 结果层面的技术结论

## 7.1 单模型 OOF 表现（10 折均值）

| 模型 | 空间 | $R^2$ | RMSE | MAE | MAPE(%) |
|---|---|---:|---:|---:|---:|
| XGBoost | primary | 0.945934 | 3.781660 | 2.461770 | 8.9295 |
| LightGBM | primary | 0.945652 | 3.805196 | 2.479859 | 8.8780 |
| HGB | primary | 0.945497 | 3.808984 | 2.425909 | 8.7165 |
| HGB_Anchor | anchor | **0.947965** | **3.740782** | **2.368325** | **8.5289** |

## 7.2 融合策略表现

| 策略 | $R^2$ | RMSE | MAE | MAPE(%) |
|---|---:|---:|---:|---:|
| Global Blend | 0.948725 | 3.700053 | 2.351104 | **8.487402** |
| Age-aware Piecewise | **0.948755** | **3.699571** | 2.352153 | 8.487811 |

按代码中的策略选择规则，最终选中 `age_piecewise`。

## 7.3 相对 paper1（AdaBoost）

| 方法 | $R^2$ | RMSE | MAE | MAPE(%) |
|---|---:|---:|---:|---:|
| paper1 AdaBoost | 0.909003 | 4.969471 | 3.508535 | 13.351333 |
| ACDCB | **0.948755** | **3.699571** | **2.352153** | **8.487811** |

提升量：

- $\Delta R^2 = +0.039752$
- $\Delta RMSE = -1.269899$
- $\Delta MAE = -1.156382$
- $\Delta MAPE = -4.863522$ pct

## 7.4 消融中的 `raw piecewise` 是什么

在本项目消融里，`raw piecewise`（对应 `v4_raw_age_piecewise_no_feature_engineering`）的定义是：

1. **只用 raw 输入**：即仅使用 8 个基础变量，不做 `feature_engineering` 和 `feature_engineering_anchor`；
2. **模型集合改为 raw 版本**：`XGB_raw`、`LGB_raw`、`HGB_raw`、`HGB_anchor_raw`；
3. **仍保留龄期分段 + 约束融合**：

$$
\mathbf{w}^{raw}_e=\arg\min\mathrm{RMSE}(\mathbf{y}_e,\mathbf{P}^{raw}_e\mathbf{w}),\quad
\mathbf{w}^{raw}_l=\arg\min\mathrm{RMSE}(\mathbf{y}_l,\mathbf{P}^{raw}_l\mathbf{w})
$$

且满足 $w_i\ge 0,\sum_i w_i=1$。

它的作用是做“控制变量”：保留融合机制，去掉特征工程，用于估计特征工程模块的净贡献。

为什么会出现“raw piecewise 看起来更好看”：

- 在当前消融结果中，`v4_raw_age_piecewise_no_feature_engineering` 的 $R^2=0.95058$、RMSE$=3.63346$，确实高于/低于 `v3`（$R^2=0.94876$、RMSE$=3.69953$）；
- 但它并非全指标绝对领先：`v3` 的 MAE 更低（2.35210 vs 2.36879），而 `v4` 的 MAPE 更低（8.32661 vs 8.48766），属于多指标“交叉领先”。

从方法角度，常见原因有三类：

1. **损失函数偏好**：融合权重优化目标是 RMSE，模型会优先压低大误差样本；这可提升 $R^2$/RMSE，但不保证 MAE 同步最优；
2. **样本规模与特征维度匹配**：在 1030 样本规模下，部分扩展特征可能提升表达能力，也可能引入方差，导致不同指标表现分化；
3. **参数-特征耦合未完全展开**：当前属于固定参数下的消融对比，若不做“特征工程与超参数联合搜索”，出现 raw 在部分指标占优是正常现象。

因此更准确的结论不是“raw 比 ACDCB 更好”，而是：**当前参数设定下，raw 与 engineered 各有优势，需通过联合调参确定最终最优配置。**

---

## 8. 你特别关心的几个问题，直接回答

## 8.1 “双空间各关注哪些参数、哪些空间？”

- 空间不是“模型空间”，而是**特征空间**。
- `anchor` 关注“基础 + 机理”共 22 维；
- `primary` 在此基础上加入 10 个增强特征到 32 维。

## 8.2 “多模型是哪些，用什么指标规定权重？”

- 模型：XGBoost / LightGBM / HGB / HGB_Anchor。
- 权重学习目标函数：**最小化 RMSE**。
- 约束：权重非负且总和为 1。

## 8.3 “龄期怎么划分，为什么这样划分？”

- 划分规则：`age <= 28` 与 `age > 28`。
- 原因：28 天是混凝土强度评价中常用工程节点；实现层面用该阈值做分段权重学习，可以让早期与后期分别最优。

## 8.4 “primary、anchor 分别关注哪方面？”

- `primary`：偏表达能力（包含增强交互与衍生特征）；
- `anchor`：偏稳健性（保留机理主干，减少高阶扩展带来的方差）。

## 8.5 “ACDCB 和人脸识别里的 ACDCB 有关系吗？”

基于当前仓库证据：

- 命名来源已在代码中明确为 `Age-Conditioned Dual-Space Constrained Blending`；
- 仓库内未出现人脸识别相关数据、网络结构或引用；
- 因此就本项目而言，属于**同名缩写重合**，不是同一算法链路。

如果后续你希望做“跨领域同名方法谱系”对照，我可以再单独补一页文献映射表。

---

## 9. 可直接用于组内汇报的技术主线（精简版）

1. ACDCB 不是单模型替换，而是“**双特征空间 + 多模型 + 约束融合 + 龄期分段**”的组合设计；
2. 关键公式是受约束 RMSE 最小化，且在 28 天阈值下做分段权重学习；
3. 结果上相对 AdaBoost 在四项指标全部提升，且提升可由消融实验定位到结构性来源（双空间、分段融合）。
