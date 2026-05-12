# ACDCB：龄期分层混凝土双空间约束融合抗压强度预测

本项目是混凝土抗压强度预测机器学习框架。该框架结合双空间特征工程（主要物理导出特征 + 锚点降维特征）、分段龄期分层与约束集成优化，以及多模型 GBDT 模型池（XGBoost、LightGBM、HistGradientBoosting）。本工作的核心发现是：仅使用原始特征训练的简洁 XGBoost 模型即可达到或超过完整 ACDCB 流水线的性能，这对“架构复杂度必然提升该领域预测精度”的假设提出了挑战。

## Python 环境

- **Python**：3.10+
- **关键包**：numpy、pandas、scikit-learn、xgboost、lightgbm、scipy、optuna、matplotlib、seaborn、xlrd
- **Conda 环境**：`torch`

安装依赖包：

```bash
pip install numpy pandas scikit-learn xgboost lightgbm scipy optuna matplotlib seaborn xlrd
```

## 数据来源

### UCI 混凝土抗压强度数据集

- **来源**：UCI Machine Learning Repository
- **参考文献**：Yeh, I-Cheng. "Modeling of strength of high performance concrete using artificial neural networks." Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)
- **样本数**：1,030
- **特征**：8 个数值输入变量（cement、slag、fly ash、water、superplasticizer、coarse aggregate、fine aggregate、age）
- **目标**：混凝土抗压强度（MPa）
- **文件**：`data/Concrete_Data.xls`

### 墨西哥数据集

- **来源**：ConcreteXAI: A multivariate dataset for concrete strength prediction via deep-learning-based methods
- **样本数**：4,420
- **数值特征（6）**：Design_F'c、Curing_age、Er（电阻率）、UPV（超声波脉冲速度）、Ts（抗拉强度）、Fs（抗折强度）
- **类别特征（4）**：Type_of_cement、Brand、Additives、Type_of_aggregates
- **目标**：Cs（抗压强度，MPa）
- **文件**：`data/Data.csv`
- **备注**：Ts 与 Fs 缺失较多（分别为 21.7% 与 60.2%）。策略 A 删除这些列；策略 B 使用中位数填补。

## 快速开始

所有命令请在项目根目录执行。

### 1. 基线复现

```bash
# Feng 2020 - AdaBoost + 决策树基学习器
python src/concrete_compressive_strength/reproduction/paper1_reproduce.py

# Yeh 1998 - 人工神经网络
python src/concrete_compressive_strength/reproduction/paper2_reproduce.py
```

### 2. ACDCB 训练与推断

```bash
# 训练完整 ACDCB 模型
python scripts/train/train_acdcb.py

# 生成预测结果
python scripts/eval/predict_acdcb.py
```

### 3. 消融实验

```bash
# UCI 数据集消融（V0-V5 + OLS 无约束）
python scripts/eval/ablation_acdcb_v2.py

# 墨西哥数据集全量消融（包含 Optuna HPO）
python scripts/new_dataset/run_ablation_new_data.py --strategy A --trials 150

# De-UPV 消融（Phase 1.1，P0）
python scripts/new_dataset/run_deupv_ablation.py --trials 100

# 细化特征空间消融（Phase 2.1，P0）
python scripts/new_dataset/run_refined_feature_ablation.py --trials 50
```

### 4. 补充实验

```bash
# 龄期阈值扫描（tau 在 3-180 天范围）
python scripts/eval/threshold_scan.py

# VIF + LASSO 特征选择验证
python scripts/eval/feature_selection_validation.py

# 异质模型池测试（MLP + GPR）
python scripts/eval/heterogeneous_pool_test.py

# Sigmoid 软加权 vs 硬阈值
python scripts/new_dataset/soft_weighting.py

# SHAP 特征重要性分析
python scripts/eval/shap_analysis.py
```

### 5. 图表生成

```bash
# 新数据集的 SCI 级图（9 面板）
python scripts/presentation/generate_newdata_figures.py

# 仅 UCI 的高分辨率图
python scripts/presentation/generate_highres_figures.py
```

## 关键结果汇总

### UCI 数据集（N=1,030，10 折交叉验证）

| 变体 | 说明 | R2 | RMSE (MPa) |
|---|---|---|---|
| V0 | AdaBoost 基线 | 0.9142 | 4.80 |
| V1 | 主空间 + 3 个 GBDT + 全局融合 | 0.9475 | 3.73 |
| V2 | 双空间 + 4 个模型 + 全局融合 | 0.9487 | 3.70 |
| V3 | ACDCB 完整版（双空间 + 分段） | 0.9488 | 3.70 |
| V4 | 原始特征 + 分段（无特征工程） | 0.9530 | 3.54 |
| V5 | OLS 无约束堆叠 | 0.9488 | 3.70 |

### 墨西哥数据集（N=4,420，10 折交叉验证，策略 A）

| 变体 | 说明 | R2 | RMSE (MPa) |
|---|---|---|---|
| V0 | AdaBoost 基线 | 0.9936 | 0.765 |
| V1 | 主空间 + 3 个 GBDT + 全局融合 | 0.9953 | 0.653 |
| V2 | 双空间 + 4 个模型 + 全局融合 | 0.9954 | 0.647 |
| V3 | ACDCB 完整版（双空间 + 分段） | 0.9954 | 0.647 |
| V4 | 原始特征 + 分段（无特征工程） | 0.9953 | 0.650 |
| V5 | OLS 无约束堆叠 | 0.9954 | 0.647 |

### 主要结论

1. **模型升级是主要贡献来源**：从 AdaBoost 切换到现代 GBDT 模型几乎带来全部性能提升。UCI 上 dR2 = +0.0333；墨西哥数据集上 dR2 = +0.0018。

2. **特征工程贡献极小**：物理导出特征带来的提升最多仅 dR2 = +0.0001（墨西哥），在 UCI 上反而有负效应（dR2 = -0.0042，V3 vs V4）。

3. **双空间架构与龄期分层贡献接近 0**：引入锚点特征空间、分段龄期融合与约束优化的综合贡献在两个数据集上均低于 0.002 R2。

4. **原始特征上的简洁 XGBoost 构成性能上限**：在两个数据集上，单个 GBDT 模型即可达到或超过完整 ACDCB 集成性能。

5. **模型间相关性超过 0.996**：XGBoost、LightGBM 与 HGB 的预测高度相关，集成融合几乎无法挖掘增益。

6. **跨数据集验证证实稳健性**：各组件贡献的模式在 UCI（N=1,030）与墨西哥数据集（N=4,420）上几乎一致。

## 引用

- Feng, D. C. et al. "Machine learning-based compressive strength prediction for concrete: an adaptive boosting approach." Construction and Building Materials, 2020.
- Yeh, I-Cheng. "Modeling of strength of high performance concrete using artificial neural networks." Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808, 1998..
- J. A. Guzmán-Torres, F. J. Domínguez-Mota, E. M. Alonso-Guzmán, G. Tinoco-Guerrero, and W. Martínez-Molina, "ConcreteXAI: A multivariate dataset for concrete strength prediction via deep-learning-based methods," Data in Brief, vol. 53, p. 110218, Apr. 2024, doi: 10.1016/j.dib.2024.110218.

## 致谢

感谢各位大佬攥写的论文参考。
感谢各位大佬提供的数据集。
感谢各位大佬提供的代码实现细节。
...

## 许可证

本项目采用 MIT 许可证。
