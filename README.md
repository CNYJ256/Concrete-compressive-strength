# 混凝土抗压强度预测：AdaBoost 复现与 ACDCB 方法

本仓库聚焦**混凝土抗压强度预测**任务，主线为：

1. 复现原论文（AdaBoost/ANN/SVM）；
2. 在相同数据集与可复现评估协议下，训练新模型；
3. 对比原论文方法与新模型的性能差异并输出图表与论文材料。

## 1. 项目背景

混凝土抗压强度受水泥、矿渣、粉煤灰、水、外加剂、骨料和龄期等因素共同影响，属于典型高维非线性回归问题。传统单模型方法在复杂分布下容易出现泛化受限。`ACDCB` 在原论文复现基线之上提出并实现了**龄期条件化双空间约束融合**（Age-Conditioned Dual-Space Constrained Blending, **ACDCB**），目标是在保持稳定性的同时进一步提升预测精度。

## 2. 仓库结构（主线）

- `data/`：原始数据集（`Concrete_Data.xls`）
- `scripts/`：公共模块（配置、数据加载、指标、模型工厂）
- `paper1/`：原论文复现脚本与说明
- `doc/`：复现与对比结果（JSON/Markdown 报告）
- `v9/`：主版本训练、推理、模型与指标
- `paper/`：论文写作与图表生成脚本

## 3. 依赖环境配置

- Python 3.9+
- 建议使用虚拟环境

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含核心依赖：`numpy`、`pandas`、`scipy`、`scikit-learn`、`xlrd`、`joblib`、`optuna`、`xgboost`、`lightgbm`、`matplotlib`。

## 4. 核心运行步骤

### 4.1 复现 paper1 基线

```bash
python paper1/scripts/reproduce_pipeline.py
```

输出文件：

- `doc/baseline_results.json`
- `doc/Baseline_Reproduction_Report.md`

### 4.2 训练 v9 模型

```bash
python v9/train.py
```

输出文件：

- `v9/model.joblib`
- `v9/metrics.json`

### 4.3 执行 v9 推理

```bash
python v9/predict.py
```

默认行为：无输入参数时直接读取原始数据 `data/Concrete_Data.xls`（全量样本）进行推理。  
输出文件：`v9/predictions.csv`

可选自定义输入：

```bash
python v9/predict.py your_input.csv your_output.csv
```

### 4.4 生成论文图表

```bash
python paper/generate_figures.py
```

输出目录：`paper/figures/`

## 5. 新模型相对原论文的主要改进

相对于原文中的单模型路径（AdaBoost/ANN/SVM），新模型的核心改进包括：

1. **年龄感知分段融合（Age-aware Piecewise Blend）**  
  对 `age <= 28` 与 `age > 28` 两个区间分别学习融合权重，降低不同龄期分布差异对单一模型的影响。

  该训练范式在 `v9` 中学术命名为：**ACDCB（Age-Conditioned Dual-Space Constrained Blending）**。

2. **多基学习器协同**  
  采用多模型加权组合，而非单一回归器，提升对复杂非线性关系的覆盖能力。

3. **统一可复现实验协议**  
  使用固定随机种子与 10 折 KFold（`shuffle=True, random_state=42`），并将关键指标落盘，保证结果可复核。

## 6. 模型结果简要展示（真实运行数据）

数据来源：`doc/baseline_results.json` 与 `v9/metrics.json`

### 6.1 paper1 复现结果

| 模型 | 测试集 R² | 测试集 RMSE |
|---|---:|---:|
| AdaBoost | 0.8929 | 5.3365 |
| ANN | 0.9160 | 4.7273 |
| SVM | 0.8713 | 5.8492 |

`paper1` AdaBoost 的 10 折结果：

- R²_mean = **0.9090**
- RMSE_mean = **4.9695**
- MAE_mean = **3.5085**
- MAPE_mean = **13.3513%**

### 6.2 v9 结果（10 折）

- R²_mean = **0.9488**
- RMSE_mean = **3.6996**
- MAE_mean = **2.3522**
- MAPE_mean = **8.4878%**

### 6.3 v9 相对 paper1（AdaBoost 10 折）提升

- R² 提升：**+0.0398**
- RMSE 降低：**-1.2699 MPa**（约 **25.55%**）
- MAE 降低：**-1.1564 MPa**
- MAPE 降低：**-4.8635%**

## 7. 数据与引用

- 数据集：UCI Concrete Compressive Strength（1030 样本、8 输入特征、1 输出）
- 推荐引用：Yeh, I. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67
- 数据许可：CC BY 4.0

## 8. 许可证

- 代码许可：MIT License（见 `LICENSE`）