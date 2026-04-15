# Final_Report

## 1. 任务概述

本项目围绕“混凝土抗压强度预测”完成了从文献学习到算法复现再到多轮创新优化的完整闭环，具体包括：

1. 对两篇核心论文进行文本解析与中文翻译；
2. 形成两份结构化学习笔记（核心问题、算法逻辑、复现指南）；
3. 复现第一篇论文（AdaBoost）核心流程并输出对比指标；
4. 按第二篇论文思路复现 ANN 与回归基线（w/c、w/b）对比代码；
5. 在 AdaBoost 上完成第一轮创新（特征工程 + 两阶段难样本重加权）；
6. 进一步引入新技术路径（Stacking 集成）完成第二轮迭代优化；


---

## 2. 论文核心摘要

### 2.1 论文A（2020, AdaBoost）

**题目**：Machine learning-based compressive strength prediction for concrete: An adaptive boosting approach  
**核心观点**：
- 将 AdaBoost 回归用于混凝土抗压强度预测；
- 输入为 8 个变量（7种配比组分 + 龄期），输出为抗压强度；
- 通过弱学习器加权集成形成强学习器，提高预测精度与鲁棒性；
- 对训练数据比例、弱学习器类型、变量敏感性与变量数量进行了系统分析。

**文献中典型结论**：
- 9:1 测试集可达到较高性能（文中报告接近 $R^2=0.982$）；
- 10 折交叉验证平均性能稳定（文中给出 $R^2$ 均值约 0.952）。

### 2.2 论文B（1998, ANN）

**题目**：Modeling of Strength of High-Performance Concrete Using Artificial Neural Networks  
**核心观点**：
- 高性能混凝土强度由多因素非线性耦合决定；
- 使用反向传播神经网络（ANN）建模比传统回归方法更准确；
- 构建 8 输入（配比+龄期）到 1 输出（抗压强度）的网络结构；
- 在多组划分下，ANN 的测试集 $R^2$ 普遍显著高于回归模型。

---

## 3. 第一篇论文复现（AdaBoost）

### 3.1 数据与工程实现

- 数据集：`data/Concrete_Data.xls`（1030条，9列）
- 复现脚本目录：`scripts/`
- 模块化划分：
  - `data_loader.py`：数据读取与清洗
  - `model_factory.py`：模型构建（AdaBoost / ANN / SVM）
  - `metrics_utils.py`：评估指标计算
  - `reproduce_pipeline.py`：基线复现实验主流程

### 3.2 复现中遇到的问题与修正

1. **`.xls` 读取失败（缺少 `xlrd`）**  
   - 现象：`pandas.read_excel` 报错 `No module named 'xlrd'`。
   - 处理：安装 `xlrd` 后恢复读取。

2. **`sklearn` 版本兼容问题（`mean_squared_error(..., squared=False)`）**  
   - 现象：旧版本不支持 `squared` 参数。
   - 处理：改为 $RMSE=\sqrt{MSE}$ 的兼容写法。

3. **重复样本处理策略**  
   - 为与原始论文数据规模一致，保留 1030 条样本（仅记录重复样本数量，不执行去重）。

### 3.3 基线实验设置

- 划分方式：训练/测试 = 9:1（随机种子 42）
- 模型：AdaBoost、ANN、SVM
- 指标：$R^2$、RMSE、MAPE、MAE
- 额外验证：AdaBoost 10 折交叉验证

### 3.4 基线结果（实测）

| 模型 | 测试集 $R^2$ | 测试集 RMSE | 测试集 MAPE(%) | 测试集 MAE |
|---|---:|---:|---:|---:|
| AdaBoost | 0.8929 | 5.3365 | 14.62 | 3.8632 |
| ANN | 0.9160 | 4.7273 | 11.91 | 3.3637 |
| SVM | 0.8713 | 5.8492 | 14.73 | 4.2099 |

**AdaBoost 10折交叉验证（均值±标准差）**：
- $R^2 = 0.9090 \pm 0.0196$
- RMSE $= 4.9695 \pm 0.5712$
- MAPE $= 13.35\% \pm 1.44\%$
- MAE $= 3.5085 \pm 0.3660$

> 说明：本地复现结果与论文报告存在差异，可能由随机划分、实现细节与库版本差异导致，但流程已可稳定运行并可重复。

---

## 4. 第二篇论文代码复现（ANN vs 回归）

为响应“按第一篇论文复现水准复现第二篇论文代码”，新增 `scripts/paper2_reproduction.py`，并输出：

- `doc/paper2_reproduction_results.json`
- `doc/Paper2_Reproduction_Report.md`

### 4.1 复现策略

1. **ANN 模型复现**：采用接近论文的 8-8-1 结构（工程化映射到 sklearn MLP）。
2. **回归基线复现**：
  $$
  f_c(t)=aX^b[c\ln(t)+d]
  $$
  其中 $X$ 分别取水灰比（w/c）与水胶比（w/b）。
3. **实验组织**：
  - S1~S4：近似来源分组（A/B/C/D，因公开数据无来源标签）；
  - R1~R4：随机 3/4 训练 + 1/4 测试。

### 4.2 核心结果（test $R^2$）

#### S1~S4（近似来源分组）
- ANN：0.8260 ~ 0.8665（均值 0.8436）
- 回归(w/c)：0.6184 ~ 0.6317（均值 0.6278）
- 回归(w/b)：0.7530 ~ 0.8130（均值 0.7796）

#### R1~R4（随机划分）
- ANN：0.8001 ~ 0.8389（均值 0.8232）
- 回归(w/c)：0.5926 ~ 0.6306（均值 0.6077）
- 回归(w/b)：0.7499 ~ 0.7913（均值 0.7726）

### 4.3 结论

- 成功复现第二篇论文“**ANN 优于回归**”的主结论方向；
- 成功复现“**水胶比回归优于水灰比回归**”的趋势。

---

## 5. 第一轮创新：改进 AdaBoost

### 5.1 方法名称

**特征工程 + 两阶段难样本重加权 AdaBoost**

### 5.2 理论动机

论文中 AdaBoost 的核心是“关注难样本”。本项目在该思想上做了工程化增强：

1. **机理相关特征增强**：加入水灰比、水胶比、减水剂胶凝材料比、龄期对数等派生特征，提升模型对非线性关系的表达能力；
2. **两阶段重加权**：
   - 阶段1训练初始模型得到残差；
   - 阶段2将残差较大的样本赋予更高权重再训练，强化模型对难样本的拟合。

若记阶段1残差为 $r_i=|y_i-\hat y_i|$，则样本权重可写为：
$$
w_i = \operatorname{clip}\left(1 + \frac{r_i}{\overline r + \epsilon},\ 1,\ 3\right)
$$
其中 $\overline r$ 为平均残差，$\epsilon$ 为数值稳定项。

### 5.3 量化结果（相对基线 AdaBoost）

在相同数据划分（同随机种子、同训练/测试集）下：

| 模型 | 测试集 $R^2$ | 测试集 RMSE | 测试集 MAPE(%) | 测试集 MAE | 训练耗时(s) |
|---|---:|---:|---:|---:|---:|
| 基线 AdaBoost | 0.8929 | 5.3365 | 14.62 | 3.8632 | 0.2749 |
| 创新 AdaBoost | 0.9199 | 4.6161 | 13.29 | 3.6085 | 0.5815 |

相对基线变化（测试集）：
- $R^2$：**+3.02%**
- RMSE：**下降 13.50%**
- MAPE：**下降 9.11%**
- MAE：**下降 6.59%**

### 5.4 分析

- 创新模型在四项测试指标上均优于基线 AdaBoost；
- 训练耗时增加（约 0.27s → 0.58s），说明精度提升是以一定计算开销为代价；
- 在课程任务目标下，该改进满足“提出新计算方式并完成量化对比”的要求。

---

## 6. 第二轮创新：新技术路径（Stacking）

根据展望部分继续迭代，新增 `scripts/new_techpath_model.py`，技术路径为：

**Stacking(HGB + ExtraTrees + SVR -> Ridge)**

并与第一轮优化 AdaBoost 做同分割对比。

### 6.1 结果对比（测试集）

| 模型 | $R^2$ | RMSE | MAPE(%) | MAE | 训练耗时(s) |
|---|---:|---:|---:|---:|---:|
| 优化 AdaBoost | 0.9199 | 4.6161 | 13.29 | 3.6085 | 0.6192 |
| 新技术路径 Stacking | 0.9497 | 3.6569 | 8.70 | 2.6396 | 6.0386 |

相对优化 AdaBoost 变化：
- $R^2$：**+3.24%**
- RMSE：**下降 20.78%**
- MAPE：**下降 34.55%**
- MAE：**下降 26.85%**

### 6.2 结论

- 新技术路径进一步显著提升了预测精度；
- 代价是训练耗时上升（约 0.62s → 6.04s）；
- 可在汇报中作为“展望落地迭代”的强支撑结果。

---

## 7. 结论与展望

### 7.1 结论

1. 已完成两篇论文的中文翻译与结构化知识提取；
2. 已实现并验证第一篇论文复现流程（数据加载、建模、评估、交叉验证）；
3. 已按课程要求补充完成第二篇论文代码复现（ANN vs 回归）；
4. 已完成两轮创新迭代：
  - 第一轮：改进 AdaBoost；
  - 第二轮：Stacking 新技术路径；
5. 两轮创新均获得可量化提升，第二轮提升更显著。

### 7.2 后续可做工作

- 进一步做超参数自动搜索（Bayesian Optimization / Optuna）；
- 引入更严格的外部验证集与分层时间切分；
- 对变量重要性与工程机理做更深入解释（SHAP / PDP）；
- 将“强度预测 + 配比优化”联动为完整设计决策工具。

---

## 8. 交付文件清单（关键）

- 文献翻译：
  - `paper/paper_1_zh.md`
  - `paper/paper_2_zh.md`
- 学习笔记：
  - `doc/Paper1_Study_Notes.md`
  - `doc/Paper2_Study_Notes.md`
- 复现代码：
  - `scripts/config.py`
  - `scripts/logger_utils.py`
  - `scripts/data_loader.py`
  - `scripts/metrics_utils.py`
  - `scripts/model_factory.py`
  - `scripts/reproduce_pipeline.py`
- 创新代码：
  - `scripts/optimized_model.py`
  - `scripts/new_techpath_model.py`
  - `scripts/paper2_reproduction.py`
- 实验结果：
  - `doc/baseline_results.json`
  - `doc/Baseline_Reproduction_Report.md`
  - `doc/optimization_results.json`
  - `doc/Optimization_Comparison_Report.md`
  - `doc/paper2_reproduction_results.json`
  - `doc/Paper2_Reproduction_Report.md`
  - `doc/new_techpath_results.json`
  - `doc/New_TechPath_Comparison_Report.md`
- 最终报告：
  - `doc/Final_Report.md`
