# 🏗️ ACDCB：龄期条件化双空间约束融合混凝土抗压强度预测框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

> **核心发现**：仅使用原始特征训练的简洁 XGBoost 模型的性能可达到或超过复杂的完整 ACDCB 架构，对"架构复杂度必然提升精度"的假设提出了挑战。

---

## 📑 快速导航

- [项目概览](#项目概览)
- [核心贡献](#核心贡献)
- [项目结构](#项目结构)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
  - [基线复现](#1基线复现)
  - [ACDCB 训练](#2acdcb-完整训练与推理)
  - [消融实验](#3消融实验)
  - [补充实验](#4补充实验)
  - [图表生成](#5图表生成)
- [数据集说明](#数据集说明)
- [技术方案](#技术方案)
- [主要结果](#主要结果)
- [论文与引用](#论文与引用)
- [常见问题](#常见问题)
- [许可证与致谢](#许可证与致谢)

---

## 📌 项目概览

### 问题定义

混凝土抗压强度预测是土木工程中的关键任务，但存在以下挑战：

| 挑战 | 说明 |
|------|------|
| **非线性映射** | 水胶比、龄期与强度的复杂非线性耦合 |
| **龄期异质性** | 早龄期（≤28天）与后龄期（>28天）的物理机理差异显著 |
| **模型多样性** | 不同机器学习模型对特征空间的拟合能力不一致 |
| **特征工程稀缺** | 混凝土材料科学领域特征挖掘的理论支撑有限 |

### ACDCB 方案的四层设计

**ACDCB**（Age-Conditioned Dual-Space Constrained Blending）通过以下方案应对：

1. **龄期条件化** — 按龄期阈值（通常 28 天）分段，早龄期与后龄期采用独立的融合权重
2. **双空间特征工程** — 构造 22 维（Anchor）和 32 维（Primary）两套特征空间，分工处理稳健性与表达能力
3. **约束融合优化** — 对多个 GBDT 子模型的融合权重施加非负与和为 1 的约束，保证可解释性
4. **多模型池策略** — 整合 XGBoost、LightGBM、HistGradientBoosting 及其锚点版本

---

## 🎯 核心贡献

### 关键发现

| # | 发现 | 数据支撑 | 含义 |
|---|------|--------|------|
| **1** | 原始特征 XGBoost (V4) 性能相当 | UCI: R²=0.953, RMSE=3.54 | 完整架构的增益 < 0.004 R² |
| **2** | 特征工程贡献极小 | UCI: ΔR² = -0.0042 (V3→V4) | 双空间特征反而有负效应 |
| **3** | 模型升级为主驱动 | UCI: ΔR² = +0.0333 (V0→V1) | 从 AdaBoost 升级到 GBDT 是关键 |
| **4** | 架构复杂度不必然优化 | 墨西哥: 三大模型相关性 > 0.996 | 集成融合增益有限 |

### 研究意义

✅ **挑战"特征为王"的传统假设** — 在该数据规模与特征维度下，模型选择的重要性远超特征工程  
✅ **提供可度量的消融框架** — 系统化地评估双空间、龄期分层、约束融合各组件的实际贡献  
✅ **为工程应用提供循证基准** — 指导混凝土强度预测模型的选型与部署策略  

---

## 📁 项目结构详解

```
ACDCB/
│
├── 📄 README.md                    # 项目说明（本文件）
├── 📄 LICENSE                      # MIT 许可证
├── 📄 requirements.txt             # Python 依赖列表
│
├── 🔬 src/                         # ⭐ 核心算法库（可复用）
│   └── concrete_compressive_strength/
│       ├── __init__.py
│       ├── core.py                 # ⭐ 特征工程 + 模型构建核心
│       ├── reproduction/           # 论文基线复现脚本
│       │   ├── paper1_reproduce.py # Feng 2020 (AdaBoost)
│       │   ├── paper2_reproduce.py # Yeh 1998 (ANN)
│       │   └── hyperparam_search.py
│       └── plotting/               # 图表生成
│           ├── generate_acdcb_figures.py
│           └── generate_comparison_figures.py
│
├── 🚀 scripts/                     # ⭐ 执行脚本库（直接可运行）
│   ├── config.py                   # ⭐ 全局配置 + 列名映射
│   ├── data_loader.py              # 数据加载工具
│   ├── model_factory.py            # 基线模型工厂
│   ├── metrics_utils.py            # 指标计算（RMSE, MAE, R²）
│   ├── logger_utils.py             # 日志工具
│   │
│   ├── 📊 train/                   # 训练脚本
│   │   ├── train_acdcb.py          # ⭐ ACDCB 主训练流程
│   │   └── hyperparam_search_raw.py # Optuna 超参搜索
│   │
│   ├── 📊 eval/                    # 评估脚本 (7 项)
│   │   ├── predict_acdcb.py        # 推理与预测生成
│   │   ├── ablation_acdcb_v2.py    # ⭐ UCI 消融 (V0-V5)
│   │   ├── threshold_scan.py       # 龄期阈值扫描
│   │   ├── feature_selection_validation.py  # VIF + LASSO
│   │   ├── heterogeneous_pool_test.py # 异质模型池 (MLP+GPR)
│   │   ├── p2_strategy_analysis.py # Phase 2 策略分析
│   │   └── shap_analysis.py        # SHAP 特征重要性
│   │
│   ├── 📊 preprocess/              # 数据预处理脚本
│   ├── 📊 presentation/            # 图表与可视化生成
│   │   └── generate_highres_figures.py
│   │
│   └── 📊 new_dataset/             # 墨西哥数据集专用 (6 脚本)
│       ├── new_data_loader.py
│       ├── run_ablation_new_data.py # ⭐ 全量消融 + HPO
│       ├── run_deupv_ablation.py   # De-UPV 消融
│       ├── run_refined_feature_ablation.py
│       ├── soft_weighting.py       # Sigmoid 软加权 vs 硬阈值
│       └── analysis_p0p1.py / analysis_p2.py
│
├── ⚙️ configs/                     # 超参数配置文件（JSON）
│   ├── acdcb_default.json          # ACDCB 优化超参数
│   └── README.md
│
├── 📊 data/                        # 输入数据集
│   ├── Concrete_Data.xls           # UCI: 1030 样本 × 8 特征
│   ├── Data.csv                    # 墨西哥: 4420 样本 × 10 特征
│   ├── Concrete_Readme.txt         # UCI 数据说明
│   ├── README.md
│   └── processed/                  # 预处理中间产物
│
└─── 📚 docs/                        # 文档与论文
    ├── README.md
    └── 📁 reports/                 # 实验报告




---

## 🖥️ 环境与安装

### 系统要求

| 项 | 要求 |
|-------|------|
| **Python** | ≥ 3.10 |
| **操作系统** | Linux / macOS / Windows |
| **内存** | ≥ 8GB（推荐 16GB 用于超参搜索） |

### 安装步骤

```bash
# 1. 进入项目根目录
cd /path/to/ACDCB

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate              # Linux/macOS
# 或
.\venv\Scripts\activate                # Windows PowerShell

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖清单

```
numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 scikit-learn>=1.0.0
xgboost>=1.5.0 lightgbm>=3.3.0 optuna>=2.10.0 shap>=0.40.0
matplotlib>=3.4.0 seaborn>=0.11.0 xlrd>=2.0.0 joblib>=1.0.0
```

### 验证安装

```bash
python -c "import xgboost, lightgbm, optuna, shap; print('✓ All OK')"
```

---

## 🚀 快速开始

所有命令请在**项目根目录**执行。

### 1️⃣ 基线复现

复现已发表论文的基线算法：

```bash
# Feng et al. (2020) - AdaBoost + 决策树
python src/concrete_compressive_strength/reproduction/paper1_reproduce.py

# Yeh & Cheng (1998) - 人工神经网络 (ANN)
python src/concrete_compressive_strength/reproduction/paper2_reproduce.py
```

**输出**：`results/metrics/baseline_results.json`

---

### 2️⃣ ACDCB 完整训练与推理

#### 训练 ACDCB 模型

```bash
# UCI 数据集上的完整训练流程
python scripts/train/train_acdcb.py
```

**输出文件**：
- `results/models/acdcb_model.joblib` — 融合模型
- `results/predictions/ablation_oof_v2.csv` — 核外预测
- `results/metrics/ablation_results_acdcb_v2.json` — 性能指标

#### 推理新数据

```bash
python scripts/eval/predict_acdcb.py
```

---

### 3️⃣ 消融实验

#### A. UCI 数据集消融（V0-V5）

```bash
python scripts/eval/ablation_acdcb_v2.py
```

#### B. 墨西哥数据集全量消融 + 超参优化

```bash
python scripts/new_dataset/run_ablation_new_data.py --strategy A --trials 150
```

#### C. De-UPV 消融

```bash
python scripts/new_dataset/run_deupv_ablation.py --trials 100
```

#### D. 细化特征空间消融

```bash
python scripts/new_dataset/run_refined_feature_ablation.py --trials 50
```

---

### 4️⃣ 补充实验

| 实验 | 命令 |
|-----|------|
| **龄期阈值扫描** | `python scripts/eval/threshold_scan.py` |
| **特征选择验证** | `python scripts/eval/feature_selection_validation.py` |
| **异质模型池** | `python scripts/eval/heterogeneous_pool_test.py` |
| **软加权 vs 硬阈值** | `python scripts/new_dataset/soft_weighting.py` |
| **SHAP 分析** | `python scripts/eval/shap_analysis.py` |

---

### 5️⃣ 图表生成

```bash
python scripts/presentation/generate_highres_figures.py
```

**输出**：`figures/presentation_highres/`

---

## 📊 数据集说明

### UCI 混凝土强度数据集

| 属性 | 说明 |
|------|------|
| **来源** | UCI Machine Learning Repository |
| **样本数** | 1,030 |
| **特征数** | 8 |
| **目标** | 混凝土抗压强度 (MPa) |
| **文件** | `data/Concrete_Data.xls` |

#### 特征详解

| 特征 | 单位 | 描述 |
|------|------|------|
| `cement` | kg/m³ | 水泥含量 |
| `slag` | kg/m³ | 高炉矿渣粉 |
| `fly_ash` | kg/m³ | 粉煤灰 |
| `water` | kg/m³ | 拌合水 |
| `superplasticizer` | kg/m³ | 高效减水剂 |
| `coarse_aggregate` | kg/m³ | 粗骨料（石子） |
| `fine_aggregate` | kg/m³ | 细骨料（砂） |
| `age` | day | 龄期（养护时间） |

---

### 墨西哥混凝土数据集

| 属性 | 说明 |
|------|------|
| **来源** | ConcreteXAI (Guzmán-Torres et al. 2024) |
| **样本数** | 4,420 |
| **特征数** | 10 (6 数值 + 4 类别) |
| **文件** | `data/Data.csv` |

#### 特征详解

| 特征 | 类型 | 缺失率 | 描述 |
|------|------|--------|------|
| `Design_F'c` | 数值 | 0% | 设计强度等级 (MPa) |
| `Curing_age` | 数值 | 0% | 养护龄期 (day) |
| `Er` | 数值 | 0% | 电阻率 (Ω·m) |
| `UPV` | 数值 | 0% | 超声波脉冲速度 (m/s) |
| `Ts` | 数值 | 21.7% ❌ | 抗拉强度 (MPa) |
| `Fs` | 数值 | 60.2% ❌ | 抗折强度 (MPa) |
| 4 类别特征 | 类别 | 0% | 水泥、品牌、外加剂、骨料类型 |

---

## 🔧 技术方案

### 双空间特征工程

构造 22 维 (Anchor) 和 32 维 (Primary) 两套特征空间，分工处理稳健性与表达能力。

### 约束融合优化

融合多个 GBDT 模型的预测，施加非负与和为 1 的约束，采用龄期分层策略。

---

## 📈 主要结果

### UCI 数据集（N=1,030, 10-fold CV）

| 变体 | R² | RMSE (MPa) |
|------|-------|-----------|
| **V0** (基线) | 0.9142 | 4.80 |
| **V1-V3** (特征工程) | 0.9475-0.9488 | 3.70-3.73 |
| **V4** (原始+分段) | **0.9530** | **3.54** |
| **V5** (OLS) | 0.9488 | 3.70 |

**结论**：V4 超过完整 ACDCB (V3)，ΔR² = +0.0042

---

### 墨西哥数据集（N=4,420, 10-fold CV）

| 变体 | R² | RMSE (MPa) |
|------|-------|-----------|
| **V0** | 0.9936 | 0.765 |
| **V1-V5** | 0.9953-0.9954 | 0.647-0.653 |

**结论**：模型升级贡献 ΔR² = +0.0018，特征工程贡献 < 0.001

---

## 📚 论文与引用

```bibtex
@article{feng2020,
  title={Machine learning-based compressive strength prediction for concrete},
  author={Feng, D. C. and others},
  journal={Construction and Building Materials},
  year={2020}
}

@article{yeh1998,
  title={Modeling of strength of high performance concrete using ANNs},
  author={Yeh, I-Cheng},
  journal={Cement and Concrete Research},
  volume={28}, number={12}, pages={1797--1808},
  year={1998}
}

@article{guzmantorres2024,
  title={ConcreteXAI: A multivariate dataset for concrete strength prediction},
  author={Guzm\'an-Torres, J. A. and others},
  journal={Data in Brief},
  volume={53}, pages={110218},
  year={2024}
}
```

---

## ❓ 常见问题

### Q: 生产中应该用哪个模型？
**A**: V4 (原始特征 + 分段融合) 性价比最优，R² = 0.953，避免使用完整 ACDCB。

### Q: 特征工程是否必要？
**A**: 在本项目中贡献 < 0.005 R²。仅当数据分布变化时才值得重新设计。

### Q: 为什么 V4 比 ACDCB 更好？
**A**: 三大 GBDT 模型高度相关 (r > 0.996)，额外特征无法提供新信息。

### Q: 能预测其他混凝土类型吗？
**A**: 可以，但高性能混凝土 (>100 MPa) 或特种混凝土需谨慎验证。

### Q: 如何复现论文结果？
**A**: 依次运行基线复现脚本、ACDCB 训练、消融实验，对比结果目录的 JSON 文件。

---

## 📄 许可证与致谢

### 许可证

MIT License — 详见 [LICENSE](./LICENSE)

### 数据集致谢

- UCI Machine Learning Repository (Yeh 1998)
- Data in Brief (Guzmán-Torres et al. 2024)

### 开源库致谢

scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandas, numpy, matplotlib

---

**最后更新**：2026 年  
**维护者**：CNYJ
**反馈渠道**：GitHub Issues