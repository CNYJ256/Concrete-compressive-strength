# 混凝土抗压强度预测（v9 主线）

本仓库聚焦混凝土抗压强度预测的**论文复现 + 工程化迭代**，当前默认使用 `v9` 作为主版本。

## 1. 项目背景

混凝土强度受胶凝材料、骨料、水胶比、外加剂与龄期等因素共同影响，呈明显非线性。项目先复现两条经典研究路径，再在此基础上持续优化：

- **paper1 复现（AdaBoost 主线）**：对比 AdaBoost / ANN / SVM；
- **paper2 复现（ANN 主线）**：验证 ANN 相对回归模型（w/c、w/b）的优势；
- **v9 工程优化主线**：在 `v8` 基础上升级为龄期分段自适应融合。

## 2. 当前仓库结构（重构后）

- `data/`：原始数据（`Concrete_Data.xls`）
- `scripts/`：公共模块（配置、数据读取、指标工具、模型工厂）
- `paper1/`：论文1复现脚本与结果
- `paper2/`：论文2复现脚本与结果
- `v9/`：当前主版本（训练、推理、指标、模型）
- `oldversion/`：归档历史版本（`v1` ~ `v8`）及早期结果文档

> 说明：`v9` 会复用 `oldversion/v8/train.py` 中的成熟特征工程与模型构造函数。

## 3. 环境与依赖配置

推荐：Python 3.9+（Windows/Linux/macOS 均可）

建议安装依赖：

- numpy
- pandas
- scipy
- scikit-learn
- xlrd
- joblib
- optuna
- xgboost
- lightgbm

## 4. 核心运行步骤（含 v9）

### 4.1 论文复现

1) 运行 `paper1` 复现：

```bash
python paper1/scripts/reproduce_pipeline.py
```

输出：

- `doc/baseline_results.json`
- `doc/Baseline_Reproduction_Report.md`

2) 运行 `paper2` 复现：

```bash
python paper2/scripts/paper2_reproduction.py
```

输出：

- `doc/paper2_reproduction_results.json`
- `doc/Paper2_Reproduction_Report.md`

### 4.2 运行 v9 训练与推理

1) 训练 v9：

```bash
python v9/train.py
```

输出：

- `v9/model.joblib`
- `v9/metrics.json`

2) 推理（默认数据前 5 行演示）：

```bash
python v9/predict.py
```

输出：

- `v9/predictions.csv`

3) 推理（可选：自定义 CSV 输入）：

```bash
python v9/predict.py your_input.csv your_output.csv
```

## 5. v9 主要改进（相对 v8）

`v9` 的核心创新是**龄期分段自适应融合（Age-aware Piecewise Blend）**：

1. 复用 `v8` 的三类强模型参数（XGBoost/LightGBM/HGB）；
2. 保留 `HGB_v7_baseline` 作为稳健锚点，增强鲁棒性；
3. 对 `age <= 28` 与 `age > 28` 分别学习融合权重；
4. 在“全局单权重融合”与“龄期分段融合”之间自动择优。

## 6. 结果简要展示（真实运行结果）

### 6.1 论文复现摘要

- **paper1（10折 AdaBoost）**：
  - R²_mean = **0.9090**
  - RMSE_mean = **4.9695**
- **paper2（Source-like 分组，test R² 均值）**：
  - ANN = **0.8436**
  - Reg(w/b) = **0.7796**
  - Reg(w/c) = **0.6278**

### 6.2 v9 与前代对比（10折 CV）

| 版本 | 策略 | R²_mean | RMSE_mean |
|---|---|---:|---:|
| v7 | HGB baseline | 0.947965 | 3.740782 |
| v8 | 全局加权融合 | 0.948725 | 3.700053 |
| **v9** | **龄期分段融合（最佳）** | **0.948755** | **3.699571** |

相对 v8，v9 增益：

- R²：`+0.00003034`
- RMSE：`-0.00048225`

## 7. 数据说明

- 数据文件：`data/Concrete_Data.xls`
- 样本量：1030
- 输入特征：8 个（`cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age`）
- 目标变量：`strength`（MPa）

## 8. 可复现性说明

- 统一随机种子：`42`
- 统一评估协议：10 折 KFold（`shuffle=True, random_state=42`）
- 所有关键结果均落盘为 JSON / Markdown，便于复核与课程汇报

---

如需查看历史版本实现细节，请进入 `oldversion/`。
