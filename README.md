# 基于机器学习的混凝土抗压强度预测

## 1. 项目学术背景

混凝土抗压强度受胶凝材料组成、骨料级配、水胶比、外加剂与龄期等多因素耦合影响，呈显著非线性关系。传统经验回归模型在复杂配比条件下泛化能力有限，因此本项目以机器学习为核心，构建可复现、可迭代、可解释的强度预测流程。

本仓库围绕两条文献主线展开：

- 论文A（AdaBoost, 2020）：验证集成学习在强度预测中的精度与鲁棒性；
- 论文B（ANN, 1998）：验证 ANN 相比传统回归的优势，并对比 w/c 与 w/b 回归趋势。

在复现基础上，本项目继续执行工程化迭代优化（特征工程、难样本重加权、Stacking 集成等）。

---

## 2. 研究目标

1. 复现并对比经典模型：AdaBoost / ANN / SVM；
2. 复现实验结论：ANN 优于传统回归，w/b 回归优于 w/c 回归；
3. 构建可持续优化管线：从 baseline（v0）到多版本迭代（v1~v5）；
4. 形成标准化文档与实验记录，支持课程汇报与科研延展。

---

## 3. 环境依赖

推荐环境：

- Python 3.9+
- Windows / Linux / macOS

核心依赖：

- pandas
- numpy
- scikit-learn
- xlrd（读取 `.xls`）
- scipy（用于公式回归参数拟合）

说明：项目已对部分旧版 `scikit-learn` 做兼容处理（如 RMSE 计算方式）。

---

## 4. 数据说明

- 主数据文件：`data/Concrete_Data.xls`
- 样本规模：1030 条
- 特征维度：8 个输入变量
- 目标变量：混凝土抗压强度（MPa）

标准化后的列名如下：

- 特征：`cement`, `slag`, `fly_ash`, `water`, `superplasticizer`, `coarse_agg`, `fine_agg`, `age`
- 目标：`strength`

---

## 5. 项目结构

- `scripts/`：公共模块（配置、数据加载、日志、指标、模型工厂）
- `paper1/`：第一篇论文复现脚本与说明文档
- `paper2/`：第二篇论文复现脚本与说明文档
- `v1/`：第一轮创新脚本（优化 AdaBoost）
- `v2/`：第二轮创新脚本（Stacking 新技术路径）
- `v3/` ~ `v7/`：受控自动化迭代版本产物目录（训练脚本、指标、模型等）
- `doc/`：创新阶段与综合阶段报告
- `paper/`：论文翻译稿
- `data/`：原始数据与数据说明

公共模块说明见：`scripts/README.md`

---

## 6. 运行说明（推荐顺序）

1. 第一篇论文复现（v0 基线）
   - 运行：`paper1/scripts/reproduce_pipeline.py`
   - 输出：`paper1/doc/baseline_results.json`、`paper1/doc/Baseline_Reproduction_Report.md`

2. 第二篇论文复现（ANN vs 回归）
   - 运行：`paper2/scripts/paper2_reproduction.py`
   - 输出：`paper2/doc/paper2_reproduction_results.json`、`paper2/doc/Paper2_Reproduction_Report.md`

3. 第一轮优化（改进 AdaBoost）
   - 运行：`v1/optimized_model.py`
   - 输出：`doc/optimization_results.json`、`doc/Optimization_Comparison_Report.md`

4. 第二轮优化（Stacking 新技术路径）
   - 运行：`v2/new_techpath_model.py`
   - 输出：`doc/new_techpath_results.json`、`doc/New_TechPath_Comparison_Report.md`

5. 多轮自动化迭代版本（v3~v7）
   - 目录：`v3/` ~ `v7/`
   - 内容：每版独立 `train.py`、`predict.py`、`metrics.json`、`model.joblib`、`CHANGELOG.md`

---

## 7. 整体架构流（逻辑）

```text
Concrete_Data.xls
   -> scripts/data_loader.py（读取/清洗/列标准化）
   -> split_features_target（X, y）
   -> paper1/paper2/v1/v2 脚本训练与评估
   -> scripts/metrics_utils.py（R², RMSE, MAPE, MAE）
   -> paper1/doc 或 paper2/doc 或 doc 报告输出
```

其中：

- `paper1/scripts/reproduce_pipeline.py` 是第一篇论文复现主入口；
- `paper2/scripts/paper2_reproduction.py` 是第二篇论文复现主入口；
- `v1/optimized_model.py` 在 baseline 上做特征工程 + 难样本重加权；
- `v2/new_techpath_model.py` 在优化思路上引入 Stacking 集成。

---

## 8. 可复现性与工程约定

- 全局随机种子：`scripts/config.py` 中 `RANDOM_STATE`
- 统一评估指标：`R²`, `RMSE`, `MAPE`, `MAE`
- 建议版本化实验在独立目录（当前为 `v3`~`v7`）隔离实现，避免互相污染
- 每次迭代需记录：改动内容、交叉验证指标变化、工程反思

---

## 9. 后续扩展方向

- 更系统的超参数优化（如 Optuna / 贝叶斯优化）
- 更严格的泛化验证（重复 K-Fold、外部测试集）
- 机理解释增强（SHAP、PDP、单调约束）
- 从“强度预测”扩展到“配比反推与多目标优化”
