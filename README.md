# Concrete-compressive-strength

混凝土抗压强度预测研究仓库（论文复现 + ACDCB 方法）。

本仓库提供了一种新的集成学习方法 ACDCB（Age-aware Constrained Dual-space Boosting），并与参考论文结果进行了系统对比。目录结构清晰划分了数据、代码、文档与结果，便于复现与扩展。

## 目录结构

```text
.
├── configs/                    # 实验配置（代码与超参数分离）
├── data/                       # 数据目录（原始数据 + 占位说明）
├── docs/                       # 文档目录
│   ├── reports/                # 实验报告（Markdown）
│   ├── papers/                 # 论文文档（Markdown/PDF）
│   └── prompt.md               # 研究提示与备忘
├── results/                    # 运行产物目录（模型/指标/预测）
├── src/
│   └── concrete_compressive_strength/
│       ├── __init__.py
│       ├── core.py             # ACDCB 核心算法模块
│       ├── plotting/           # 图表生成脚本
│       └── reproduction/       # 论文复现脚本
├── scripts/
│   ├── train/                  # 训练脚本
│   ├── eval/                   # 评估与推理脚本
│   ├── preprocess/             # 数据预处理脚本
│   └── *.py                    # 论文复现复用模块
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.9+

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行入口

### 1) 数据预处理/校验

```bash
python scripts/preprocess/prepare_dataset.py
```

输出：`results/metrics/dataset_profile.json`

### 2) 训练 ACDCB

```bash
python scripts/train/train_acdcb.py
```

可选：指定配置文件

```bash
python scripts/train/train_acdcb.py configs/acdcb_default.json
```

输出：

- `results/models/acdcb_model.joblib`
- `results/metrics/acdcb_metrics.json`

### 3) 推理 ACDCB

```bash
python scripts/eval/predict_acdcb.py
```

可选：自定义输入输出

```bash
python scripts/eval/predict_acdcb.py your_input.csv your_output.csv
```

默认输出：`results/predictions/acdcb_predictions.csv`

### 4) ACDCB 消融实验

```bash
python scripts/eval/ablation_acdcb.py
```

输出：

- `results/metrics/ablation_results_acdcb.json`
- `results/predictions/ablation_oof_predictions.csv`

### 5) 论文复现脚本

```bash
python src/concrete_compressive_strength/reproduction/paper1_reproduce.py
python src/concrete_compressive_strength/reproduction/paper2_reproduce.py
```

输出将写入：

- `results/metrics/*.json`
- `docs/reports/*.md`

### 6) 生成图表

```bash
python src/concrete_compressive_strength/plotting/generate_acdcb_figures.py
python src/concrete_compressive_strength/plotting/generate_comparison_figures.py
```

输出目录：`figures/`

## 数据来源与引用

- UCI Concrete Compressive Strength（1030 样本，8输入，1输出）
- Yeh, I. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67

## License

MIT（见 `LICENSE`）