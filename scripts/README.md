# scripts 目录说明

本目录已按“训练 / 评估 / 预处理”进行分层：

- `train/`
   - `train_acdcb.py`：ACDCB 训练入口（读取 `configs/acdcb_default.json`）
- `eval/`
   - `predict_acdcb.py`：ACDCB 推理入口
   - `ablation_acdcb.py`：ACDCB 消融实验入口
- `preprocess/`
   - `prepare_dataset.py`：数据基础校验与摘要生成

同时保留以下公共模块（供论文复现脚本复用）：

- `config.py`：全局路径与实验常量
- `data_loader.py`：数据读取与清洗
- `logger_utils.py`：日志工具
- `metrics_utils.py`：指标计算
- `model_factory.py`：基线模型工厂

论文复现脚本现位于：

- `src/concrete_compressive_strength/reproduction/paper1_reproduce.py`
- `src/concrete_compressive_strength/reproduction/paper2_reproduce.py`

## 运行建议

在项目根目录执行：

- 训练：`python scripts/train/train_acdcb.py`
- 推理：`python scripts/eval/predict_acdcb.py`
- 消融：`python scripts/eval/ablation_acdcb.py`
- 数据检查：`python scripts/preprocess/prepare_dataset.py`
