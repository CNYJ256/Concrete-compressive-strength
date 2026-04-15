# scripts 目录说明

本目录现仅保留**公共模块**，供 `paper1/`、`paper2/`、`v1/`、`v2/` 以及版本化目录脚本复用。

---

## 1. 当前目录职责

- `config.py`：全局配置（路径、随机种子、模型超参数）
- `data_loader.py`：数据读取、列名标准化、基础清洗
- `logger_utils.py`：统一日志格式与 logger 构建
- `metrics_utils.py`：`R²/RMSE/MAPE/MAE` 指标计算
- `model_factory.py`：基线与优化模型构建工厂

> 说明：复现与创新的可执行脚本已迁出，不再位于本目录。

---

## 2. 已迁出的脚本位置

- 第一篇论文复现：`paper1/scripts/reproduce_pipeline.py`
- 第二篇论文复现：`paper2/scripts/paper2_reproduction.py`
- 第一轮创新：`v1/optimized_model.py`
- 第二轮创新：`v2/new_techpath_model.py`

---

## 3. 公共模块依赖关系（简版）

```text
scripts/config.py
scripts/data_loader.py
scripts/logger_utils.py
scripts/metrics_utils.py
scripts/model_factory.py
   └─ 被 paper1/paper2/v1/v2 与版本化脚本引用
```

---

## 4. 维护建议

- 若新增实验脚本，建议放在独立版本目录，不直接放在 `scripts/`。
- `scripts/` 只保留“可复用公共能力”，避免目录职责混乱。
- 修改公共模块后，建议至少回归运行：
  - `paper1/scripts/reproduce_pipeline.py`
  - `paper2/scripts/paper2_reproduction.py`
  - `v1/optimized_model.py`
  - `v2/new_techpath_model.py`
