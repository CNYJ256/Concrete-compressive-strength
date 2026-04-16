# v9 算法说明（独立版）

## 1. 方法命名与定位

`v9` 训练范式学术命名为：

**龄期条件化双空间约束融合**  
**Age-Conditioned Dual-Space Constrained Blending (ACDCB)**

定位：

- `v9` 代码独立维护，不依赖 `oldversion/` 目录；
- 以 10 折 OOF 为基础进行多模型约束融合；
- 输出完整四指标：`R² / RMSE / MAE / MAPE`。

---

## 2. 方法总览

### 2.1 候选模型池

- `XGBoost`
- `LightGBM`
- `HGB`
- `HGB_Anchor`（锚点模型）

### 2.2 双特征空间

- `primary`：扩展机理特征空间
- `anchor`：紧凑锚点特征空间

融合时按模型所属空间分别生成特征，保证训练-推理一致。

### 2.3 融合策略

1. **Global Blend**：全样本共享同一权重；
2. **Age-aware Piecewise Blend**：
   - 早龄期：`age <= 28`
   - 后龄期：`age > 28`
   - 分段独立优化权重。

最终以 `R²` 为主指标、`RMSE` 为次指标自动选择最优策略。

---

## 3. 最新结果（10 折）

### 3.1 v9 内部策略对比

| 策略 | R²_mean | RMSE_mean | MAE_mean | MAPE_mean(%) |
|---|---:|---:|---:|---:|
| Global Blend | 0.948724848 | 3.700053450 | 2.351103886 | 8.4874 |
| **Age-aware Piecewise** | **0.948755187** | **3.699571200** | 2.352152847 | 8.4878 |

最终选择：`age_piecewise`。

### 3.2 对比 paper1 AdaBoost（10 折）

| 方法 | R²_mean | RMSE_mean | MAE_mean | MAPE_mean(%) |
|---|---:|---:|---:|---:|
| paper1 AdaBoost | 0.909002814 | 4.969470675 | 3.508534979 | 13.3513 |
| **v9 (ACDCB)** | **0.948755187** | **3.699571200** | **2.352152847** | **8.4878** |

增益：

- `R²_gain = +0.039752`
- `RMSE_drop = 1.269899`
- `MAE_drop = 1.156382`
- `MAPE_drop = 4.863522`

---

## 4. 关键权重解释（最优分段策略）

### 4.1 早龄期权重（<= 28 天）

- `XGBoost`: 0.313813
- `LightGBM`: 0.039930
- `HGB`: 0.013239
- `HGB_Anchor`: 0.633019

### 4.2 后龄期权重（> 28 天）

- `XGBoost`: 0.463224
- `LightGBM`: 0.000000
- `HGB`: 0.122167
- `HGB_Anchor`: 0.414610

解释：早龄期更依赖锚点模型的稳健性，后龄期对 XGBoost 的非线性刻画能力依赖更高。

---

## 5. 产物文件

- `v9/core.py`：数据/特征/模型构造核心模块
- `v9/train.py`：独立训练与融合权重学习
- `v9/predict.py`：按 `selected_strategy` 自动推理
- `v9/model.joblib`：v9 最终模型包
- `v9/metrics.json`：详细四指标与权重信息
- `v9/predictions.csv`：默认全量原始数据推理输出
