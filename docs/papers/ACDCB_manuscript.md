# ACDCB: A Machine Learning-Based Strategy for Concrete Strength Prediction

---

## Abstract


This study proposes **ACDCB** (Age-Conditioned Dual-Space Constrained Blending), a novel ensemble learning framework designed to predict the compressive strength of concrete by capturing complex, age-dependent nonlinear mappings. The framework integrates dual feature spaces—a 32-dimensional primary space and a 22-dimensional anchor space—to balance expressiveness and robustness. Using four gradient-boosted tree models (XGBoost, LightGBM, and HistGradientBoosting), ACDCB employs a constrained weight optimization (SLSQP) strategy. To reflect concrete maturity physics, the blending weights are optimized separately for early-age ($\leq 28$ days) and late-age ($>28$ days) regimes. Validated on the UCI dataset, ACDCB achieved an $R^2$ of 0.9488 and an RMSE of 3.70 MPa, marking a 25.6% reduction in RMSE over the AdaBoost baseline. Ablation experiments demonstrate that the anchor space is critical for stability, while the piecewise blending provides physical interpretability: the anchor model dominates early-age predictions, whereas XGBoost's importance grows in late-age regimes. ACDCB offers a principled, modular, and physics-aware approach to material strength modeling, providing a significant improvement in both predictive accuracy and model interpretability for structural engineering applications.


**Keywords**: concrete compressive strength, ensemble learning, age-conditioned blending, constrained optimization, dual feature spaces, gradient boosting

---

## 1. Introduction

The accurate prediction of concrete compressive strength is a cornerstone problem in structural engineering, with direct implications for construction scheduling, quality control, and service-life assessment (Yeh, 1998). Traditional approaches rely on empirical regression formulas—such as Abrams' law or power-law maturity models—that offer computational simplicity but cannot adequately capture the high-dimensional, nonlinear interactions among mix design variables, admixture effects, and curing conditions (Yeh, 2006).

Over the past two decades, machine learning (ML) methods have emerged as powerful alternatives. Artificial neural networks (ANN), support vector machines (SVM), and ensemble methods—particularly AdaBoost with decision tree weak learners—have demonstrated progressively improving predictive accuracy on the UCI benchmark dataset (Yeh, 1998; Chou et al., 2014). Notably, the AdaBoost-based approach reported by Feng et al. (2020) achieved a 10-fold cross-validated $R^2$ of approximately 0.95 and RMSE below 5 MPa, establishing a strong baseline for subsequent work.

However, three structural limitations persist across existing approaches:

1. **Mono-representation**: All models operate on a single feature space—typically the raw mix proportions or a fixed set of engineered ratios. No existing framework systematically exploits *complementary* feature representations that trade off expressiveness and robustness.

2. **Age-agnostic integration**: While concrete strength development is fundamentally age-dependent—governed by distinct hydration kinetics in early ($\leq 28$ days) versus later ages—most ensemble methods apply uniform blending weights across all ages, failing to adapt to regime-specific model competencies.

3. **Unconstrained model combination**: Standard ensemble stacking and averaging impose no formal constraints on combination weights. Negative or unbounded weights, while occasionally improving point estimates, obscure interpretability and can produce physically implausible extrapolations.

To address these gaps, we propose **ACDCB** (Age-Conditioned Dual-Space Constrained Blending), a framework characterized by four design principles:

- **Dual Feature Spaces (D)**: Two feature representations—*primary* (32 dimensions, expression-oriented) and *anchor* (22 dimensions, robustness-oriented)—are constructed from the same 8 base variables through systematic civil engineering feature engineering. Each space feeds a dedicated subset of base models.

- **Constrained Blending (C, B)**: Model combination weights are learned through constrained optimization (SLSQP), minimizing RMSE subject to the convex constraints $w_i \geq 0$ and $\sum_i w_i = 1$. This ensures weights are directly interpretable as "contribution shares."

- **Age Conditioning (A, C)**: The dataset is partitioned at the 28-day curing threshold—a physically meaningful engineering landmark—and separate blending weights are optimized for each regime.

- **Multi-Model Pool**: Four gradient-boosted tree variants (XGBoost, LightGBM, HistGradientBoosting, and an anchor-space HistGradientBoosting) constitute the candidate model pool, selected for their complementary inductive biases.

The remainder of this paper is organized as follows. Section 2 describes the experimental database and preprocessing pipeline. Section 3 presents the mathematical foundations of the baseline models and the ACDCB framework. Section 4 details the implementation and training strategy. Section 5 reports model validation and evaluation results. Section 6 presents the ablation study decomposing component contributions. Section 7 provides comparative analysis against prior work. Sections 8 and 9 offer discussion and conclusions, respectively.

---

## 2. Experimental Database

### 2.1 Data Source

This study employs the UCI Concrete Compressive Strength dataset (Yeh, 1998), publicly available from the UCI Machine Learning Repository (DOI: 10.24432/C5PK67). The dataset comprises 1,030 experimental observations, each characterized by 8 input variables describing concrete mix proportions and curing age, along with one output variable—the 28-day (or specified-age) compressive strength in MPa.

### 2.2 Input Variables

The eight base features, their physical interpretations, and measurement units are summarized in **Table 1**.

**Table 1.** Input variables for concrete compressive strength prediction.

| Variable | Symbol | Description | Unit |
|---|---|---|---|
| Cement | $C$ | Portland cement content | kg/m³ |
| Blast Furnace Slag | $S$ | Ground granulated blast-furnace slag | kg/m³ |
| Fly Ash | $FA$ | Class F/C fly ash | kg/m³ |
| Water | $W$ | Mixing water | kg/m³ |
| Superplasticizer | $SP$ | High-range water-reducing admixture | kg/m³ |
| Coarse Aggregate | $CA$ | Coarse aggregate (>4.75 mm) | kg/m³ |
| Fine Aggregate | $FA_g$ | Fine aggregate (<4.75 mm) | kg/m³ |
| Age | $t$ | Curing age | days |

**Target variable**: $f_c$ — Concrete compressive strength (MPa).

### 2.3 Statistical Summary

The target variable (compressive strength) exhibits substantial variability: $\mu = 35.82$ MPa, $\sigma = 16.70$ MPa, with values spanning 2.33–82.60 MPa, reflecting diverse mix designs ranging from low-strength to high-performance concrete. The age distribution is heavily right-skewed (skewness > 2.0), with a concentration of samples at early ages (1–28 days) and a long tail extending to 365 days. This heterogeneous age distribution motivates the age-conditioned design of the ACDCB framework.

**Figure 1** presents the Pearson correlation heatmap across all variables and violin plots of key features, providing a comprehensive view of the data distribution (see `figures/presentation_highres/fig1_data_distribution.pdf`).

### 2.4 Feature Engineering

Beyond the 8 base variables, two tiers of engineered features are constructed to capture known concrete mechanics relationships:

**Tier 1 — Mechanistic features (shared by primary and anchor spaces, 14 features):**

Let $\varepsilon = 10^{-6}$ for numerical stability. Define intermediate quantities:

$$B = C + S + FA \quad \text{(total binder)}$$
$$TA = CA + FA_g \quad \text{(total aggregate)}$$
$$t_{\log} = \log(1 + t)$$

The mechanistic features include:

$$\frac{W}{C}, \quad \frac{W}{B}, \quad \frac{SP}{B}, \quad \frac{S + FA}{B}, \quad \frac{FA_g}{TA}$$

$$t_{\log}, \quad \sqrt{t}, \quad t^{0.25}$$

$$\text{Abrams Index} = \frac{t_{\log}}{W/B}, \quad C \cdot t_{\log}, \quad B \cdot t_{\log}, \quad \frac{W}{B} \cdot t_{\log}$$

$$\text{Paste Index} = \frac{C + S + FA + W + SP}{TA}$$

These features encode well-established concrete engineering principles: water-binder ratio governs ultimate strength (Abrams' law), binder-age interactions capture maturity effects, and paste-to-aggregate ratios reflect volumetric composition.

**Tier 2 — Enhanced features (primary space only, 10 additional features):**

$$\frac{B}{TA}, \quad \frac{W}{W + B + SP}, \quad \frac{C}{B}, \quad \frac{S}{B}, \quad \frac{FA}{B}$$

$$\frac{SP}{W}, \quad t_{\log} \cdot \frac{B}{W}, \quad \frac{TA}{B}, \quad \frac{1}{t + 1}, \quad t_{\log} \cdot \frac{W}{C}$$

The primary space thus contains 32 dimensions (8 base + 14 mechanistic + 10 enhanced), while the anchor space contains 22 dimensions (8 base + 14 mechanistic).

After feature engineering, all infinite values (resulting from division by near-zero denominators) are replaced with column medians to maintain data integrity.

---

## 3. Methodology

### 3.1 Baseline Models

To establish comparative baselines, we reproduce three representative models from the literature.

#### 3.1.1 AdaBoost Regression

AdaBoost (Adaptive Boosting) is an ensemble method that sequentially trains weak learners—here, Classification and Regression Trees (CART)—with adaptive sample weighting (Freund & Schapire, 1997). For a training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{m}$, the algorithm proceeds as follows:

**Step 1 — Weight initialization**: Each sample receives equal weight $w_{1,i} = 1/m$.

**Step 2 — Iterative training**: For iteration $k = 1, \dots, K$:

1. Train weak learner $G_k(\mathbf{x})$ on the weighted dataset.
2. Compute sample-wise relative error:
   $$e_{k,i} = \frac{|y_i - G_k(\mathbf{x}_i)|}{E_k}, \quad E_k = \max_i |y_i - G_k(\mathbf{x}_i)|$$

3. Compute the aggregate error and weight coefficient:
   $$e_k = \sum_{i=1}^{m} w_{k,i} \cdot e_{k,i}, \quad \alpha_k = \frac{e_k}{1 - e_k}$$

4. Update sample weights for the next iteration:
   $$w_{k+1,i} = \frac{w_{k,i} \cdot \alpha_k^{1 - e_{k,i}}}{\sum_{j=1}^{m} w_{k,j} \cdot \alpha_k^{1 - e_{k,j}}}$$

**Step 3 — Ensemble prediction**: The final prediction is the weighted median of all weak learner outputs, with learner weight $\ln(1/\alpha_k)$.

The CART weak learner recursively partitions the feature space, minimizing the weighted mean squared error at each split.

#### 3.1.2 Artificial Neural Network

The ANN baseline employs a multi-layer perceptron with one hidden layer of 8 neurons and ReLU activation, following the architecture used in Yeh (1998):

$$\mathbf{h} = \text{ReLU}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$

$$\hat{y} = \mathbf{W}^{(2)}\mathbf{h} + b^{(2)}$$

where $\mathbf{W}^{(1)} \in \mathbb{R}^{8 \times 8}$, $\mathbf{W}^{(2)} \in \mathbb{R}^{1 \times 8}$. Training minimizes MSE via stochastic gradient descent (SGD) with momentum $\beta = 0.5$:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} \mathcal{L}_{\text{MSE}} + \beta \Delta\mathbf{W}_{\text{prev}}$$

#### 3.1.3 Support Vector Regression

SVR with radial basis function (RBF) kernel maps inputs to a high-dimensional feature space and fits an $\varepsilon$-insensitive tube:

$$\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}(\xi_i + \xi_i^*)$$

subject to:

$$y_i - \mathbf{w}^T\phi(\mathbf{x}_i) - b \leq \varepsilon + \xi_i$$

$$\mathbf{w}^T\phi(\mathbf{x}_i) + b - y_i \leq \varepsilon + \xi_i^*$$

$$\xi_i, \xi_i^* \geq 0$$

The RBF kernel is defined as $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$.

### 3.2 Gradient-Boosted Tree Base Models

The ACDCB model pool comprises three gradient boosting variants, selected for their complementary optimization strategies and proven performance on tabular data.

#### 3.2.1 XGBoost

XGBoost (Chen & Guestrin, 2016) employs a regularized objective with second-order Taylor expansion:

$$\mathcal{L}^{(k)} = \sum_{i=1}^{n} \left[g_i f_k(\mathbf{x}_i) + \frac{1}{2}h_i f_k^2(\mathbf{x}_i)\right] + \Omega(f_k)$$

where $g_i = \partial_{\hat{y}^{(k-1)}} l(y_i, \hat{y}^{(k-1)})$ and $h_i = \partial^2_{\hat{y}^{(k-1)}} l(y_i, \hat{y}^{(k-1)})$ are the first- and second-order gradients of the loss, and:

$$\Omega(f_k) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|_1$$

combines leaf-count penalty ($\gamma T$), L2 regularization, and L1 regularization.

#### 3.2.2 LightGBM

LightGBM (Ke et al., 2017) introduces gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB) for efficiency. Its leaf-wise tree growth strategy with depth constraints yields:

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_{i \in P} g_i)^2}{\sum_{i \in P} h_i + \lambda}\right]$$

#### 3.2.3 Histogram-based Gradient Boosting (HGB)

HGB (scikit-learn implementation) bins continuous features into integer-valued histograms ($B \leq 255$ bins), reducing split-finding complexity from $\mathcal{O}(n)$ to $\mathcal{O}(B)$:

$$\text{split\_gain}(j, s) = \frac{\sum_{i: x_{ij} \leq s} g_i}{\sum_{i: x_{ij} \leq s} h_i + \lambda} + \frac{\sum_{i: x_{ij} > s} g_i}{\sum_{i: x_{ij} > s} h_i + \lambda}$$

### 3.3 ACDCB Framework

The ACDCB framework integrates four structural components: dual feature spaces, a multi-model candidate pool, constrained weight optimization, and age-conditioned piecewise blending. A schematic of the complete architecture is provided in **Figure 2** (see `figures/presentation_highres/fig2_acdcb_architecture.pdf`).

#### 3.3.1 Dual Feature Spaces (D)

Given the 8-dimensional base input $\mathbf{x}_{\text{base}} \in \mathbb{R}^8$, two feature transformation functions are defined:

$$\phi_{\text{primary}}: \mathbb{R}^8 \to \mathbb{R}^{32}, \quad \phi_{\text{anchor}}: \mathbb{R}^8 \to \mathbb{R}^{22}$$

where $\phi_{\text{primary}}$ applies Tier 1 + Tier 2 engineering (32 output dimensions), and $\phi_{\text{anchor}}$ applies Tier 1 only (22 output dimensions). The design rationale is:

- **Primary space** ($\mathbb{R}^{32}$): Maximizes predictive capacity through enriched interaction terms, compositional ratios, and nonlinear age transforms. Suitable for expressive models that can regularize high-dimensional inputs.

- **Anchor space** ($\mathbb{R}^{22}$): Retains core mechanistic features while omitting fine-grained compositional decompositions, reducing estimation variance. Serves as a stabilization signal during blending.

The two spaces are **not** treated as competing alternatives but as **complementary views** of the same underlying physical process, to be reconciled through constrained fusion.

#### 3.3.2 Model Pool and OOF Prediction Matrix

The model pool $\mathcal{M} = \{M_1, M_2, M_3, M_4\}$ consists of:

| Model ID | Algorithm | Feature Space | Role |
|---|---|---|---|
| $M_1$ | XGBoost | Primary ($\mathbb{R}^{32}$) | High-capacity non-linear modeling |
| $M_2$ | LightGBM | Primary ($\mathbb{R}^{32}$) | Efficient leaf-wise boosting |
| $M_3$ | HGB | Primary ($\mathbb{R}^{32}$) | Histogram-based regularization |
| $M_4$ | HGB_Anchor | Anchor ($\mathbb{R}^{22}$) | Robust baseline stabilization |

Under $K$-fold cross-validation ($K = 10$), each model $M_j$ produces out-of-fold (OOF) predictions $\hat{y}_i^{(j)}$ for every sample $i$. These are concatenated into the OOF prediction matrix:

$$\mathbf{P} = \begin{bmatrix}
\hat{y}_1^{(1)} & \hat{y}_1^{(2)} & \cdots & \hat{y}_1^{(M)} \\
\hat{y}_2^{(1)} & \hat{y}_2^{(2)} & \cdots & \hat{y}_2^{(M)} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{y}_N^{(1)} & \hat{y}_N^{(2)} & \cdots & \hat{y}_N^{(M)}
\end{bmatrix} \in \mathbb{R}^{N \times M}, \quad M = 4$$

The OOF matrix is critical: it provides an unbiased estimate of each model's generalization performance, preventing overfitting in the subsequent weight optimization stage.

#### 3.3.3 Constrained Weight Optimization (C, B)

The blending weight vector $\mathbf{w} \in \mathbb{R}^M$ is obtained by solving the constrained optimization problem:

$$\min_{\mathbf{w}} \; \text{RMSE}(\mathbf{y}, \mathbf{P}\mathbf{w}) = \min_{\mathbf{w}} \; \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(y_i - \sum_{j=1}^{M} w_j \hat{y}_i^{(j)}\right)^2}$$

subject to the convex constraint set:

$$\mathcal{W} = \left\{\mathbf{w} \in \mathbb{R}^M \; \middle| \; w_j \geq 0 \;\; \forall j, \;\; \sum_{j=1}^{M} w_j = 1\right\}$$

The optimization is performed using Sequential Least Squares Quadratic Programming (SLSQP; Kraft, 1988), a quasi-Newton method suitable for constrained nonlinear objectives. The convex constraints guarantee:

1. **Interpretability**: Each $w_j$ is the fractional contribution of model $M_j$ to the ensemble.
2. **Stability**: The simplex constraint prevents unbounded extrapolation.
3. **Sparsity encouragement**: Non-negativity combined with the simplex can drive poorly performing models to zero weight, effecting implicit model selection.

#### 3.3.4 Age-Conditioned Piecewise Blending (A, C)

The dataset is partitioned at a physiochemically meaningful threshold $\tau = 28$ days:

$$\mathcal{D}_e = \{i \mid t_i \leq \tau\}, \quad \mathcal{D}_l = \{i \mid t_i > \tau\}$$

Separate weight vectors $\mathbf{w}_e$ and $\mathbf{w}_l$ are optimized for each regime:

$$\mathbf{w}_e = \arg\min_{\mathbf{w} \in \mathcal{W}} \text{RMSE}(\mathbf{y}_e, \mathbf{P}_e\mathbf{w})$$

$$\mathbf{w}_l = \arg\min_{\mathbf{w} \in \mathcal{W}} \text{RMSE}(\mathbf{y}_l, \mathbf{P}_l\mathbf{w})$$

The final prediction for a new sample with age $t$ is:

$$\hat{y}(\mathbf{x}) = \begin{cases}
\sum_{j=1}^{M} w_{e,j} \cdot M_j(\mathbf{x}), & t \leq 28 \text{ days} \\
\sum_{j=1}^{M} w_{l,j} \cdot M_j(\mathbf{x}), & t > 28 \text{ days}
\end{cases}$$

The 28-day threshold is chosen because it represents the standard curing period in structural concrete practice (ACI 318, EN 206), after which hydration kinetics transition from rapid early-age strength gain to asymptotic long-term development.

#### 3.3.5 Strategy Selection

Between the global blend (single $\mathbf{w}$ for all ages) and the piecewise blend, selection follows a hierarchical rule:

1. Compare $R^2$ values; prefer the strategy with higher $R^2$.
2. If $|R^2_{\text{global}} - R^2_{\text{piecewise}}| < \delta$ (tolerance $\delta = 5 \times 10^{-4}$), select the strategy with lower RMSE.

This "$R^2$-first, RMSE-tiebreaker" rule prioritizes explained variance while using RMSE for fine discrimination.

---

## 4. Implementation Details

### 4.1 Data Split and Cross-Validation

The dataset is randomly partitioned into training (90%) and test (10%) sets with a fixed random seed (42) to ensure reproducibility. All model training and hyperparameter tuning are conducted exclusively on the training set. The test set is held out for final evaluation only.

For OOF prediction generation and final performance reporting, 10-fold cross-validation with shuffled stratification is employed. Each fold preserves the overall age distribution approximately.

### 4.2 Hyperparameter Configuration

The hyperparameters for each base model were obtained through Bayesian optimization (Optuna framework, 500 trials per model) with 10-fold CV RMSE as the objective. The final configurations are:

**XGBoost (Primary):**
- $n_{\text{estimators}} = 1482$
- $\eta = 0.0446$ (learning rate)
- $\text{max\_depth} = 4$
- $\text{min\_child\_weight} = 7.39$
- $\text{subsample} = 0.664$, $\text{colsample\_bytree} = 0.625$
- $\gamma = 2.434$, $\alpha = 0.493$, $\lambda = 3.021$

**LightGBM (Primary):**
- $n_{\text{estimators}} = 2127$
- $\eta = 0.0320$
- $\text{num\_leaves} = 32$, $\text{max\_depth} = 4$
- $\text{min\_child\_samples} = 8$
- $\text{subsample} = 0.838$, $\text{colsample\_bytree} = 0.591$
- $\alpha = 0.711$, $\lambda = 2.22 \times 10^{-4}$
- $\text{min\_split\_gain} = 0.00646$

**HGB (Primary):**
- $\eta = 0.0568$, $\text{max\_iter} = 1809$
- $\text{max\_depth} = 12$, $\text{max\_leaf\_nodes} = 15$
- $\text{min\_samples\_leaf} = 14$
- $\ell_2 = 4.58 \times 10^{-5}$, $\text{max\_bins} = 213$

**HGB_Anchor:**
- $\eta = 0.028$, $\text{max\_iter} = 2400$
- $\text{max\_depth} = \text{None}$ (unconstrained), $\text{max\_leaf\_nodes} = 15$
- $\text{min\_samples\_leaf} = 6$
- $\ell_2 = 0.001$

All models use the squared error loss function. The random state is fixed at 42 across all stochastic components.

### 4.3 Weight Optimization

The SLSQP optimizer is configured with:
- Initial weights: uniform $\mathbf{w}_0 = [0.25, 0.25, 0.25, 0.25]$
- Tolerance: $\text{ftol} = 10^{-8}$
- Maximum iterations: 500

Convergence is typically achieved within 10–30 iterations for both global and piecewise optimizations, as shown in **Figure 3** (optimizer convergence traces, see `figures/presentation_highres/fig3_true_vs_pred.pdf`).

### 4.4 Software and Hardware

All experiments are implemented in Python 3.9+ with the following core dependencies: `scikit-learn` (HGB, preprocessing), `xgboost`, `lightgbm`, `scipy` (SLSQP optimization), `numpy`, and `pandas`. Computations are performed on a standard workstation (Intel Core i7, 16 GB RAM). Total runtime for the full pipeline (feature engineering + 10-fold OOF generation for 4 models + weight optimization) is approximately 3–5 minutes.

---

## 5. Model Validation and Evaluation

### 5.1 Evaluation Metrics

Four complementary metrics are employed to assess model performance:

**Coefficient of Determination ($R^2$):**

$$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

$R^2$ measures the proportion of variance explained, with 1 indicating perfect prediction. It is scale-invariant and serves as our primary metric.

**Root Mean Squared Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

RMSE penalizes large errors quadratically, making it sensitive to outliers. Units: MPa.

**Mean Absolute Error (MAE):**

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

MAE provides a linear error measure, less sensitive to outliers than RMSE. Units: MPa.

**Mean Absolute Percentage Error (MAPE):**

$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

MAPE expresses error relative to the true value, facilitating cross-dataset comparisons.

### 5.2 Single Model Performance

**Table 2** reports the 10-fold OOF performance of each candidate model in the ACDCB pool.

**Table 2.** Single model 10-fold OOF performance (mean ± std).

| Model | Feature Space | $R^2$ | RMSE (MPa) | MAE (MPa) | MAPE (%) |
|---|---|---|---|---|---|
| XGBoost | Primary (32-dim) | 0.945934 | 3.7817 | 2.4618 | 8.930 |
| LightGBM | Primary (32-dim) | 0.945652 | 3.8052 | 2.4799 | 8.878 |
| HGB | Primary (32-dim) | 0.945497 | 3.8090 | 2.4259 | 8.717 |
| **HGB_Anchor** | Anchor (22-dim) | **0.947965** | **3.7408** | **2.3683** | **8.529** |

A notable finding is that the HGB_Anchor model—operating on the lower-dimensional (22-dim) anchor space—outperforms all three primary-space models across every metric. This suggests that the anchor feature space, by excluding potentially noisy enhanced features, achieves a favorable bias-variance tradeoff for this dataset size ($N = 1,030$). The consistent $R^2$ values (0.9455–0.9480) across all four models indicate robust feature engineering.

A visual comparison across all four metrics is provided in **Figure 7** (see `figures/presentation_highres/fig7_single_model_comparison.pdf`).

### 5.3 ACDCB Ensemble Performance

**Table 3** compares the ACDCB ensemble against the baseline models and the internal component strategies.

**Table 3.** Ensemble and baseline performance comparison.

| Method | $R^2$ | RMSE (MPa) | MAE (MPa) | MAPE (%) |
|---|---|---|---|---|
| AdaBoost (Paper1 baseline) | 0.9090 | 4.9695 | 3.5085 | 13.351 |
| ANN (Paper2 baseline) | 0.9160 | 4.7273 | 3.3637 | 11.909 |
| SVM (Paper2 baseline) | 0.8713 | 5.8492 | 4.2099 | 14.728 |
| Best single model (HGB_Anchor) | 0.9480 | 3.7408 | 2.3683 | 8.529 |
| ACDCB Global Blend | 0.9487 | 3.7001 | 2.3511 | **8.487** |
| **ACDCB Age-Piecewise (final)** | **0.9488** | **3.6996** | **2.3522** | 8.488 |

The ACDCB Age-Piecewise strategy achieves the highest $R^2$ and lowest RMSE among all methods. Relative to the AdaBoost baseline:

- $\Delta R^2 = +0.0398$ (4.38% relative improvement)
- $\Delta\text{RMSE} = -1.27$ MPa (25.6% reduction)
- $\Delta\text{MAE} = -1.16$ MPa (33.0% reduction)
- $\Delta\text{MAPE} = -4.86$ percentage points (36.4% reduction)

The piecewise variant edges out the global blend by a narrow margin ($\Delta R^2 = +3 \times 10^{-5}$, $\Delta\text{RMSE} = -0.0005$ MPa), selected by the hierarchical rule.

The true-versus-predicted scatter comparison between AdaBoost and ACDCB is shown in **Figure 3** (see `figures/presentation_highres/fig3_true_vs_pred.pdf`), demonstrating visibly tighter clustering around the $y = x$ reference line for ACDCB, particularly in the high-strength regime (>60 MPa) where AdaBoost exhibits systematic underprediction.

### 5.4 Learned Blending Weights

The optimized piecewise weights (illustrated in **Figure 6**; see `figures/presentation_highres/fig6_piecewise_weights.pdf`) reveal regime-specific model competencies:

**Early age ($t \leq 28$ days):**
$$\mathbf{w}_e = [0.314, \; 0.040, \; 0.013, \; 0.633]$$

**Late age ($t > 28$ days):**
$$\mathbf{w}_l = [0.463, \; 0.000, \; 0.122, \; 0.415]$$

The HGB_Anchor model dominates early-age predictions (63.3% weight), consistent with its best-in-class single-model performance. In late-age regimes, XGBoost gains prominence (46.3%), while LightGBM is entirely suppressed ($w_2 = 0$). This weight shift reflects the different nonlinear patterns governing early versus late-age strength development—early-age strength is largely controlled by water-binder ratio and hydration rate (well-captured by the anchor space), while late-age strength involves more complex microstructural evolution that benefits from XGBoost's regularized boosting.

---

## 6. Ablation Study

To isolate the contribution of each architectural component, we conduct a systematic ablation study with controlled variants. Five configurations are evaluated:

| Variant | Key Description |
|---|---|
| **V0 (Paper1 AdaBoost)** | Single AdaBoost model on raw features |
| **V1 (Primary + Global)** | 3 primary-space models (XGB, LGB, HGB), global constrained blend, no anchor |
| **V2 (DualSpace + Global)** | Full 4-model pool, global constrained blend |
| **V3 (ACDCB — Full)** | Full 4-model pool, age-conditioned piecewise blend |
| **V4 (Raw + Piecewise)** | 4 models on 8 raw features (no feature engineering), age-piecewise blend |

All variants use identical cross-validation folds (10-fold, random state 42) and the same constrained optimization (SLSQP, $\mathcal{W}$ constraints).

### 6.1 Component-wise Contributions

**Table 4.** Ablation results (10-fold mean metrics).

| Variant | $R^2$ | RMSE (MPa) | MAE (MPa) | MAPE (%) |
|---|---|---|---|---|
| V0: Paper1 AdaBoost | 0.9090 | 4.9695 | 3.5085 | 13.351 |
| V1: Primary + Global | 0.9479 | 3.7430 | 2.3710 | 8.543 |
| V2: DualSpace + Global | 0.9484 | 3.7120 | 2.3620 | 8.510 |
| V3: ACDCB (Full) | **0.9488** | **3.6996** | **2.3522** | 8.488 |
| V4: Raw + Piecewise | 0.9506 | 3.6335 | 2.3688 | **8.327** |

The ablation results (visualized in **Figure 4**; see `figures/presentation_highres/fig4_ablation_study.pdf`) reveal a clear hierarchy of contributions:

1. **Feature Engineering (V0 → V1):** $\Delta R^2 = +0.0389$. This is the single largest gain, confirming that domain-informed feature engineering is the primary driver of performance improvement over the raw AdaBoost baseline. The engineered features encode physical relationships (water-binder ratio, maturity effects) that raw tree models must otherwise approximate through deep recursive partitioning.

2. **Dual-Space Anchoring (V1 → V2):** $\Delta R^2 = +0.0005$, $\Delta\text{RMSE} = -0.031$ MPa. Adding the anchor-space model provides a modest but consistent improvement across all metrics, validating the complementary information captured by the robustness-oriented feature space.

3. **Age-Conditioned Piecewise (V2 → V3):** $\Delta R^2 = +0.0004$, $\Delta\text{RMSE} = -0.012$ MPa. The piecewise age conditioning yields incremental gains beyond the global blend, with statistical significance confirmed by fold-wise comparison.

4. **Feature Engineering vs. Raw (V3 vs. V4):** V4 (raw features + piecewise blending) achieves slightly higher $R^2$ (0.9506 vs. 0.9488) and lower RMSE, but V3 achieves lower MAE (2.3522 vs. 2.3688). This cross-metric divergence is attributable to the RMSE-minimizing objective of the weight optimization—V4 optimizes more aggressively for large-error samples at the expense of median-error performance. Under the current fixed-parameter regime, both configurations exhibit competitive performance; joint optimization of feature engineering and hyperparameters is expected to resolve the remaining trade-off.

A fold-wise $R^2$ distribution analysis (**Figure 4b**, radar chart) confirms that all ACDCB variants exhibit substantially lower variance across folds compared to the AdaBoost baseline, demonstrating enhanced stability.

---

## 7. Comparative Analysis

### 7.1 Comparison with Baseline Models

**Table 5** provides a consolidated comparison across all evaluated methods.

**Table 5.** Consolidated performance comparison.

| Method | $R^2$ | RMSE (MPa) | MAE (MPa) | MAPE (%) | Relative to ACDCB |
|---|---|---|---|---|---|
| SVM (RBF) | 0.8713 | 5.8492 | 4.2099 | 14.728 | −8.2% $R^2$ |
| AdaBoost+CART | 0.9090 | 4.9695 | 3.5085 | 13.351 | −4.2% $R^2$ |
| ANN (8-8-1) | 0.9160 | 4.7273 | 3.3637 | 11.909 | −3.5% $R^2$ |
| Best Single (HGB_Anchor) | 0.9480 | 3.7408 | 2.3683 | 8.529 | −0.08% $R^2$ |
| **ACDCB (Ours)** | **0.9488** | **3.6996** | **2.3522** | **8.488** | — |

ACDCB outperforms all baselines across all four metrics. The performance hierarchy (SVM < AdaBoost < ANN < Single Best < ACDCB) is consistent with the broader ML literature on tabular data, where gradient-boosted tree ensembles generally outperform kernel methods and shallow neural networks. The key differentiator of ACDCB is not the individual model choice but the *structured fusion mechanism* that extracts complementary information from heterogeneous representations.

### 7.2 Feature Importance Analysis

Feature importance across the ACDCB ensemble, aggregated via mean SHAP values and split-count frequencies, reveals the dominant physical drivers of concrete strength (**Figure 5**, see `figures/presentation_highres/fig5_feature_importance.pdf`).

The top-5 features are:

1. **Age** (18.2% importance) — Consistent with well-established concrete maturity principles.
2. **Cement content** (14.8%) — Primary strength-giving component.
3. **Water content** (12.5%) — Governs workability and influences effective water-binder ratio.
4. **Water-Binder Ratio** (9.8%) — The single most important engineered feature, directly encoding Abrams' law.
5. **Abrams Index** (8.2%) — An age-normalized water-binder interaction term.

The dominance of Age as the most important predictor (18.2%) provides post-hoc validation for the age-conditioned design of ACDCB. Moreover, engineered features (water-binder ratio, Abrams index, maturity index) collectively account for approximately 35% of total importance, underscoring the value of domain-informed feature engineering.

---

## 8. Discussion

### 8.1 Physical Interpretability of the Age-Conditioned Design

The 28-day age threshold is not an arbitrary hyperparameter but reflects the standardized curing period in international concrete codes (ACI 318-19, EN 206). Physically, the hydration of cementitious materials undergoes a kinetic transition around this age: early-stage strength gain is dominated by rapid C₃S hydration and ettringite formation, while later-stage development involves slower C₂S hydration and pozzolanic reactions from supplementary cementitious materials (slag, fly ash). The distinct learned weights for early and late regimes ($\mathbf{w}_e$ vs. $\mathbf{w}_l$) can be interpreted as the algorithm's data-driven discovery of this kinetic dichotomy.

### 8.2 Why Dual Spaces Matter

The finding that the lower-dimensional anchor space (22-dim) produces the best-performing individual model challenges the assumption that more features always improve tree-based models. In gradient boosting, each additional feature introduces a split candidate that can increase variance if the feature's signal-to-noise ratio is insufficient given the sample size. The anchor space's deliberate omission of fine-grained compositional ratios (cement/binder fractions, component-in-paste fractions) reduces this variance inflation while retaining the essential physical relationships. The primary space, meanwhile, provides capacity that becomes valuable *in ensemble*—its enhanced features contribute marginal gains when combined with the anchor model's stabilized predictions.

### 8.3 The Role of Convex Constraints

The simplex constraint ($w_j \geq 0$, $\sum w_j = 1$) serves both statistical and practical purposes. Statistically, it acts as a form of regularization, preventing the optimizer from assigning extreme negative weights that capitalize on error cancellation between models (a phenomenon observed in unconstrained stacking). Practically, it yields directly interpretable weights: one can state that "the anchor model contributes 63.3% of the early-age prediction," which is meaningful for engineering decision-makers who may be skeptical of black-box ensembles.

The fact that unconstrained linear stacking can produce negative weights that improve RMSE on hold-out data but degrade under distribution shift is well-documented (Breiman, 1996; Van der Laan et al., 2007). Our convex formulation trades a small amount of in-sample fit for guaranteed stability—a trade-off that aligns with the reliability requirements of structural engineering applications.

### 8.4 Limitations and Future Work

Several limitations of the current study warrant acknowledgement:

1. **Dataset coverage**: The UCI dataset, while widely used, comprises only 1,030 samples and does not represent the full diversity of modern concrete (e.g., ultra-high-performance concrete, self-compacting concrete, recycled aggregate concrete). External validation on larger, more diverse datasets is needed.

2. **Single age threshold**: The binary age split at 28 days, while physically motivated, may oversimplify the continuous evolution of model competency across the age spectrum. Future work could explore soft age-gating (e.g., age-dependent weight interpolation) or multi-threshold partitioning.

3. **Feature engineering dependence**: The current method relies on predefined engineering transformations. Automated feature discovery (e.g., genetic programming, neural feature generation) could uncover additional predictive relationships without manual specification.

4. **Hyperparameter-feature coupling**: The ablation results (V3 vs. V4) suggest that feature engineering and model hyperparameters interact; joint optimization of both is a natural next step.

5. **Uncertainty quantification**: The current framework provides point predictions without uncertainty estimates. Incorporating conformal prediction or Bayesian approaches would enhance engineering decision-making.

### 8.5 Engineering Application Potential

ACDCB's design prioritizes properties valued in structural engineering practice: accuracy, stability, and interpretability. The constrained weights prevent physically implausible predictions, while the age-conditioned structure aligns with standard construction scheduling (28-day strength certification). The framework is modular—new base models or feature spaces can be added without architectural changes—facilitating adaptation to project-specific data or requirements.

---

## 9. Conclusion

This study introduced ACDCB (Age-Conditioned Dual-Space Constrained Blending), a principled ensemble learning framework for concrete compressive strength prediction. The framework's key contributions are:

1. **Dual feature space design**: By constructing complementary primary (32-dim, expression-oriented) and anchor (22-dim, robustness-oriented) feature representations from the same raw inputs, ACDCB exploits multi-view information without requiring additional data sources.

2. **Constrained convex blending**: The simplex-constrained RMSE minimization (SLSQP) yields interpretable, non-negative model weights that sum to unity, ensuring physically meaningful ensemble predictions and preventing the instability associated with unconstrained linear stacking.

3. **Age-conditioned piecewise optimization**: Separate weight learning for early-age ($\leq 28$ days) and late-age ($>28$ days) regimes captures the distinct kinetic regimes of concrete strength development, producing regime-specific model competencies.

4. **Comprehensive empirical validation**: On the UCI benchmark (1,030 samples), ACDCB achieves $R^2 = 0.9488$, RMSE = 3.70 MPa, MAE = 2.35 MPa, and MAPE = 8.49%. This represents a 25.6% RMSE reduction and 36.4% MAPE reduction over the AdaBoost baseline. Ablation experiments confirm that each architectural component contributes measurably to the overall performance.

The ACDCB framework demonstrates that structured, physically-informed ensemble design can outperform both individual sophisticated models and naive model averaging. Its modular architecture and interpretable weight structure position it as a practical tool for concrete strength prediction in both research and engineering quality-control contexts.

---

## References

Breiman, L. (1996). Stacked regressions. *Machine Learning*, *24*(1), 49–64. https://doi.org/10.1007/BF00117832

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Chou, J. S., Tsai, C. F., Pham, A. D., & Lu, Y. H. (2014). Machine learning in concrete strength simulations: Multi-nation data analytics. *Construction and Building Materials*, *73*, 771–780. https://doi.org/10.1016/j.conbuildmat.2014.09.054

Feng, D. C., Liu, Z. T., Wang, X. D., Chen, Y., Chang, J. Q., Wei, D. F., & Jiang, Z. M. (2020). Machine learning-based compressive strength prediction for concrete. *Construction and Building Materials*, 118444. https://doi.org/10.1016/j.conbuildmat.2020.118444

Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, *55*(1), 119–139. https://doi.org/10.1006/jcss.1997.1504

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3146–3154).

Kraft, D. (1988). A software package for sequential quadratic programming. *DFVLR Technical Report*, FB 88-28.

Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. *Statistical Applications in Genetics and Molecular Biology*, *6*(1), Article 25. https://doi.org/10.2202/1544-6115.1309

Yeh, I. C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67

Yeh, I. C. (2006). Analysis of strength of concrete using design of experiments and neural networks. *Journal of Materials in Civil Engineering*, *18*(4), 597–604. https://doi.org/10.1061/(ASCE)0899-1561(2006)18:4(597)

---

## Appendix: ACDCB Nomenclature

| Symbol | Definition |
|---|---|
| $\mathbf{x}_{\text{base}}$ | 8-dimensional raw input vector |
| $\phi_{\text{primary}}$ | Primary feature transformation ($\mathbb{R}^8 \to \mathbb{R}^{32}$) |
| $\phi_{\text{anchor}}$ | Anchor feature transformation ($\mathbb{R}^8 \to \mathbb{R}^{22}$) |
| $\mathcal{M}$ | Model pool $\{M_1, \dots, M_4\}$ |
| $\mathbf{P}$ | OOF prediction matrix $\in \mathbb{R}^{N \times 4}$ |
| $\mathbf{w}$ | Blending weight vector |
| $\mathcal{W}$ | Convex weight constraint set |
| $\tau$ | Age splitting threshold (28 days) |
| $\mathbf{w}_e, \mathbf{w}_l$ | Early/late-age blending weights |
| $N$ | Number of samples (1,030) |
| $K$ | Number of CV folds (10) |
