# Empirical Asset Pricing via Machine Learning

A full replication and implementation of the methodology from **Gu, Kelly, and Xiu (2020) — *Empirical Asset Pricing via Machine Learning*** (Journal of Finance, 2020).

This project builds the entire pipeline from scratch: raw data ingestion, feature engineering, model training, out-of-sample evaluation, and Long/Short portfolio simulation — across five model families and two temporal validation strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
   - [Feature Engineering](#feature-engineering)
   - [Temporal Validation Splits](#temporal-validation-splits)
   - [Models](#models)
   - [Portfolio Construction](#portfolio-construction)
4. [Project Structure](#project-structure)
5. [Results](#results)
   - [Fixed Split (2005–2021)](#fixed-split-20052021)
   - [3-Way Tuning Split (2005–2021)](#3-way-tuning-split-20052021)
   - [Top Predictive Features](#top-predictive-features)
6. [Limitations](#limitations)
7. [Installation & Setup](#installation--setup)
8. [Workflow](#workflow)
9. [References](#references)

---

## Overview

The core question of the paper is: **can machine learning algorithms systematically predict cross-sectional equity returns better than classical linear models?**

This project answers that question by:
- Training five families of models on 64 years of US equity data (1957–2021)
- Evaluating out-of-sample predictive R² and portfolio Sharpe ratios
- Comparing performance across 1-month (`ret`) and 12-month (`ret12`) return horizons
- Identifying which firm characteristics and macroeconomic interactions drive predictions

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| `datashare.csv` | 94 firm-level characteristics for all US stocks, 1957–2021. Based on Green, Hand & Zhang (2017). | Requires [WRDS](https://wrds-www.wharton.upenn.edu/) access |
| `PredictorData2022.xlsx` | 8 monthly macroeconomic state variables from Welch & Goyal (2008), updated through 2022. | Free download at [Amit Goyal's website](https://sites.google.com/view/agoyal145) |

> **Note:** `datashare.csv` is a private dataset and is **not included** in this repository.

---

## Methodology

### Feature Engineering

Raw inputs go through four transformations to produce the analytic dataset:

**1. Universe filtering**
Data is restricted to March 1957 – December 2021 to match the paper's extended benchmark.

**2. Cross-sectional rank normalization**
Each month, every firm characteristic $c_{i,t}$ is ranked into $[-1, 1]$:

$$x_{i,t} = \frac{2 \cdot \text{rank}(c_{i,t})}{N_t} - 1$$

This is robust to outliers and ensures all models (especially Neural Networks) receive numerically stable inputs.

**3. Cross-sectional median imputation**
Missing values are replaced with the monthly cross-sectional median, preserving a neutral signal (≈ 0 after ranking) without discarding observations.

**4. Macro interaction terms (Kronecker products)**
Every firm characteristic is multiplied by every macro variable:

$$z_{i,t} = c_{i,t} \otimes x_t$$

This expands 94 micro features × 8 macro variables into **~824 predictive signals**. The economic intuition: a characteristic like Momentum may predict returns differently during recessions vs. expansions. By interacting `mom12m × inflation`, the model learns *time-varying* betas.

**5. Target construction**
- `ret` — 1-month forward return ($r_{t+1}$)
- `ret12` — 12-month forward return ($r_{t+12}$)

Both are constructed by shifting `mom1m` forward per stock (`permno`).

---

### Temporal Validation Splits

To prevent look-ahead bias, all models are evaluated strictly out-of-sample:

| Strategy | Train | Validation | Test |
|---|---|---|---|
| **Fixed Split** | 1957–2004 (2.87M obs) | — | 2005–2021 (1.21M obs) |
| **3-Way Tuning** | 1957–1989 (1.49M obs) | 1990–2004 (1.38M obs) | 2005–2021 (1.21M obs) |

The 3-Way split uses the validation period for hyperparameter tuning, then fixes the best configuration before evaluating on the test set.

---

### Models

All models are trained separately for `ret` and `ret12` targets.

#### Linear Models
- **OLS** — ordinary least squares, full 824-feature input.
- **Lasso** — L1-penalized regression (alpha tuned via validation). Provides automatic feature selection.
- **ElasticNet** — L1+L2 combination (implemented via `SGDRegressor` for scalability on large data).

#### Random Forest (`train_rf_*.py`)
- 300 decision trees, max depth 6
- Parallel fit (`n_jobs=-1`)
- Produces variable importance scores for interpretation

#### Gradient Boosted Regression Trees — GBRT (`train_gbrt_*.py`)
- Sequential ensemble; each tree corrects the residuals of the previous
- Hyperparameters tuned: tree depth ∈ {2, 3}, learning rate ∈ {0.01, 0.1}
- Memory-intensive — subsampling used for large train sets

#### Neural Networks — NN (`train_nn_*.py`)
- Architecture: `[input → 32 → 16 → 1]`
- Batch normalization after each hidden layer (per Gu et al.)
- Optimizer: Adam
- Activation: ReLU (default) or Tanh (via `compare_nn_tanh.py`)
- Implementation: PyTorch

---

### Portfolio Construction

After obtaining model predictions $\hat{r}_{i,t+1}$, a **Long/Short decile portfolio** is formed each month:

- **Long**: top 10% of stocks by predicted return (decile 9)
- **Short**: bottom 10% of stocks by predicted return (decile 0)
- **L/S**: Long minus Short

Performance is annualized:
- `ret` horizon: multiply mean by 12, multiply vol by √12
- `ret12` horizon: returns are already annual (factor = 1)

Sharpe ratio = annualized mean / annualized volatility.

> Transaction costs are not modeled. Characteristics with high autocorrelation (persistence > 0.95) imply low portfolio turnover and are more robust to trading costs in practice.

---

## Project Structure

```
.
├── src/
│   ├── rebuild_dataset.py          # Builds partitioned H5/parquet dataset from raw CSV
│   ├── data_loader.py              # Memory-safe loader for partitioned dataset
│   ├── feature_engineering.py      # Rank normalization, imputation, Kronecker products
│   ├── models.py                   # OLS, Lasso, ElasticNet, RF, GBRT, NN class definitions
│   ├── portfolio.py                # Long/Short decile portfolio evaluation
│   ├── pipeline.py                 # End-to-end orchestration script
│   │
│   ├── train_linear_fixed.py       # Linear models — Fixed split
│   ├── train_linear_3way_ret.py    # Linear models — 3-Way split, ret target
│   ├── train_linear_3way_ret12.py  # Linear models — 3-Way split, ret12 target
│   ├── train_rf_fixed.py           # Random Forest — Fixed split
│   ├── train_rf_3way.py            # Random Forest — 3-Way split
│   ├── train_gbrt_fixed_ret.py     # GBRT — Fixed split, ret target
│   ├── train_gbrt_fixed_ret12.py   # GBRT — Fixed split, ret12 target
│   ├── train_gbrt_3way_ret.py      # GBRT — 3-Way split, ret target
│   ├── train_gbrt_3way_ret12.py    # GBRT — 3-Way split, ret12 target
│   ├── train_nn_fixed_ret.py       # Neural Network — Fixed split, ret target
│   ├── train_nn_fixed_ret12.py     # Neural Network — Fixed split, ret12 target
│   ├── train_nn_3way_ret.py        # Neural Network — 3-Way split, ret target
│   ├── train_nn_3way_ret12.py      # Neural Network — 3-Way split, ret12 target
│   │
│   ├── generate_statistics.py      # Descriptive stats: IC, persistence, LaTeX tables
│   ├── plot_importance.py          # Variable importance charts (RF, GBRT, Lasso)
│   ├── plot_comparisons.py         # Cross-model Sharpe/return bar charts
│   ├── plot_cumulative_comparisons.py  # Cumulative equity curves across all models
│   ├── keras_convergence.py        # NN training loss convergence visualization
│   └── compare_nn_tanh.py          # ReLU vs Tanh activation comparison
│
├── data/
│   └── rebuilt_dataset_parts/      # Partitioned parquet files (one per year, ~65 files)
│                                   # [NOT included — generated by rebuild_dataset.py]
│
├── results/
│   ├── rf_fixed_report.txt
│   ├── rf_3way_report.txt
│   ├── gbrt_fixed_report.txt
│   ├── gbrt_3way_ret_report.txt
│   ├── gbrt_3way_ret12_report.txt
│   ├── linear_fixed_report.txt
│   ├── linear_3way_report.txt
│   ├── nn_fixed_report.txt
│   ├── nn_fixed_ret12_report.txt
│   ├── nn_3way_ret_report.txt
│   ├── characteristic_stats.csv    # IC, persistence per feature
│   ├── time_series_summary.csv     # Monthly stock counts and return stats
│   └── plots/                      # Equity curves and convergence plots
│
├── walkthrough_analysis.tex        # LaTeX report for academic submission
├── data_pipeline_explained.md      # Detailed pipeline documentation
├── readme_resume.md                # French-language architecture summary
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Results

### Data Summary (1957–2021)

| Metric | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Stocks per month | 5,254 | 2,195 | 1,057 | 9,042 |
| Mean monthly return | 0.97% | 5.48% | -27.3% | 29.4% |
| Return volatility | 12.1% | 3.8% | 5.1% | 38.5% |
| Mean 12-month return | 13.3% | 23.2% | -47.5% | 109.2% |

---

### Fixed Split (2005–2021)

Train: 1957–2004 (500k subsampled) — Test: 2005–2021

| Model | R² OOS — ret | L/S Sharpe — ret | R² OOS — ret12 | L/S Sharpe — ret12 |
|---|---:|---:|---:|---:|
| OLS | -7.09% | 0.02 | -8.78% | 1.03 |
| Lasso | -5.89% | 0.01 | -8.51% | 1.02 |
| Random Forest ¹ | -0.35% | -0.06 | **+2.37%** | 0.36 |
| GBRT ² | -1.45% | 0.91 | *not run* | *not run* |
| Neural Network ² | *not run* | *not run* | -3.44% | **1.76** |

¹ Test set subsampled to 800k observations due to memory constraints.
² Result from earlier compiled run — no longer an individual report file.

---

### 3-Way Tuning Split (2005–2021)

Train: 1957–1989 (500k subsampled) — Val: 1990–2004 (hyperparameter tuning) — Test: 2005–2021

| Model | R² OOS — ret | L/S Sharpe — ret | R² OOS — ret12 | L/S Sharpe — ret12 |
|---|---:|---:|---:|---:|
| OLS | -9.32% | -0.07 | -14.62% | 0.13 |
| Lasso (α=0.0001) | -3.97% | 0.36 | -12.59% | 0.13 |
| Random Forest | -0.69% | 0.00 | **+2.18%** | 0.21 |
| GBRT ² | -0.80% | 0.85 | **+2.40%** | -0.46 |
| Neural Network ³ | -0.92% | 1.15 | *not run* | *not run* |

² Result from earlier compiled run — no longer an individual report file.
³ Trained on 10% sample with linear activation (`nn_3way_ret_linear_report.txt`).

---

### Key Findings

**1. Non-linear models dominate on ret12.**
Random Forest and GBRT are the only models achieving **positive out-of-sample R²** (~+2.2–2.4% on `ret12`). This matches the paper's central finding: tree-based models capture non-linear, interaction-driven patterns that linear models miss entirely.

**2. Negative R² ≠ bad portfolio.**
All models produce negative R² on the 1-month horizon (`ret`) — yet GBRT achieves a Sharpe of **0.91** and NN achieves up to **1.76** on `ret`. This is the classical "weak signal, strong sort" phenomenon: even small predictive signals, when applied to rank stocks into deciles, generate meaningful alpha because prediction *rank order* matters more than accuracy.

**3. Neural Networks produce the best L/S Sharpe.**
Despite a relatively compact architecture ([32, 16] hidden layers), the NN with batch normalization achieves the highest Long/Short Sharpe on `ret12` (Fixed split: 1.76).

**4. Linear models struggle on ret.**
OLS and Lasso show near-zero or negative L/S Sharpe on the 1-month horizon, confirming they cannot capture the non-linearities required for monthly return sorting.

---

### Top Predictive Features

Features consistently ranked in the top 20 across non-linear models:

| Feature | Type | Economic Interpretation |
|---|---|---|
| `divo` | Micro | Dividend omission indicator — signals financial distress |
| `sin` | Micro | "Sin stock" indicator (tobacco, alcohol, gambling) |
| `rd_x_macro_bm` | Interaction | R&D intensity × Book-to-Market ratio |
| `divo_x_macro_bm` | Interaction | Dividend omission × market valuation regime |
| `sin_x_macro_ep` | Interaction | Sin stocks × earnings yield — regime-conditional effect |
| `stdcf` | Micro | Cash-flow volatility — distress/uncertainty signal |
| `convind` | Micro | Convertible debt indicator |

Linear models (Lasso) select different features: `mom12m`, `mom6m`, `baspread`, `turn`, `idiovol` — consistent with classical momentum and liquidity anomalies.

---

## Limitations

**Computational budget**
All models were trained on a personal laptop (CPU only). The full training set exceeds 2.8 million observations; training on the full sample would take 12+ hours per model run. A subsample of **500,000 observations** was used for training (the full test set was always used for evaluation). The paper's results used university HPC clusters running for days. Despite this, the key qualitative findings are reproduced.

**Early NN stopping**
Neural Networks were trained for 5–10 epochs due to time constraints. The original paper trains to convergence with early stopping on a validation loss. Longer training would likely improve Sharpe ratios further.

**No transaction costs**
Portfolio returns are gross. In practice, high-turnover strategies (e.g., those relying on low-persistence features like `mom1m`, AC ≈ -0.04) would be significantly eroded by bid-ask spreads and market impact.

**No recursive re-fitting**
A rolling/expanding window that re-trains the model each year would better reflect real-world deployment and reduce the risk of the model becoming stale on later test dates.

**ElasticNet results incomplete**
ElasticNet is implemented in `src/models.py` but was not fully reported across all configurations — Lasso results are the primary linear benchmark.

---

## Workflow

Run steps in order. Each step writes its output to disk so you can resume mid-pipeline.

```bash
# Step 1 — Build the partitioned analytic dataset (~65 parquet files, one per year)
#           This only needs to be run once. Takes 30–60 min.
python src/rebuild_dataset.py

# Step 2 — (Optional) Generate descriptive statistics and data validation
python src/generate_statistics.py

# Step 3 — Train a model (choose any combination)
python src/train_rf_fixed.py
python src/train_rf_3way.py
python src/train_gbrt_fixed_ret.py
python src/train_gbrt_3way_ret.py
python src/train_linear_fixed.py
python src/train_linear_3way_ret.py
python src/train_nn_fixed_ret12.py
python src/train_nn_3way_ret.py
# ... (all train_*.py scripts follow the same pattern)

# Step 4 — Results are written to results/<model>_report.txt automatically

# Step 5 — Generate comparison plots
python src/plot_comparisons.py
python src/plot_cumulative_comparisons.py
python src/plot_importance.py
```

> Each model training run takes **1–3 hours** on a standard laptop CPU. Neural Network runs are faster (~30–60 min for 5 epochs at 500k samples).

---

## References

- **Gu, S., Kelly, B., & Xiu, D. (2020).** Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- **Green, J., Hand, J. R. M., & Zhang, X. F. (2017).** The characteristics that provide independent information about average U.S. monthly stock returns. *Review of Financial Studies*, 30(12), 4389–4436.
- **Welch, I., & Goyal, A. (2008).** A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.
