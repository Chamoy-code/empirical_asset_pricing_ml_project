# Data Pipeline & Methodology Explanation

This document explains the transformation process from raw data to the final predictive dataset, detailing the **why** and **how** of each step and its critical role in the subsequent phases of the project (Modeling, Validation, Portfolio Construction).

## 1. Input Data Sources

-   **`datashare.csv` (Micro)**: Contains stock-level characteristics for thousands of US firms over time.
    -   *Key Content*: Permnos (IDs), Dates, Returns (`mom1m`), and 94 financial characteristics (e.g., `bm` for Book-to-Market, `mvel1` for Market Equity).
-   **`PredictorData2022 (1).xlsx` (Macro)**: Contains monthly macroeconomic indicators from Welch & Goyal.
    -   *Key Content*: 8 State variables describing the economy (e.g., `tb` for Treasury Bills, `dp` for Dividend Yield, `infl` for Inflation).

## 2. The Transformation Process (`src/rebuild_dataset.py`)

We transformed these raw inputs into an "Analytic Dataset" (`data/rebuilt_dataset_parts/`) specifically designed for Machine Learning in Asset Pricing (Gu, Kelly, & Xiu, 2020).

### Step A: Universe Filtering & Cleaning
-   **Action**: Select data from **March 1957 to December 2021**.
-   **Why?**: This period matches the extended benchmark study (Project 2020 Update). It ensures we have a consistent universe of stocks and avoids "pre-sample" bias where data quality might be low.
-   **Impact on Phase 2 (Models)**: Ensures models are trained on reliable, standard data, making your results comparable to literature.

### Step B: Feature Engineering (Micro)
-   **Action 1: Rank Normalization via Cross-Section**:
    -   For each month $t$, strictly rank every characteristic $c_{i,t}$ into the interval $[-1, 1]$.
    -   *Formula*: $x_{rank} = \frac{2 \cdot \text{rank}(x)}{N} - 1$.
    -   **Why?**: Stock characteristics have extreme outliers (e.g., a small firm with massive BM ratio). Machine Learning models (especially Neural Networks in **Phases 2 & 5**) fail if inputs are not scaled. Ranking is robust to outliers and static scaling.
-   **Action 2: Missing Data Imputation**:
    -   Replace missing values with the cross-sectional median for that month.
    -   **Why?**: Discarding rows with 1 missing feature would delete 90% of the data. Median imputation preserves the "neutral" signal (0 after ranking).

### Step C: Macro Interactions (The "conditionality" Step)
-   **Action**: Compute Kronecker products $z_{i,t} = c_{i,t} \times x_t$.
    -   Multiplies *every* stock characteristic ($c$) by *every* macro variable ($x$).
    -   **Result**: Expands 94 features $\rightarrow$ ~920 predictive signals.
-   **Why?**: **This is the core economic insight.** A characteristic like "Momentum" might work well in expansions but crash in recessions. By interacting Momentum $\times$ Inflation, the model can learn a *dynamic* (time-varying) beta.
-   **Impact on Phase 5 (Interpretation)**: Allows us to interpret *when* a strategy works (e.g., "Value works when Inflation is high").

### Step D: Target Generation
-   **Action**: Created `ret` (1-month forward return) and `ret12` (12-month forward return).
-   **Why?**: We want to predict *future* returns ($r_{t+1}$), not contemporaneous ones. `ret12` allows checking if signals persist over longer horizons (1 year).

## 3. Statistical Analysis (`src/generate_statistics.py`)

We generated statistics in `results/` to validate data quality *before* trusting the models.

-   **`time_series_summary.csv`**:
    -   *Tracks*: Number of stocks over time.
    -   *Why?*: Ensures our universe isn't artificially small in 1970 or dropping to zero in 2010. Consistent sample size is vital for **Phase 3 (Validation)**.
-   **`characteristic_stats.csv`**:
    -   *Metric 1: Rank IC (Information Coefficient)*: The correlation between a feature and future return. A high IC (like 0.05) implies the feature is a strong predictor on its own.
    -   *Metric 2: Persistence (Autocorrelation)*: How slowly a feature changes. High persistence (0.95+) means the signal is stable (low turnover).
    -   **Impact on Phase 4 (Portfolio)**: If models pick features with **low persistence**, your portfolio turnover will be huge, and transaction costs will kill profits. Knowing this *ex-ante* helps explaining poor Net Returns later.

## 4. Relevance to Future Phases

### PHASE 2: Model Implementation (Fixed Period)
-   **Data Role**: The normalized inputs ($[-1, 1]$) ensure OLS, ElasticNet, and Neural Networks can all train on the *same* data without numerical instability/exploding gradients.
-   **Stats Role**: If models fail (R2 < 0), we check `data_summary_stats.csv` to ensure targets weren't outliers/errors.

### PHASE 3: Validation & Tuning (Recursive Refitting)
-   **Data Role**: The dataset is ordered by `DATE`. We will expand the training window year-by-year (e.g., train 1957-1990, valid 1991... train 1957-1991, valid 1992).
-   **Why?**: Financial markets evolve. The "Interactions" (Step C) allow the model to adapt to changing macroeconomic regimes during this refitting.

### PHASE 4: Portfolio Construction
-   **Data Role**: We use the predicted returns ($\hat{r}_{i, t+1}$) from models to sort stocks into Deciles.
-   **Why?**: We buy Top 10% / Sell Bottom 10%. The quality of this sort depends entirely on the signal-to-noise ratio of the 920 input features we built.

### PHASE 5: Interpretation
-   **Data Role**: Variable Importance analysis will tell us *which* of the 920 features drive returns.
-   **Connection**: We will compare the model's "Top Features" vs. the "High Rank IC" features from `characteristic_stats.csv` to see if the Machine Learning model found "hidden" patterns that simple correlation missed.
