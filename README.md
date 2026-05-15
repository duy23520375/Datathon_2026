# Datathon 2026: The Gridbreakers

<div align="center">

🏆 **Top 15 / 527 Teams — Top 3%**  
*VinUniversity Datathon 2026, organized by VinTelligence — VinUni DS&AI Club*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Score](https://img.shields.io/badge/Score-93%2F100-brightgreen)

</div>

---

## 📌 Overview

This repository contains the full solution of team **UIT-Innovators** for **Datathon 2026: The Gridbreakers** — the first Data Science competition at VinUniversity. The challenge simulates a Vietnamese fashion e-commerce company's operations (2012–2022) across 15 interconnected CSV tables.

Our approach covers three components: data-driven MCQ analysis, deep exploratory data analysis with prescriptive business insights, and an ensemble machine learning model for revenue forecasting.

---

## 🏅 Results

| Component | Score |
|-----------|-------|
| Part 1 — MCQ | 20 / 20 |
| Part 2 — Visualization Quality | 14 / 15 |
| Part 2 — Analysis Depth | 24 / 25 |
| Part 2 — Business Insight | 14 / 15 |
| Part 2 — Creativity & Storytelling | 5 / 5 |
| Part 3 — Model Performance | 10 / 12 |
| Part 3 — Technical Report | 6 / 8 |
| **Total** | **93 / 100** |

> **Final Test MAE: 670,366.68** on the Kaggle leaderboard.

---

## 👥 Team & Contributions

| Member | Student ID | Role |
|--------|-----------|------|
| Nguyễn Hữu Khánh Duy | 23520375 | EDA (Part 2) — Diagnostic analysis (root cause identification: Pricing Paradox, Promo exploitation loophole); predictive extrapolation (inventory trap forecast, customer LTV decay); data storytelling & report writing |
| Tăng Hoàng Phúc | 23521219 | EDA (Part 2) — Uncovered the **Phantom Inventory Crisis** (30,495 contradictory records; mathematically proven via inventory accounting identity); prescriptive strategy design (What-if scenario modeling, Dynamic Replenishment framework, No-stacking Rule proposal); quantified business recommendations; report writing |
| Võ Hoàng Minh | 23520961 | MCQ (Part 1) — Data computation for all 10 questions; ML Engineer (Part 3) — Forecasting pipeline architecture, feature engineering (Tet Factor, cyclical encoding, payday distance), technical report |
| Hồ Hoàng Quân | 23521252 | ML Engineer (Part 3) — Heterogeneous Ensemble modeling (LightGBM + MLP), anchor mechanism design, SHAP explainability analysis, technical report |

---

## 📂 Repository Structure

```
Datathon_2026/
├── Data/                        # 15 competition CSV files (not tracked by git)
├── Part_1/
│   └── QA.ipynb                 # MCQ computation (Pandas-based)
├── Part_2/
│   ├── insight1-datathon.ipynb  # "The Chain of Destruction" analysis
│   └── insight2-datathon.ipynb  # Web traffic & CVR ecosystem analysis
├── Part_3/
│   └── model.py                 # Heterogeneous Ensemble forecasting pipeline
├── requirements.txt
└── README.md
```

---

## 🔍 Part 2 — Exploratory Data Analysis

We applied a **4-level analytical framework** (Descriptive → Diagnostic → Predictive → Prescriptive) across two independent investigation threads.

### Insight 1: "The Chain of Destruction" (`insight1-datathon.ipynb`)

A cross-system analysis connecting Inventory, Pricing, and Promotions data to decode how a single operational failure cascades into a full profit collapse.

**Key Findings:**

- **P&L Waterfall:** Gross revenue of 13.12B VND yielded only 1.21B VND net profit. Promotion spending consumed ~600M VND — nearly 50% of net profit.
- **Pricing Paradox:** 106,058 transactions (18.58% of all items) were sold below COGS, creating direct margin leakage before any discount was applied.
- **Phantom Inventory Crisis:** 30,495 records (50.6%) simultaneously showed `stockout_days > 0` and `overstock_flag = 1` — a logical impossibility proven mathematically (avg. 251.8 units available, yet system reported stockout).
- **Promo Exploitation Loophole:** 99.4% of loss-making orders had a promo code applied, extracting 296.8M VND from the marketing budget on already-negative-margin items.
- **10x Inventory Trap:** Days of Supply surged from 161 days (2012) to 1,638 days (2022). Linear regression (R² = 0.98) projects 2,055 days by 2025.
- **Dead Capital:** 96.1% of working capital (4.93B VND out of 5.13B VND) frozen in overstock inventory by end of 2022.

**Prescriptive Actions:**
1. Deploy ERP cross-validation logic: auto-lock markdown triggers when `stock_on_hand > 0` but `stockout_days > 15` — force physical audit.
2. Implement No-stacking Rule: disable all promo codes on SKUs where `UnitPrice ≤ COGS`.
3. Dynamic Replenishment: cut Open-To-Buy budget for Everyday & Activewear by 50%; reallocate to high-velocity Hero products using Seasonality Index.
4. What-if Scenario: applying 60% markdown on 540,081 stranded Outdoor units reduces net cash burn by 0.13B VND vs. status quo holding.

### Insight 2: Web Traffic & CVR Ecosystem (`insight2-datathon.ipynb`)

Analyzes web traffic as a proxy for marketing budget allocation and uncovers the relationship between traffic sources, conversion rate, and seasonal demand patterns.

**Key Findings:**
- Organic customers generate 68% higher LTV (168,323 VND) vs. sale-hunter customers acquired via promotions (100,193 VND), with 6.9 vs. 4.3 lifetime orders respectively.
- Seasonal failure map reveals stockouts peak precisely during high-demand months (April, August), confirming procurement is misaligned with demand cycles.

---

## 🤖 Part 3 — Revenue Forecasting

### Methodology

A **Heterogeneous Ensemble** combining gradient boosting and neural network models, augmented by a domain-specific anchor mechanism.

### Feature Engineering

| Feature Group | Description |
|---------------|-------------|
| **Tet Factor (`tf`)** | Decay-weighted coefficient around Vietnamese Lunar New Year |
| **Day Factor (`df`)** | Day-of-week signal with Tet noise removed |
| **Month-Day Factor (`mf`)** | Fixed public holiday capture |
| **Cyclical Encoding** | `sin/cos` transformation for month, weekday, day-of-month (T ∈ {7, 12, 31}) |
| **Payday Distance** | Distance to 1st and 15th of month (Vietnamese salary cycle) |
| **Lag Features** | t−364, t−371, t−728 (year-aligned lags) |

### Anchor Mechanism

To control long-range forecast drift, predictions are anchored to a growth baseline:

```
ŷ_anchor = b₂₂ · γ^(year−2022) · tf · df · mf
```

Where `b₂₂` is the 2022 baseline revenue and `γ = 1.12` is the expected growth multiplier.

### Model Ensemble

| Model | Objective |
|-------|-----------|
| LightGBM | L2 (MSE) |
| LightGBM | Tweedie (skewed distributions) |
| LightGBM | Huber (outlier-robust) |
| LightGBM | Trained on `log(1+x)` space |
| LightGBM | Trained on `√x` space |
| MLP (128→64→32) | High-order feature interactions |
| MLP (256→128→64) | High-order feature interactions |

Final output: ensemble average → post-processing normalization to match `TARGET_MEAN` from anchor.

### Results

| Metric | Value |
|--------|-------|
| Final Test MAE | **670,366.68** |
| Random Seed | 42 (fully reproducible) |

Explainability outputs (SHAP summary plots, permutation importance, dependence plots) are auto-generated into `explainpicture/` on each run.

---

## 🚀 Reproducing Results

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare data

Place all 15 competition `.csv` files inside the `Data/` directory at the project root.

### Step 3: Run each part

**Part 1 & Part 2 (Notebooks):**

Open `.ipynb` files in Jupyter. Replace Kaggle paths with local paths:

```python
# Replace:
pd.read_csv('/kaggle/input/.../orders.csv')

# With:
pd.read_csv('../Data/orders.csv')
```

Then run all cells top to bottom.

**Part 3 (Forecasting Model):**

Update the three data paths at the top of `model.py`:

```python
train_raw = pd.read_csv('../Data/sales.csv', parse_dates=['Date'])
test_raw  = pd.read_csv('../Data/sample_submission.csv', parse_dates=['Date'])
promos    = pd.read_csv('../Data/promotions.csv', parse_dates=['start_date', 'end_date'])
```

Then run:

```bash
cd Part_3
python model.py
```

Output: `final_submission.csv` + `explainpicture/` folder with SHAP visualizations.

---

## 🛠️ Tech Stack

**EDA & Visualization:** `pandas` · `plotly` · `seaborn` · `matplotlib`  
**Machine Learning:** `lightgbm` · `scikit-learn` (MLPRegressor)  
**Explainability:** `shap` · Permutation Importance  
**Optimization:** `optuna`  
**Reproducibility:** `SEED = 42` throughout all stochastic components

---

## 📄 Report

Full technical report (NeurIPS format, 4 pages + appendix) is included in the repository, covering:
- Business insight narrative with 10 supporting visualizations
- Model pipeline architecture and feature engineering rationale
- SHAP-based model explainability analysis
- What-if scenario quantification for inventory decisions

---

*University of Information Technology (UIT) — Vietnam National University Ho Chi Minh City*  
*Contact: {23520375, 23521219, 23520961, 23521252}@gm.uit.edu.vn*
