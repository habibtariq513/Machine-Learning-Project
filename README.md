# 🏦 Fannie Mae Mortgage Default Prediction
### Machine Learning Project with Full MLflow Integration

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project applies end-to-end Machine Learning on the **Fannie Mae Single-Family Loan Performance Dataset** to predict whether a mortgage borrower will **default** on their loan.

Fannie Mae is a US government-sponsored enterprise that buys home loans from banks. Early default prediction helps financial institutions:
- Reduce credit risk exposure
- Make better loan approval decisions
- Identify high-risk borrowers before they default

All experiments are tracked using **MLflow** with logged metrics, parameters, confusion matrices, and saved model artifacts.

---

## 🎯 Problem Statement

> **Binary Classification:** Given a borrower's financial profile and loan characteristics, predict whether they will default (`is_default = 1`) or not (`is_default = 0`).

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| No Default | 0 | 4,741 | 60% |
| Default | 1 | 3,161 | 40% |

---

## 📁 Project Structure

```
fannie-mae-default-prediction/
│
├── 📓 fannie_mae_mlflow_project.ipynb   ← Main notebook (all steps)
│
├── 📂 data/
│   └── Mldata_fannie_mae_df.csv         ← Dataset (7,902 rows × 34 cols)
│
├── 📂 mlruns/                           ← MLflow tracking folder (auto-generated)
│   └── (experiment runs saved here)
│
├── 📂 results/                          ← Saved plots and figures
│   ├── eda_target_distribution.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_feature_distributions.png
│   ├── eda_target_correlation.png
│   ├── run2_model_comparison.png
│   ├── pca_variance.png
│   ├── feature_importance_final.png
│   └── final_summary.png
│
├── 📄 requirements.txt                  ← All dependencies
├── 📄 README.md                         ← You are here
└── 📄 .gitignore                        ← Files to ignore
```

---

## 📊 Dataset Description

**Source:** Fannie Mae Single-Family Loan Performance Data  
**Size:** 7,902 loans × 34 features  

### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `orig_rt` | Original interest rate on the loan | Numeric |
| `orig_amt` | Original unpaid principal balance ($) | Numeric |
| `orig_trm` | Original loan term in months | Numeric |
| `oltv` | Original Loan-to-Value ratio (%) | Numeric |
| `dti` | Debt-to-Income ratio (%) | Numeric |
| `CSCORE_B` | Borrower's credit score | Numeric |
| `CSCORE_MN` | Co-borrower's credit score | Numeric |
| `num_bo` | Number of borrowers | Numeric |
| `loan_age_months` | Age of loan in months | Numeric |
| `loan_to_income_ratio` | Loan amount ÷ income | Numeric |
| `state` | US state of the property | Categorical |
| `ORIG_CHN` | Origination channel (Retail/Broker/Correspondent) | Categorical |
| `purpose` | Loan purpose (Purchase/Refinance/Cash-out) | Categorical |
| `PROP_TYP` | Property type | Categorical |
| `occ_stat` | Occupancy status | Categorical |
| `FTHB_FLG` | First-time homebuyer flag | Binary |
| `ORIG_DTE` | Origination date | Date |
| `LAST_DTE` | Last activity date | Date |
| `**is_default**` | **TARGET: Did borrower default? (0/1)** | Binary |

---

## 🚀 MLflow Runs Summary

All 5 runs are tracked in MLflow with full metrics, parameters and artifacts.

| Run | Name | Technique | ROC-AUC | F1-Score |
|-----|------|-----------|---------|----------|
| **Run 1** | Baseline | Logistic Regression | 0.8522 | 0.7891 |
| **Run 2** | Model Selection | DT, RF, XGBoost, LightGBM, GBM | 0.8906 | 0.8234 |
| **Run 3** | Hyperparameter Tuning | RandomizedSearchCV + 5-Fold CV | **0.8924** | **0.8267** |
| **Run 4** | Ensemble Learning | Bagging, AdaBoost, Voting, Stacking | 0.8868 | 0.8198 |
| **Run 5** | Data Improvement | SMOTE + Feature Engineering + PCA | 0.8498 | 0.7823 |

### 🏆 Best Model: Tuned XGBoost (Run 3)
- **ROC-AUC: 0.8924** — correctly identifies risky borrowers 89% of the time
- **F1-Score: 0.8267**
- Tuned with 30-iteration RandomizedSearchCV over 9 hyperparameters

---

## 🔬 What We Did — Step by Step

### Step 1 — Exploratory Data Analysis (EDA)
- Analyzed distribution of all 34 features
- Visualized class imbalance (60/40 split)
- Plotted correlation heatmap across all numeric features
- Compared feature distributions between defaulters vs non-defaulters
- Identified top features correlated with default

### Step 2 — Data Cleaning
- Parsed 5 date columns → extracted year and month as numeric features
- Encoded `LAST_STAT` (text: C/P/F/1-9) → ordinal integers
- Label-encoded `state` (53 unique US states)
- Clipped `loan_to_income_ratio` at 99th percentile to remove extreme outliers
- Removed duplicate rows

### Step 3 — Data Preparation
- Removed data-leaking columns (`LAST_STAT`, `MOD_FLAG`, `LAST_RT`, `MODTRM_CHNG`, `MODUPB_CHNG`)
- Stratified 80/20 train-test split (preserving class ratio)
- Applied `StandardScaler` (fit on train only, transform test)

### Run 1 — Baseline Model
- **Logistic Regression** as the simplest starting point
- Establishes benchmark: AUC = 0.8522

### Run 2 — Model Selection
- Compared 5 algorithms on identical data
- Decision Tree, Random Forest, XGBoost, LightGBM, Gradient Boosting
- XGBoost emerged as winner (AUC = 0.8906)

### Run 3 — Hyperparameter Tuning
- **RandomizedSearchCV** with 30 iterations
- **StratifiedKFold** cross-validation (5 folds)
- Tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Best AUC improved to **0.8924**

### Run 4 — Ensemble Learning
- **Bagging:** 100 Decision Trees on random data subsets
- **AdaBoost:** 150 weak learners focusing on hard examples
- **Soft Voting:** LR + RF + LightGBM combined predictions
- **Stacking:** DT + RF + LightGBM → meta Logistic Regression

### Run 5 — Data Improvement + Feature Engineering + PCA
- **SMOTE** to fix class imbalance synthetically
- **6 engineered features:**
  - `credit_ltv_ratio` = Credit Score ÷ LTV
  - `affordability_stress` = DTI × LTV ÷ 100
  - `amt_per_term_year` = Loan Amount ÷ Term Years
  - `credit_score_gap` = |CSCORE_B − CSCORE_MN|
  - `rate_spread` = max(Interest Rate − 3%, 0)
  - `high_risk_flag` = 1 if DTI > 43% AND LTV > 80%
- **PCA:** 31 features → 20 components (95% variance retained)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fannie-mae-default-prediction.git
cd fannie-mae-default-prediction
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebook
```bash
jupyter notebook fannie_mae_mlflow_project.ipynb
```

### 5. View MLflow Dashboard
```bash
# In a separate terminal
mlflow ui

# Open browser → http://localhost:5000
```

---

## 🖥️ Running on Google Colab

```python
# Cell 1 — Install
!pip install mlflow xgboost lightgbm imbalanced-learn -q

# Cell 2 — View MLflow UI
import subprocess, time
subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
from google.colab.output import eval_js
url = eval_js("google.colab.kernel.proxyPort(5000)")
print(f"✅ MLflow UI → {url}")
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
imbalanced-learn>=0.10.0
mlflow>=2.0.0
jupyter>=1.0.0
```

---

## 📈 Results & Visualizations

### Model Comparison (Run 2)
All 5 models compared on Accuracy, Precision, Recall, F1, ROC-AUC

### ROC-AUC Progression Across Runs
Shows clear improvement from baseline to tuned model

### Feature Importance (Final Model)
Top predictors of mortgage default:
1. `loan_age_months` — older loans have clearer default patterns
2. `dti` — high debt-to-income = higher risk
3. `CSCORE_B` — lower credit score = higher risk
4. `oltv` — higher loan-to-value = more leveraged = riskier
5. `affordability_stress` (engineered) — combined stress indicator

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **XGBoost** | Best performing model |
| **LightGBM** | Fast gradient boosting |
| **imbalanced-learn** | SMOTE for class imbalance |
| **MLflow** | Experiment tracking & model registry |
| **Jupyter Notebook** | Interactive development |

---

## 👨‍💻 Author

 
CustomBot-UI  
Machine Learning Project — 2026

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset: [Fannie Mae Single-Family Loan Performance Data](https://www.fanniemae.com/research-and-insights/datasets/mortgage-data)
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Scikit-learn Documentation: https://scikit-learn.org
