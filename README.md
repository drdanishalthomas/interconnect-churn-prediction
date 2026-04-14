# Interconnect Telecom Churn Prediction

**Dr. Danisha L. Thomas** | TripleTen Data Science Bootcamp | Sprint 17 Final Project

---

## Project Overview

Interconnect, a telecom operator, seeks to proactively identify customers at risk of churning so that targeted retention offers can be made before they disengage. This project builds a binary classification model to predict churn using customer contract, personal, internet, and phone service data.

**Primary Metric:** AUC-ROC  
**Target:** ≥ 0.85  
**Final Test AUC-ROC:** 0.8540  
**Final Test Accuracy:** 78.96%

---

## Dataset

Four datasets were merged on `customerID`:

| Dataset | Records | Features |
|---------|---------|----------|
| contract.csv | 7,043 | 8 |
| personal.csv | 7,043 | 5 |
| internet.csv | 5,517 | 8 |
| phone.csv | 6,361 | 2 |

**Target Variable:** `EndDate != 'No'` → churn = 1 (26.6% of customers)

---

## Methodology

### Data Preprocessing
- Fixed `TotalCharges` data type (object → float)
- Filled NaN values in internet/phone columns with `'No service'` (non-subscribers, not missing data)
- Engineered `tenure_approx` feature: `TotalCharges / MonthlyCharges`
- Applied one-hot encoding for categorical features
- Applied StandardScaler to numerical features
- Stratified train/validation/test split: 60/20/20

### Models Trained

| Model | Validation AUC-ROC |
|-------|-------------------|
| **CatBoost (Tuned)** | **0.8580** ✅ |
| Logistic Regression | 0.8460 |
| XGBoost (Tuned) | 0.8467 |
| CatBoost (Default) | 0.8536 |
| LightGBM (Tuned) | 0.8354 |
| LightGBM (Default) | 0.8349 |
| XGBoost (Default) | 0.8242 |

### Winning Model: CatBoost (Tuned)

```python
CatBoostClassifier(
    random_state=12345,
    verbose=0,
    iterations=300,
    learning_rate=0.1,
    depth=1,
    class_weights=[1, 2]
)
```

Key finding: Shallow trees (depth=1) outperformed deeper configurations, indicating the dataset is largely explained by simple patterns and that deeper trees lead to overfitting.

---

## Results

| Metric | Score |
|--------|-------|
| Final Test AUC-ROC | **0.8540** |
| Final Test Accuracy | **78.96%** |

---

## Key Steps

- Correctly identifying the target variable from `EndDate` column
- Feature engineering `tenure_approx` as a strong churn signal
- Isolating raw feature sets for CatBoost before one-hot encoding (critical for avoiding feature mismatch)
- Class imbalance handling with `class_weights=[1, 2]`

---

## Recommendations

- **Target month-to-month contract customers** — highest churn risk; prioritize retention offers such as contract upgrade incentives
- **Leverage tenure as an early warning signal** — customers with low tenure and high monthly charges relative to total charges are at elevated churn risk
- **Deploy CatBoost in production** — AUC-ROC of 0.854 makes it well-suited for real-time churn scoring to prioritize proactive outreach

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `CatBoost` `XGBoost` `LightGBM` `Matplotlib` `Seaborn` `Jupyter Notebook`

---

## Author

**Dr. Danisha L. Thomas, PhD**  
Data Scientist | Behavioral Intelligence & Healthcare Analytics  
[LinkedIn](https://linkedin.com/in/drdlthomas) | [GitHub](https://github.com/drdanishalthomas)
