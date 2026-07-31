# 📊 Model Experiment Results

This document summarizes the performance of different machine learning experiments.

---

## 🚀 Run 1 — Baseline Logistic Regression

- **Model:** Logistic Regression
- **AUC:** **0.8522**

### Notes
- Simple baseline model.
- Served as a good benchmark for comparing more advanced models.

---

## 🔍 Run 2 — Model Selection

Tested five different machine learning models.

### Best Model
- **Gradient Boosting**
- **AUC:** **0.8906**

### Notes
- Gradient Boosting outperformed all other tested models.
- Interestingly, it achieved a higher AUC than the default XGBoost model during this round.

---

## 🏆 Run 3 — Tuned XGBoost (Best Overall)

- **Model:** XGBoost
- **Test AUC:** **0.8924**
- **Cross-Validation AUC:** **0.8957**

### Best Hyperparameters
- **Number of Trees:** 300
- **Learning Rate:** 0.05
- **Max Depth:** 3

### Notes
- Best-performing model across all experiments.
- A low learning rate with many trees and shallow depth produced a well-regularized model with excellent generalization.

---

## 🤝 Run 4 — Ensemble Methods

### Best Ensemble
- **Bagging Classifier**
- **AUC:** **0.8868**

### Notes
- Performed well but remained slightly below the tuned XGBoost model.
- Demonstrates that a carefully tuned single model can outperform more complex ensemble approaches.

---

## 🧪 Run 5 — Feature Engineering + PCA + SMOTE

- **Techniques Used:**
  - Feature Engineering
  - Principal Component Analysis (PCA)
  - SMOTE
- **AUC:** **0.8498**

### Notes
- Performance decreased compared to previous experiments.
- PCA likely removed some informative features.
- SMOTE did not provide a benefit for this financial dataset.

---

# 📈 Overall Ranking

| Rank | Model | AUC |
|------|-------|------|
| 🥇 1 | Tuned XGBoost | **0.8924** |
| 🥈 2 | Gradient Boosting | **0.8906** |
| 🥉 3 | Bagging Classifier | **0.8868** |
| 4 | Logistic Regression | **0.8522** |
| 5 | Feature Engineering + PCA + SMOTE | **0.8498** |

---

## ✅ Final Conclusion

The **tuned XGBoost** model achieved the best overall performance with a **Test AUC of 0.8924** and a **Cross-Validation AUC of 0.8957**. Its combination of **300 trees**, **learning rate of 0.05**, and **maximum depth of 3** provided the best balance between predictive performance and generalization. More complex preprocessing techniques such as **PCA** and **SMOTE** did not improve performance for this dataset, highlighting that careful model tuning was more effective than additional feature transformations.
