# IEE-Fraud-Detection

This directory contains a complete fraud detection workflow using the **IEEE-CIS Fraud Detection** dataset from Kaggle.  
It includes Exploratory Data Analysis (EDA), model training with multiple classifiers, and hyperparameter tuning of the final selected model (XGBoost).

---

## 📁 Files in This Directory

### **1. Exploratory Data Analysis.ipynb**
A comprehensive EDA notebook covering:
- Dataset overview and inspection  
- Target-focused exploration (fraud vs non-fraud)  
- **Numerical features**
  - Distribution plots  
  - Outlier visualization using boxplots  
- **Categorical features**
  - Target mean plots  
  - Category distribution visualizations  
- **Statistical tests**
  - t-tests for categorical columns with 2 unique values  
  - One-way ANOVA for categorical columns with more than 2 unique values  
- **Correlation analysis**
  - Spearman correlation using a **5,000-row sample** to reduce memory usage

These insights directly guide feature selection and preprocessing decisions in later notebooks.

---

### **2. Training Multiple Classifier.ipynb**

This notebook constructs and evaluates multiple machine learning models using a robust preprocessing framework.

#### ✔ Feature Selection
- Numeric columns are extracted:
  ```python
  num_cols = df.select_dtypes(include=['int64', 'float64']).drop('isFraud', axis=1).columns.tolist()
````

* Spearman correlation calculated using a **10,000-row sample**
* Features with **|correlation| > 0.9** are dropped:

  ```python
  to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
  df_reduced = df.drop(columns=to_drop)
  ```

#### ✔ Data Splitting

Train / validation / test sets are created.

#### ✔ Feature Categorization by Skewness

Based on absolute skewness:

* **Normal features:** < 1
* **Moderately skewed:** 1–5
* **Heavily skewed:** ≥ 5

Moderately skewed features undergo **quantile capping** (1%–99%).

#### ✔ Preprocessing Pipelines

**Numeric pipelines:**

* Standard scaling
* Robust scaling
* Power transform (Yeo-Johnson) + Robust scaling

**Categorical pipelines:**

* **TargetEncoder** for linear-type models (Logistic, SVM, NB, KNN)
* **OrdinalEncoder** for tree-based models

All combined using `ColumnTransformer`.

#### ✔ Machine Learning Models Trained

| Model               |
| ------------------- |
| Logistic Regression |
| Random Forest       |
| Gradient Boosting   |
| AdaBoost            |
| Naive Bayes         |
| KNN                 |
| XGBoost             |

**Best performer:** **XGBoost**, with validation metrics:

```
Accuracy: 0.9837
F1 Score: 0.7145
AUC: 0.9538
```

---

### **3. Hyperparameter Tuned Model.ipynb**

This notebook focuses on tuning the XGBoost model using a two-stage approach:

1. **Initial RandomizedSearchCV**
2. **Refined RandomizedSearchCV** using narrowed ranges from Stage 1

#### ✔ Final Model Results

```
Accuracy: 0.9817
F1 Score: 0.6642
ROC AUC: 0.9468
```

**Classification Report**

```
              precision    recall  f1-score   support
0              0.98        1.00      0.99     113975
1              0.92        0.52      0.66       4133

accuracy                                0.98     118108
macro avg        0.95        0.76      0.83     118108
weighted avg     0.98        0.98      0.98     118108
```

#### ✔ Additional Outputs

* Learning curves
* Feature importance plot (XGBoost)

---

## 🧭 End-to-End Workflow Summary

1. Exploratory data analysis
2. Correlation-based feature reduction
3. Skewness handling, outlier capping, and categorical encoding
4. Training/validation across multiple models
5. Selecting XGBoost as the best performer
6. Hyperparameter tuning for performance optimization
7. Final evaluation with F1, AUC, and classification report
8. Visualization of learning behavior and feature influence

---

## 📦 Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

---

## 📜 License

This project is intended for educational and research purposes.
Please refer to Kaggle’s dataset license for usage restrictions.

```
## Kaggle Notebook
https://www.kaggle.com/code/bijaybeezoe/ieee-cis-fraud-detection-train-multiple-classifier
https://www.kaggle.com/code/bijaybeezoe/ieee-cis-fraud-detection-hyperparameter-tuned
