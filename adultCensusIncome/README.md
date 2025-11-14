# **Adult Census Income Classification**

📂 *Directory:* `Classification/adultCensusIncome/`

Predicting whether an individual earns **>50K** or **≤50K** annually based on demographic and socioeconomic features from the **UCI Adult Census Income** dataset.
This project demonstrates complete EDA, preprocessing, model comparison, evaluation, and baseline model saving.

---

## **1. Project Overview**

This project addresses a binary classification problem:

> **Can we predict whether a person earns more than $50,000 per year?**

Using demographic factors such as age, education, marital status, workclass, hours per week, and more, various machine-learning algorithms are developed and compared.

The goal is to explore the dataset thoroughly, build robust preprocessing pipelines, analyze model performance, and establish a replicable workflow suitable for academic or research use.

---

## **2. Repository Structure**

```
adultCensusIncome/
│
├── basemodel.ipynb          # Full EDA + model training + evaluation + learning curves
├── model.ipynb              # Same workflow, minor differences in visualization
├── README.md                # Documentation (this file)
```

**Note:**
Both notebooks follow the same methodology.
`basemodel.ipynb` includes learning curves for selected models (RF, DT, LR, SVC, KNN).

---

## **3. Dataset Description**

* **Source:** `adult.csv` (Kaggle's Adult Census Income dataset)
* **Instances:** **32,561**
* **Features:** **14 predictors + 1 target**
* **Target Variable:** `income`

  * Encoded as: `<=50K` → **0**, `>50K` → **1**
* **Missing Data:** Represented as `"?"`

  * Appears mainly in:

    * `workclass`
    * `occupation`
    * `native.country`

### **Feature Types**

* **Numerical:** 6 (e.g., age, hours.per.week, capital.gain)
* **Categorical:** 8 (e.g., marital.status, education, relationship)

---

## **4. Exploratory Data Analysis (Key Findings)**

### **4.1 Class Imbalance**

| Class   | Percentage |
| ------- | ---------- |
| `<=50K` | ~76%       |
| `>50K`  | ~24%       |

### **4.2 Missing Data Patterns**

* `workclass` and `occupation` missing together **1,836/1,843 times**
* Correlation between their missingness: **0.998**

### **4.3 Feature Relationships**

* `education` and `education.num` have **perfect mapping**
* High correlation among:

  * `age`, `hours.per.week`, `education.num`, `capital.gain`

### **4.4 Outliers**

Strong right-skew observed in:

* `capital.gain`
* `capital.loss`
* `fnlwgt`

---

## **5. Preprocessing Pipeline**

To accommodate different algorithm families, data is processed in **two parallel tracks**:

### **Track A – Tree-Based Models (`df1`)**

* **Drop:** `education.num`
* **Encoder:** `OrdinalEncoder`
* Algorithms: RF, DT, LightGBM, CatBoost, XGBoost

### **Track B – Linear / Distance-Based Models (`df2`)**

* **Drop:** `education` (keep numeric `education.num`)
* **Encoder:** `OneHotEncoder`
* Algorithms: Logistic Regression, SVM, KNN, Naive Bayes

### **Common Preprocessing Steps**

* **Numerical Imputation:** Median
* **Categorical Imputation:** Most frequent
* **Scaling:**

  * Tree models → *No scaling*
  * Linear & KNN models → StandardScaler
* **Outlier Handling:**

  * If |skew| > 1 → IQR (3× rule)
  * Else → Z-score (|z| > 3)

---

## **6. Models Trained and Results**

Below are the final test-set scores (F1 prioritized due to imbalance):

| Model                   | Type     | F1 Score   | Accuracy | Notes                                       |
| ----------------------- | -------- | ---------- | -------- | ------------------------------------------- |
| **CatBoost**            | Tree     | **0.6253** | 0.8397   | Best overall                                |
| **LightGBM**            | Tree     | 0.6232     | 0.8386   | Excellent generalization                    |
| **Logistic Regression** | Linear   | 0.6106     | 0.8337   | Best linear model                           |
| **KNN**                 | Distance | 0.6113     | 0.8262   |                                             |
| **Linear SVC**          | Linear   | 0.6059     | 0.8320   |                                             |
| **Random Forest**       | Tree     | 0.5887     | 0.8224   |                                             |
| **Decision Tree**       | Tree     | 0.5271     | 0.7697   |                                             |
| **XGBoost**             | Tree     | —          | —        | Trained but not evaluated in both notebooks |
| **Naive Bayes**         | Linear   | —          | —        | Failed due to sparse matrix limitations     |

**All working models are saved as `.joblib` pipelines** in `/kaggle/working/base_models/`.

---

## **7. Evaluation Metrics**

Each model is evaluated with:

* Classification Report
* Confusion Matrix
* Accuracy, Precision, Recall, F1
* ROC-AUC (if supported)
* Learning Curves (select models only)

Visualizations include:

* Distribution plots
* Correlation heatmap
* Missingness map
* ROC curves
* F1 vs training size learning curves

---

## **8. How to Run the Project**

### **Step 1 — Install dependencies**

```bash
!pip install -q scikit-learn pandas numpy seaborn matplotlib xgboost lightgbm catboost joblib
```

### **Step 2 — Ensure dataset path is correct**

```
/kaggle/input/adult-census-income/adult.csv
```

### **Step 3 — Run either notebook**

* `basemodel.ipynb` → full workflow with learning curves
* `model.ipynb` → same pipeline, fewer plots

### **Step 4 — Models are saved automatically**

Saved to:

```
/kaggle/working/base_models/
```

---

## **9. Future Work**

* [ ] SMOTE or class weights
* [ ] Hyperparameter tuning (Optuna / GridSearchCV)
* [ ] Feature engineering: `capital.net = gain - loss`
* [ ] Model ensembling (stacking / voting)
* [ ] SHAP interpretability for tree models
* [ ] Calibration curves (for probabilistic accuracy)

---


