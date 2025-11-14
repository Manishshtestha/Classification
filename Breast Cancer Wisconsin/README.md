# **Breast Cancer Wisconsin – Classification Model**

This directory contains a full end-to-end machine learning workflow for classifying breast tumors as **malignant** or **benign** using the **Breast Cancer Wisconsin Diagnostic Dataset** from `sklearn.datasets`.

The notebook includes Exploratory Data Analysis (EDA), preprocessing, outlier handling, feature selection, model training using a pipeline with SMOTE, evaluation, learning curves, SHAP interpretability, and model saving.

---

## 📁 **File**

### **`model.ipynb`**

A complete analysis and modeling pipeline with the following stages:

---

##  **1. Exploratory Data Analysis (EDA)**

### ✔ Dataset Loading

The dataset is loaded using `load_breast_cancer()` from scikit-learn.
Key additions:

* Convert to pandas DataFrame
* Add `target` and human-readable `diagnosis` column (`malignant` / `benign`)

### ✔ Initial Data Inspection

Includes:

* Shape
* Data types
* Summary statistics (`describe`)
* Unique values per column
* Missing values analysis

### ✔ Missing Value Visualization

Using `missingno`:

* `msno.matrix()`
* `msno.heatmap()`

### ✔ Target Distribution

Bar plot of **benign vs malignant** counts.

### ✔ Numerical Distribution + Outliers

For each numerical feature:

* Histogram with KDE
* Boxplot grouped by diagnosis
* Display skewness

### ✔ Correlation & Pairplots

* Full heatmap of numeric correlations
* Pairplot for selected features

### ✔ Statistical Tests

Independent **t-tests** for each numerical feature to compare malignant vs benign classes.

Most features show **p-values < 0.05**, indicating significant differences.

---

##  **2. Feature Selection (VIF Analysis)**

Variance Inflation Factor (VIF) is computed to evaluate multicollinearity among predictors.

Top features have extremely high VIF (e.g., mean radius, mean perimeter), reflecting strong correlation patterns typical for this dataset.

---

##  **3. Train-Test Split**

```
Train: 80%
Test:  20%
Stratified Sampling: ✔
```

---

##  **4. Outlier Detection & Removal**

Outliers handled differently based on skewness:

### ✦ For highly skewed features (|skew| > 1):

* IQR-based filtering with **3×IQR rule**

### ✦ For near-normal features:

* Z-score filtering (> |3|)

Before/after counts printed for each feature.

---

##  **5. Preprocessing Pipeline**

### Numeric pipeline

```python
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

Wrapped in a `ColumnTransformer` for all numeric features.

---

##  **6. Model Pipeline with SMOTE**

To handle class imbalance, SMOTE is integrated:

```python
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        max_iter=1000
    ))
])
```

---

##  **7. Model Training**

Pipeline is fit on the training data:

```python
pipeline.fit(X_train, y_train)
```

Predictions and probabilities:

```python
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
```

---

##  **8. Model Evaluation**

### **Classification Report**

```
Accuracy:     0.9649
Precision:    0.9857
Recall:       0.9583
F1 Score:     0.9718
```

### ✔ Confusion Matrix (plotted)

### ✔ ROC Curve with AUC

### ✔ Learning Curve

Shows stable training/test curves → model generalizes well.

---

##  **9. Model Explainability (SHAP)**

SHAP is used for feature importance:

```python
explainer = shap.Explainer(pipeline.named_steps['classifier'], X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

Outputs:

* Global importance ranking
* Interpretation of how features influence predictions

---

##  **10. Saving the Model**

The trained pipeline is saved as:

```
breast_cancer_model.joblib
```

Using:

```python
joblib.dump(pipeline, "breast_cancer_model.joblib")
```

---

##  **Workflow Summary**

1. Load & explore dataset
2. Inspect data: stats, missing, distributions, skewness
3. Analyze correlations + t-tests
4. Outlier handling with skewness-aware strategy
5. Build preprocessing pipeline
6. Integrate SMOTE for class balance
7. Train Elastic Net Logistic Regression
8. Evaluate with accuracy, F1, precision/recall
9. Interpret via SHAP
10. Save final model

---

##  **Dataset**

Dataset: **Breast Cancer Wisconsin Diagnostic Dataset**
Source: `sklearn.datasets`

---

##  **License**

This project is for learning and research.
Dataset License: Permissible for research/educational use via scikit-learn.

---
