# Customer Churn Prediction

> End-to-end machine learning project to predict customer churn in the telecom industry using multiple machine learning models, feature engineering, hyperparameter tuning, and SHAP explainability.

## Business Problem

Customer churn is a major challenge in the telecom industry. Retaining an existing customer is significantly cheaper than acquiring a new one. This project predicts customers who are likely to churn so the business can proactively retain them.

## Project Objectives

- Predict whether a customer will churn.
- Compare multiple machine learning models.
- Optimize the best-performing model.
- Explain predictions using SHAP.
- Generate actionable business recommendations.
- Deploy the final model using Streamlit.

---

# Dataset

**Source:** IBM Telco Customer Churn Dataset (Kaggle)

- 7,043 customers
- 21 original features
- Binary target:
  - Yes = Churn
  - No = No Churn

---

# Methodology

## Data Preprocessing

- Converted `TotalCharges` to numeric
- Handled missing values
- Removed unnecessary columns
- Encoded target variable
- Built preprocessing pipeline using `ColumnTransformer`

## Feature Engineering

- `tenure_group`
- `NumberOfServices`

## Modeling

Models evaluated:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

Used:

- Train/Test Split
- Stratified 5-Fold Cross Validation
- Hyperparameter Tuning
- Threshold Optimization

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# Best Model

**Tuned XGBoost**

Performance:

| Metric | Value |
|--------|------:|
| Accuracy | 79.7% |
| Recall | 68.2% |
| Precision | 60.4% |
| F1 Score | 64.1% |
| ROC-AUC | 0.848 |

---

# Key Business Insights

- Month-to-month customers churn far more than annual contract customers.
- New customers are more likely to churn.
- Fiber optic customers show higher churn.
- Customers without Online Security or Tech Support churn more.
- Electronic check users have higher churn.
- Higher monthly charges increase churn risk.
- Customers using multiple services are more loyal.

---

# Recommendations

1. Convert month-to-month customers to annual contracts.
2. Improve onboarding for new customers.
3. Launch dedicated retention campaigns for fiber customers.
4. Offer free security and tech support trials.
5. Encourage automatic payments.
6. Provide loyalty offers to high-value customers.

---

# Explainability

SHAP was used to interpret model predictions.

Included:

- SHAP Summary Plot
- SHAP Feature Importance (Bar Plot)

---

# Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

---

# Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   ├── app.py
│   
│
├── data/
│   ├── raw/
│   
│
├── models/
│   └── pipeline.pkl
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

```bash
git clone https://github.com/ATHAR-HUSSAIN-2910/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Download the IBM Telco Customer Churn dataset and place it inside:

```text
data/raw/
```

---

# Usage

Run the notebook:

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```


# Author

**Athar Hussain**

GitHub: https://github.com/ATHAR-HUSSAIN-2910

LinkedIn: https://www.linkedin.com/in/athar-hussain-datascience/

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- IBM Telco Customer Churn Dataset
- Scikit-learn
- XGBoost
- SHAP
