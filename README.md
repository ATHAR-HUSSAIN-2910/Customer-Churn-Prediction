# 📉 Customer Churn Prediction

> End-to-end machine learning project to predict customer churn in the telecom industry using multiple machine learning models, feature engineering, hyperparameter tuning, threshold optimization, and SHAP explainability.

## 🚀 Live Demo

### Try the deployed application

👉 **[Launch Customer Churn Prediction App - https://athars-customer-churn-prediction.streamlit.app/](https://athars-customer-churn-prediction.streamlit.app/)**

The Streamlit application allows users to enter customer information and receive a churn prediction from the trained machine learning pipeline.

> **Note:** The application is intended for educational and portfolio purposes and should not be treated as a production customer-retention system without further validation.

---

# 💼 Business Problem

Customer churn is a major challenge in the telecom industry.

Retaining an existing customer is generally more cost-effective than acquiring a new one. Therefore, identifying customers who are likely to churn can help businesses take proactive retention actions.

This project builds a machine learning system that predicts customers who are likely to churn and uses model explainability to understand the factors contributing to those predictions.

---

# 🎯 Project Objectives

* Predict whether a customer will churn.
* Compare multiple machine learning models.
* Optimize the best-performing model.
* Optimize the classification threshold.
* Explain predictions using SHAP.
* Identify important customer churn patterns.
* Generate actionable business recommendations.
* Deploy the final model using Streamlit.

---

# 📊 Dataset

**Source:** IBM Telco Customer Churn Dataset (Kaggle)

### Dataset Summary

* **7,043 customers**
* **21 original features**
* Binary target variable

| Target | Meaning  |
| ------ | -------- |
| `Yes`  | Churn    |
| `No`   | No Churn |

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
    ↓
Data Understanding
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Train / Test Split
    ↓
Preprocessing Pipeline
    ↓
Baseline Model
    ↓
Multiple Model Comparison
    ↓
Cross Validation
    ↓
Hyperparameter Tuning
    ↓
Threshold Optimization
    ↓
Final Model Evaluation
    ↓
SHAP Explainability
    ↓
Model Pipeline Saving
    ↓
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Converted `TotalCharges` to numeric.
* Handled missing values.
* Removed unnecessary columns.
* Encoded the target variable.
* Encoded categorical variables.
* Built a preprocessing pipeline using `ColumnTransformer`.

Using a preprocessing pipeline ensures that the same transformations are consistently applied during both training and inference.

---

# 🛠️ Feature Engineering

Two additional features were created to provide the models with more useful information about customer behavior.

### `tenure_group`

Groups customers according to their tenure with the company.

### `NumberOfServices`

Represents the number of services used by a customer.

These features were designed to capture customer lifecycle and service-usage patterns that may be useful for predicting churn.

---

# 🤖 Modeling

Multiple machine learning algorithms were evaluated.

### Models Compared

* Dummy Classifier
* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

### Model Development

The project used:

* Train/Test Split
* Stratified 5-Fold Cross Validation
* Hyperparameter Tuning
* Threshold Optimization

### Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

ROC-AUC was particularly useful for evaluating the model's ability to distinguish between churn and non-churn customers across different classification thresholds.

---

# 🏆 Best Model

## Tuned XGBoost

The tuned XGBoost model achieved the following performance:

| Metric    |     Value |
| --------- | --------: |
| Accuracy  |     79.7% |
| Recall    |     68.2% |
| Precision |     60.4% |
| F1 Score  |     64.1% |
| ROC-AUC   | **0.848** |

The model achieved a **ROC-AUC of 0.848**, indicating good ability to distinguish between customers who churn and those who do not.

---

# 🎚️ Threshold Optimization

A classification threshold of `0.50` is not necessarily optimal for every business problem.

Because customer churn prediction is often focused on identifying potential churners, different classification thresholds were evaluated to understand the trade-off between:

* Precision
* Recall
* F1 Score
* Accuracy

The final threshold was selected based on validation performance rather than using the default `0.50` threshold blindly.

This demonstrates an important practical machine learning concept:

> **The probability threshold should be aligned with the business objective rather than automatically assuming 0.50 is optimal.**

---

# 🔎 Model Explainability with SHAP

**SHAP (SHapley Additive exPlanations)** was used to understand the predictions made by the final model.

This helps answer questions such as:

* Which features are driving churn predictions?
* Which customer characteristics are associated with higher churn risk?
* Which features contribute to lower churn risk?

### Explainability Outputs

The project includes:

* SHAP Summary Plot
* SHAP Feature Importance Bar Plot

SHAP provides a way to move beyond simply predicting churn and understand **why the model is making those predictions**.

---

# 💡 Key Business Insights

The analysis identified several important churn patterns:

* Month-to-month customers churn significantly more than annual contract customers.
* New customers are more likely to churn.
* Fiber optic customers show higher churn.
* Customers without Online Security or Tech Support churn more.
* Electronic check users have higher churn.
* Higher monthly charges are associated with increased churn risk.
* Customers using multiple services tend to be more loyal.

These findings help connect the machine learning model to potential business actions.

---

# 📈 Business Recommendations

Based on the analysis, the following strategies could help reduce customer churn:

### 1. Encourage Long-Term Contracts

Provide incentives for month-to-month customers to move toward annual contracts.

### 2. Improve New Customer Onboarding

Create stronger onboarding and early-stage engagement programs for new customers.

### 3. Target High-Risk Fiber Customers

Develop dedicated retention campaigns for customers using fiber optic services.

### 4. Promote Security & Technical Support

Offer free trials, discounts, or bundled packages for Online Security and Tech Support.

### 5. Encourage Automatic Payments

Provide incentives for customers to switch from electronic check payments to automatic payment methods.

### 6. Reward High-Value Loyal Customers

Provide loyalty benefits and personalized offers to customers using multiple services.

---

# 🧠 What This Project Demonstrates

This project goes beyond simply training a classification model.

It demonstrates an end-to-end machine learning workflow involving:

```text
Business Problem
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Preprocessing Pipeline
      ↓
Model Comparison
      ↓
Cross Validation
      ↓
Hyperparameter Tuning
      ↓
Threshold Optimization
      ↓
Model Evaluation
      ↓
SHAP Explainability
      ↓
Model Serialization
      ↓
Streamlit Deployment
```

---

# 🌐 Streamlit Deployment

The final machine learning pipeline has been deployed as an interactive **Streamlit web application**.

### Live Application

**[Open the Customer Churn Prediction App →](https://athars-customer-churn-prediction.streamlit.app/)**

The application uses the saved machine learning pipeline to process user inputs and generate churn predictions.

The Streamlit application is located at:

```text
app/app.py
```

---

# 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
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

# 🛠️ Tech Stack

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Programming language             |
| Pandas           | Data manipulation                |
| NumPy            | Numerical computing              |
| Matplotlib       | Data visualization               |
| Seaborn          | Statistical visualization        |
| Scikit-learn     | Machine learning & preprocessing |
| XGBoost          | Gradient boosting model          |
| SHAP             | Model explainability             |
| Joblib           | Model serialization              |
| Streamlit        | Web application & deployment     |
| Jupyter Notebook | Model development                |

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ATHAR-HUSSAIN-2910/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

Download the IBM Telco Customer Churn dataset and place the dataset inside:

```text
data/raw/
```

The notebook can then be used to reproduce the data preparation, feature engineering, model training, and evaluation workflow.

---

# ▶️ Usage

## Run the Jupyter Notebook

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

Run the notebook from beginning to end to reproduce the machine learning workflow.

## Run the Streamlit Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# 💾 Saved Model

The final trained preprocessing and modeling pipeline is stored as:

```text
models/pipeline.pkl
```

The saved pipeline allows the Streamlit application to reuse the trained preprocessing and model without retraining every time the application starts.

---

# 📊 Project Results

The final tuned XGBoost model achieved:

```text
ROC-AUC : 0.848
Accuracy: 79.7%
Recall  : 68.2%
Precision: 60.4%
F1 Score: 64.1%
```

The model demonstrates a reasonable ability to identify customers at risk of churn while maintaining a balance between precision and recall.

---

# 📚 Key Learnings

Through this project, I gained practical experience with:

* End-to-end supervised machine learning
* Binary classification
* Data cleaning
* Feature engineering
* Categorical feature encoding
* `ColumnTransformer`
* Scikit-learn pipelines
* Cross-validation
* Model comparison
* Hyperparameter tuning
* XGBoost
* Classification threshold optimization
* Precision/Recall trade-offs
* ROC-AUC
* SHAP explainability
* Model serialization
* Streamlit deployment
* Connecting machine learning predictions with business recommendations

---

# ⚠️ Limitations

The model's performance is specific to the dataset and modeling pipeline used in this project.

The predictions should not be interpreted as guaranteed customer behavior.

For a production churn prediction system, additional work would be required, including:

* Monitoring model performance over time
* Handling data drift
* Probability calibration
* Production-scale data pipelines
* Experiment tracking
* Automated retraining
* Business cost-based threshold selection
* Model monitoring and logging

---

# 👨‍💻 Author

**Athar Hussain**

Aspiring Data Scientist & Machine Learning Engineer

### GitHub

**[ATHAR-HUSSAIN-2910](https://github.com/ATHAR-HUSSAIN-2910)**

### LinkedIn

**[Athar Hussain — Data Science](https://www.linkedin.com/in/athar-hussain-datascience/)**

---

# ⭐ Project Highlights

* Built an end-to-end **customer churn prediction system**
* Compared **7 machine learning models**
* Used **ColumnTransformer and Scikit-learn pipelines**
* Performed **stratified 5-fold cross-validation**
* Tuned **XGBoost hyperparameters**
* Performed **classification threshold optimization**
* Achieved **0.848 ROC-AUC**
* Used **SHAP for model explainability**
* Generated actionable **business recommendations**
* Saved the complete preprocessing and modeling pipeline
* Built an interactive **Streamlit application**
* Successfully **deployed the model online**

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

* IBM Telco Customer Churn Dataset
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
