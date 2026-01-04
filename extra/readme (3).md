# Loan Payback Prediction using Machine Learning  
### Predicting whether Loan will be paid back with XGBoost & FastAPI | Dockerized & Deployed on Render

## Problem Description
Banks give several types of loans to customers, like home loan and personal loan. 
Before getting a loan, customers provide personal information such as Annual Income, Existing Debt, Age, Gender,Marital Status,  Employment Status, Education level etc.  Banks usually face the problem about unpaid and outstanding loans.  Banks can develop a ML application using the personal information provided by customer to predict the probability whether customer will pay back the loan or not .Then they can decide whether or not to lend the money to the prospective customer.

The objective of this project is to develop a machine learning–based Loan Payback Prediction system that predicts whether the borrower will payback the loan

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)  
- Feature engineering and selection
- Model training and comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)  
- Hyperparameter tuning
- Training final model
- Deploying the model using FastAPI and Docker

The final solution is built using the XGBoost model....***write..more here***

The model is deployed as a REST API using FastAPI, containerized using Docker, and hosted on Render for real-time inference.


## How the Solution Is Used

### 1. Transaction Prediction 

When a transaction is processed, its details are sent to the `/predict` endpoint. The API responds with loanpayback probability and classification. 
***please change output here instead of true /false, print something else***

#### Example request:
```json
[
  {
"employment_status": "employed",
"education_level": "bachelors",
"grade_subgrade": "b3",
"credit_score": 689,
"annual_income": 82000,
"debt_to_income_ratio": 17.3,
"loan_amount": 15000,
"interest_rate": 12.5
}
]

  

#### Example Response
```json
{
  "loan_pay_back_probability": 0.9875,
  "loan_pay_back": "true"
}
```
Based on the model response:

| Prediction | Recommended Action                            |
|------------|-----------------------------------------------|
| True       | Approve the loan
| False      | Do not Approve the loan                         

### Summary of integration

- Supports real-time single Loan Payback prediction
- Enables batch loan payback prediction for multiple transactions
- Returns a probability score and final payback prediction
- Suitable for integration into live loan application systems or periodic reviews of customer
- Helps reduce bad and outstanding loans


```python

```

## Exploratory Data Analysis (EDA)

### 1. Dataset Overview

- **Total transactions:** `599,394`
- **Loan Paid Back Prediction:** `474,494` (~79.88%)
- **Loan not paid back Predictions:** `119,500` (~20.11%)
- **Missing values:** `None`

**Data source:** [Loan Payback Prediction Dataset(Kaggle)]  
(https://www.kaggle.com/competitions/playground-series-s5e11/data)


---

### 2. Feature Overview

| Feature Type | Features |
|--------------|----------|
| **Identifier** | `id`|
| **Numerical** | `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate` |
| **Categorical** | `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade` (Education_level and grade_subgrade was tranformed to numeric using Ordinal encoding) |
| **Target variable** | `loan_paid_back` (0 = Customer defaults, 1 = Customer will pay back) |

---

### 3. Class Distribution (Class Imbalance)

The target variable is quite imbalanced:
- **Loan Paid Back:** `474,494`
- **Loan Not Paid Back:** `119,500`

**Image in repository at:** `images/class_distribution.png`

![Loan Payback Prediction Counts](images/class_distribution.png)



**Key insight:**
- Accuracy is not sufficient to judge the models → we focused on **ROC-AUC and F1-score**

### 4. Mutual Information (Categorical/Binary Features)

| Feature | MI Score |
|---------|----------|
| `employment_status` | 0.175941 |
| `grade_subgrade` | 0.026769 |
| `loan_purpose` | 0.000325 |
| `education_level` | 0.0048 |
| `gender ` | 0.000028 |
| `marital_status ` | 0.000003 |


**Insight:**
- Employment Status is a strong feature in dataset. It indicates whether customer will pay back the loan. Grade Subgrade is also a good feature
- Some features like loan_purpose and education_level have less influence, but can prove to be useful after encoding
- gender and marital status dont have much influence on loan payback intention hence these will be dropped

### 5. Correlation of Numeric Features

**Image in repository:** `images/correlation_matrix.png`

![Correlation Matrix of Numeric Features](images/correlation_matrix.png)

**Important correlations with `is_fraud`:**
- `debt_to_income_ratio` → 0.335758
- `credit_score` → 0.234319
- `interest_rate` → -0.130789

#### Interpretation:
* **High Credit Score** → Loan Paid Pack probability is higher 
* **Low Debt to Income ratio** → Loan Paid Pack probability is higher 
* **Lower interest rates** →Loan Paid Pack probability is higher 

## Feature Engineering

It was necessary to encode certain categorical features into numeric to improve model performance. Some features were dropped as they were not influencing the Loan Pay Back Variable.
The `FeatureEngineering` is implemented  **src/features.py**. This transformation is applied as the first step of the ML pipeline to ensure consistency during both training and inference.

### Key Transformations

| Feature | Description | Motivation |
|--------|-------------|------------|
| `education_level`|`education_encoded` | Education Level is encoded using the Ordinal encoder |
| `grade_subgrade` |`grade_code`| Grade Subgrade is encoded using the Ordinal Encoder |
| `loan_purpose` | loan_purpose_te |Loan Purpose is encoded using target encoder|
| **Dropped:** `id`| Removed unique identifiers | Prevent data leakage and overfitting |
| **Dropped:** `education_level`, `grade_subgrade`, `loan_purpose` | Replaced by encoded features | Avoid redundancy |
| **Dropped:** `marital_status`, `gender` | Has negligible impact on target variable loan_paid_back | feature reduction |

### Why This Matters

- Most machine-learning algorithms operate on numbers, not labels or strings.
- By converting education_level using ordinal encoder , it helps the model capture the inherent distance between various education levels
- By converting grade_subgrade using ordinal encoder, it helps model capture the distance between various grades.
- Loan Purpose was converted using target encoding because some loans like education and medical loan and riskier to give.
- Prevents **data leakage** by excluding unique identifiers
- Dropped columns like gender and marital status so that model can be built on important features.

### Pipeline Integration

The transformer is used as part of the final ML pipeline:

```python
pipeline = Pipeline(
    steps=[
        ("featureengineering", FeatureEngineering()),
        ("vectorizer", DictVectorizer(sparse=False)),
        ("model", final_model),  # XGBClassifier
    ]
)
```

## Model Training & Selection

The dataset was split into:
* 60% Training
* 20% Validation
* 20% Testing

Multiple models were trained using the training set and evaluated against the validation set. Hyperparameter tuning and threshold optimization were performed to maximize predictive performance, especially focusing on F1-score and Recall, which are critical for fraud detection.

### Models Evaluated

| Model | Tuned Parameters | Decision Threshold | ROC-AUC | Precision | Recall | F1 Score |
|-------|------------------|-------------------|---------|-----------|--------|----------|
| Logistic Regression | `C=0.001, max_iter=3000, n_jobs=-1` | 0.50 | 0.9435 | 0.7714 | 0.4732 | 0.5866 |
| Decision Tree | `max_depth=9, min_samples_leaf=60, random_state=42` | 0.30 | 0.9709 | 0.8833 | 0.7663 | 0.8206 |
| Random Forest | `class_weight='balanced', max_depth=10, min_samples_leaf=10, n_estimators=500, n_jobs=-1, random_state=42` | 0.70 | 0.9758 | 0.6885 | 0.8092 | 0.7440 |
| XGBoost (Final) | `objective='binary:logistic', eval_metric='auc', subsample=1.0, scale_pos_weight=25, n_estimators=500, min_child_weight=40, max_depth=5, learning_rate=0.01, colsample_bytree=0.7, tree_method='hist', n_jobs=-1, random_state=42` | 0.80 | 0.9776 | 0.7944 | 0.8014 | 0.7979 |



```python

```
