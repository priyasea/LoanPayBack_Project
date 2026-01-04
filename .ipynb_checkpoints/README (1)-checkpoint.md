# E-Commerce Fraud Detection using Machine Learning  
### Predicting fraudulent transactions with XGBoost & FastAPI | Dockerized & Deployed on Render

## Problem Description

E-commerce platforms process millions of transactions every day. Even a small number of fraudulent transactions can lead to major financial losses, chargebacks, operational overhead, and customer dissatisfaction.

Fraud detection is a challenging problem due to:

- Highly imbalanced datasets (fraudulent cases are extremely rare)  
- Constantly evolving fraud patterns  
- The requirement for near real-time decision-making  
- The need to avoid false positives, which can negatively impact genuine users  

The objective of this project is to develop a machine learning–based fraud detection system that predicts whether a transaction is fraudulent using historical data and engineered features.

The project includes:

- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)  
- Hyperparameter optimization and threshold tuning  

The final solution is built using the XGBoost model, optimized with a custom decision threshold of `0.80`, which balances fraud detection coverage and false positives.

The model is deployed as a REST API using FastAPI, containerized using Docker, and hosted on Render for real-time inference.

---

## How the Solution Is Used

### 1. Single Transaction Prediction 

When a transaction is processed, its details are sent to the `/predict` endpoint. The API responds with fraud probability and classification. 

```json
{
  "fraud_probability": 0.9875,
  "prediction": "Fraud"
}
```
Based on the model response:

| Prediction | Recommended Action                            |
|------------|-----------------------------------------------|
| Fraud      | Block transaction or send for manual review   |
| Legitimate | Approve transaction                           

### 2. Multiple Transactions (Batch Prediction)

The API also supports batch processing via the `/predict_batch endpoint`. This enables fraud detection for bulk transaction processing, which is particularly useful for nightly batch reviews or risk scoring at scale.

#### Example request:
```json
[
  {
    "transaction_id": 1,
    "user_id": 100,
    "account_age_days": 530,
    "total_transactions_user": 45,
    "avg_amount_user": 125.75,
    "amount": 180.50,
    "country": "US",
    "bin_country": "US",
    "channel": "web",
    "merchant_category": "electronics",
    "promo_used": 0,
    "avs_match": 1,
    "cvv_result": 1,
    "three_ds_flag": 1,
    "transaction_time": "2024-02-19T14:05:00Z",
    "shipping_distance_km": 350.2
  },
  {
    "transaction_id": 2,
    "user_id": 101,
    "account_age_days": 90,
    "total_transactions_user": 50,
    "avg_amount_user": 80.10,
    "amount": 950.00,
    "country": "FR",
    "bin_country": "RO",
    "channel": "app",
    "merchant_category": "gaming",
    "promo_used": 1,
    "avs_match": 0,
    "cvv_result": 0,
    "three_ds_flag": 0,
    "transaction_time": "2024-02-19T02:23:00Z",
    "shipping_distance_km": 2200.8
  }
]
```
#### Example Response
```json
[
  {
    "fraud_probability": 0.0383,
    "prediction": "Legitimate"
  },
  {
    "fraud_probability": 0.9342,
    "prediction": "Fraud"
  }
]
```

---

### Summary of integration

- Supports real-time single transaction scoring
- Enables batch fraud detection for multiple transactions
- Returns a probability score and final fraud prediction
- Suitable for integration into live payment systems or periodic transaction reviews
- Helps reduce financial losses while minimizing false positives

The system is suitable for real-world integration into transaction processing pipelines to minimize fraud risk and enhance payment security.

## Exploratory Data Analysis (EDA)

### 1. Dataset Overview

- **Total transactions:** `299,695`
- **Fraudulent transactions:** `6,612` (~2.2%)
- **Legitimate transactions:** `293,083` (~97.8%)
- **Unique users:** `6,000`
- **Missing values:** `None`

**Data source:** [E-Commerce Fraud Detection Dataset(Kaggle)]  
(https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)

This synthetic but realistic dataset simulates e-commerce transactions across countries and platforms. It models patterns similar to actual financial fraud scenarios while preserving privacy.

---

### 2. Feature Overview

| Feature Type | Features |
|--------------|----------|
| **Identifier** | `transaction_id`, `user_id` |
| **Numerical** | `account_age_days`, `total_transactions_user`, `avg_amount_user`, `amount`, `shipping_distance_km` |
| **Categorical** | `country`, `bin_country`, `channel`, `merchant_category` |
| **Binary flags** | `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag` |
| **Time-based** | `transaction_time` (later transformed) |
| **Target variable** | `is_fraud` (0 = legitimate, 1 = fraud) |

---

### 3. Fraud Distribution (Class Imbalance)

The target variable is heavily imbalanced:
- **Legitimate:** `293,083`
- **Fraud:** `6,612`

**Image in repository at:** `images/fraud_distribution.png`

![Fraud vs Non-Fraud Transaction Counts](images/fraud_distribution.png)

**Key insight:**
- Fraudulent transactions represent only ~2.2% of the dataset
- Accuracy alone is misleading → we focused on **ROC-AUC, Precision, Recall, and F1-score**

---

### 4. Correlation of Numeric Features

**Image in repository:** `images/correlation_matrix.png`

![Correlation Matrix of Numeric Features](images/correlation_matrix.png)

**Important correlations with `is_fraud`:**
- `shipping_distance_km` → 0.27
- `amount` → 0.20
- `account_age_days` → -0.12

#### Interpretation:
* **Large shipping distances** → suspicious (possible cross-border fraud) 
* **Higher transaction amount** → higher fraud risk 
* **New accounts (low age)** → more likely involved in fraud

---

### 5. Mutual Information (Categorical/Binary Features)

| Feature | MI Score |
|---------|----------|
| `avs_match` | 0.0169 |
| `cvv_result` | 0.0149 |
| `three_ds_flag` | 0.0103 |
| `channel` | 0.0048 |
| `promo_used` | 0.0019 |
| `country` | 0.0002 |
| `bin_country` | 0.0001 |
| `merchant_category` | 0.0000 |

**Insight:**
- Security-related checks (`avs_match`, `cvv_result`, `three_ds_flag`) are strong indicators of fraud
- Country-based features have low standalone influence but improve performance when encoded effectively

---

### 6. Key Takeaways from EDA

1. **Severe class imbalance (~2.2% fraud)** → Metrics like accuracy are incorrect for model evaluation, use ROC-AUC, Precision, Recall, and F1-score.
2. **Transaction amount and shipping distance** are important fraud indicators.
3. **Newer accounts** (`low account_age_days`) are more likely to commit fraud.
4. **Security flag features** (`avs_match`, `cvv_result`, `three_ds_flag`) highly correlate with fraud.
5. **No missing data** → focus was on feature engineering rather than cleaning.

---

## Feature Engineering

To improve model performance and capture fraud-related behavioral patterns, a custom feature transformation pipeline was implemented using `FeatureEngineering` (located in **src/features.py**). This transformation is applied as the first step of the ML pipeline to ensure consistency during both training and inference.

### Key Transformations

| Feature | Description | Motivation |
|--------|-------------|------------|
| `amount_per_avg_ratio` | `amount / avg_amount_user` | Detect unusually high-value transactions specific to the user’s past behavior |
| `cross_country_flag` | 1 if `bin_country ≠ country`, else 0 | Flags transactions where the issuing card country differs from shipping country |
| `country_freq`, `bin_country_freq` | Frequency encoding of country columns | Capture population-based risk while handling cardinality |
| `hour`, `day_of_week` | Extracted from `transaction_time` | Fraud tends to follow unusual temporal patterns |
| `is_night` | 1 if time between 00:00–06:00 | Fraud typically increases during low monitoring hours |
| **Dropped:** `transaction_id`, `user_id` | Removed unique identifiers | Prevent data leakage and overfitting |
| **Dropped:** `country`, `bin_country`, `transaction_time` | Replaced by engineered temporal/geographic features | Avoid redundancy |

### Why This Matters

- Helps the model capture **behavioral anomalies** and **geographical inconsistencies**  
- Prevents **data leakage** by excluding unique identifiers  
- Temporal features (hour, day_of_week, is_night) helped improve detection of night-time fraud  
- The engineered features contributed to improved predictive performance, especially **precision and recall**

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
| Logistic Regression | `C=0.001, max_iter=3000, n_jobs=-1` | 0.30 | 0.9435 | 0.7714 | 0.4732 | 0.5866 |
| Decision Tree | `max_depth=9, min_samples_leaf=60, random_state=42` | 0.30 | 0.9709 | 0.8833 | 0.7663 | 0.8206 |
| Random Forest | `class_weight='balanced', max_depth=10, min_samples_leaf=10, n_estimators=500, n_jobs=-1, random_state=42` | 0.70 | 0.9758 | 0.6885 | 0.8092 | 0.7440 |
| XGBoost (Final) | `objective='binary:logistic', eval_metric='auc', subsample=1.0, scale_pos_weight=25, n_estimators=500, min_child_weight=40, max_depth=5, learning_rate=0.01, colsample_bytree=0.7, tree_method='hist', n_jobs=-1, random_state=42` | 0.80 | 0.9776 | 0.7944 | 0.8014 | 0.7979 |

### Threshold Optimization Visualizations

To determine the optimal probability threshold for classification, Precision–Recall–F1 curves were plotted for each model:
#### Logistic Regression 
![Precision-Recall-F1 curve for different thresholds](images/logistic_regression_threshold_performance.png)
#### Decision Tree
![Precision-Recall-F1 curve for different thresholds for decision tree](images/decision_threshold_performance.png)
#### Random Forest
![Precision-Recall-F1 curve for different thresholds for Random Forest](images/rf_threshold_performance.png)
#### XGBClassifier (selected model)
![Precision-Recall-F1 curve for different thresholds for XGBClassifier](images/xgb_classifier_performance.png)

### Final Model Selection

After comparing performance across models, **XGBClassifier** was selected as the final production model based on the following:

* Highest ROC-AUC on validation  
* Balanced Precision and Recall  
* Best F1-score, indicating strong fraud detection capability with minimal false positives  
* Captures non-linear relationships effectively  
* Optimized using `scale_pos_weight` to handle class imbalance

### Final Model Evaluation (on Test Set)

After selecting XGBClassifier, the model was retrained using full train + testing datasets, and final evaluation was performed on the test set.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9781 |
| Precision | 0.7823 |
| Recall | 0.8238 |
| F1 Score | 0.8025 |
| Decision Threshold | 0.80 |

### Why Threshold = 0.80?

* A lower threshold catches more fraud but increases false positives.
* A higher threshold reduces false alerts but may miss fraud cases.
* **0.80 provides the best trade-off, maximizing F1-score (0.8025).**

## Exporting Notebook to Script

To comply with project requirements and ensure reproducibility, all essential machine learning steps developed in the notebook (`notebooks/notebook.ipynb`) were fully converted into Python scripts.

### Scripts Created

| Script | Purpose |
|--------|---------|
| `src/train.py` | Contains final model training pipeline and saves the trained model. |
| `src/predict.py` | Loads the trained model and serves predictions via a FastAPI REST endpoint. |
| `src/features.py` | Implements the custom feature engineering logic. |

### What Was Exported from Notebook

The following core logic developed and validated in `notebooks/notebook.ipynb` was migrated into standalone scripts for production readiness:

| Exported Component | Implemented In | Description |
|-------------------|----------------|-------------|
| Data loading | `train.py` | Reads transaction dataset from `data/transactions.csv`. |
| Feature engineering logic | `features.py` | Custom transformer class `FeatureEngineering`. |
| Model training & hyperparameter tuning | `train.py` | Uses tuned XGBoost model parameters finalized from notebook experiments. |
| Decision threshold selection (`0.80`) | `train.py` | Threshold locked based on best F1 score from validation results. |
| Final model training | `train.py` | Trains full XGBoost model on entire training dataset. |
| Model serialization (pipeline + threshold) | `train.py` | Saved using `pickle` as `models/fraud_detection_xgb_pipeline.bin`. |
| API-based prediction logic | `predict.py` | Loads trained pipeline and serves predictions via FastAPI. |

### Example: Model Saving in `train.py`
```python
model_path = "models/fraud_detection_xgb_pipeline.bin"

with open(model_path, "wb") as f_out:
    pickle.dump({"pipeline": pipeline, "threshold": best_threshold}, f_out)
```

### 
Example: Model Loading in `predict.py`
```python
with open(MODEL_PATH, "rb") as f_in:
    model_data = pickle.load(f_in)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]
```

## Reproducibility

This project is fully reproducible. The dataset, notebook, and training scripts are included in the repository, allowing seamless re-execution.

- Dataset available in `data/transactions.csv`
- Full analysis in `notebooks/notebook.ipynb`
- Feature Training function in `src/features.py`
- Final model training located in `src/train.py`
- Inference logic exposed via `src/predict.py`
- Trained pipeline saved at `models/fraud_detection_xgb_pipeline.bin`

### How to Reproduce
```bash
# Install dependencies and set up environment
uv sync

# Run training script
uv run python -m src.train

# Start inference API
uv run uvicorn src.predict:app --reload --port 8000
```

---

## Model Deployment (Local)

The trained machine learning model is deployed locally using **FastAPI** and served via **Uvicorn**.

### Start API Locally
```bash
uv run uvicorn src.predict:app --reload --port 8000
```

Once running:
- **Swagger UI:** `http://localhost:8000/docs`
- **Root endpoint:** `http://localhost:8000/`
- **Supports:**
  - Single transaction prediction → `/predict`
  - Batch prediction → `/predict_batch`
  
---

## Dependency & Environment Management

The project uses **uv** to manage dependencies and execution. All required packages are defined in `pyproject.toml` and `requirements.txt`.

### Install Dependencies
```bash
uv sync
```
### Example Execution Commands
```bash
uv run python -m src.train      # Train model
uv run uvicorn src.predict:app --reload --port 8000   # Launch API
```
---

## Dependency Files

### `requirements.txt`
```txt
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
pydantic
pickle-mixin
```

### `pyproject.toml`
```toml
[project]
name = "ml-ecommerce-fraud-detection"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.121.2",
    "numpy>=2.3.5",
    "pandas>=2.3.3",
    "pydantic>=2.12.4",
    "scikit-learn>=1.7.2",
    "uvicorn>=0.38.0",
    "xgboost>=3.1.1",
]
```
---

## Containerization (Docker)

The project is fully containerized using **Docker**, allowing consistent deployment across environments.

### Dockerfile Used
```dockerfile
# Use lightweight Python
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
```
---

### Build the Docker Image

Run the following command inside the project folder:
```bash
docker build -t fraud-detection-api:latest .
```
---

### Run the Docker Container
```bash
docker run -p 8000:8000 fraud-detection-api:latest
```

Once started, the API will be available at:
- **Local URL** → `http://localhost:8000/docs/`
- **Swagger UI** → `http://localhost:8000/docs/`
---

## Cloud Deployment

The fraud detection API is deployed using FastAPI + Docker on Render, allowing real-time inference via REST API.

**Live API URL**:
[Render API URL ]
(https://fraud-detection-api-hsyb.onrender.com/docs)

**(Interactive Swagger UI for API testing)**

### Deployment Steps (Docker + Render)

#### 1. Push complete project to GitHub
[github repo link]
(https://github.com/codevalhalla/ml-ecommerce-fraud-detection)


#### 2. On Render Dashboard → “New Web Service”

#### 3. Select Deployment Settings

| Setting | Value |
|---------|-------|
| Environment |	Docker |
| Repository | `codevalhalla/ml-ecommerce-fraud-detection` |
| Branch | main |
| Root Directory | `(leave empty)` |
| Environment Variables | `PORT=8000` |
| Instance Type | Free Tier |

#### 4. Click "Deploy Web Service"

Render automatically:

* Pulls repo

* Builds Docker image

* Runs FastAPI service using command from Dockerfile
```CSS
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
```
---
##  Proof of Successful Deployment

### Render Deployment Configuration
![Deployment video](images/cloud_deployment/cloud_deployment_video.webm)
![Deployment Pic](images/cloud_deployment/render_main_api.png)


### API Testing Examples

#### 1. Single Transaction (POST /predict)

##### Request

![API request](images/cloud_deployment/predict_request.png)

```json
{
  "transaction_id": 1001,
  "user_id": 25,
  "account_age_days": 720,
  "total_transactions_user": 52,
  "avg_amount_user": 120.50,
  "amount": 480.75,
  "country": "US",
  "bin_country": "US",
  "channel": "web",
  "merchant_category": "electronics",
  "promo_used": 0,
  "avs_match": 1,
  "cvv_result": 1,
  "three_ds_flag": 1,
  "transaction_time": "2024-03-10T14:35:00Z",
  "shipping_distance_km": 650.40
}
```

##### Response:

![API request](images/cloud_deployment/predict_response.png)

```json
{
  "fraud_probability": 0.8438,
  "prediction": "Fraud"
}
```

#### 2. Batch Transactions (POST /predict_batch)

##### Batch Request

![API Batch request](images/cloud_deployment/predict_batch_request.png)

```json
[
  {
    "transaction_id": 1101,
    "user_id": 41,
    "account_age_days": 300,
    "total_transactions_user": 45,
    "avg_amount_user": 95.40,
    "amount": 950.00,
    "country": "DE",
    "bin_country": "TR",
    "channel": "app",
    "merchant_category": "gaming",
    "promo_used": 1,
    "avs_match": 0,
    "cvv_result": 0,
    "three_ds_flag": 0,
    "transaction_time": "2024-03-05T02:10:00Z",
    "shipping_distance_km": 3500.00
  },
  {
    "transaction_id": 1102,
    "user_id": 102,
    "account_age_days": 820,
    "total_transactions_user": 58,
    "avg_amount_user": 180.00,
    "amount": 150.75,
    "country": "FR",
    "bin_country": "FR",
    "channel": "web",
    "merchant_category": "fashion",
    "promo_used": 0,
    "avs_match": 1,
    "cvv_result": 1,
    "three_ds_flag": 1,
    "transaction_time": "2024-03-05T17:00:00Z",
    "shipping_distance_km": 120.00
  }
]
```

##### Batch Response

![API Batch request](images/cloud_deployment/predict_batch_request.png)

```json
[
  {
    "fraud_probability": 0.9824,
    "prediction": "Fraud"
  },
  {
    "fraud_probability": 0.0365,
    "prediction": "Legitimate"
  }
]
```
## Conclusion

This project successfully implements an end-to-end fraud detection system for e-commerce transactions using machine learning. After performing detailed exploratory data analysis, feature engineering, and model comparison, XGBoost was selected as the final model due to its high performance:

ROC-AUC: 0.9781

F1-Score: 0.8025

Decision Threshold: 0.80

The final solution was packaged with FastAPI, containerized using Docker, and deployed on Render, enabling real-time prediction of single or multiple transactions. The system is ready for integration into production environments to help minimize fraud risk and improve transactional security.

Future enhancements may include real-time streaming, model monitoring, explainability (SHAP), and additional behavioral analytics


