📘 Loan Payback Prediction — Midterm Project
This project predicts whether a loan applicant will successfully pay back a loan based on demographic and financial features.
This repository contains the Loan Payback Prediction project implemented in the Jupyter Notebook LoanPayback_MidTerm_with_EDA.ipynb.
The notebook includes complete data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and insights.
This notebook trains the datsset on some of the ML models we learnt and then chooses the Final Model which gives best ROC_AUC_SCORE

📂 Project Structure
LoanPayback_Project/
│
├── README.md
├── notebook.ipynb              # EDA + model development
│
├── train.py                    # Training pipeline
├── predict.py                  # FastAPI prediction service
│
├── model.bin                   # Trained model
├── dv.bin                      # DictVectorizer
├── encoders.bin                # Label/Ordinal encoders
│
├── train.csv                   # Dataset (53 MB)
│
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependencies
│
├── Dockerfile                  # Container for prediction service
└── deployment_screenshot.png   # Optional (deployment proof)

📂 Project Overview
Financial institutions want to estimate whether a borrower is likely to repay a loan.
Using customer and loan features, this project builds a model to classify loan repayment behavior.
The workflow includes:
The goal of this project is to analyze a loan dataset and predict whether a loan will be paid back using machine learning techniques.
This includes:

Data cleaning and preprocessing
Exploratory Data Analysis (EDA)
Feature engineering and selection
Model selection and hyperparameter tuning
Training final model

Deploying the model using FastAPI and Docker

🧠 Key Steps in the Notebook
1️⃣ Dataset Preview

The notebook starts by loading the dataset and visually inspecting the first few rows to understand the structure of the data.

2️⃣ Checking Data Types

A detailed check of column data types ensures correct handling of numerical and categorical variables.

3️⃣ Analysis of categorical variables

The notebook performs:

Analysis has been done on the categorical variables to determine Risk Ratio and Mutual Information score.

Also some categorical variables exhibited a ordinal relationship and hence has been encoded to numeric like education level and grade_subgrade.

4️⃣ Analysis of numeric Variables:- Numeric variables have been assigned using Correlation Heatmap.


5️⃣ Dropping Non-Predictive Columns

Columns with no predictive value (like IDs) are removed to avoid noise in the model.

6️⃣ Exploratory Data Analysis (EDA)

Although headings were limited, typical EDA steps include:

Histograms

Correlation heatmaps

Distribution checks

Relationship between variables and target

7️⃣ Feature Engineering

The variables that really affects the Loan Paid Back variable like Employment Status has been kept while rest like gender, marital status, loan purpose has been dropped
Based on the EDA, categorical variables may be encoded and numeric variables normalized/cleaned.

8️⃣ Model Training

Machine learning models that were trained are

Logistic Regression

Decision Trees

Random Forest

XGBoost

Hyperparameters are tuned to improve model performance.

9️⃣ Model Evaluation

Common evaluation metrics:

Accuracy

ROC–AUC

📊 Technologies Used

Python

Pandas, NumPy

Matplotlib / Seaborn

Scikit-learn

XGBoost

Jupyter Notebook

🚀 How to Run the Project


📁 Repository Structure
.
├── LoanPayback_MidTerm_with_EDA.ipynb     # Main notebook
├── train.py                                # (Optional) Script version
├── README.md                               # Project documentation
└── data/                                   # Dataset (if included)
