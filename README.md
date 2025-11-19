📘 Loan Payback Prediction — Midterm Project

This repository contains the Loan Payback Prediction project implemented in the Jupyter Notebook LoanPayback_MidTerm_with_EDA.ipynb.
The notebook includes complete data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and insights.
This notebook trains the datsset on some of the ML models we learnt and then chooses the Final Model which gives best ROC_AUC_SCORE

📂 Project Overview

The goal of this project is to analyze a loan dataset and predict whether a loan will be paid back using machine learning techniques.
This includes:

Understanding the dataset

Cleaning and preprocessing

Performing exploratory data analysis (EDA)

Building predictive models

Evaluating model performance

Drawing insights from the results

🧠 Key Steps in the Notebook
1️⃣ Dataset Preview

The notebook starts by loading the dataset and visually inspecting the first few rows to understand the structure of the data.

2️⃣ Checking Data Types

A detailed check of column data types ensures correct handling of numerical and categorical variables.

3️⃣ Analysis of categorical variables

The notebook performs:

Analysis has been done on the categorical variables to determine Risk Ratio and Mutual Information score. The variables that really affects the Loan Paid Back variable has been kept while rest like gender, marital status, loan purpose has been dropped

Also some categorical variables exhibited a ordinal relationship and hence has been encoded to numeric like education level and grade_subgrade.

4️⃣ Statistical Summary

Summary statistics help identify:

Data distribution

Outliers

Feature behavior

Potential transformation requirements

5️⃣ Dropping Non-Predictive Columns

Columns with no predictive value (like IDs) are removed to avoid noise in the model.

6️⃣ Exploratory Data Analysis (EDA)

Although headings were limited, typical EDA steps include:

Histograms

Correlation heatmaps

Distribution checks

Relationship between variables and target

7️⃣ Feature Engineering

Based on the EDA, categorical variables may be encoded and numeric variables normalized/cleaned.

8️⃣ Model Training

Machine learning models may include:

Logistic Regression

Random Forest

XGBoost

Other classifiers

Hyperparameters are tuned to improve model performance.

9️⃣ Model Evaluation

Common evaluation metrics:

Accuracy

Precision, Recall, F1

ROC–AUC

Confusion Matrix

📊 Technologies Used

Python

Pandas, NumPy

Matplotlib / Seaborn

Scikit-learn

XGBoost (if used)

Jupyter Notebook

🚀 How to Run the Notebook
1. Install Dependencies
pip install -r requirements.txt


(If you don’t have requirements.txt, I can generate it based on the notebook.)

2. Open the Notebook
jupyter notebook LoanPayback_MidTerm_with_EDA.ipynb

📁 Repository Structure
.
├── LoanPayback_MidTerm_with_EDA.ipynb     # Main notebook
├── train.py                                # (Optional) Script version
├── README.md                               # Project documentation
└── data/                                   # Dataset (if included)
