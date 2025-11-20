📘 Loan Payback Prediction — Midterm Project <br>
 This project predicts whether a loan applicant will successfully pay back a loan based on demographic and financial features.
 This repository contains the Loan Payback Prediction project implemented in the Jupyter Notebook LoanPayback_MidTerm_with_EDA.ipynb.
 The notebook includes complete data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and insights.
 This notebook trains the datsset on some of the ML models we learnt and then chooses the Final Model which gives best ROC_AUC_SCORE

📂 Project Structure<br>
LoanPayback_Project/
│
├── README.md
├── LoanPayback_with_EDA.ipynb              # EDA + model development
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

📂 **Project Overview**
Financial institutions want to estimate whether a borrower is likely to repay a loan.
Using customer and loan features, this project builds a model to classify loan repayment behavior.
The workflow includes:
The goal of this project is to analyze a loan dataset and predict whether a loan will be paid back using machine learning techniques.
This includes:
<li>
<ol> Data cleaning and preprocessing </ol>
<ol> Exploratory Data Analysis (EDA) </ol>
<ol> Feature engineering and selection </ol>
<ol> Model selection and hyperparameter tuning </ol>
<ol> Training final model</ol>
<ul> Deploying the model using FastAPI and Docker</ul>
</li>



🧠 **1. Key Steps in the Notebook LoanPayback_with_EDA****
1️⃣ Dataset Preview <br>
The notebook starts by loading the dataset and visually inspecting the first few rows to understand the structure of the data.<br>
2️⃣ Checking Data Types<br>
A detailed check of column data types ensures correct handling of numerical and categorical variables.<br>
3️⃣ Analysis of categorical variables<br>
The notebook performs:<br>
Analysis has been done on the categorical variables to determine Risk Ratio and Mutual Information score.<br>
Also some categorical variables exhibited a ordinal relationship and hence has been encoded to numeric like education level and grade_subgrade.<br>
4️⃣ Analysis of numeric Variables:- Numeric variables have been assigned using Correlation Heatmap.
5️⃣ Dropping Non-Predictive Columns<br>
Columns with no predictive value (like IDs) are removed to avoid noise in the model.<br>
6️⃣ Exploratory Data Analysis (EDA)<br>
Although headings were limited, typical EDA steps include:<br>
Histograms<br>
Correlation heatmaps<br>
Distribution checks<br>
Relationship between variables and target<br>
7️⃣ Feature Engineering<br>
The variables that really affects the Loan Paid Back variable like Employment Status has been kept while rest like gender, marital status, loan purpose has been dropped<br>
Based on the EDA, categorical variables may be encoded and numeric variables normalized/cleaned.<br>

8️⃣ Model Training<br>
Machine learning models that were trained are<br>
Logistic Regression<br>
Decision Trees<br>
Random Forest<br>
XGBoost<br>
Hyperparameters are tuned to improve model performance.<br>

9️⃣ Model Evaluation<br>
Common evaluation metrics:<br>
Accuracy<br>
ROC–AUC<br>

**2. Dataset<br>**
File: train.csv<br>
Size: ~53 MB<br>
Includes features such as:<br>
Income<br>
Employment<br>
Education<br>
Loan amount<br>
Payment history<br>
Grade / Rating<br>
If dataset is missing after cloning the repo:<br>
👉 Download your dataset and place it in the project root as:<br>
LoanPayback_Project/train.csv<br>

**🛠 3. Training Script (train.py) <br>**
The training pipeline:<br>
Preprocesses data<br>
Encodes categorical variables<br>
Vectorizes features<br>
Trains the final model<br>
Saves the following artifacts:<br>
model.bin<br>
dv.bin<br>
encoders.bin<br>
Run training:<br>
python train.py<br>

**4. Prediction Service (predict.py)<br>**
FastAPI-based service that:<br>
Loads model.bin, dv.bin, encoders.bin<br>
Accepts JSON input<br>
Returns the loan payback probability & classification<br> 

**5. Dependencies**
Dependencies are managed with uv and stored in pyproject.toml and uv.lock.

Install:
uv sync

**7.  Docker Deployment**
🌐 Deployment
This project includes a Dockerized FastAPI prediction service.

Run the service locally:<br>
docker build -t loanpayback . <br> 
docker run -it --rm -p 9696:9696 loanpayback <br>

Test prediction endpoint:
POST http://localhost:9696/predict

 {<br>
	"employment_status": "employed",<br>
  "education_level": "bachelors",<br>
  "grade_subgrade": "b3",<br>
	"credit_score": 689,<br>
	 "annual_income": 82000,<br>
   "debt_to_income_ratio": 17.3,<br>
   "loan_amount": 15000,<br>
    "interest_rate": 12.5<br>
    }<br>

📊 Technologies Used<br>
Python<br>
Pandas, NumPy<br>
Matplotlib / Seaborn<br>
Scikit-learn<br>
XGBoost<br>
Jupyter Notebook<br>

🚀 How to Run the Project


📁 Repository Structure
.
├── LoanPayback_MidTerm_with_EDA.ipynb     # Main notebook
├── train.py                                # (Optional) Script version
├── README.md                               # Project documentation
└── data/                                   # Dataset (if included)
