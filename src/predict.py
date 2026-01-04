# predict.py

from typing import Dict, Any
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from fastapi.responses import HTMLResponse
from pathlib import Path
# =========================
# Load saved model pipeline
# =========================
MODEL_PATH = "models/loanpayback_pipeline.bin"
BASE_DIR = Path(__file__).resolve().parent.parent
with open(MODEL_PATH, "rb") as f_in:
    model_data = pickle.load(f_in)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]

print(f"Model loaded successfully with threshold = {threshold}")

# =========================
# FastAPI App Initialization
# =========================

# =========================
# Pydantic Input Schema
# =========================


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employment_status: Literal[
        "employed",         
        "retired"  ,         
        "self-employed" ,    
        "student",           
        "unemployed" 
    ]
    education_level: Literal[
    "bachelors" ,  
    "high_school" ,  
    "masters", 
    "other",     
    "phd"      
    ]
    grade_subgrade: Literal["a1","a2","a3","a4","a5","b1","b2","b3","b4","b5","c1","c2","c3","c4","c5","d1","d2","d3","d4","d5","e1","e2","e3","e4","e5"
    ,"f1","f2","f3","f4","f5"]
    loan_purpose: Literal["business" ,"car" ,"debt_consolidation" 
    ,"education"             
    ,"home"                  
    ,"medical"               
    ,"other"                 
    ,"vacation"]
    credit_score: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)
    debt_to_income_ratio: float = Field(..., ge=0.0)
    loan_amount: float = Field(..., ge=0.0)
    interest_rate: float = Field(..., ge=0.0)
    

# -------------------------------------------------------
# Preprocess input
# -------------------------------------------------------

class LoanPaybackPrediction(BaseModel):
    loan_paid_back_probability: float
    loan_paid_back: str

app = FastAPI(title="Loan Payback Prediction API", version="1.0")

# =========================
# Prediction Function
# =========================

def predict_Customer(customer):
    df = pd.DataFrame([customer])
    prob = pipeline.predict_proba(df)[0][1]
    return prob

# =========================
# API Endpoints
# =========================
@app.get("/")
def home():
    return {"message": "Welcome to Loan Payback Preiction API"}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html_path = BASE_DIR / "templates" / "index.html"
    return html_path.read_text()

@app.post("/predict",response_model=LoanPaybackPrediction)
def predict_api(customer: Customer) -> LoanPaybackPrediction:
    prob = predict_Customer(customer.model_dump())
    pred = "Loan shall be paid back" if prob >= threshold else "Loan will not be paid back"
    return LoanPaybackPrediction(
        loan_paid_back_probability=round(float(prob), 4),
        loan_paid_back= pred
     )
# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)