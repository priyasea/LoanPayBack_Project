import pickle
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field,ConfigDict

# -------------------------------------------------------
# Load artifacts
# -------------------------------------------------------
with open("model.bin", "rb") as f_in:
    model = pickle.load(f_in)

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

with open("encoders.bin", "rb") as f_in:
    enc = pickle.load(f_in)

edu_encoder = enc["edu_encoder"]
grade_encoder = enc["grade_encoder"]




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
    credit_score: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)
    debt_to_income_ratio: float = Field(..., ge=0.0)
    loan_amount: float = Field(..., ge=0.0)
    interest_rate: float = Field(..., ge=0.0)
    

# -------------------------------------------------------
# Preprocess input
# -------------------------------------------------------

class PredictResponse(BaseModel):
    loan_pay_back_probability: float
    loan_pay_back: bool

app = FastAPI(title="customer-churn-prediction")

def preprocess(features):

    # Convert input to DataFrame
    df = pd.DataFrame([features])

    # --- Education cleanup ---
    df["education_level"] = df["education_level"].str.lower().str.replace(" ", "_")
    df["education_level"] = df["education_level"].replace("master's", "masters")
    df["education_level"] = df["education_level"].replace("bachelor's", "bachelors")

    df["education_encoded"] = edu_encoder.transform(df[["education_level"]])

    # --- Grade cleanup ---
    df["grade_subgrade"] = df["grade_subgrade"].str.lower()
    df["grade_code"] = grade_encoder.transform(df[["grade_subgrade"]])

    # Drop original text input columns
    df = df.drop(["education_level", "grade_subgrade"], axis=1)

    return df


# -------------------------------------------------------
# Predict single record
# -------------------------------------------------------

#def predict_single(features):
  #  df = preprocess(features)

    # Columns: numeric + encoded
   # X = dv.transform(df.to_dict(orient="records"))
    #dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))

    #return float(model.predict(dmatrix)[0])

def predict_single(customer):
   # result = pipeline.predict_proba(customer)[0, 1]
   # return float(result)
   df = preprocess(customer)

    # Columns: numeric + encoded
   X = dv.transform(df.to_dict(orient="records"))
   dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
   return float(model.predict(dmatrix)[0]) 

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:

    prob = predict_single(customer.model_dump())

    return PredictResponse(
        loan_pay_back_probability=prob,
        loan_pay_back=prob >= 0.5
    )

# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
        #sample = {
         #   "loan_amount": 15000,
         #   "interest_rate": 12.5,
         #   "credit_score": 689,
         #   "annual_income": 82000,
         #   "debt_to_income_ratio": 17.3,
         #   "employment_status": "student",
         #   "education_level": "bachelors",
          #  "grade_subgrade": "b3"
        #}

    #prediction = predict_single(sample)
    #print("Predicted probability of loan being paid back:", prediction)