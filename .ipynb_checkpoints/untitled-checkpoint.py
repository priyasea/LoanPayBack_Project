import pickle
import pandas as pd
import xgboost as xgb


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


# -------------------------------------------------------
# Preprocess input
# -------------------------------------------------------
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
def predict_single(features):

    df = preprocess(features)

    # Columns: numeric + encoded
    X = dv.transform(df.to_dict(orient="records"))
    dmatrix = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

    return float(model.predict(dmatrix)[0])


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":

    sample = {
        "loan_amount": 15000,
        "interest_rate": 12.5,
        "term": 36,
        "employment_length": 5,
        "annual_income": 82000,
        "dti_ratio": 17.3,
        "education_level": "bachelors",
        "grade_subgrade": "b3"
    }

    prediction = predict_single(sample)
    print("Predicted probability of loan being paid back:", prediction)