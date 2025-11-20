import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb


# -------------------------------------------------------
# Data loading + preprocessing (same logic as before)
# -------------------------------------------------------
def load_and_preprocess_data():

    df = pd.read_csv("train.csv")

    df = df.drop("id", axis=1)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Identify categorical and numerical
    categorical = list(df.dtypes[df.dtypes == "object"].index)
    numerical = list(df.dtypes[df.dtypes != "object"].index)

    for c in categorical:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    # Remove from numerical
    if "loan_paid_back" in numerical:
        numerical.remove("loan_paid_back")

    # drop unused
    df = df.drop(["gender", "marital_status"], axis=1)

    # cleanup education
    df["education_level"] = df["education_level"].replace("master's", "masters")
    df["education_level"] = df["education_level"].replace("bachelor's", "bachelors")

    # Education Encoder
    edu_categories = [["high_school", "other", "bachelors", "masters", "phd"]]
    edu_encoder = OrdinalEncoder(categories=edu_categories)
    df["education_encoded"] = edu_encoder.fit_transform(df[["education_level"]])
    df = df.drop(["education_level", "loan_purpose"], axis=1)

    # Grade encoder
    grade_categories = [[
        "f5", "f4", "f3", "f2", "f1",
        "e5", "e4", "e3", "e2", "e1",
        "d5", "d4", "d3", "d2", "d1",
        "c5", "c4", "c3", "c2", "c1",
        "b5", "b4", "b3", "b2", "b1",
        "a5", "a4", "a3", "a2", "a1"
    ]]
    grade_encoder = OrdinalEncoder(categories=grade_categories)
    df["grade_code"] = grade_encoder.fit_transform(df[["grade_subgrade"]])
    df = df.drop(["grade_subgrade"], axis=1)

    # Target
    y = df.loan_paid_back.values
    df = df.drop("loan_paid_back", axis=1)

    return df, y, edu_encoder, grade_encoder



# -------------------------------------------------------
# Train model
# -------------------------------------------------------
def train_model(df, y):

    categorical = list(df.dtypes[df.dtypes == "object"].index)
    numerical = list(df.dtypes[df.dtypes != "object"].index)

    dv = DictVectorizer(sparse=False)
    train_dict = df[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y, feature_names=features)

    params = {
        "eta": 0.2,
        "max_depth": 6,
        "min_child_weight": 30,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 1,
        "verbosity": 0
    }

    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    return model, dv


# -------------------------------------------------------
# Save all artifacts
# -------------------------------------------------------
def save_artifacts(model, dv, edu_encoder, grade_encoder):

    with open("model.bin", "wb") as f_out:
        pickle.dump(model, f_out)

    with open("dv.bin", "wb") as f_out:
        pickle.dump(dv, f_out)

    with open("encoders.bin", "wb") as f_out:
        pickle.dump(
            {"edu_encoder": edu_encoder, "grade_encoder": grade_encoder},
            f_out
        )

    print("Saved: model.bin, dv.bin, encoders.bin")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
df, y, edu_encoder, grade_encoder = load_and_preprocess_data()
model, dv = train_model(df, y)
save_artifacts(model, dv, edu_encoder, grade_encoder)
