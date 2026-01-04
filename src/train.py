
# train.py

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from src.featureproc import FeatureEngineering


#  Load dataset
df = pd.read_csv("data/train.csv")

y = df["loan_paid_back"]
X = df.drop(columns=["loan_paid_back"])


# Final tuned XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    tree_method='hist', 

    # Tuned parameters
    n_estimators=450,
    min_child_weight=30,
    max_depth=6,
    learning_rate=0.1,
)


# 3Build pipeline: FeatureEngineering → DictVectorizer → XGB
pipeline = Pipeline(
    steps=[
        ("featureengineering", FeatureEngineering()),
        ("vectorizer", DictVectorizer(sparse=False)),
        ("model", xgb_model),
    ]
)


# Train on full dataset
print("\nTraining model on full dataset...")
pipeline.fit(X, y)
print(" Model trained successfully!")


# Store best decision threshold (from your notebook: 0.80)
best_threshold = 0.45


# Save pipeline + threshold
model_path = "models/loanpayback_pipeline.bin"

with open(model_path, "wb") as f_out:
    pickle.dump({"pipeline": pipeline, "threshold": best_threshold}, f_out)

print(f"\n Model saved to: {model_path}")
print(" Saved object contains full pipeline (FE + DictVectorizer + XGB) and threshold.")
