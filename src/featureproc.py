import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    """
      Custom transformer that:
    - Drops ID columns 
    - Drops columns that dont influence target
    - Encodes Education Level feature
    - Encodes Grade Subgrade feature
    - Encodes Loan Purpose feature
    - Returns list[dict] suitable for DictVectorizer
    """
    def __init__(self):
        self.edu_encoder = None
        self.grade_encoder = None
        self.purpose_te = None
     
         
    def fit(self, X, y):
        X = X.copy()

        categorical_columns = list(X.dtypes[X.dtypes == 'object'].index)
        for c in categorical_columns:
            X[c] = (
                X[c]
                .str.lower()
                .str.replace(" ", "_")
            )


        # Normalize education labels
        X['education_level'] = (
            X['education_level']
            .replace({"master's": "masters", "bachelor's": "bachelors"})
        )

        # Ordinal encoders
        self.edu_encoder = OrdinalEncoder(
            categories=[['high_school', 'other', 'bachelors', 'masters', 'phd']],
            handle_unknown="use_encoded_value",
            unknown_value=-1
            )

        self.grade_encoder = OrdinalEncoder(
            categories=[[
                'f5','f4','f3','f2','f1',
                'e5','e4','e3','e2','e1',
                'd5','d4','d3','d2','d1',
                'c5','c4','c3','c2','c1',
                'b5','b4','b3','b2','b1',
                'a5','a4','a3','a2','a1'
            ]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        self.edu_encoder.fit(X[['education_level']])
        self.grade_encoder.fit(X[['grade_subgrade']])

        # Target encoding (SAFE: only in fit)
        self.purpose_te = pd.Series(y).groupby(X['loan_purpose']).mean()

        return self



    def transform(self, X):
        X = X.copy()

        # Drop identifiers
        X = X.drop(
            ['id', 'gender', 'marital_status'],
            axis=1,
            errors='ignore'
            )

        # Normalize education
        X['education_level'] = (
            X['education_level']
            .replace({"master's": "masters", "bachelor's": "bachelors"})
        )

        # Ordinal encoding
        X['education_encoded'] = self.edu_encoder.transform(
            X[['education_level']]
        )
        X['grade_code'] = self.grade_encoder.transform(
            X[['grade_subgrade']]
        )
        global_mean = self.purpose_te.mean()
        X['loan_purpose_te'] = (
            X['loan_purpose']
            .map(self.purpose_te)
            .fillna(global_mean)
        )

        X = X.drop(
            ['education_level', 'grade_subgrade', 'loan_purpose'],
            axis=1
        )
       # VERY IMPORTANT: DictVectorizer expects list of dicts
        return X.to_dict(orient="records")
     

