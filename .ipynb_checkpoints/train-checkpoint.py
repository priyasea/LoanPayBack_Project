#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# In[13]:


def load_and_prepocess_data():
    #data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv('train.csv')
    df = df.drop('id' , axis = 1)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    numerical_columns = list(df.dtypes[df.dtypes != 'object'].index)
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
   
    if 'loan_paid_back' in numerical_columns:
        numerical_columns.remove('loan_paid_back')

    df = df.drop(['gender', 'marital_status'], axis = 1)
    df['education_level'] = df['education_level'].replace("master's", "masters")
    df['education_level'] = df['education_level'].replace("bachelor's", "bachelors")
    edu_category_order = [['high_school', 'other', 'bachelors', 'masters', 'phd']]
    edu_encoder = OrdinalEncoder(categories=edu_category_order)
    df['education_encoded'] = edu_encoder.fit_transform(df[['education_level']])
    df = df.drop(['education_level','loan_purpose'], axis=1)
    grade_category_order = [['f5', 'f4', 'f3', 'f2', 'f1', 'e5', 'e4','e3','e2','e1', 'd5', 'd4','d3','d2','d1', 'c5', 'c4','c3','c2','c1', 'b5', 'b4','b3','b2','b1',  'a5', 'a4','a3','a2','a1']]
    grade_encoder = OrdinalEncoder(categories=grade_category_order)
    df['grade_code'] = grade_encoder.fit_transform(df[['grade_subgrade']])
    df  = df.drop(['grade_subgrade'], axis = 1)

    df = df.reset_index(drop=True)
    y_train = df.loan_paid_back.values
    del df['loan_paid_back']
    
 
    return df, y_train




def train_model(df, y_train):
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    numerical_columns = list(df.dtypes[df.dtypes != 'object'].index)


    dv = DictVectorizer(sparse=False)

    train_dict = df[categorical_columns + numerical_columns].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    # Model
    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    xgb_params = {
     'eta': 0.2, 
     'max_depth': 6,
     'min_child_weight': 30,
    
     'objective': 'binary:logistic',
     'eval_metric': 'auc',

     'nthread': 8,
     'seed': 1,
     'verbosity': 1,
     }


    # Fit
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=False)

    
   
    

    return xgb_model   

def save_model(filename ,model):   
    with open('model.bin', 'wb') as f_out:
        pickle.dump(model, f_out)
    print ('model saved to model.bin')


df , y_train = load_and_prepocess_data()
model = train_model(df,y_train)
save_model('model.bin', model)


# In[ ]:





# In[ ]:




