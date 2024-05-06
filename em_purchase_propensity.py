# Library importing

import pandas as pd
import numpy as np
from sklearn import set_config
set_config(transform_output = 'pandas')
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import joblib

# Data importing

em_propensity_dc = pd.read_pickle('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_dc')
em_propensity_emc = pd.read_pickle('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_emc')
em_propensity_pp = pd.read_pickle('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_pp')

# Functions

def validation_strategy_cl(dataframe, TARGET):
    X_train_, X_val, y_train_, y_val = train_test_split(dataframe.drop(TARGET, axis=1), dataframe[TARGET], test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size = 0.20, random_state = 52)
    return X_train, X_test, y_train, y_test, X_val, y_val

def propensity_prediction(new_df, target, model, precision_rate):
    cid = new_df.drop(target, axis = 1).reset_index()[['pk_cid']]
    X = new_df.drop(target, axis = 1).reset_index().drop('pk_cid', axis = 1)

    X_dmatrix = xgb.DMatrix(X)
    predictions = model.predict(X_dmatrix)

    prediction = {'Prediction': predictions}
    predictions = pd.DataFrame(prediction)

    final_df = cid.merge(predictions, left_index=True, right_index=True).groupby('pk_cid')['Prediction'].agg(
    Propensity = 'mean')
    final_df = final_df[final_df['Propensity']> precision_rate]
    return final_df

# Classification models

TARGET_DC = 'debit_card'
TARGET_PP = 'pension_plan'
TARGET_EMC = 'emc_account'

## Debit Card model
X_train, X_test, y_train, y_test, X_val, y_val = validation_strategy_cl(em_propensity_dc, TARGET_DC)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val = xgb.DMatrix(X_val, label=y_val)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'auc'
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
#num_boost_round = 400
num_boost_round = 65

xgb_model_dc = xgb.train(params=param, dtrain=xgb_train, num_boost_round=num_boost_round, evals=watchlist)

## Pension Plan model
X_train, X_test, y_train, y_test, X_val, y_val = validation_strategy_cl(em_propensity_pp, TARGET_PP)
undersampler = RandomUnderSampler(sampling_strategy=0.15, random_state=42)
X_train_r, y_train_r = undersampler.fit_resample(X_train, y_train)

xgb_train = xgb.DMatrix(X_train_r, label=y_train_r)
xgb_val = xgb.DMatrix(X_val, label=y_val)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'auc'
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
#num_boost_round = 400
num_boost_round = 65

xgb_model_pp = xgb.train(params=param, dtrain=xgb_train, num_boost_round=num_boost_round, evals=watchlist)

## Emc Account model
X_train, X_test, y_train, y_test, X_val, y_val = validation_strategy_cl(em_propensity_emc, TARGET_EMC)
X_train_r, y_train_r = undersampler.fit_resample(X_train, y_train)

xgb_train = xgb.DMatrix(X_train_r, label=y_train_r)
xgb_val = xgb.DMatrix(X_val, label=y_val)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'auc'
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
#num_boost_round = 400
num_boost_round = 65

xgb_model_emc = xgb.train(params=param, dtrain=xgb_train, num_boost_round=num_boost_round, evals=watchlist)



# Saving models

joblib.dump(xgb_model_dc, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_dc.pkl')
joblib.dump(xgb_model_pp, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_pp.pkl')
joblib.dump(xgb_model_emc, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_emc.pkl')

# Purchase propensity

## Loading data
xgb_model_dc = joblib.load('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_dc.pkl')
xgb_model_pp = joblib.load('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_pp.pkl')
xgb_model_emc = joblib.load('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/models/xgb_model_emc.pkl')

# Development ppm

dc_prop_cust = em_propensity_dc[em_propensity_dc[TARGET_DC]==0]
emc_prop_cust = em_propensity_emc[em_propensity_emc[TARGET_EMC]==0]
pp_prop_cust = em_propensity_pp[em_propensity_pp[TARGET_PP]==0]

## Debit card
dc_prop_pred = propensity_prediction(dc_prop_cust, TARGET_DC, xgb_model_dc, 0.8)

## Pension plan
pp_prop_pred = propensity_prediction(pp_prop_cust, TARGET_PP, xgb_model_pp, 0.8)

## Emc account
emc_prop_pred = propensity_prediction(emc_prop_cust, TARGET_EMC, xgb_model_emc, 0.8)

# Pickles

pd.to_pickle(dc_prop_pred, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/dc_prop_pred')
pd.to_pickle(pp_prop_pred, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/pp_prop_pred')
pd.to_pickle(emc_prop_pred, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/emc_prop_pred')


