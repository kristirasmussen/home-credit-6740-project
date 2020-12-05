# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:39:57 2020

@author: Kristi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# =============================================================================
# Read in datasets
# =============================================================================

app_train = pd.read_csv("application_train.csv")
bureau = pd.read_csv("bureau.csv")
bureau_bal = pd.read_csv("bureau_balance.csv")
credit_card_bal = pd.read_csv("credit_card_balance.csv")
install_pymts =  pd.read_csv("installments_payments.csv")
pos_cash_bal = pd.read_csv("POS_CASH_balance.csv")
prev_app = pd.read_csv("previous_application.csv")

# =============================================================================
# Aggregate and Clean Data
# =============================================================================

# Aggregate Bureau Values
bureau_cnt = bureau.groupby('SK_ID_CURR', as_index = False).count()[['SK_ID_CURR','SK_ID_BUREAU']].rename(columns = {'SK_ID_BUREAU':'SK_ID_BUREAU_CNT'})



bureau_mean = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg('mean').reset_index()[['SK_ID_CURR','DAYS_CREDIT',
                                                                                                                          'CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE',
                                                                                                                          'AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT',
                                                                                                                          'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']].rename(columns = {'DAYS_CREDIT':'DAYS_CREDIT_mean',
                                                                                                                                          'CREDIT_DAY_OVERDUE':'CREDIT_DAY_OVERDUE_mean',
                                                                                                                                          'AMT_CREDIT_MAX_OVERDUE':'AMT_CREDIT_MAX_OVERDUE_mean',
                                                                                                                                          'AMT_CREDIT_SUM':'AMT_CREDIT_SUM_mean',
                                                                                                                                          'AMT_CREDIT_SUM_DEBT':'AMT_CREDIT_SUM_DEBT_mean',
                                                                                                                                          'AMT_CREDIT_SUM_LIMIT':'AMT_CREDIT_SUM_LIMIT_mean',
                                                                                                                                          'AMT_CREDIT_SUM_OVERDUE':'AMT_CREDIT_SUM_OVERDUE_mean',
                                                                                                                                          'AMT_ANNUITY':'AMT_ANNUITY_mean'})

bureau_sum = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg('sum').reset_index()[['SK_ID_CURR','AMT_CREDIT_SUM',
                                                                                                                          'AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT',
                                                                                                                          'AMT_CREDIT_SUM_OVERDUE']].rename(columns = {'AMT_CREDIT_SUM':'AMT_CREDIT_SUM_sum',
                                                                                                                                          'AMT_CREDIT_SUM_DEBT':'AMT_CREDIT_SUM_DEBT_sum',
                                                                                                                                          'AMT_CREDIT_SUM_LIMIT':'AMT_CREDIT_SUM_LIMIT_sum',
                                                                                                                                          'AMT_CREDIT_SUM_OVERDUE':'AMT_CREDIT_SUM_OVERDUE_sum'})

bureau_max = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg('max').reset_index()[['SK_ID_CURR','DAYS_CREDIT',
                                                                                                                          'CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE']].rename(columns = {'DAYS_CREDIT':'DAYS_CREDIT_max',
                                                                                                                                          'CREDIT_DAY_OVERDUE':'CREDIT_DAY_OVERDUE_max',
                                                                                                                                          'AMT_CREDIT_MAX_OVERDUE':'AMT_CREDIT_MAX_OVERDUE_max'})

bureau_min = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg('min').reset_index()[['SK_ID_CURR','DAYS_CREDIT',
                                                                                                                          'CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE']].rename(columns = {'DAYS_CREDIT':'DAYS_CREDIT_min',
                                                                                                                                          'CREDIT_DAY_OVERDUE':'CREDIT_DAY_OVERDUE_min',
                                                                                                                                          'AMT_CREDIT_MAX_OVERDUE':'AMT_CREDIT_MAX_OVERDUE_min'})

# pivot and count loan statuses
bureau_loan_statuses = bureau[['SK_ID_CURR','CREDIT_ACTIVE']].pivot_table(index='SK_ID_CURR',columns='CREDIT_ACTIVE',aggfunc=len,fill_value=0)

# pivot and count credit_type values
bureau_credit_types = bureau[['SK_ID_CURR','CREDIT_TYPE']].pivot_table(index='SK_ID_CURR',columns='CREDIT_TYPE',aggfunc=len,fill_value=0)
                                                                                                                                                                                            
# merge together aggregated dataframes
from functools import reduce
bureau_agg = reduce(lambda  left,right: pd.merge(left,right,on=['SK_ID_CURR'],how='left'), [bureau_cnt,bureau_mean,bureau_sum,bureau_max,bureau_min,bureau_loan_statuses,bureau_credit_types])


# Aggregate Bureau Balance Values
bureau_bal_statuses = bureau_bal[['SK_ID_BUREAU','STATUS']].pivot_table(index='SK_ID_BUREAU',columns='STATUS',aggfunc=len,fill_value=0).reset_index()
bureau_bal_cnt = bureau_bal.groupby('SK_ID_BUREAU', as_index = False).count()[['SK_ID_BUREAU','MONTHS_BALANCE']].rename(columns = {'MONTHS_BALANCE':'MONTHS_cnt'})

# merge subsets
bureau_bal_agg = pd.merge(bureau_bal_statuses,bureau_bal_cnt,how='inner',on='SK_ID_BUREAU')

# scale statuses to get percent of months account is in each status
bureau_bal_agg.iloc[:,1:-1] = bureau_bal_agg.iloc[:,1:-1].div(bureau_bal_agg.MONTHS_cnt, axis=0)

# drop MONTHS_cnt column
bureau_bal_agg = bureau_bal_agg.drop(columns=['MONTHS_cnt'])

# merge bureau_bal_agg with bureau_agg and aggregate again so we have 1 row per SK_ID_CURR
bureau_bal_agg_to_SK_ID_CURR = pd.merge(bureau[['SK_ID_CURR','SK_ID_BUREAU']],bureau_bal_agg, on='SK_ID_BUREAU', how='inner')
bureau_agg_status = bureau_bal_agg_to_SK_ID_CURR.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg('mean')

# merge together final bureau data
bureau_agg_final = pd.merge(bureau_agg,bureau_agg_status,how='left',on='SK_ID_CURR')

# =============================================================================
# Merge Cleaned Data
# =============================================================================

app_train_expanded = reduce(lambda left,right: pd.merge(left,right,on=['SK_ID_CURR'],how='left'), [app_train,bureau_agg_final])


# =============================================================================
# Perform Variable Selection/Feature Reduction
# =============================================================================


# =============================================================================
# Split into Train/Test Sets, Model, and Report Accuracy
# =============================================================================

data = app_train_expanded.loc[:, app_train.columns != "TARGET"] 
label = app_train_expanded[["TARGET"]]

random_seed = 42

# randomly split data to 80% training, 20% testing
X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=.2,random_state=random_seed)



