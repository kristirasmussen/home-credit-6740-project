# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:39:57 2020

@author: Kristi
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import time

start_time = time.time()

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

#####################

# Installment data feature engineering
modified_installment = install_pymts.copy()
modified_installment['PAYMENT_PERCENTAGE'] = modified_installment['AMT_PAYMENT'] / modified_installment['AMT_INSTALMENT']
modified_installment['PAYMENT_DIFFERENCE'] = modified_installment['AMT_INSTALMENT'] - modified_installment['AMT_PAYMENT']

#It's relative to the application date - and you want one column for tardy and one for eager.
# Pay attention to order in the first line below - entry payment - days installment (because its relative to application date)
# Second part - essentially zero-ing out non-tardy (vice versa for section two)
modified_installment['DAYS_PAST_DUE'] = modified_installment['DAYS_ENTRY_PAYMENT'] - modified_installment['DAYS_INSTALMENT']
modified_installment['DAYS_PAST_DUE'] = modified_installment['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)

modified_installment['DAYS_BEFORE_DUE'] = modified_installment['DAYS_INSTALMENT'] - modified_installment['DAYS_ENTRY_PAYMENT']
modified_installment['DAYS_BEFORE_DUE'] = modified_installment['DAYS_BEFORE_DUE'].apply(lambda x: x if x > 0 else 0)

#Found a cool way to do aggregations at one go 
aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_PAST_DUE': ['max', 'mean', 'sum'],
        'DAYS_BEFORE_DUE': ['max', 'mean', 'sum'],
        'PAYMENT_PERCENTAGE': ['max', 'mean', 'var'],
        'PAYMENT_DIFFERENCE': ['max', 'mean', 'var']
    }

modified_installment = modified_installment.groupby('SK_ID_CURR').agg(aggregations)
modified_installment.columns = pd.Index(['INSTALLMENT_' + e[0] + "_" + e[1].upper() for e in modified_installment.columns.tolist()])

modified_installment = modified_installment.reset_index()

#####################3

## Point of Sale and Cash Loan Data
pos_cash_bal['DPD_FLAG'] = np.where(pos_cash_bal['SK_DPD_DEF'] > 0.0, 1, 0)

# Calculate Past Due Metrics for previous POS and Cash loans
pcb_prev_acct_cnt = pos_cash_bal[['SK_ID_CURR','SK_ID_PREV']].drop_duplicates().groupby('SK_ID_CURR', as_index=False).agg('count').rename(columns={'SK_ID_PREV':'PREVIOUS_ACCOUNTS'})
cash_bal_pastdue_cnt = pos_cash_bal[['SK_ID_CURR','DPD_FLAG']].groupby('SK_ID_CURR', as_index=False).agg('sum').rename(columns={'DPD_FLAG':'TIMES_PAST_DUE'})
cash_bal_pastdue_total_days = pos_cash_bal[['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR', as_index=False).agg('sum').rename(columns={'SK_DPD_DEF':'TOTAL_DPD'})
cash_bal_pastdue_max = pos_cash_bal[['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR', as_index=False).agg('max').rename(columns={'SK_DPD_DEF':'MAX_DPD'})

# Merge dataframes
prev_pos_cash_agg = reduce(lambda  left,right: pd.merge(left,right,on=['SK_ID_CURR'],how='left'), [pcb_prev_acct_cnt,cash_bal_pastdue_cnt,cash_bal_pastdue_total_days,cash_bal_pastdue_max])
# Remove temporary dataframes
del pcb_prev_acct_cnt, cash_bal_pastdue_cnt, cash_bal_pastdue_total_days, cash_bal_pastdue_max

## Previous Credit Account Data
credit_card_bal['DPD_FLAG'] = np.where(credit_card_bal['SK_DPD_DEF'] > 0.0, 1, 0)

# Calculate Past Due Metrics for previous credit accounts
credit_prev_acct_cnt = credit_card_bal[['SK_ID_CURR','SK_ID_PREV']].drop_duplicates().groupby('SK_ID_CURR', as_index=False).agg('count').rename(columns={'SK_ID_PREV':'PREVIOUS_ACCOUNTS_CREDIT'})
credit_bal_pastdue_cnt = credit_card_bal[['SK_ID_CURR','DPD_FLAG']].groupby('SK_ID_CURR', as_index=False).agg('sum').rename(columns={'DPD_FLAG':'TIMES_PAST_DUE_CREDIT'})
credit_bal_pastdue_total_days = credit_card_bal[['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR', as_index=False).agg('sum').rename(columns={'SK_DPD_DEF':'TOTAL_DPD_CREDIT'})
credit_bal_pastdue_max = credit_card_bal[['SK_ID_CURR','SK_DPD_DEF']].groupby('SK_ID_CURR', as_index=False).agg('max').rename(columns={'SK_DPD_DEF':'MAX_DPD_CREDIT'})
# Calculate avg total amount receivable among previous credit accounts over all months
credit_bal_avg_receivable = credit_card_bal[['SK_ID_CURR','AMT_TOTAL_RECEIVABLE']].groupby('SK_ID_CURR', as_index=False).agg('mean').rename(columns={'AMT_TOTAL_RECEIVABLE':'AVG_AMT_TOT_RECEIVABLE'})

# Merge dataframes
prev_credit_agg = reduce(lambda  left,right: pd.merge(left,right,on=['SK_ID_CURR'],how='left'), [credit_prev_acct_cnt,credit_bal_pastdue_cnt,credit_bal_pastdue_total_days,credit_bal_pastdue_max,credit_bal_avg_receivable])
# remove temporary dataframes
del credit_prev_acct_cnt, credit_bal_pastdue_cnt, credit_bal_pastdue_total_days, credit_bal_pastdue_max, credit_bal_avg_receivable

#Create interaction between Avg Total Receivable and Times Past Due
prev_credit_agg['AVG_TOT_REC_PAST_DUE_INTERR'] = prev_credit_agg['AVG_AMT_TOT_RECEIVABLE'] * prev_credit_agg['TIMES_PAST_DUE_CREDIT']



# =============================================================================
# Merge Cleaned Data, Make Categorical Dummy Variables, and Split to Train/Test
# =============================================================================

app_train_expanded = reduce(lambda left,right: pd.merge(left,right,on=['SK_ID_CURR'],how='left'), [app_train,bureau_agg_final,modified_installment,prev_pos_cash_agg,prev_credit_agg])

app_train_expanded = app_train_expanded.replace(np.inf,np.nan)

# split into numeric/categorical categories
app_train_expanded_categorical = app_train_expanded.loc[:,app_train_expanded.dtypes==np.object]

# create dummy variables for all categorical columns
categorical_cols = app_train_expanded_categorical.columns
app_train_dummies = pd.get_dummies(app_train_expanded, columns=categorical_cols,dummy_na='nan_as_category')

# remove values where 1,000 years employed
app_train_dummies['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# impute missing values
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(strategy='median')
imp_median.fit(app_train_dummies)
imp_median_data = imp_median.transform(app_train_dummies)

app_train_dummies = pd.DataFrame(imp_median_data, columns=app_train_dummies.columns)


# split into data and label sets
data = app_train_dummies.drop(columns=["TARGET","SK_ID_CURR"])
label = app_train_dummies[["TARGET"]]

scaler = StandardScaler().fit(data)
data_scaled = scaler.transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

random_seed = 42

# Scaled - randomly split data to 80% training, 20% testing
X_train,X_test,y_train,y_test=train_test_split(data_scaled,label,test_size=.2,random_state=random_seed)

# downsample train data - scaled
X_train_full = X_train.merge(y_train, how='inner', left_index=True, right_index=True)
X_train_full['TARGET'].value_counts()
X_train_zero = X_train_full[X_train_full["TARGET"] == 0]
X_train_one = X_train_full[X_train_full["TARGET"] == 1]
X_train_sample = X_train_zero.sample(n=19876)
X_train_final = pd.concat([X_train_sample,X_train_one], axis=0)

X_train = X_train_final.drop(columns=["TARGET"])
y_train = X_train_final[["TARGET"]]


# =============================================================================
# Perform Variable Selection/Feature Reduction
# =============================================================================

# run LASSO regression to find important columns
clf = LassoCV().fit(X_train, y_train)
importance = np.abs(clf.coef_)

# extract important columns
importance_df = pd.DataFrame(importance).reset_index()
importance_df.columns = ['index','importance']
skinny_cols = list(importance_df[importance_df['importance'] >= 0.02]['index'])

# skinny down the numeric columns that are important
skinny_data = X_train.iloc[:,skinny_cols]
skinny_data_cols = skinny_data.columns

# list of column names to include for our modeling
keep_cols = list(skinny_data_cols)
good_data = X_train[keep_cols]

# =============================================================================
# Make Models
# =============================================================================

results = []

def get_metrics(model_name, y_test, ypred_test):
    accuracy = round(accuracy_score(y_test, ypred_test),4)
    mae = metrics.mean_absolute_error(y_test, ypred_test)
    mse = metrics.mean_squared_error(y_test, ypred_test)
    rmse = np.sqrt(mse)
    return {'Model':model_name,'Accuracy':accuracy,'MAE':mae,'MSE':mse,'RMSE':rmse}


######## CART ########
cart = DecisionTreeClassifier(criterion='gini', max_depth=20,random_state=random_seed).fit(X_train,y_train)
cart_ypred_test = cart.predict(X_test)
results.append(get_metrics('CART', y_test, cart_ypred_test))
print('CART:\n', confusion_matrix(y_test, cart_ypred_test))


######## Random Forest ########
rf = RandomForestClassifier(n_estimators=12,random_state=random_seed).fit(X_train, y_train)
rf_ypred_test = rf.predict(X_test)
results.append(get_metrics('Random Forest', y_test, rf_ypred_test))
print('RF:\n', confusion_matrix(y_test, rf_ypred_test))


######## Naive Bayes ########
gnb = GaussianNB(var_smoothing=.001).fit(X_train, y_train)
nb_ypred_test = gnb.predict(X_test)
results.append(get_metrics('Naive Bayes', y_test, nb_ypred_test))
print('NB:\n', confusion_matrix(y_test, nb_ypred_test))


######## KNN ########
knn = KNeighborsClassifier(n_neighbors=3,n_jobs=-1).fit(X_train, y_train)
knn_ypred_test = knn.predict(X_test)
results.append(get_metrics('KNN', y_test, knn_ypred_test))
print('KNN:\n', confusion_matrix(y_test, knn_ypred_test))


######## Logistic Regression ########
logReg = LogisticRegression(max_iter=200, solver='liblinear').fit(X_train, y_train)
logReg_ypred_test = logReg.predict(X_test)
results.append(get_metrics('Logistic Regression', y_test, logReg_ypred_test))
print('Log Reg:\n', confusion_matrix(y_test, logReg_ypred_test))


######## Linear SVM ########
svm = SVC(max_iter=200, kernel='linear').fit(X_train, y_train)
svm_ypred_test = svm.predict(X_test)
results.append(get_metrics('Linear SVM', y_test, svm_ypred_test))
print('Linear SVM:\n', confusion_matrix(y_test, svm_ypred_test))


results_df = pd.DataFrame(results)

# code runtime
end_time = time.time()
print(round((end_time - start_time)/60,4), 'minutes')



