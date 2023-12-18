# %%
import pandas as pd
import numpy as np

from datetime import datetime

from autogluon.tabular import TabularDataset, TabularPredictor

# %%
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('public_dataset_without_gt.csv')

# %% [markdown]
# # Data Preprocessing

# %%
# Merge Dataset
df = pd.concat([df_train, df_test])
df.info()

# %% [markdown]
# ## Identify

# %%
label = 'default_12month' # target to predict

# %%
categoricals = ["APP_Area", "APP_Province", "gender", "marital_status",
                "type_of_residence",
                "c_business_type", 'c_position', 'c_occupation','c_employment_status',
                'c_salary_payment_methods', 'media', 'place_for_sending_information',
                'r_propose', 'r_generalcode3', 'apply']

numericals = ["number_of_children", "number_of_resident", "living_period_year",
             "c_number_of_employee", 'c_monthly_salary', 'c_number_of_working_year',
             'r_expected_credit_limit', 'r_allloan_case', 'r_allloan_amount', 'r_additional_income', 'r_spouse_income', 'FICO_Score']

date = ["date_of_birth", "APP_date"]

# %% [markdown]
# # Feature engineering

# %% [markdown]
# ## Transform

# %% [markdown]
# ### Date

# %%
df['APP_date'] = pd.to_datetime(df['APP_date'])
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
current_date = pd.to_datetime('today')

df['applicant_age'] = (current_date - df['date_of_birth']).dt.days
df['application_age'] = (current_date - df['APP_date']).dt.days

df = df.drop(['APP_date'], axis=1)
df = df.drop(['date_of_birth'], axis=1)

# %% [markdown]
# ### FICO Score
# https://www.linkedin.com/pulse/understanding-fico-score-comprehensive-guide-financemagnates/

# %%
df['fico_score_category'] = pd.cut(df['FICO_Score'], 
                                   bins=[0, 300, 579, 669, 739, 799, 850], 
                                   labels=['No Credit Info', 'Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])

fico_mapping = {'No Credit Info': 0, 'Poor': 1, 'Fair': 2, 'Good': 3, 'Very Good': 4, 'Exceptional': 5}
df['fico_score_category_numerical'] = df['fico_score_category'].map(fico_mapping)

df = df.drop(['FICO_Score'], axis=1)

# %% [markdown]
# ## New

# %% [markdown]
# - credit_utilization \
# ref. https://www.investopedia.com/terms/c/credit-utilization-rate.asp#toc-how-credit-utilization-impacts-borrowers

# %%
# numericals
df['employment_duration'] = df['c_number_of_working_year'] + df['c_number_of_working_month'] / 12
df['residence_duration'] = df['living_period_year'] + df['living_period_month'] / 12
df['income_per_person'] = df['c_monthly_salary'] / (df['number_of_resident'] + 1)
df['total_income'] = df['c_monthly_salary'] + df['r_additional_income'] + df['r_spouse_income']
df['credit_utilization'] = df['r_expected_credit_limit'] / df['total_income']
df['income_to_creditlimit'] = df['c_monthly_salary'] / df['r_expected_credit_limit']

# categoricals
df['applicant_age_group'] = pd.cut(df['applicant_age'], 
                                   bins=[0, 25, 45, 65, 100], 
                                   labels=['young', 'middle-aged', 'senior', 'old'])

categoricals += ['applicant_age_group']

# %%
df = df.drop(['c_monthly_salary','r_additional_income','r_spouse_income'], axis=1)

# %% [markdown]
# ## Drop

# %%
drop_cols = ['no', 'r_generalcode1', 'r_generalcode2',
             'date_of_birth_week', 'c_date_of_salary_payment',
             'postal_code', 'c_postal_code', 'APP_Shop Name',
             "c_number_of_working_month", "living_period_month","tel_category"]
df = df.drop(drop_cols, axis=1)

# %% [markdown]
# ## Encoding

# %%
# One-Hot Encoding
df = pd.get_dummies(df, columns=categoricals)

# %% [markdown]
# # Modeling

# %%
# Data Splitting
len_train = len(df_train)

df_train = df[:len_train]
df_test = df[len_train:]

# %%
time_limit = 60*15

predictor = TabularPredictor(label=label, eval_metric='roc_auc')
predictor.fit(df_train,
              time_limit=time_limit,
              presets='best_quality')

# %%
predictor.model_best

# %% [markdown]
# # Evaluation

# %%
predictor.leaderboard(df_train)

# %% [markdown]
# # Submission

# %%
df_test_nolabel = df_test.drop(label, axis=1)
public_dataset = pd.read_csv('public_dataset_without_gt.csv') #for no_column (index)

# %%
# ...!brk # break run

# %% [markdown]
# ## Single Export

# %%
# model_name = 'CatBoost_BAG_L1'

# y_pred = predictor.predict_proba(df_test_nolabel, model=model_name)

# result_df = pd.DataFrame({'no': public_dataset['no'], 'default_12month': y_pred[1]})

# output_name = 'output_' + str(time_limit//60)+'min_'+ model_name + '.csv'
# # Export
# result_df.to_csv(output_name, index=False, header=['no', 'default_12month'])

# %% [markdown]
# ## Multi Export

# %%
# models_name = ['WeightedEnsemble_L3', 'WeightedEnsemble_L2',
#                'LightGBM_BAG_L2', 'LightGBMXT_BAG_L2',
#                'CatBoost_BAG_L2', 'CatBoost_r177_BAG_L1', 'CatBoost_BAG_L1'
#                'XGBoost_BAG_L2', 'XGBoost_BAG_L1']
models_name = predictor.model_names()

for model_name in models_name:
    
    y_pred = predictor.predict_proba(df_test_nolabel, model=model_name)
    
    result_df = pd.DataFrame({'no': public_dataset['no'], 'default_12month': y_pred[1]})
    
    output_name = 'output_' + str(time_limit//60)+'min_'+ model_name + '.csv'
    # Export
    result_df.to_csv(output_name, index=False, header=['no', 'default_12month'])

# %% [markdown]
# # Feature Importance

# %%
# feaImp = predictor.feature_importance(df_test, 
#                              model='CatBoost_BAG_L1', 
#                              time_limit=60*1)


