# %% [markdown]
# # Import Data

# %%
import pandas as pd

# %%

train_dataset = pd.read_csv('train_dataset.csv')

# %%
train_dataset.info()

# %%
train_dataset.nunique()

# %% [markdown]
# # Data Preprocessing

# %%
train_dataset['APP_date'] = train_dataset['APP_date'].astype('datetime64')
train_dataset['date_of_birth'] = train_dataset['date_of_birth'].astype('datetime64')

train_dataset = train_dataset.drop(['r_generalcode1', 'r_generalcode2'], axis=1)

train_dataset.info()

# %% [markdown]
# ## Data Splitting
# - Data (sample / full)
#     - data : 90%
#         - train : 80% /90 = 72 /100
#         - test : 20% /90 = 18 /100
#     - data_unseen : 10%

# %%
dataset = train_dataset
# dataset = train_dataset.sample(5000)

# %%
data = dataset.sample(frac=0.9, random_state=123)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# %%
data.columns

# %% [markdown]
# # PyCaret

# %% [markdown]
# ## Setup

# %%
from pycaret.classification import *

# %%
# All features
categoricals = ['APP_Area', 'APP_Province', 'APP_Shop Name', 'gender', 'date_of_birth_week', 'marital_status', 'postal_code', 'tel_category', 'living_period_year', 'living_period_month', 'type_of_residence', 'c_postal_code', 'c_business_type', 'c_position', 'c_occupation', 'c_employment_status', 'c_salary_payment_methods', 'c_date_of_salary_payment', 'media', 'place_for_sending_information', 'r_propose', 'r_generalcode3', 'apply']
numericals = ['c_number_of_employee', 'c_monthly_salary', 'c_number_of_working_year', 'c_number_of_working_month', 'r_expected_credit_limit', 'r_allloan_case', 'r_allloan_amount', 'r_additional_income', 'r_spouse_income', 'FICO_Score', 'number_of_children', 'number_of_resident']
date = ['APP_date', 'date_of_birth']
ordinals = {'c_position':[6,5,4,3,2,1]}

# Top 10 features from catboost
keep = ['FICO_Score',
'postal_code',
'c_postal_code',
'c_number_of_working_year',
'date_of_birth_year',
'r_alloan_amount',
'living_period_year',
'c_monthly_salary',
'APP_Shop Name',
'number_of_children']


# %%
exp = ClassificationExperiment()
exp.setup(data,  target= 'default_12month', session_id= 7, train_size= 0.8, 
          preprocess= True, 
          categorical_features= categoricals,
          numeric_features= numericals,
          numeric_imputation= 'median',
          date_features= date,
          ordinal_features= ordinals,
        #   keep_features= keep
          remove_outliers= True,
          normalize= True,
          )

# %% [markdown]
# ## Model

# %%
exp.models()

# %%
exp.compare_models(sort='AUC')

# %%
model_list = ['catboost', 'gbc', 'lightgbm', 'ada', 'xgboost']
# model_list = ['catboost']

# %%
my_models = exp.compare_models(include=model_list, n_select=5, sort='AUC')
# my_models = exp.compare_models(include=model_list, n_select=1)

# %% [markdown]
# ## Tune Model

# %%
my_models

# %%
tuned_models = [exp.tune_model(model, optimize='AUC') for model in my_models]
# tuned_models = exp.tune_model(my_models, optimize='AUC')

# %%
tuned_models

# %% [markdown]
# ## Create Models

# %%
catboost = exp.create_model('catboost')
catboost_tuned = exp.tune_model(catboost)

# %%
lightgbm = exp.create_model('lightgbm')
lightgbm_tuned = exp.tune_model(lightgbm)

# %%
xgboost = exp.create_model('xgboost')
xgboost_tuned = exp.tune_model(xgboost)

# %% [markdown]
# ### Logistic Regression

# %%
lr = exp.create_model('lr')
lr_tuned = exp.tune_model(lr)

# %% [markdown]
# ### Naive Bayes

# %%
nb = exp.create_model('nb')
nb_tuned = exp.tune_model(nb)

# %% [markdown]
# # Plot Winning Model

# %%
exp.plot_model(catboost_tuned, plot = 'parameter')

# %%
exp.plot_model(catboost_tuned, plot = 'auc')

# %%
exp.plot_model(catboost_tuned, plot = 'class_report')

# %%
exp.plot_model(catboost_tuned, plot = 'boundary')

# %%
exp.plot_model(catboost_tuned, plot = 'learning')

# %%
exp.plot_model(catboost_tuned, plot = 'feature')

# %%
exp.plot_model(catboost_tuned, plot = 'feature_all')

# %% [markdown]
# # Predict Test Model

# %%
holdout_pred = exp.predict_model(catboost_tuned)

# %%
holdout_pred

# %% [markdown]
# # Finalize Model

# %%
# final_model = exp.finalize_model(catboost_tuned)
final_model = catboost_tuned
final_model

# %% [markdown]
# ## Predict Unseen Data

# %%
unseen_predictions = exp.predict_model(final_model, data=data_unseen)
unseen_predictions

# %% [markdown]
# # Predict Output

# %%
public_dataset = pd.read_csv('public_dataset_without_gt.csv')

# %%
public_dataset['APP_date'] = public_dataset['APP_date'].astype('datetime64')
public_dataset['date_of_birth'] = public_dataset['date_of_birth'].astype('datetime64')

public_dataset = public_dataset.drop(['r_generalcode1', 'r_generalcode2'], axis=1)

public_dataset.info()

# %%
pred = exp.predict_model(final_model, data = public_dataset, raw_score = True)
pred

# %%
pred['prediction_score_1']

# %% [markdown]
# ## Export to CSV

# %%
result_df = pd.DataFrame({'no': public_dataset['no'], 'default_12month': pred['prediction_score_1']})

result_df = result_df.sort_values(by='no', ascending=True)

result_df.to_csv('output_pyc.csv', index=False, header=['no', 'default_12month'])


