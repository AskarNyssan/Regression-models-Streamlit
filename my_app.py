# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 01:02:15 2021

@author: NysanAskar
"""
import streamlit as st 
# Suppress warning
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Import dependent packages
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

st.title('Regression models')

st.write("""
## Explore different regression methods
""")

st.write("""
 We will use "Seoul Bike" dataset, it can can be downloaded:    
""")
link = '[Seoul Bike](https://www.kaggle.com/c/seoul-bike-rental-prediction/data)'
st.markdown(link, unsafe_allow_html=True) 

st.write("""
The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information. The target variable is a bike count required at each hour.
""")

regression_name = st.sidebar.selectbox(
    'Select regression model',
    ('Ridge regression', 'Lasso regression', 'SVR', 'k-nn', 'Random Forest', 
                 'AdaBoost regression', 'Gradient tree boosting')
)

# Download data 
X_matrix = pd.read_csv('Dataset/SeoulBikeData.csv', sep=',', encoding= 'unicode_escape').drop(['Date','Rented Bike Count'], axis = 'columns')
Y_variable = pd.read_csv('Dataset/SeoulBikeData.csv', sep=',', encoding= 'unicode_escape', usecols=['Rented Bike Count'])

# function to convert categoricl variabes into one-hot-coding
def encode_and_bind(data, feature):
    dummies = pd.get_dummies(data[[feature]])
    res = pd.concat([data, dummies], axis=1)
    return(res)

X_data = X_matrix
categ_columns = ['Seasons', 'Holiday', 'Functioning Day'] 
for i in range(len(categ_columns)): 
    X_data = X_data.merge(encode_and_bind(X_matrix, categ_columns[i]), how='inner')

X_data.drop(categ_columns, axis=1, inplace=True)


def add_parameter_ui(clf_name):
    params = dict()  
    if clf_name == 'Ridge regression':
        alpha = st.sidebar.slider('alpha', 0.01, 10.0)
        params['alpha'] = alpha
    elif clf_name == 'Lasso regression':
        alpha = st.sidebar.slider('alpha', 0.01, 10.0)
        params['alpha'] = alpha
    elif clf_name == 'SVR':       
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C             
    elif clf_name == 'k-nn':
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 10)
        params['n_neighbors'] = n_neighbors
    elif clf_name == 'Random Forest':     
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        params['max_depth'] = max_depth
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10)
        params['min_samples_leaf'] = min_samples_leaf
        n_estimators = st.sidebar.slider('n_estimators', 10, 100)
        params['n_estimators'] = n_estimators
    elif clf_name == 'AdaBoost regression':     
        n_estimators = st.sidebar.slider('n_estimators', 10, 100)
        params['n_estimators'] = n_estimators
        learning_rate = st.sidebar.slider('learning_rate', 1, 10)
        params['learning_rate'] = learning_rate
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 10, 100)
        params['n_estimators'] = n_estimators
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10)
        params['min_samples_leaf'] = min_samples_leaf
        learning_rate = st.sidebar.slider('learning_rate', 1, 10)
        params['learning_rate'] = learning_rate
    return params

params = add_parameter_ui(regression_name)

def get_regression(clf_name, params):
    clf = None
    if clf_name == 'Ridge regression':
        clf = Ridge(alpha=params['alpha'])
    elif clf_name == 'Lasso regression':
        clf = Lasso(alpha=params['alpha'])    
    elif clf_name == 'SVR':
        clf = svm.SVR(C=params['C'])        
    elif clf_name == 'k-nn':
        clf = KNeighborsRegressor(n_neighbors=params['n_neighbors'])       
    elif clf_name == 'Random Forest':
        clf = RandomForestRegressor(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'],  min_samples_leaf =params['min_samples_leaf'])       
    elif clf_name == 'AdaBoost regression':
        clf = AdaBoostRegressor(n_estimators=params['n_estimators'], 
              learning_rate =params['learning_rate'])       
    else:
        clf = GradientBoostingRegressor(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], min_samples_leaf =params['min_samples_leaf'], learning_rate =params['learning_rate'])
    return clf


clf = get_regression(regression_name, params)


#### REGRESSION ####

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_variable, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
R_score = r2_score(y_test, y_pred)

st.write(f'Regression model = {regression_name}')
st.write(f'R_score =', R_score)


def predict(y_pred):
    if len(y_pred.shape) == 2:
        y_pred = y_pred.flatten()
        return y_pred
    else:
        return y_pred
    
    
st.write("""
## True value vs Predicted value
""")

y_test, y_pred = pd.Series(y_test['Rented Bike Count'], name="y_test"), pd.Series(predict(y_pred), name="y_pred")
fig, ax = plt.subplots()
ax = sns.regplot(x=y_test, y=y_pred, ci=None, color="b")
st.pyplot(fig)

st.write("""
## Residuals vs Predicted value
""")

y_residual = pd.Series(np.array(y_test) - np.array(predict(y_pred)), name="y_residual")
fig, ax = plt.subplots()
ax = sns.regplot(x=y_test, y=y_residual, ci=None, color="b")
st.pyplot(fig)









