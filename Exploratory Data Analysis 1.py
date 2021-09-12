#!/usr/bin/env python
# coding: utf-8

#CRISP - DM Method
#BARRY - BUSINESS UNDERSTANDING
#DROVE - DATA UNDERSTANDING
#DIRECTLY TO THE - DATA PREP
#MEDICAL - MODELLING
#EMERGENCY - EVALUATION
#DEPARTMENT - DEPLOYMENT



#1. Business Understanding
#Forecasting transaction
#likely regression
#data for 3 years
#advised data quality is ok


#Data Understanding
import pandas as pd
df = pd.read_csv('regression.csv')
df.head(5)



df.info()



for column in df.columns:
    print(column, len(df[column].unique()),df[column].unique())


# Visualize the data
from matplotlib import pyplot as plt
import seaborn as sns



plt.figure(figsize=(20,6))
sns.violinplot(x='Account Type', y = 'Amount', data = df).set_title('Account Type ViolinPlot')



plt.figure(figsize=(20,6))
sns.violinplot(x='Account Type', y = 'Amount', data = df[df['Account Type'] == 'Liability']).set_title('Liability ViolinPlot')


## Review Trends
df.head()


monthmap = {
    'Jan':1,
    'Feb':2,
    'Mar':3,
    'Apr':4,
    'May':5,
    'Jun':6,
    'Jul':7,
    'Aug':8,
    'Sep':9,
    'Oct':10,
    'Nov':11,
    'Dec':12,
}


df['Period'] = df['Month'].apply(lambda x: monthmap[x])


df[df['Month']=='Dec'].head()
df['Day'] = 1


df['Date'] =df['Year'].astype(str) + '-' + df['Period'].astype(str) + '-' + df['Day'].astype(str)
df['Date'] = pd.to_datetime(df['Date'])



plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Type'] == 'Revenue'])
plt.show()


plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Description'] == 'Product Sales'])
plt.show()


plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Description'] == 'Service Revenue'])
plt.show()

corrdict = {}
for key,row in df.join(pd.get_dummies(df['Account'])).iterrows():
    corrdict[key] = {int(row['Account']):row['Account']}
corrdf = pd.DataFrame.from_dict(corrdict).T.fillna(0)


plt.figure(figsize=(20,6))
sns.heatmap(corrdf.corr()).set_title('Account Correlation')
plt.show()

import numpy as np

for account in df['Account'].unique():
    plt.figure(figsize=(20,6))
    sns.lineplot(x='Date',y='Amount',estimator = np.median,hue='Account Description', data = df[df['Account'] == account]).set_title(' {} by month'.format(account))
    plt.show()


df = df[df['Account'] != 3000001]
df['Account'].unique()


df['Account'] = 'ACC ' + df['Account'].astype(str)


df.head()

df['Year'] = df['Year'].astype(str)

df.dtypes

df.drop(['Period','Day','Date'], axis=1, inplace = True)


len(df['Account'].unique())

len(df['Account Description'].unique())


df['AccountVal'] = df['Account'] + df['Account Description']

df.head()


df.drop(['Account Description','AccountVal'], axis = 1, inplace = True)


# one hot encoding
# unique column for each value
pd.get_dummies(df)
df = pd.get_dummies(df)
df

x = df.drop('Amount', axis=1)
y = df['Amount']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 1234)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


pipelines = {
    'rf':make_pipeline(RandomForestRegressor(random_state=1234)),
    'gb':make_pipeline(GradientBoostingRegressor(random_state=1234)),
    'ridge':make_pipeline(Ridge(random_state=1234)),
    'lasso':make_pipeline(Lasso(random_state=1234)),
    'enet':make_pipeline(ElasticNet(random_state=1234)),
}
RandomForestRegressor().get_params()


#hyperparameter grid
hypergrid = {
    'rf': {
        'randomforestregressor__min_samples_split':[2,4,6],
        'randomforestregressor__min_samples_leaf':[1,2,3]
    },
    'gb': {
        'gradientboostingregressor__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,0.99]
    },
    'ridge': {
        'ridge__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,0.99]
    },
    'lasso':{
        'lasso__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,0.99]
    },
    'enet': {
        'elasticnet__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,0.99]
    }
}


from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

fit_models = {}
for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10,n_jobs=-1)
    try:
        print('Starting training for {}'.format(algo))
        model.fit(X_train,y_train)
        fit_models[algo] = model
        print('{} has been successfully fit.'.format(algo))
    except NotFittedError as e:
        print(repr(e))
        

fit_models['rf'].predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error

for algo,model in fit_models.items():
    yhat= model.predict(X_test)
    print('{} scores - R2: {} MAE : {}'.format(algo,r2_score(y_test,yhat), mean_absolute_error(y_test,yhat)))

best_model = fit_models['rf']

