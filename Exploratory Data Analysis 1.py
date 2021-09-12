#!/usr/bin/env python
# coding: utf-8

# In[3]:


#CRISP - DM Method
#BARRY - BUSINESS UNDERSTANDING
#DROVE - DATA UNDERSTANDING
#DIRECTLY TO THE - DATA PREP
#MEDICAL - MODELLING
#EMERGENCY - EVALUATION
#DEPARTMENT - DEPLOYMENT


# In[ ]:


#1. Business Understanding
#Forecasting transaction
#likely regression
#data for 3 years
#advised data quality is ok


# In[4]:


#Data Understanding
import pandas as pd
df = pd.read_csv('regression.csv')
df.head(5)


# In[5]:


df.info()


# In[6]:


for column in df.columns:
    print(column, len(df[column].unique()),df[column].unique())


# In[7]:


# Visualize the data
from matplotlib import pyplot as plt
import seaborn as sns


# In[11]:


plt.figure(figsize=(20,6))
sns.violinplot(x='Account Type', y = 'Amount', data = df).set_title('Account Type ViolinPlot')


# In[13]:


plt.figure(figsize=(20,6))
sns.violinplot(x='Account Type', y = 'Amount', data = df[df['Account Type'] == 'Liability']).set_title('Liability ViolinPlot')


# In[14]:


## Review Trends
df.head()


# In[15]:


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


# In[17]:


df['Period'] = df['Month'].apply(lambda x: monthmap[x])


# In[22]:


df[df['Month']=='Dec'].head()
df['Day'] = 1


# In[24]:


df['Date'] =df['Year'].astype(str) + '-' + df['Period'].astype(str) + '-' + df['Day'].astype(str)


# In[26]:


df['Date'] = pd.to_datetime(df['Date'])


# In[32]:


plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Type'] == 'Revenue'])
plt.show()


# In[34]:


plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Description'] == 'Product Sales'])
plt.show()


# In[35]:


plt.figure(figsize= (20,6))
sns.lineplot(x='Date',y='Amount',hue = 'Account Description', estimator = None, data = df[df['Account Description'] == 'Service Revenue'])
plt.show()


# In[37]:


corrdict = {}
for key,row in df.join(pd.get_dummies(df['Account'])).iterrows():
    corrdict[key] = {int(row['Account']):row['Account']}
    


# In[41]:


corrdf = pd.DataFrame.from_dict(corrdict).T.fillna(0)


# In[42]:


plt.figure(figsize=(20,6))
sns.heatmap(corrdf.corr()).set_title('Account Correlation')
plt.show()


# In[43]:


import numpy as np


# In[46]:


for account in df['Account'].unique():
    plt.figure(figsize=(20,6))
    sns.lineplot(x='Date',y='Amount',estimator = np.median,hue='Account Description', data = df[df['Account'] == account]).set_title(' {} by month'.format(account))
    plt.show()


# In[47]:


df = df[df['Account'] != 3000001]
df['Account'].unique()


# In[52]:


df['Account'] = 'ACC ' + df['Account'].astype(str)


# In[53]:


df.head()


# In[54]:


df['Year'] = df['Year'].astype(str)


# In[55]:


df.dtypes


# In[58]:


df.drop(['Period','Day','Date'], axis=1, inplace = True)


# In[59]:


len(df['Account'].unique())


# In[60]:


len(df['Account Description'].unique())


# In[61]:


df['AccountVal'] = df['Account'] + df['Account Description']


# In[62]:


df.head()


# In[63]:


df.drop(['Account Description','AccountVal'], axis = 1, inplace = True)


# In[64]:


# one hot encoding
# unique column for each value
pd.get_dummies(df)


# In[65]:


df = pd.get_dummies(df)


# In[66]:


df


# In[70]:


x = df.drop('Amount', axis=1)
y = df['Amount']


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 1234)


# In[74]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[75]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[77]:


pipelines = {
    'rf':make_pipeline(RandomForestRegressor(random_state=1234)),
    'gb':make_pipeline(GradientBoostingRegressor(random_state=1234)),
    'ridge':make_pipeline(Ridge(random_state=1234)),
    'lasso':make_pipeline(Lasso(random_state=1234)),
    'enet':make_pipeline(ElasticNet(random_state=1234)),
}


# In[88]:


RandomForestRegressor().get_params()


# In[95]:


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


# In[96]:


from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError


# In[97]:


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
        


# In[98]:


fit_models['rf'].predict(X_test)


# In[99]:


from sklearn.metrics import r2_score, mean_absolute_error


# In[101]:


for algo,model in fit_models.items():
    yhat= model.predict(X_test)
    print('{} scores - R2: {} MAE : {}'.format(algo,r2_score(y_test,yhat), mean_absolute_error(y_test,yhat)))


# In[102]:


best_model = fit_models['rf']


# In[ ]:





# In[ ]:





# In[ ]:




