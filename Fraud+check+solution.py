
# coding: utf-8

# In[287]:


#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[288]:


Fraud_data = pd.read_csv("Fraud.csv")
Fraud_data.columns


# In[289]:


Fraud_data.head()


# In[290]:


Fraud_data['Fraud_Var'] = Fraud_data['Taxable.Income'] <= 30000
Fraud_data.dtypes
Fraud_data['Fraud_Var'].value_counts()


# In[291]:


string_col = ['Undergrad','Marital.Status','Urban','Fraud_Var']


# In[292]:


#to convert string fields to numeric
from sklearn import preprocessing
for i in string_col:
    number = preprocessing.LabelEncoder()
    Fraud_data[i] = number.fit_transform(Fraud_data[i])
    


# In[293]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_data,test_size = 0.2)


# In[294]:


from sklearn.tree import  DecisionTreeClassifier


# In[295]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:6],train['Fraud_Var'])


# In[296]:


preds = model.predict(test.iloc[:,0:6])
pd.Series(preds).value_counts()
pd.crosstab(test['Fraud_Var'],preds)


# In[297]:


# Accuracy = train
np.mean(train['Fraud_Var'] == model.predict(train.iloc[:,0:6])) # 1


# In[299]:


# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1


# In[300]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,0:6],train['Fraud_Var'])
preds = rf.predict(test.iloc[:,0:6])


# In[301]:


# Accuracy = train
np.mean(train['Fraud_Var'] == rf.predict(train.iloc[:,0:6])) # 1


# In[302]:


# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1

