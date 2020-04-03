
# coding: utf-8

# In[1]:


#Company_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


computer_sale = pd.read_csv("Company_Data.csv")


# In[3]:


computer_sale.columns
computer_sale.dtypes


# In[4]:


#lets considersales are high if > 7.5, creating the variable to identify sales as high or low
computer_sale['Sale_Var'] = computer_sale['Sales'] >= 7.5
computer_sale.drop("Sales",axis=1,inplace=True)
computer_sale_str_columns = ['ShelveLoc','Urban','US','Sale_Var']


# In[5]:


colnames = list(computer_sale.columns)
predictors = colnames[:4]
target = colnames[4]


# In[6]:


# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
computer_sale['is_train'] = np.random.uniform(0, 1, len(computer_sale))<= 0.75
computer_sale['is_train']
train,test = computer_sale[computer_sale['is_train'] == True],computer_sale[computer_sale['is_train']==False]


# In[7]:


#convert the string columns from numeric
from sklearn import preprocessing
for i in computer_sale_str_columns:
    number = preprocessing.LabelEncoder()
    computer_sale[i] = number.fit_transform(computer_sale[i])
    


# In[8]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(computer_sale,test_size = 0.2)


# In[9]:


from sklearn.tree import  DecisionTreeClassifier


# In[10]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,1:11],train['Sale_Var'])


# In[11]:


np.mean(train['Sale_Var'] == model.predict(train.iloc[:,1:11])) #1

np.mean(test['Sale_Var'] == model.predict(test.iloc[:,1:11])) #0.81


# In[12]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,1:11],train['Sale_Var'])

np.mean(train['Sale_Var'] == rf.predict(train.iloc[:,1:11])) #0.99

np.mean(test['Sale_Var'] == rf.predict(test.iloc[:,1:11])) #0.95

