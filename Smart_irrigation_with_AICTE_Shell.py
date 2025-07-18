#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler
import joblib


# In[3]:


df = pd.read_csv("Smart_irrigation.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df = df.drop('Unnamed: 0', axis=1)
df.head()


# In[8]:


df.describe()


# In[9]:


X = df.iloc[:, 0:20]  


y = df.iloc[:, 20:]


# In[10]:


X.sample(10)


# In[11]:


y.sample(10)


# In[12]:


X.info()


# In[13]:


y.info()


# In[14]:


X


# In[15]:


X.shape, y.shape


# In[ ]:




