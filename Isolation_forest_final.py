#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[4]:


train_df = pd.read_csv('G:/train_features.csv')


# In[5]:


train_df = train_df[['building_id', 'meter_reading', 'square_feet',
       'year_built', 'floor_count','air_temperature', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure','is_holiday','anomaly']]


# In[6]:


train_df = train_df.dropna()


# In[7]:


train_df['anomaly'] = train_df['anomaly'].astype(int)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(train_df.drop('anomaly', axis=1), train_df['anomaly'], test_size=0.3, random_state=42)


# In[9]:

# normalization
outliers_fraction = float(.01)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train,y_train)
X_test = scaler.fit_transform(X_test,y_test)


# In[10]:


model = IsolationForest(contamination=outliers_fraction)
model.fit(X_train)


# In[11]:


y_pred = model.predict(X_test)


# In[12]:


y_pred[y_pred == 1] = 0  # inliers = 0
y_pred[y_pred == -1] = 1  # outliers = 1


# In[13]:


accuracy = accuracy_score(y_test, y_pred )
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted' )
f1 = f1_score(y_test, y_pred,average='weighted' )
conf_matrix = confusion_matrix(y_test, y_pred)


# In[14]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
print(f"Confusion matrix: \n{conf_matrix}")


# In[18]:


confusion_matrix = [[TN, FP], [FN, TP]]

FPR = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
