#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


data = pd.read_csv('D:\Data\Flood\Worksheet.csv')


# In[6]:


data


# In[7]:


data.head(10)


# In[8]:


import csv


# In[9]:


import string
# Open the input CSV file
with open('D:\Data\Flood\Worksheet.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Create a list to store the modified rows
    modified_rows = []
    
    for row in reader:
        modified_row = []
        
        for item in row:
            # Remove punctuation from each item in the row
            item_without_punctuation = ''.join(char for char in item if char not in string.punctuation)
            modified_row.append(item_without_punctuation)
        
        modified_rows.append(modified_row)
        
        # Open a new CSV file for writing
        with open('D:\Data\Flood\Worksheet_New.csv', 'w', newline='') as file:
            writer = csv.writer(file)
        # Write the modified rows to the new CSV file
        # writer.writerows(modified_rows)       
            writer.writerows(modified_rows)


# In[10]:


data1 = pd.read_csv('D:\Data\Flood\Worksheet_New.csv')


# In[11]:


data1


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


plt.figure(figsize=(8, 6))
sns.histplot(data1['Age'], bins=10, kde=True)
plt.xlabel('Age')
plt.ylabel('Occupation')
plt.title('Distribution of Age')
plt.show()


# In[14]:


# Cropping Pattern Variables
# Crop diversity analysis
plt.figure(figsize=(8, 6))
data1['Main Crops Cultivated'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Crop Type Distribution')
plt.show()


# In[15]:


# Crop yield analysis by crop type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Main Crops Cultivated', y='Approximate Annual Crop Yield in Kg', data=data)
plt.xlabel('Main Crops Cultivated')
plt.ylabel('Approximate Annual Crop Yield in Kg')
plt.title('Crop Yield by Crop Type')
plt.xticks(rotation=45)
plt.show()


# In[17]:


# Crop yield analysis by crop type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Farmland Size', y='Approximate Annual Crop Yield in Kg', data=data)
plt.xlabel('Main Crops Cultivated')
plt.ylabel('Approximate Annual Crop Yield in Kg')
plt.title('Crop Yield by Crop Type')
plt.xticks(rotation=45)
plt.show()


# In[19]:


# Flooding Impact Variables
# Property damage analysis by flood severity
plt.figure(figsize=(8, 6))
sns.barplot(x='Flood Severity Rating', y='Damage to Property', data=data1)
plt.xlabel('Flood Severity Rating')
plt.ylabel('Damage to Property')
plt.title('Property Damage by Flood Severity')
plt.show()


# In[21]:


# Perception of Flood Vulnerability Variables
# Boxplot of perception scores by age group
plt.figure(figsize=(8, 6))
sns.boxplot(x='Age', y='Perception of Flood Vulnerability', data=data1)
plt.xlabel('Age')
plt.ylabel('Perception of Flood Vulnerability')
plt.title('Perception of Flood Vulnerability by Age Group')
plt.show()


# In[22]:


# Adaptation Variables
# Adoption rates analysis by flood 
plt.figure(figsize=(10, 6))
sns.barplot(x='Education Level', y='Adaptation Strategies', hue='Gender', data=data1)
plt.xlabel('Education Level')
plt.ylabel('Adaptation Strategies')
plt.title('Adoption Rate by Education Level')
plt.legend(title='Gender')
plt.show()


# In[23]:


# Correlation analysis of adaptation variables
adaptation_variables = ['Main Crops Cultivated', 'Damage to Property', 'Flood Severity Rating']  # Replace with your adaptation variables
adaptation_data = data1[adaptation_variables]

plt.figure(figsize=(8, 6))
sns.heatmap(adaptation_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Adaptation Variables')
plt.show()


# In[ ]:




