#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing pandas

import pandas as pd


# # Cleaning Data of Regression Dataset 1

# In[9]:


#loading the first data set


df1 = pd.read_csv(r'C:\Users\HP\Downloads\Used Cars Data-20240721T072109Z-001\Used Cars Data\pakwheels_used_cars.csv')
df1


# In[11]:


#displaying its first 15 rows

df1.head(15)


# In[13]:


#next first i decided to find out how many missing values are there in the data set


missing_values = df1.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])


# In[14]:


#now since there are a large amount of missing values in the data, i decided to simply drop the rows having missing data, instead of ruining the data set by mean or median
df1_new = df1.dropna()

#displaying my data set after dropping missing values
df1_new


# In[21]:


#first i made a list of all of my categorical columns, by comparing with the csv file
categorical_columns = ['assembly', 'body', 'ad_city', 'color', 'fuel_type', 'make', 'model', 'registered', 'transmission']

# the i applied one-hot encoding to the categorical columns
df1_one_hot = pd.get_dummies(df1, columns=categorical_columns)


print("Dataset after applying One-Hot Encoding :")
df1_one_hot.head(14)


# In[22]:


import pandas as pd

#now after applying one hot encoding to my data set, i first identified the numerical columns and then normalized them


numerical_columns = df1_one_hot.select_dtypes(include=['int64', 'float64']).columns

df1_normalized = df1_one_hot.copy()
df1_normalized[numerical_columns] = (df1_one_hot[numerical_columns] - df1_one_hot[numerical_columns].min()) / (df1_one_hot[numerical_columns].max() - df1_one_hot[numerical_columns].min())

print("Normalized DataFrame ")
df1_normalized.head(14)


# In[38]:


import pandas as pd
from sklearn.model_selection import train_test_split

#next i split my dataset into training and testing, 80-20


# In[39]:


X = df1_normalized.drop('price', axis=1)  #defining the features, which are all of my other columns except price column
y = df1_normalized['price']    #and my target variable which is my price column


# In[40]:


#splitting my dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size (X_train):", X_train.shape)
print("Testing set size (X_test):", X_test.shape)
print("Training target size (y_train):", y_train.shape)
print("Testing target size (y_test):", y_test.shape)


# # Cleaning of Regression Dataset 2

# In[23]:


#loading the first data set


df2 = pd.read_csv(r"C:\Users\HP\Downloads\Used Cars Data-20240721T072109Z-001\Used Cars Data\pakwheels_used_car_data_v02.csv")
df2


# In[26]:


#displaying its first 15 rows

df2.head(15)


# In[27]:


#next first i decided to find out how many missing values are there in the data set


missing_values = df2.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])


# In[29]:


#first i made a list of all of my categorical columns, by comparing with the csv file
categorical_columns2 = ['city', 'assembly', 'body', 'make', 'model', 'transmission', 'fuel', 'color', 'registered']

# the i applied one-hot encoding to the categorical columns
df2_one_hot = pd.get_dummies(df2, columns=categorical_columns2)


print("Dataset Number 2 after applying One-Hot Encoding :")
df2_one_hot.head(14)


# In[32]:


import pandas as pd

#now after applying one hot encoding to my data set, i first identified the numerical columns and then normalized them


numerical_columns2 = df2_one_hot.select_dtypes(include=['int64', 'float64']).columns

df2_normalized2 = df2_one_hot.copy()
df2_normalized2[numerical_columns2] = (df2_one_hot[numerical_columns2] - df2_one_hot[numerical_columns2].min()) / (df2_one_hot[numerical_columns2].max() - df2_one_hot[numerical_columns2].min())

print("Normalized DataFrame (first 5 rows):")
df2_normalized2.head(14)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

#next i split my dataset into training and testing, 80-20


# In[ ]:


X = df2_normalized2.drop('price', axis=1)  #defining the features, which are all of my other columns except price column
y = df2_normalized2['price']    #and my target variable which is my price column


# In[ ]:


#splitting my dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size (X_train):", X_train.shape)
print("Testing set size (X_test):", X_test.shape)
print("Training target size (y_train):", y_train.shape)
print("Testing target size (y_test):", y_test.shape)


# # Data Analysis and Visualization

# ### For Dataset 1

# In[44]:


# Calculate mean, median, and mode for all numerical columns

# Mean
mean_values = df1_normalized.mean()

# Median
median_values = df1_normalized.median()

# Mode (Note: mode can return multiple values if there's more than one mode)
mode_values = df1_normalized.mode().iloc[0]  # `.iloc[0]` selects the first mode if there are multiple

# Display the results
print("Mean values for numerical columns:")
print(mean_values)

print("\nMedian values for numerical columns:")
print(median_values)

print("\nMode values for numerical columns:")
print(mode_values)


# In[47]:


print("Summary statistics for df1:")
df1_normalized.describe(include='all')


# In[54]:


pip install --upgrade matplotlib


# In[55]:


import matplotlib.pyplot as plt


# In[57]:


#plotting my histogram
df1_normalized.hist(figsize=(12, 8), bins=30, edgecolor='black')

#displaying using .show()
plt.show()


# In[58]:


import pandas as pd
import matplotlib.pyplot as plt


#lisiting features
feature_pairs = [
    ('ad_ref', 'price'),
    ('engine_cc', 'price'),
    ('mileage', 'price'),
    ('year', 'price'),
    ('ad_ref', 'engine_cc'),
    ('mileage', 'year')
]


plt.figure(figsize=(14, 10))
for i, (feature1, feature2) in enumerate(feature_pairs, 1):
    plt.subplot(3, 3, i) 
    plt.scatter(df1_normalized[feature1], df1_normalized[feature2], alpha=0.7)
    plt.title(f'Scatter Plot: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




corr_matrix = df1_normalized.corr()


plt.figure(figsize=(12, 10))
cax = plt.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(cax)


plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)


plt.title('Correlation Heatmap')


plt.show()


# ### For Dataset 2 

# In[ ]:


# Calculate mean, median, and mode for all numerical columns

# Mean
mean_values = df2_normalized2.mean()

# Median
median_values = df2_normalized2.median()

mode_values = df2_normalized2.mode().iloc[0] 

print("Mean values for numerical columns:")
print(mean_values)

print("\nMedian values for numerical columns:")
print(median_values)

print("\nMode values for numerical columns:")
print(mode_values)


# In[ ]:


print("Summary statistics for df1:")
df2_normalized2.describe(include='all')


# In[ ]:


df2_normalized2.hist(figsize=(12, 8), bins=30, edgecolor='black')

plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


feature_pairs = [
    ('ad_ref', 'price'),
    ('engine_cc', 'price'),
    ('mileage', 'price'),
    ('year', 'price'),
    ('ad_ref', 'engine_cc'),
    ('mileage', 'year')
]


plt.figure(figsize=(14, 10))
for i, (feature1, feature2) in enumerate(feature_pairs, 1):
    plt.subplot(3, 3, i)  
    plt.scatter(df2_normalized2[feature1], df2_normalized2[feature2], alpha=0.7)
    plt.title(f'Scatter Plot: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)

plt.tight_layout()
plt.show()

