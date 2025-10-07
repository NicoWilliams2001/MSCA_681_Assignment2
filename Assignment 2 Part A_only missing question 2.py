#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_decision_regions
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

import seaborn as sns

#Loading the dataset
carseats_df = pd.read_csv('/Users/nicolaswilliams/Desktop/MSCA 681/Datasets-20251004/Carseats.csv')

print(carseats_df.head())



# # A. Data Exploration and Exploration

# ## 1.1 Explore and visualize the dataset. Look into dataset structure, missing values, summary statistics.

# In[45]:


carseats_df = pd.read_csv('/Users/nicolaswilliams/Desktop/MSCA 681/Datasets-20251004/Carseats.csv')
# Data Structure 
print('\n1. BASIC DATASET INFO:')
print(f'Dataset shape: {carseats_df.shape}')
print(f'Dataset rows: {carseats_df.shape[0]}')
print(f'Dataset columns: {carseats_df.shape[1]}')

print('\n2. COLUMN INFORMATION:')
print(carseats_df.info())
numeric_variables = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']
categorical_variables = ['ShelveLoc', 'Urban', 'US']
print("Numerical Variables Columns:", numeric_variables)
print("Categorical Variables Columns:", categorical_variables)

print('\n3. DATA TYPES:')
print(carseats_df.dtypes)

# Missing Value 
print('\n4. MISSING VALUES COUNT:')
missing_values = carseats_df.isnull().sum() 
print(missing_values)

# Looking into the categorical values
print('\n5. CATEGORICAL VALUES:')
for col in categorical_variables:
  print(f"\nValue counts for {col}:")
  print(carseats_df[col].value_counts())  

# Summary statistics 
print('\n6. SUMMARY STATISTICS:')
carseats_df.describe()



# ## 1.2 Create at least 5 histograms, 5 box plots, 5 bar charts, 5 scatterplots using the numerical and categorical input variables and the output variable in your dataset.

# ### Histogram

# In[106]:


# Create a list of titles 
titles = [
    "Sales in thousands ($)", "Competition Price in thousands ($)", "Income in thousands ($)",
    "Advertising Budget in thousands ($)", "Populationin thousands", 
    "Price", "Age", "Years of Education"]

# Prepare the variables
histo_vars = numeric_variables

# Create a figure and axes for subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()  # Flatten the axes array for easier indexing

for i, col in enumerate(histo_vars):
    sns.histplot(data=carseats_df, x=col, kde=True, ax=axes[i])
    
    # Set the title and labels
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# ### Box Plots

# In[107]:


# Create a list of titles 
titles = [
    "Sales in thousands ($)", "Competition Price in thousands ($)", "Income in thousands ($)",
    "Advertising Budget in thousands ($)", "Populationin thousands", 
    "Price", "Age", "Years of Education"]
# Prepare the variables
box_vars = numeric_variables

# Create a figure and axes for subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()  # Flatten the axes array for easier indexing

for i, col in enumerate(box_vars):
    sns.boxplot(data=carseats_df, x=col, ax=axes[i])
    
    # Set the title and labels
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(col)
 

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# ### Bar Charts

# In[99]:


carseats_df = pd.read_csv('/Users/nicolaswilliams/Desktop/MSCA 681/Datasets-20251004/Carseats.csv')

categorical_variables = ['ShelveLoc', 'Urban', 'US']

# Create a list of titles 
titles = ['Quality of Shelve Location', 'Store Is Located in an Urban Location', 'Store Is Located in the US']

# Bar charts for categorical counts
bar_vars = categorical_variables
fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # Changed to 1 row, 3 columns
axes = axes.ravel()  # Flattens axes into a 1D array

for i, col in enumerate(bar_vars):
    sns.countplot(data=carseats_df, x=col, ax=axes[i])
    # Added colour and smoothened edges
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Total Amount")

plt.tight_layout()
plt.show()


# In[103]:


carseats_df = pd.read_csv('/Users/nicolaswilliams/Desktop/MSCA 681/Datasets-20251004/Carseats.csv')
group_vars = [('ShelveLoc', 'Sales'), ('Urban', 'Sales'), ('US', 'Sales')]

# Create a list of titles 
titles = ["Sales by Quality of Shelve Location", "Sales by if the Store is in an Urban Location", "Sales by if the Store is in the US"]

# Bar charts for categorical counts
fig, axes = plt.subplots(1, 3, figsize=(14, 5)) 
axes = axes.ravel()  # Flattens axes into a 1D array

for i, (cat_col, num_col) in enumerate(group_vars):
    sns.barplot(data=carseats_df, x=cat_col, y=num_col, ax=axes[i])
    # Added colour and smoothened edges
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(cat_col)
    axes[i].set_ylabel("Average Sales")

plt.tight_layout()
plt.show()


# ### Scatterplots

# In[110]:


#Scatter charts (numerical vs sales)
#income #population #advertising #age #price
scatter_vars = numeric_variables
titles= [
    "Sales in thousands ($)", "Competition Price in thousands ($)", "Income in thousands ($)",
    "Advertising Budget in thousands ($)", "Populationin thousands", 
    "Price", "Age", "Years of Education"]

fig, axes = plt.subplots(3,3, figsize=(14,10))
axes = axes.ravel()

for i, col in enumerate(scatter_vars):
    sns.scatterplot(data=carseats_df, x=col, y='Sales', ax=axes[i])
    axes[i].set_title(f"{(titles[i])} vs Sales")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Sales")

plt.tight_layout()
plt.show()


# ## 2. Examine relationships between numerical variables. Identify highly correlated variables with each other and with the target. Discuss whether these variables should be dropped or not, considering the type of models you will use later.

# In[16]:


# Remove the unnamed variable
carseats_df = carseats_df.drop('Unnamed: 0', axis=1)

# Calculate the numerical variable correlation matrix for numeric columns only
corr_train = carseats_df.corr(numeric_only=True)

# Plot the correlation matrix with heatmap
plt.figure(figsize=(12, 10))  
sns.heatmap(corr_train,  
            annot=True,              
            cmap='coolwarm',         
            fmt='.2f',               
            square=True,             
            cbar_kws={"shrink": .8}, 
            linewidths=0.5,          
            center=0)                
plt.title('Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)               
plt.tight_layout()                   
plt.show()

The variable most highly correlated with Sales, our target, is Price with a correlation of -0.44, indicating that higher prices are associated with lower sales.

Among the variables, CompPrice and Price show the highest correlation at 0.58, suggesting that competitor prices and actual prices tend to move together. Advertising and Population (0.27) show a weak positive correlation. Overall, the correlation is not a significant concern in this dataset because there's no predictor pairs exceed the 0.7 threshold.
# ## 3. Convert categorical variables into numeric form using dummy variables.

# In[46]:


carseats_df = pd.read_csv('/Users/nicolaswilliams/Desktop/MSCA 681/Datasets-20251004/Carseats.csv')

# Let's one-hot encode our categorical variables
carseats_df = pd.get_dummies(carseats_df, columns=["ShelveLoc", "Urban", "US"], drop_first = True, dtype = 'int')

# With the new dummy variables
print(carseats_df.head(10))

# Looking into the categorical values
categorical_variables = ['ShelveLoc_Good','ShelveLoc_Medium', 'Urban_Yes', 'US_Yes']
for col in categorical_variables:
  print(f"\nValue counts for {col}:")
  print(carseats_df[col].value_counts()) 


# ## 4. Partition the data into a training set (60%), validation set (20%), and test set (20%) (set random_state=42).

# In[21]:


from sklearn.model_selection import train_test_split

#Split the data into temp and test sets
temp_df, train_df = train_test_split(carseats_df, test_size=0.6, random_state=42)

#Split the data into train and validation sets
test_df, validation_df = train_test_split(temp_df, test_size=0.5, random_state=42)

#Print the shapes
print(f"Train set: {train_df.shape[0]}")
print(f"Validation shape: {validation_df.shape[0]}")
print(f"Test shape: {test_df.shape[0]}")

