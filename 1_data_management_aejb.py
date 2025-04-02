# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:12:03 2020

@author: annic
"""

# Importing the necessary libraries and modules

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="whitegrid")
#sns.set(style="darkgrid")

import statsmodels.formula.api as smf  # statsmodels
import statsmodels.stats.multicomp as multi  # statsmodels and posthoc test
import statsmodels.api as sm  # Statsmodel for the qqplots
import scipy.stats  # For the Chi-Square test of independance

#from sklearn.cross_validation import train_test_split older codes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
import sklearn.metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
#from sklearn.linear_model import lassoLarsCV older codes
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

# Feature Importance - for the random trees
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

# Feature importance - for the Kmeans
from sklearn import preprocessing
from sklearn.cluster import KMeans

import graphviz


# =============================================================================
# Potential bugg fixs
pd.set_option('display.float_format', lambda x: '%.1f' % x) # Supress the scientific notation
"""
In order to revert Pandas behaviour to defaul use .reset_option().
pd.reset_option('display.float_format')
"""
# This will supress the copy warning from a previous dataset that refers to same variable names
pd.options.mode.chained_assignment = None  # default='warn'

# =============================================================================
data = pd.read_csv('worldbank.csv', low_memory=False) # low_memory=False pandas will try to see in each columns what's the type of the variable

data.shape
data.head()


# =============================================================================
# Data management
# =============================================================================
# Convert all the variables that we will need to numeric

# Dependent variable
# 2013: Foreing direct investment (current us $)
data['x132_2013'] = pd.to_numeric(data['x132_2013'])

# Adjusted savings : consumption of fixed capital (current us$)
data['x15_2012'] = pd.to_numeric(data['x15_2012'])

# GDP per capita at market prices (current us$)
data['x139_2012'] = pd.to_numeric(data['x139_2012'])

# GDP per capita (current us$)
data['x142_2012'] = pd.to_numeric(data['x142_2012'])


# Adjusted net national income per capita (current us $)
data['x11_2012'] = pd.to_numeric(data['x11_2012'])

# Adjusted net national income (current us$)
data['x9_2012'] = pd.to_numeric(data['x9_2012'])

# =============================================================================
# ### New World Bank classification of countries based on revenues:
# Gross national income per capita
# 
# - LOW-INCOME ECONOMIES (1,035 OR LESS)   
# - LOWER-MIDDLE INCOME ECONOMIES (1,036 TO 4,045) 
# - UPPER-MIDDLE-INCOME ECONOMIES (4,046 TO 12,535) 
# - HIGH-INCOME ECONOMIES (12,536 OR MORE)

# Let's work with a subset of the data to make it more convinent to work with

df = data[['country', 'x132_2013', 'x15_2012', 'x139_2012', 'x142_2012', 'x11_2012', 'x9_2012']]


df.dtypes

# Data Cleaning: Handling Missing Data
df = df.dropna()
df.isnull().any() # Is there missing value for the variable/culumns ?


# =============================================================================
# Descriptive statistics
# Mean
df.mean()

# Minimum
df.min()

# Maximum
df.max()

# =============================================================================
len(df.columns)
df.columns



# Because zero is not a valid value for these variables,
# I will center them by subtracting the mean 



# ### New World Bank classification of countries based on revenues:
# Gross national income per capita

# - LOW-INCOME ECONOMIES (1,035 OR LESS)   
# - LOWER-MIDDLE INCOME ECONOMIES (1,036 TO 4,045) 
# - UPPER-MIDDLE-INCOME ECONOMIES (4,046 TO 12,535) 
# - HIGH-INCOME ECONOMIES (12,536 OR MORE)

def country_classification(row):
    if row['x11_2012'] > 12536:
        return 3
    elif (row['x11_2012'] > 4046) & (row['x11_2012'] < 12535):
        return 2
    elif (row['x11_2012'] > 1036) & (row['x11_2012'] < 4045):
        return 1
    elif row['x11_2012'] < 1035:
        return 0

df['country_classification'] = df.apply(lambda row: country_classification(row), axis = 1)


df[['country', 'country_classification']].head(10)

plot_country_classification = df['country_classification'].value_counts(sort = True, dropna = False)
print(plot_country_classification)

sns.countplot(df['country_classification'])
plt.title('Country classification')
plt.xlabel('Country classification')


