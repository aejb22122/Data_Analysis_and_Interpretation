# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:47:10 2020

@author: annick-eudes
"""

# Importing the necessary libraries and modules

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
#sns.set(style="whitegrid")
sns.set(style="darkgrid")

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

# Foreing direct investment (current us $)
data['x132_2012'] = pd.to_numeric(data['x132_2012'])
data['x132_2013'] = pd.to_numeric(data['x132_2013'])

# Adjusted savings : consumption of fixed capital (current us$)
data['x15_2012'] = pd.to_numeric(data['x15_2012'])
data['x15_2013'] = pd.to_numeric(data['x15_2013'])

# GDP per capita at market prices (current us$)
data['x139_2012'] = pd.to_numeric(data['x139_2012'])
data['x139_2013'] = pd.to_numeric(data['x139_2013'])

# GDP per capita (current us$)
data['x142_2012'] = pd.to_numeric(data['x142_2012'])
data['x142_2013'] = pd.to_numeric(data['x142_2013'])

# Adjusted net national income per capita (current us $)
data['x11_2012'] = pd.to_numeric(data['x11_2012'])
data['x11_2013'] = pd.to_numeric(data['x11_2013'])

# Adjusted net national income (current us$)
data['x9_2012'] = pd.to_numeric(data['x9_2012'])
data['x9_2013'] = pd.to_numeric(data['x9_2013'])

# GDP per capita growth annual (annual %)
# data['x143_2012'] = pd.to_numeric(data['x143_2012'])
# data['x143_2013'] = pd.to_numeric(data['x143_2013'])

# # Gross domestic savings (% of GDP)
# data['x146_2012'] = pd.to_numeric(data['x146_2012'])
# data['x146_2013'] = pd.to_numeric(data['x146_2013'])

# =============================================================================

# Let's select only low income countries

# ### New World Bank classification of countries based on revenues:
# Gross national income per capita
# 
# - LOW-INCOME ECONOMIES (1,035 OR LESS)   
# - LOWER-MIDDLE INCOME ECONOMIES (1,036 TO 4,045) 
# - UPPER-MIDDLE-INCOME ECONOMIES (4,046 TO 12,535) 
# - HIGH-INCOME ECONOMIES (12,536 OR MORE)

#data_lic = data.loc[(data['x11_2012'] < 1025 )]



#.loc[row_indexer,col_indexer] = value instead

#data_lic.head()

# Let's work with a subset of the data to make it more convinent to work with

df = data[['country', 
          'x132_2012','x132_2013',
          'x15_2012', 'x15_2013',
          'x139_2012','x139_2013',  
          'x142_2012','x142_2013',
          'x11_2012', 'x11_2013',
          'x9_2012','x9_2013']]


df.dtypes

# Data Cleaning: Handling Missing Data
df = df.dropna()
df.isnull().any() # Is there missing value for the variable/culumns ?


# Creating new variables

# Panel data with two time periods : "Before and After" comparisons. 
# We're considering the changes from the two time.

df['FDI'] = abs(df['x132_2013'] - df['x132_2012'])
df['consumption_fixed_capital'] = abs(df['x15_2013'] - df['x15_2012'])
df['GDP_market_prices'] = abs(df['x139_2013'] - df['x139_2012'])
df['GDP_per_capita_current'] = abs(df['x142_2013'] - df['x142_2012'])
df['net_nat_income_per_capita'] = abs(df['x11_2013'] - df['x11_2012'])
df['net_nat_income_current'] = abs(df['x9_2013'] - df['x9_2012'])

#df['GDP_per_capita_growth'] = abs(df['x143_2013'] - df['x143_2012'])
#df['gross_savings_perc_GDP'] = abs(df['x146_2013'] - df['x146_2012'])


# =============================================================================
# Descriptive statistics
# Mean
df['x132_2013'].mean()
df['consumption_fixed_capital'].mean()
df['GDP_market_prices'].mean()
df['GDP_per_capita_current'].mean()
df['net_nat_income_per_capita'].mean()
df['net_nat_income_current'].mean()


# Minimum
df['FDI'].min()
df['consumption_fixed_capital'].min()
df['GDP_market_prices'].min()
df['GDP_per_capita_current'].min()
df['net_nat_income_per_capita'].min()
df['net_nat_income_current'].min()


# Maximum
df['FDI'].max()
df['consumption_fixed_capital'].max()
df['GDP_market_prices'].max()
df['GDP_per_capita_current'].max()
df['net_nat_income_per_capita'].max()
df['net_nat_income_current'].max()


# =============================================================================
len(df.columns)
df.columns

# Centering the explanatory variables that do not have any 0s
# Do we have 0 values?
df.loc[df['FDI'] == 0]
df.loc[df['consumption_fixed_capital'] == 0]
df.loc[df['GDP_market_prices'] == 0] # The're a 0 here
df.loc[df['GDP_per_capita_current'] == 0]
df.loc[df['net_nat_income_per_capita'] == 0]
df.loc[df['net_nat_income_current'] == 0] # The're two 0 here
#df.loc[df['GDP_per_capita_growth'] == 0]
#df.loc[df['gross_savings_perc_GDP'] == 0]

# Because zero is not a valid value for these variables,
# I will center them by subtracting the mean 
df['consumption_fixed_capital_c'] = (df['consumption_fixed_capital'] - df['consumption_fixed_capital'].mean())
df['GDP_per_capita_current_c'] = (df['GDP_per_capita_current'] - df['GDP_per_capita_current'].mean())
df['net_nat_income_per_capita_c'] = (df['net_nat_income_per_capita'] - df['net_nat_income_per_capita'].mean())
df['net_nat_income_current_c'] = (df['net_nat_income_current'] - df['net_nat_income_current'].mean())
#df['GDP_per_capita_growth_c'] = (df['GDP_per_capita_growth'] - df['GDP_per_capita_growth'].mean())
#df['gross_savings_perc_GDP_c'] = (df['gross_savings_perc_GDP'] - df['gross_savings_perc_GDP'].mean())


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

plot_country_classification.plot.bar()
plt.title('Country classification')
#plt.savefig('country_classification.png')

x = df['country_classification']
labels =  np.arange(len(df['country_classification']))
plt.bar(x, labels, align='edge', width=-0.4);

sns.countplot(df['country_classification'])
plt.title('Country classification')
plt.xlabel('Country classification')


