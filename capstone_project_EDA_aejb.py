# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:46:39 2020

@author: annic
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
# DO NOT RUN THESE LINES !!!
# =============================================================================

data = pd.read_csv('worldbank.csv', low_memory=False) # low_memory=False pandas will try to see in each columns what's the type of the variable

data.shape
data.head()

df = data[['country', 'x132_2013', 'x15_2012', 
           'x139_2012', 'x142_2012', 'x11_2012', 
           'x9_2012']]


# =============================================================================
# Exploratory data analysis
# =============================================================================
df.columns

# Histograme of the Income country's distribution
plt.hist(df['x11_2012'], density=False, alpha = 0.7)
plt.title('Adjusted net national income per capita (CURRENT US$) in low income countries')

# ### Bivariate plots
# FDI and income

x = df['x15_2012']
y = df['x132_2013']

plt.scatter(x, y, alpha = 0.7, marker = 'o')
plt.xlabel('Consumption of Fixed Capital (Current USD)')
plt.ylabel('FDI, net inflows (BOP, current US$)')
plt.title('Consumption of Fixed Capital (Current USD) and FDI');

# =============================================================================
x = df['x139_2012'] 
y = df['x132_2013']

plt.scatter(x, y, alpha = 0.7, marker = 'o')
plt.xlabel('GDP At Market Prices (current USD)')
plt.ylabel('FDI, net inflows (BOP, current US$)')
plt.title('GDP At Market Prices and FDI');


# =============================================================================
x = df['x142_2012']
y = df['x132_2013']

plt.scatter(x, y, alpha = 0.7, marker = 'o')
plt.xlabel('GDP per capita (current US$)')
plt.ylabel('FDI, net inflows (BOP, current US$)')
plt.title('GDP per capita (current US$) and FDI');

# =============================================================================
x = df['x11_2012']
y = df['x132_2013']

plt.scatter(x, y, alpha = 0.7, marker = 'o')
plt.xlabel('Adjusted Net National Income Per Capita (Current USD)')
plt.ylabel('FDI, net inflows (BOP, current US$)')
plt.title('Adjusted Net National Income Per Capita and FDI');

# =============================================================================
x = df['x9_2012']
y = df['x132_2013']

plt.scatter(x, y, alpha = 0.7, marker = 'o')
plt.xlabel('Adjusted Net National Income (Current USD)')
plt.ylabel('FDI, net inflows (BOP, current US$)')
plt.title('Adjusted Net National Income (Current USD) and FDI');


# =============================================================================
# Data analysis and testing
# =============================================================================
# Correlations :
    
df_corr = df.dropna()
print('Pearson correlations :')
print('=============================================================================')
print('association between consumption of fixed capital and FDI')
print(scipy.stats.pearsonr(df_corr['x15_2012'], df_corr['x132_2013']))
print('')
print('association between GDP market price  and FDI')
print(scipy.stats.pearsonr(df_corr['x139_2012'],df_corr['x132_2013']))
print('')
print('=============================================================================')
print('association between GDP per capita current and FDI')
print(scipy.stats.pearsonr(df_corr['x142_2012'], df_corr['x132_2013']))
print('')
print('=============================================================================')
print('association between Adjusted Net National Income Per Capita (Current USD) and FDI')
print(scipy.stats.pearsonr(df_corr['x11_2012'], df_corr['x132_2013']))
print('')
print('=============================================================================')
print('association between Adjusted Net National Income  and FDI')
print(scipy.stats.pearsonr(df_corr['x9_2012'], df_corr['x132_2013']))
print('')
print('=============================================================================')

# =============================================================================
# ANOVA
# ***What are the effects of country’s income level (GDP), 
# Domestic Capital Stock (Gross capital formation) and Foreign Direct Investments?***
# Is income associated with FDI and is this dependant on revenue level?

anova1 = smf.ols(formula = 'x132_2013 ~ C(country_classification)', data = df).fit()
print(anova1.summary())

# To interpret the results fully, we need to examine the means 
anova_df = df[['x132_2013', 'country_classification']].dropna()

print('Means for FDI by country classification')
print(anova_df.groupby('country_classification').mean())

print('Standard deviation for FDI by country classification')
print(anova_df.groupby('country_classification').std())

# All the means are equal with the 


# Plot the orbital period with horizontal boxes

# - LOW-INCOME ECONOMIES (1,035 OR LESS)   
# - LOWER-MIDDLE INCOME ECONOMIES (1,036 TO 4,045) 
# - UPPER-MIDDLE-INCOME ECONOMIES (4,046 TO 12,535) 
# - HIGH-INCOME ECONOMIES (12,536 OR MORE)


def rev_country_classification(row):
    if row['country_classification'] == 3:
        return 'High income'
    elif row['country_classification'] == 2:
        return 'Upper middle'
    elif row['country_classification'] == 1:
        return 'Low middle'
    elif row['country_classification'] == 0:
        return 'Low income'

anova_df['country_classification2'] = df.apply(lambda row: rev_country_classification(row), axis = 1)


sns.boxplot(x="country_classification2", y="x132_2013", data=anova_df, palette="YlGnBu_d", showfliers = False)
plt.xlabel('Country classification')
plt.ylabel('FDI')

sns.factorplot(x="country_classification", y="x132_2013", data=anova_df)



# Post hoc test
mc1 = multi.MultiComparison(df['x132_2013'], df['country_classification'])
res1 = mc1.tukeyhsd()
print(res1.summary())


# =============================================================================
# Multiple regressions

# Centering the explanatory variables that do not have any 0s
# Do we have 0 values?

df.loc[df['x15_2012'] == 0] # Needs to be cented
df.loc[df['x139_2012'] == 0] # Needs to be cented
df.loc[df['x142_2012'] == 0] # Needs to be cented
df.loc[df['x11_2012'] == 0] # Needs to be cented
df.loc[df['x9_2012'] == 0] # Needs to be cented

# adding number of cigarettes smoked as an explanatory variable 
# center quantitative IVs for regression analysis
df['x15_2012_c'] = (df['x15_2012'] - df['x15_2012'].mean())
df['x139_2012_c'] = (df['x139_2012'] - df['x139_2012'].mean())
df['x142_2012_c'] = (df['x142_2012'] - df['x142_2012'].mean())
df['x11_2012_c'] = (df['x11_2012'] - df['x11_2012'].mean())
df['x9_2012_c'] = (df['x9_2012'] - df['x9_2012'].mean())

# Verifications
print(df['x15_2012_c'] .mean()) 
print(df['x139_2012_c'].mean())
print(df['x142_2012_c'].mean())
print(df['x11_2012_c'].mean())
print(df['x9_2012_c'].mean())



# First regression  with all the variables :
df.columns
reg0 = smf.ols('x132_2013 ~ C(country_classification) + x15_2012_c + x139_2012_c + x142_2012_c + x11_2012_c + x9_2012_c',
               data = df).fit()
print(reg0.summary())


reg1 = smf.ols('x132_2013 ~ C(country_classification) + x15_2012_c + x139_2012_c + x9_2012_c',
               data = df).fit()
print(reg1.summary())


reg3 = smf.ols('x132_2013 ~ x15_2012_c + x139_2012_c + x9_2012_c',
               data = df).fit()
print(reg3.summary())

# Diagnostic



