#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:36:49 2019

@author: samuelghatan
"""

""" Many American cities have communal bike sharing stations where you can rent bicycles by the hour or day. Washington, D.C. is one of these cities. 
The District collects detailed data on the number of bicycles people rent by the hour and day.

Hadi Fanaee-T at the University of Porto compiled this data into a CSV file, which you'll be working with in this project. 
The file contains 17380 rows, with each row representing the number of bike rentals for a single hour of a single day. 
You can download the data from the University of California, Irvine's website. 

In this project I will create and employ a few different machine learning models to predict the total number of bikes people rented in a given hour (cnt).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor

bike_rentals = pd.read_csv('bike_rental_hour.csv')

# First I want to see the disturbution of rentals
#plt.hist(bike_rentals['cnt'])
#plt.show()
# Most hours rentals are between 0-100

# Next I will covert the hr column into morrning, afternoon, evening and night for better predictions

def assign_label(hour):
    if hour >=0 and hour < 6:
        return 4
    elif hour >=6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour <=24:
        return 3

bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)

# Next I would like to explore how each column is correlated with cnt.
correlations = bike_rentals.corr()
#sns.heatmap(correlations)
correlations['cnt'].sort_values()
# Causal and registered are sub-features of the count column so are excluded, the next most correlated features is temp

# Select Training and test data
train_max_row = math.floor(bike_rentals.shape[0] * .8)
train = bike_rentals.iloc[:train_max_row]
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]

# Apply linear regression, Linear regression will probably work fairly well on this data, given that many of the columns are highly correlated with cnt
pred_columns = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'time_label']
lr = LinearRegression()
lr.fit(train[pred_columns],train['cnt'])
predictions = lr.predict(test[pred_columns])

# Calculate errors
mae = mean_absolute_error(test['cnt'],predictions)
mse = numpy.mean((predictions - test["cnt"]) ** 2)
# The error is very high, which may be due to the fact that the data has a few extremely high rental counts, but otherwise mostly low counts.

# Decision trees can more accurately predict outcomes, lets try this algorithm next
clf = DecisionTreeClassifier(min_samples_leaf = 2)
clf.fit(train[pred_columns],train['cnt'])
predictions_2 = clf.predict(test[pred_columns])
dc_mse = numpy.mean((predictions_2 - test["cnt"]) ** 2)
print(dc_mse)
# By taking the nonlinear predictors into account, the decision tree regressor appears to have much higher accuracy than linear regression.

# Next ill apply the random forest algorithm to the dataset to improve prediction accuracy further
reg = RandomForestRegressor(min_samples_leaf=5)
reg.fit(train[pred_columns], train["cnt"])
predictions_3 = reg.predict(test[pred_columns])

numpy.mean((predictions_3 - test["cnt"]) ** 2)
# By removing some of the sources of overfitting, the random forest accuracy is improved over the decision tree accuracy.