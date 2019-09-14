#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:28:51 2019

@author: samuelghatan
"""

""" 
This is a guided project from Dataquest. 
Working with housing data for the city of Ames, Iowa, United States from 2006 to 2010.
The goal of this project is to build a linear regression model that can accurately predict house prices.
Information on the data columns can be found here: https://s3.amazonaws.com/dq-content/307/data_description.txt
DF shape = (2930, 82)
"""

import pandas as pd
pd.options.display.max_columns = 999
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold
import seaborn as sns

""" My Pipeline consists of three functions  pipeline of functions 
    transform_features()
    select_features()
    train_and_test()
these will allow me to quickly iterate on different models. """

def transform_features(df):
    num_missing = df.isnull().sum()
    drop_missing_cols = num_missing[(num_missing > len(df)/20)].sort_values() # Remove features with with more than 5% missing values
    df = df.drop(drop_missing_cols.index, axis=1) 
    
    # Clean and transform text columns
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    
    # Clean and transform numeric columns
    num_missing = df.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(df)/20) & (num_missing > 0)].sort_values()
    replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0] # For columns with missing values, fill in with the most common value in that column
    df = df.fillna(replacement_values_dict)
    
    ## What new features can we create, that better capture the information in some of the features?
    years_sold = df['Yr Sold'] - df['Year Built'] # How many years before house was sold
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add'] # How many years before the house was remodeled
    
    ## Create new columns
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod
    
    ## Drop rows with negative values for both of these new features
    df = df.drop([1702, 2180, 2181], axis=0)
    
    ## No longer need original year columns
    df = df.drop(["Year Built", "Year Remod/Add"], axis = 1)
    
    # Drop columns that aren't useful for ML
    df = df.drop(["PID", "Order"], axis=1) # Parcel identification number, Observation number
    
    ## Drop columns that leak info about the final sale
    df = df.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
    return df

def select_features(transform_df):
    numeric_df = transform_df.select_dtypes(include=['int', 'float'])
    correlations = abs(numeric_df.corr())
    ## Drop columns with less than 0.5 correlation with SalePrice
    numeric_df = numeric_df.drop(correlations[correlations > 0.5].index,axis = 1)
    
    # Which categorical columns should we keep?
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                        "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                        "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                        "Misc Feature", "Sale Type", "Sale Condition"]
    
    ## Which categorical/nominal columns do we still have in the transformed Dataframe?
    transform_cat_cols = []
    for col in nominal_features:
        if col in transform_df.columns:
            transform_cat_cols.append(col)
    
    ## How many unique values in each categorical column?
    uniqueness_counts = transform_df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
    ## Cutoff of 10 unique values ()
    drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > 10].index
    # Update transform_df to drop those categorical columns that have too many unique values
    transform_df = transform_df.drop(drop_nonuniq_cols, axis=1)
    
    ## Select just the remaining text columns and convert to categorical
    text_cols = transform_df.select_dtypes(include=['object'])
    for col in text_cols:
        transform_df[col] = transform_df[col].astype('category')
        
    ## Create dummy columns and add back to the dataframe
    for col in text_cols:
        dummy_cols = pd.get_dummies(transform_df[col])
        transform_df= pd.concat([transform_df,dummy_cols],axis = 1)
        del transform_df[col]
    
    
    return transform_df

def train_and_test(df, k = 0):
    numeric_train = df.select_dtypes(include=['int', 'float'])
    numeric_test = df.select_dtypes(include=['int', 'float'])
    features = numeric_train.columns.drop('SalePrice')
    lr = linear_model.LinearRegression()
    
    # When k equals 0, perform holdout validation
    if k == 0:
        train = df[:1460]
        test = df[1460:]
        lr.fit(train[features], train["SalePrice"])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test["SalePrice"], predictions)
        rmse = np.sqrt(mse)
        return rmse
    # When k = 1 perfrom a simple cross validation
    if k == 1:
        # Randomize *all* rows (frac=1) from `df` and return
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        
        lr.fit(train[features], train["SalePrice"])
        predictions_one = lr.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        lr.fit(test[features], test["SalePrice"])
        predictions_two = lr.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        return avg_rmse
    # When K is greater than 1 perform k-fold cross validation using k folds
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df): #Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds 
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        #print( rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse
        

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
k_rank = {}
for i in range(2,20):
    rmse = train_and_test(filtered_df, k=i)
    k_rank[i] = rmse
print(k_rank)

    
    
    