'''
Created on Oct 19, 2016

@author: mot16
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing

def readData(filePath):

    df = pd.read_csv(filePath)
    # df.head(3)
    
    # features
    X = df.ix[:, df.columns != 'wviral_rawres']
    # X.head(5)
    
    # labels
    y = df.wviral_rawres
    
    return X, y

def clean_data(X, y):
    
    # drop all but 7 vars
    X = X[["age", "sex", "cty_cod", "ethnic", "race", "center", "group_nb"]]
    # X.head(2)
    
    # String to int
    #le = preprocessing.LabelEncoder()
    #X.cty_cod = le.fit_transform(X.cty_cod)
    #X.sex = le.fit_transform(X.sex)
    # X.head(2)
    
    # X.group_nb.unique()
    # replace group 2 with 1
    X.group_nb.replace(to_replace=2, value=1, inplace=True)
    
    # check if any columns contain Null value
    X.isnull().any()
   
    # binaries class variable
    y.fillna(value='N', inplace=True)
    y.replace(to_replace='A', value='P', inplace=True)
    y.replace(to_replace='B', value='P', inplace=True)
    
    return X, y

def saveData(X, y, filePath):
    data = pd.concat([X, y], axis=1)
    data.to_csv(filePath, index=False)
    
X, y = readData("C:\\Users\\mot16\\projects\\master\\data\\joined_data_no_dates.csv");
X, y = clean_data(X, y)
saveData(X, y, "C:\\Users\\mot16\\projects\\master\\data\\joined_data_no_dates_cleanedin_python.csv")
