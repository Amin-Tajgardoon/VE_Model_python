'''
Created on Sep 27, 2016

@author: mot16
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing

def readData(filePath):

    df = pd.read_csv(filePath)
    #df.head(3)
    
    # features
    X= df.ix[:, df.columns != 'wviral_rawres']
    #X.head(5)
    
    # labels
    y = df.wviral_rawres
    
    # drop pid and treatmnt, group_nb is the same as treatmnt
    X = X.drop(['pid','treatmnt'], axis =1 )
    #X.head(2)
    
    # String to int
    le = preprocessing.LabelEncoder()
    X.cty_cod = le.fit_transform(X.cty_cod)
    X.sex = le.fit_transform(X.sex)
    #X.head(2)
    
    # replace missing values
    X.days_to_infect = X.days_to_infect.replace(to_replace='NaN',value=-1)
    X.loc[(pd.isnull(X.days_to_ili_start)) & (X.ili_nb > 0),'days_to_ili_start'] = np.nanmean(X.days_to_ili_start)
    X.loc[(pd.isnull(X.days_to_ili_start)) & (X.ili_nb == 0) & (X.days_to_infect == -1 ),'days_to_ili_start'] = -1
    
    # reolace lables with 0 and 1
    y.replace(to_replace='N',value=0, inplace=True)
    y.replace(to_replace='P',value=1, inplace=True)
    
    return X, y

def saveData(X, y, filePath):
    data = pd.concat([X,y],axis=1)
    data.to_csv(filePath, index = False)