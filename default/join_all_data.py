'''
Created on Oct 21, 2016

@author: mot16
'''

import pandas as pd
from os import listdir
from pandas import io
import numpy as np

'''
    prepare_wconvac():
        1- binaries TRADNAME variable
        2- selects PID and binary variables (drops other variables !!!)
        3- replaces 0 and 1 with N and Y in binary variables
        4- drops duplicate rows
'''
def prepare_wconvac(wconvac):
    
    df = pd.get_dummies(data=wconvac, prefix='concvac', prefix_sep='_', dummy_na=False, columns=['TRADNAME'], sparse=False, drop_first=False)
    cols = np.r_[list(wconvac.columns).index('PID'), np.arange(wconvac.shape[1]-1, df.shape[1])]
    df = df[:, cols]
    df = pd.concat([df, df.iloc[:,list(range(1,df.shape[1]))].replace(0,'N')], axis = 1)
    df = pd.concat([df, df.iloc[:,list(range(1,df.shape[1]))].replace(1,'Y')], axis = 1)
    df = df.drop_duplicates()
    return df

def addPrefixToColumnNames(columns, prefix, exclude):
    colnames = []
    for col in list(columns):
        if col == exclude:
            colnames.append(col)
        else:
            colnames.append(col + prefix)
    return colnames

'''
main section
'''
dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-108134\\R_analysis\\"

wconvac = pd.read_csv()
prepare_wconvac

dataFiles = [f for f in listdir(dataDir) if f.count('_info.csv') == 0]
 
for f in list(dataFiles):
    with open(dataDir + f) as myfile:
        head = next(myfile)
    if head.split(sep=',').count('PID') == 0:
        dataFiles.remove(f)

noTimeSeries = ['gsk_108134_expogn.csv', 'gsk_108134_pid.csv', 'gsk_108134_reaccod.csv', 'gsk_108134_wconc.csv', 'gsk_108134_wdemog.csv', 'gsk_108134_welig.csv', 'gsk_108134_wlabo.csv', 'gsk_108134_wnoadm.csv', 'gsk_108134_wnpap.csv', 'gsk_108134_wphist.csv', 'gsk_108134_wpneumo.csv', 'gsk_108134_wsolpre.csv']
timeSeries = list(set(dataFiles) - set(noTimeSeries))
labResults = 'gsk_108134_wviral.csv'
demog = "gsk_108134_wdemog.csv"

df = pd.read_csv(dataDir + demog)
   
df.columns = addPrefixToColumnNames(df.columns, '_' + demog[11:].split('.')[0], 'PID')
noTimeSeries.remove(demog)
suffix = '_duplicate'
for f in noTimeSeries:
    print(f)
    print(df.shape)
    try:
        newDf = pd.read_csv(dataDir + f)
        newDf.columns = addPrefixToColumnNames(newDf.columns, '_' + f[11:].split('.')[0], 'PID')
        df = pd.merge(left=df, right=newDf, how='left', on='PID', copy=True, indicator=False)
#         dupCols = [c for c in list(df.columns) if suffix in c]
#         df = df.drop(labels=dupCols, axis=1)
        #df = df.T.drop_duplicates().T
    except Exception as err:
        print('cannot read ' + f + "\n" + str(err))
        

print(df.shape)
df2 = df.dropna(axis=1, how='all')
df2.shape
df3 = df2.loc[:, df2.apply(pd.Series.nunique) != 1]

dupCols = [c for c in list(df3.columns) if 'duplicate' in c]
df3 = df3.drop(labels=dupCols, axis=1)
df3.shape
df3.isnull().any()
r = df3.isnull().any()
r.index[r == True]
# df3.to_csv('C:\\Users\\mot16\\projects\\master\\data\\df3.csv')

