'''
Created on Oct 21, 2016

@author: mot16
'''

import pandas as pd
from os import listdir
import numpy as np

def addPrefixToColumnNames(columns, prefix, exclude):
    colnames = []
    for col in list(columns):
        if col == exclude:
            colnames.append(col)
        else:
            colnames.append(col + prefix)
    return colnames


'''
    add_dummies():
        1- binaries TRADNAME variable
        2- selects PID and binary variables (drops other variables !!!)
        3- For each PID, sets the binary values to 'Y', for which data exist
        4- adds 'conc_vac' variable which indicates whether subject had conc_vaccination
'''
def add_dummies(dataset, medicine_colname, pid_colname, target_colname, prefix):
    
    df = pd.get_dummies(data=dataset, prefix='', prefix_sep='', dummy_na=False, columns=[medicine_colname], sparse=False, drop_first=False)
    cols = np.r_[list(dataset.columns).index(pid_colname), np.arange(dataset.shape[1] - 1, df.shape[1])]
    df = df.iloc[:, cols]
    df_result = pd.DataFrame(columns=df.columns)
    df_result.PID = df.PID.unique()
    df_result.fillna('N', inplace=True)
    for i in range(0, df.shape[1]):
        trdname = dataset.ix[i, medicine_colname]
        pid = dataset.ix[i, pid_colname]
        df_result.loc[df_result[pid_colname] == pid, trdname] = 'Y'
    
    cols = addPrefixToColumnNames( df_result.columns, prefix, pid_colname)
    
    df_result.columns = cols
    df_result[target_colname] = 'Y'

    return df_result

'''

main section

'''
dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-108134\\R_analysis\\"

wconvac = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_wconvac_iso_strdate2.csv")
convac = add_dummies(wconvac, 'TRADNAME', 'PID', 'concvac', 'concvac_')

dataFiles = [f for f in listdir(dataDir) if f.count('_info.csv') == 0]
 
for f in list(dataFiles):
    with open(dataDir + f) as myfile:
        head = next(myfile)
    if head.split(sep=',').count('PID') == 0:
        dataFiles.remove(f)

noTimeSeries = ['gsk_108134_expogn.csv', 'gsk_108134_pid.csv', 'gsk_108134_reaccod.csv',
                'gsk_108134_wconc.csv', 'gsk_108134_wdemog.csv', 'gsk_108134_welig.csv',
                'gsk_108134_wlabo.csv', 'gsk_108134_wnoadm.csv', 'gsk_108134_wnpap.csv',
                'gsk_108134_wphist.csv', 'gsk_108134_wpneumo.csv', 'gsk_108134_wsolpre.csv']

timeSeries = list(set(dataFiles) - set(noTimeSeries))

labResults = 'gsk_108134_wviral.csv'
demog = "gsk_108134_wdemog.csv"


''' read wdemog as first dataset and removes it from noTimeSeries'''
df = pd.read_csv(dataDir + demog)
noTimeSeries.remove(demog)

''' adds filename to columns as prefix'''   
df.columns = addPrefixToColumnNames(df.columns, '_' + demog[11:].split('.')[0], 'PID')

suffix = '_duplicate'

''' 
joins all datasets in noTimeSeries list
'''
for f in noTimeSeries:
    print(f)
    print(df.shape)
    try:
        newDf = pd.read_csv(dataDir + f)
        newDf.columns = addPrefixToColumnNames(newDf.columns, '_' + f[11:].split('.')[0], 'PID')
        df = pd.merge(left=df, right=newDf, how='left', on='PID', copy=True, indicator=False)

    except Exception as err:
        print('cannot read ' + f + "\n" + str(err))

''' joins wconcvac '''        
df = pd.merge(left=df, right=convac, how='left', on='PID', copy=True, indicator=False)

''' TODO: join wmedic '''

''' TODO: join lab-results (both culture-confirmed and rtPCR) from wviral and wili '''

print(df.shape)
df2 = df.dropna(axis=1, how='all')
print(df2.shape)
df3 = df2.loc[:, df2.apply(pd.Series.nunique) != 1]

dupCols = [c for c in list(df3.columns) if 'duplicate' in c]
df3 = df3.drop(labels=dupCols, axis=1)
print(df3.shape)
df3.isnull().any()
r = df3.isnull().any()
r.index[r == True]
# df3.to_csv('C:\\Users\\mot16\\projects\\master\\data\\df3.csv')

