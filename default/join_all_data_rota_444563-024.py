'''
Created on Oct 21, 2016

@author: mot16
'''

import pandas as pd
from os import listdir
import numpy as np
from csv import QUOTE_NONNUMERIC

def modify_column_names(columns, prefix, suffix, exclude):
    colnames = []
    for col in list(columns):
        if col == exclude:
            colnames.append(col)
        else:
            colnames.append(prefix + col + suffix)
    return colnames


'''
    add_dummies():
        1- adds dummies for "medicine_colname" variable
        2- selects PID and binary variables (drops other variables !!!)
        3- For each PID, sets the binary values to 'Y', for which data exist
        4- adds 'target_colname' variable which indicates whether any dummy variable has a value for each PID
'''
def add_dummies(dataset, medicine_colname, pid_colname, target_colname, prefix):
    dataset[medicine_colname] = dataset[medicine_colname].str.upper()
    dataset[medicine_colname] = dataset[medicine_colname].str.strip()  
    
    df = pd.get_dummies(data=dataset, prefix='', prefix_sep='', dummy_na=False, columns=[medicine_colname], sparse=False, drop_first=False)
    cols = np.r_[list(df.columns).index(pid_colname), np.arange(dataset.shape[1] - 1, df.shape[1])]
    df = df.iloc[:, cols]
    df_result = pd.DataFrame(columns=df.columns)
    df_result.PID = df.PID.unique()
    df_result.fillna('N', inplace=True)
    for i in range(0, dataset.shape[0]):
        trdname = dataset.ix[i, medicine_colname]
        pid = dataset.ix[i, pid_colname]
        df_result.loc[df_result[pid_colname] == pid, str(trdname)] = 'Y'
    
    cols = modify_column_names(df_result.columns, prefix, '', pid_colname)
    
    df_result.columns = cols
    df_result[target_colname] = 'Y'

    return df_result


'''
    add_diognosis():
        1- adds dummies for DIAGTERM variable (General medical history)
        2- Creates a dataframe with PID, and dummies
        3- For each PID, sets the dummy variable to its DIAGSTAT value
        4- adds a prefix to dummy vars
        5- adds 'target' variable which indicates whether any of dummies has a value for each PID
'''
def add_diognosis(dataset, diagterm_colname, diagstat_cloname, pid_colname, target_colname, prefix):
    # 1
    #dataset[diagterm_colname] = dataset[diagterm_colname].str.upper()
    #dataset[diagterm_colname] = dataset[diagterm_colname].str.strip()  
    
    df = pd.get_dummies(data=dataset, prefix='', prefix_sep='', dummy_na=False, columns=[diagterm_colname], sparse=False, drop_first=False)
    # 2
    dummies = np.r_[list(df.columns).index(pid_colname), np.arange(dataset.shape[1] - 1, df.shape[1]) ]
    df = df.iloc[:, dummies]
    df_result = pd.DataFrame(columns=df.columns)
    df_result.PID = df.PID.unique()
    # df_result.fillna('N', inplace=True)
    
    # 3
    for i in range(0, dataset.shape[0]):
        diagterm = dataset.ix[i, diagterm_colname]
        pid = dataset.ix[i, pid_colname]
        diagstat = dataset.ix[i, diagstat_cloname]
        df_result.loc[df_result[pid_colname] == pid, str(diagterm)] = diagstat
    
    # 4
    cols = modify_column_names(df_result.columns, prefix, '', pid_colname)
    
    # 5
    df_result.columns = cols
    df_result[target_colname] = 'Y'

    return df_result


'''
    ######################### MAIN  ###############################
'''

dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-444563-024\\444563-024-csv-rawdata\\R_raw\\"

''' read gsk_444563_024_ge_gn_no_duplicate_pid dataset and 
    keep only "TYPE" variable as test_results 
'''
test_results = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + 'gsk_444563_024_ge_gn_no_duplicate_pid.csv')
test_results = test_results[['PID', 'TYPE']]

demog = "gsk_444563_024_demog.csv"
pid = 'gsk_444563_024_pid.csv'
noTimeSeries = [demog, pid]


''' read wdemog '''
df = pd.read_csv(dataDir + demog)
noTimeSeries.remove(demog)

''' reads conc vaccination info
     and adds dummy variables for vaccine name, a binary for conc_vac and removes other vars
'''
'''
TODO: remove conc vaccinations occured after Rota infection 
'''
wconvac = pd.read_csv(dataDir + "gsk_444563_024_vacc_con.csv")
concvac_prefix = 'concvac_'
any_conc_var = 'any_conc_vac'
convac = add_dummies(wconvac, 'MOD_TRAD', 'PID', any_conc_var, concvac_prefix)

''' reads medication info from 
    and adds dummy variables for medicine name, a binary for med_gsk_cod and removes other vars
'''
'''
TODO: remove medications taken after Rota infection 
'''

wmedic = pd.read_csv(dataDir + "gsk_444563_024_medic.csv")
medic_prefix = 'medic_'
any_med = 'any_medicine'
medic = add_dummies(wmedic, 'TRADNAME', 'PID', any_med, medic_prefix)


''' reads wgenmd and adds dummy variables for each distinct diagnosis, a binary_var for diganosis and removes other vars'''
'''
TODO: using DIAGNOSI variable instead of DIAGTERM creates 912 variables instead of 13
'''
wgenmd = pd.read_csv(dataDir + "gsk_444563_024_diagnos.csv")
diag_prefix = 'diag_'
any_diag = 'any_diagnosis'
diag = add_diognosis(wgenmd, 'DIAGTERM', 'DIAGSTAT', 'PID', any_diag, diag_prefix)


'''
    ######################### JOIN DATASETS ###############################
'''

''' adds filename to columns as prefix'''   
df.columns = modify_column_names(df.columns, '', '_' + demog[15:].split('.')[0], 'PID')

''' 
joins all datasets in noTimeSeries list
'''
for f in noTimeSeries:
    print(f)
    print(df.shape)
    try:
        newDf = pd.read_csv(dataDir + f)
        newDf.columns = modify_column_names(newDf.columns, '', '_' + f[15:].split('.')[0], 'PID')
        df = pd.merge(left=df, right=newDf, how='left', on='PID', copy=True, indicator=False)

    except Exception as err:
        print('cannot read ' + f + "\n" + str(err))


'''
    ######################### Drop NULL / CONSTANT / DUPLICATE COLUMNS ###############################
'''

''' drops null variables'''
df2 = df.copy()
df2 = df2.dropna(axis=1, how='all')
print(df2.shape)

''' drops constant variables, do not exclude Null values '''
df2 = df2.loc[:, df2.apply(pd.Series.nunique, args=(False,)) != 1]
print(df2.shape)

''' drops duplicate columns '''
df2 = df2.T.drop_duplicates().T
print(df2.shape)

'''
    ######################### ADD DUMMY VARIABLES FOR MEDICATION, VACCINES, DIAGNOSIS ###############################
'''
''' joins wconcvac '''        
df2 = pd.merge(left=df2, right=convac, how='left', on='PID', copy=True, indicator=False)

''' joins wmedic '''
df2 = pd.merge(left=df2, right=medic, how='left', on='PID', copy=True, indicator=False)

''' joins wgenmd '''
df2 = pd.merge(left=df2, right=diag, how='left', on='PID', copy=True, indicator=False)

'''
    ######################### ADD OUTCOME VARIABLES ###############################
'''
df2 = pd.merge(left=df2, right=test_results, how='left', on='PID', copy=True, indicator=False)

print(df2.shape)


'''
    ######################### Drop Selected Variables ###############################
'''

df2.drop(labels=['ELIG_MA_pid', 'ELIM_RMA_pid', 'ELIM_SMA_pid', 'DOB_RAW_demog', 'GROUP_NB_pid', 'BLOCK_NB_pid',
    'CTY_NAM_pid', 'COUNTRY_pid'], axis=1, inplace=True)

'''
    ######################### Fill blank values ###############################
'''

''' fill  missing values with 'N' '''
concvac_cols = [c for c in list(df2.columns) if concvac_prefix in c]
medic_cols = [c for c in list(df2.columns) if medic_prefix in c]
diag_cols = [c for c in list(df2.columns) if diag_prefix in c]
other = [any_med, 'TYPE', any_conc_var, any_diag]

cols = diag_cols + concvac_cols + medic_cols + other
for label in cols:
    df2.loc[:, label].fillna(value='N', inplace=True)
    
df2.TYPE.replace(to_replace=['NT', 'NC', 'MS'], value='N', inplace=True)

''' fill other missing values with 'NA' '''
df2.fillna(value='NA', inplace=True)

'''
    ######################### concatenate int values with a string so not confused with continuos values in GeNie ###############################
'''
df2['temp'] = 's'
df2.CENTER_demog = df2.CENTER_demog.astype('str').str.cat(df2.temp)
df2.RACE_demog = df2.RACE_demog.astype('str').str.cat(df2.temp)
df2.drop(labels=['temp'], axis=1, inplace=True)


df2.to_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_444563_024_all_vars.csv", index=False)


'''
    ######################### ATP cohort only ###############################
'''
df2 = df2[df2.ELI_F3MA_pid != 1]



df2.to_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_444563_024_all_vars_ATP.csv", index=False)


'''
    ######################### no medicine ###############################
'''
non_medic_cols = list(set(df2.columns)- set(medic_cols))
non_medic_cols.remove(any_med)
df2 = df2[non_medic_cols]
df2.to_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_444563_024_all_vars_ATP_no_medicine.csv", index=False)


