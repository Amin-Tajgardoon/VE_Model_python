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
        df_result.loc[df_result[pid_colname] == pid, str(trdname)] = 'Y'
    
    cols = modify_column_names(df_result.columns, prefix, '', pid_colname)
    
    df_result.columns = cols
    df_result[target_colname] = 'Y'

    return df_result

'''
    returns (PID,RAWRES) pairs for FLU-A or FLU-B cases
'''
def getPositives(wviral):
    df = wviral[['PID', 'RAWRES']]
    df = df[df['RAWRES'].isin(['A', 'B'])]
    return df


'''

main section

'''
dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-108134\\R_analysis\\"

''' reads wconcvac from the sql output and adds dummy variables for vaccine name, a binary for conc_vac and removes other vars'''
wconvac = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_wconvac_iso_strdate2.csv")
concvac_prefix = 'concvac_'
convac = add_dummies(wconvac, 'TRADNAME', 'PID', 'concvac', concvac_prefix)


''' reads wmedic from the sql output and adds dummy variables for medicine name, a binary for med_gsk_cod and removes other vars'''
wmedic = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_wmedic_iso_date2.csv")
medic_prefix = 'medic_'
medic = add_dummies(wmedic, 'GSK_COD', 'PID', 'med_gsk_cod', medic_prefix)

noTimeSeries = ['gsk_108134_expogn.csv', 'gsk_108134_pid.csv', 'gsk_108134_reaccod.csv',
                'gsk_108134_wconc.csv', 'gsk_108134_wdemog.csv', 'gsk_108134_welig.csv',
                'gsk_108134_wlabo.csv', 'gsk_108134_wnoadm.csv', 'gsk_108134_wnpap.csv',
                'gsk_108134_wphist.csv', 'gsk_108134_wpneumo.csv', 'gsk_108134_wsolpre.csv']


''' read wviral dataset and retrieve positive cases as A or B and add a binary column for target variable '''
wviral = 'gsk_108134_wviral.csv'
culture_viral = getPositives(pd.read_csv(dataDir + wviral))
culture_viral['viral_res_binary'] = 'P'

''' reads lab_result table containg rt_PCR, H1, H3, B (MATCH, UNMATCH, IR, NULL) related to culture-confirmed results '''
lab_result = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "lab_result.csv")


''' read wdemog as first dataset and removes it from noTimeSeries'''
demog = "gsk_108134_wdemog.csv"
df = pd.read_csv(dataDir + demog)
noTimeSeries.remove(demog)

''' adds filename to columns as prefix'''   
df.columns = modify_column_names(df.columns, '', '_' + demog[11:].split('.')[0], 'PID')

''' 
joins all datasets in noTimeSeries list
'''
for f in noTimeSeries:
    print(f)
    print(df.shape)
    try:
        newDf = pd.read_csv(dataDir + f)
        newDf.columns = modify_column_names(newDf.columns, '', '_' + f[11:].split('.')[0], 'PID')
        df = pd.merge(left=df, right=newDf, how='left', on='PID', copy=True, indicator=False)

    except Exception as err:
        print('cannot read ' + f + "\n" + str(err))

''' joins wconcvac '''        
df = pd.merge(left=df, right=convac, how='left', on='PID', copy=True, indicator=False)

''' joins wmedic '''
df = pd.merge(left=df, right=medic, how='left', on='PID', copy=True, indicator=False)

''' joins lab-results (both culture-confirmed and rtPCR) from wviral and wili '''
df = pd.merge(left=df, right=culture_viral, how='left', on='PID', copy=True, indicator=False)
df = pd.merge(left=df, right=lab_result, how='left', on='PID', copy=True, indicator=False)


print(df.shape)

''' drops null variables'''
df2 = df.dropna(axis=1, how='all')
print(df2.shape)

''' drops constant variables, do not exclude Null values '''
df3 = df2.loc[:, df2.apply(pd.Series.nunique, args=(False,)) != 1]
print(df3.shape)

''' drops duplicate columns '''
# dupCols = [c for c in list(df3.columns) if 'duplicate' in c]
# df3 = df3.drop(labels=dupCols, axis=1)
# print(df3.shape)
df4 = df3.T.drop_duplicates().T
print(df4.shape)

''' finds columns with missing values'''
r = df4.isnull().any()
print(r.index[r == True])
# df4.to_csv('C:\\Users\\mot16\\projects\\master\\data\\df3.csv')
df5 = df4.copy()

df5.drop(labels=['RDE_SCHD_welig', 'RND_ID_welig', 'GROUP_NB_welig', 'CENTER_welig', 'CTY_COD_welig', 'CTY_NAM_welig', 'ELIG_MA_welig', 'ELIM_SMA_welig', 'TREATMNT_welig'], axis=1, inplace=True)
df5.drop(labels=['RDE_SCHD_wpneumo', 'RND_ID_wpneumo', 'GROUP_NB_wpneumo', 'CENTER_wpneumo', 'CTY_COD_wpneumo', 'CTY_NAM_wpneumo', 'TREATMNT_wpneumo'], axis=1, inplace=True)

df5.ELIG_MA_wdemog.fillna(value='NA', inplace=True)

df5.ELIM_RMA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELIM_RMA_wdemog.fillna(value='N', inplace=True)


df5.ELIM_SMA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELIM_SMA_wdemog.fillna(value='N', inplace=True)


df5.ELI_F3MA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELI_F3MA_wdemog.fillna(value='N', inplace=True)


df5.ELI_F4MA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELI_F4MA_wdemog.fillna(value='N', inplace=True)


df5.ELIM_RMA_pid.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELIM_RMA_pid.fillna(value='N', inplace=True)

df5.ELIM_SMA_pid.replace(to_replace=1.0, value='Y', inplace=True)
df5.ELIM_SMA_pid.fillna(value='N', inplace=True)

''' the only missing age_cat belogs to subject with age = 66. so age_cat should be 2 for this subject Agecat = 2 where age >=50 years'''
df5.AGECAT_pid.fillna(value=2, inplace=True)


df5.P_APSIDE_reaccod.fillna(value='NA', inplace=True)


df5.P_APSITE_reaccod.replace(to_replace=1.0, value='DELTOID', inplace=True)
df5.P_APSITE_reaccod.fillna(value='NA', inplace=True)


df5.EFF_VIAL_reaccod.fillna(value=0, inplace=True)


df5.DECISION_wconc.fillna(value='NA', inplace=True)


df5.BRK_RDAT_wconc.fillna(value='NA', inplace=True)

df5.LC_GC_wconc.fillna(value='NA', inplace=True)


df5.LC_RDAT_wconc.fillna(value='NA', inplace=True)


df5.NOPROTCA_wconc.fillna(value='NA', inplace=True)


df5.PREGNANT_wconc.fillna(value='NA', inplace=True)

flu_seasons = ['SEASON1_wphist', 'SEASON2_wphist', 'SEASON3_wphist']
concvac_cols = [c for c in list(df5.columns) if concvac_prefix in c]
medic_cols = [c for c in list(df5.columns) if medic_prefix in c]
other = ['med_gsk_cod', 'RAWRES', 'viral_res_binary' , 'rt_PCR', 'concvac']

cols = flu_seasons + concvac_cols + medic_cols + other
for label in cols:
    df5.loc[:, label].fillna(value='N', inplace=True)
    
# df5.loc[:, flu_seasons + concvac_cols + medic_cols + other].fillna(value= 'N', inplace = True)

''' fill other missing values with 'NA' '''
df5.fillna(value='NA', inplace=True)

print(df5.shape)
# df6 = df5.T.drop_duplicates().T
# print(df6.shape)
df5.to_csv(dataDir + "gsk_108134_joined_allvars.csv", index=False, quoting=QUOTE_NONNUMERIC)

''' drops conc vac and medic dummies '''
df6 = df5.drop(labels=concvac_cols + medic_cols, axis=1, inplace=False)
df6.to_csv(dataDir + "gsk_108134_joined_no_dummies.csv", index=False, quoting=QUOTE_NONNUMERIC)










