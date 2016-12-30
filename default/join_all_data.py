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
    returns (PID,RAWRES) pairs for FLU-A or FLU-B cases
'''
def getPositives(wviral):
    df = wviral[['PID', 'RAWRES']]
    df = df[df['RAWRES'].isin(['A', 'B'])]
    return df


'''
    ######################### MAIN  ###############################
'''

dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-108134\\R_analysis\\"

''' reads wconcvac from the sql output (excludes conc vaccines after infection)
     and adds dummy variables for vaccine name, a binary for conc_vac and removes other vars
'''
wconvac = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_wconvac_iso_strdate2.csv")
concvac_prefix = 'concvac_'
any_conc_var = 'any_conc_vac'
convac = add_dummies(wconvac, 'TRADNAME', 'PID', any_conc_var, concvac_prefix)

''' reads wmedic from the sql output (excludes conc vaccines after infection)
    and adds dummy variables for medicine name, a binary for med_gsk_cod and removes other vars
'''
wmedic = pd.read_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_wmedic_iso_date2.csv")
medic_prefix = 'medic_'
any_med = 'any_medicine'
medic = add_dummies(wmedic, 'GSK_COD', 'PID', any_med, medic_prefix)


''' reads wgenmd and adds dummy variables for each distinct diagnosis, a binary_var for diganosis and removes other vars'''
wgenmd = pd.read_csv(dataDir + "gsk_108134_wgenmd.csv")
diag_prefix = 'diag_'
any_diag = 'any_diagnosis'
diag = add_diognosis(wgenmd, 'DIAGTERM', 'DIAGSTAT', 'PID', any_diag, diag_prefix)

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


'''
    ######################### JOIN DATASETS ###############################
'''

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
''' joins lab-results (both culture-confirmed and rtPCR) from wviral and wili '''
df2 = pd.merge(left=df2, right=culture_viral, how='left', on='PID', copy=True, indicator=False)
df2 = pd.merge(left=df2, right=lab_result, how='left', on='PID', copy=True, indicator=False)

print(df2.shape)

'''
    ######################### Drop Selected Variables ###############################
'''


'''
Drop variables related to ATP analysis:
    ELIG_MA_wdemog, ELIM_RMA_wdemog, ELIM_SMA_wdemog, ELI_F3MA_wdemog, ELI_F4MA_wdemog, ELIM_RMA_pid,
    P_APSIDE_reaccod, P_APSITE_reaccod, EFF_VIAL_reaccod, DECISION_wconc, ELIMCRIT_wconc, NOPROTCA_wconc, LINKAE_wconc,
    ELIG_QST_welig, CRIT_ANS_welig, CRIT_NR_welig, ELI_NAM_welig, P_APROUT_wnoadm, 
'''
df2.drop(labels=['ELIG_MA_wdemog', 'ELIM_RMA_wdemog', 'ELIM_SMA_wdemog', 'ELI_F3MA_wdemog', 'ELI_F4MA_wdemog', 'ELIM_RMA_pid',
    'P_APSIDE_reaccod', 'P_APSITE_reaccod', 'EFF_VIAL_reaccod', 'DECISION_wconc', 'ELIMCRIT_wconc', 'NOPROTCA_wconc', 'LINKAE_wconc',
    'ELIG_QST_welig', 'CRIT_ANS_welig', 'CRIT_NR_welig', 'ELI_NAM_welig', 'P_APROUT_wnoadm'], axis=1, inplace=True)

'''
drop activity and internal use variables
ACTIVITY_welig, ACT_DESC_welig, ORIGIN_wlabo,  
'''
df2.drop(labels=['ACTIVITY_welig', 'ACT_DESC_welig', 'ORIGIN_wlabo'], axis=1, inplace=True)


'''
drop randomization variables:

BLOCK_NB: Randomization Block number
RND_ID: Treatment number (randomization)
'''
df2.drop(labels=['RND_ID_wdemog', 'RND_ID_welig', 'RND_ID_wpneumo', 'BLOCK_NB_pid'], axis=1, inplace=True)

'''
drop date variables

DATE_REF_wdemog, DOB_RAW_wdemog, STRT_DAT_wpneumo, STOP_DAT_wpneumo, ACTRDATE_wnpap (date of visit), LC_RDAT_wconc (last contact), BRK_RDAT_wconc (of unblinding)
'''
df2.drop(labels=['DATE_REF_wdemog', 'DOB_RAW_wdemog', 'STRT_DAT_wpneumo', 'STOP_DAT_wpneumo', 'ACTRDATE_wnpap', 'LC_RDAT_wconc', 'BRK_RDAT_wconc'], axis=1, inplace=True)

'''
drop duplicates: have similar name to the one in demog but different in number of nulls

center: CENTER_wdemog, CENTER_wpneumo, CENTER_welig
schedulte: RDE_SCHD_wdemog, RDE_SCHD_wpneumo, RDE_SCHD_welig
Country: CTY_COD_wdemog, CTY_NAM_wdemog, CTY_COD_ wpneumo, CTY_NAM_ wpneumo, CTY_COD_ welig, CTY_NAM_ welig
GROUP_NB: GROUP_NB_wdemog, GROUP_NB_welig, GROUP_NB_wpneumo, 'GRP_VIAL_wnpap'
Eligibility: ELIG_MA_welig
elimination: ELIM_SMA_pid, ELIM_SMA_welig
Treatment: TREATMNT_welig, TREATMNT_wpneumo

vaccine: VACCINE_expogn, P_CODE_expogn

vial_type: VIAL_TYP_expogn, INJECT_wnpap

vaccine side: L_SIDE_wnpap

'''
df2.drop(labels=['CENTER_wpneumo', 'CENTER_welig', 'RDE_SCHD_wpneumo', 'RDE_SCHD_welig',
                'CTY_NAM_wdemog', 'CTY_COD_wpneumo', 'CTY_NAM_wpneumo', 'CTY_COD_welig',
                'CTY_NAM_welig', 'GROUP_NB_welig', 'GROUP_NB_wpneumo', 'GRP_VIAL_wnpap',
                'ELIG_MA_welig', 'ELIM_SMA_pid', 'ELIM_SMA_welig', 'TREATMNT_welig',
                'TREATMNT_wpneumo', 'P_CODE_expogn', 'INJECT_wnpap', 'L_SIDE_wnpap'], axis=1, inplace=True)

''''
drop GROUP_NB_wdemog as it is a duplicate of treatment
'''
df2.drop('GROUP_NB_wdemog', axis=1, inplace=True)


''''
drop ETHNIC_wdemog as it is a duplicate of ETHN_NEW_wdemog
'''
df2.drop('ETHNIC_wdemog', axis=1, inplace=True)


'''
drop variables from wpneumo dataset, keep one binary variable for pneumonia
'''
wpneumo_cols = [c for c in list(df2.columns) if '_wpneumo' in c]
wpneumo_cols.remove('OUTCOME_wpneumo')
df2.drop(labels=wpneumo_cols, axis=1, inplace=True)

''' finds columns with missing values'''
r = df2.isnull().any()
print(r.index[r == True])
# df4.to_csv('C:\\Users\\mot16\\projects\\master\\data\\df3.csv')

'''
    ######################### FILL MISSING VALUES ###############################
'''
# df2.ELIG_MA_wdemog.fillna(value='NA', inplace=True)

'''
Elimination and eligibility variables
'''
# df2.ELIM_RMA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELIM_RMA_wdemog.fillna(value='N', inplace=True)
# 
# 
# df2.ELIM_SMA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELIM_SMA_wdemog.fillna(value='N', inplace=True)
# 
# 
# df2.ELI_F3MA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELI_F3MA_wdemog.fillna(value='N', inplace=True)
# 
# 
# df2.ELI_F4MA_wdemog.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELI_F4MA_wdemog.fillna(value='N', inplace=True)
# 
# 
# df2.ELIM_RMA_pid.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELIM_RMA_pid.fillna(value='N', inplace=True)

# df2.ELIM_SMA_pid.replace(to_replace=1.0, value='Y', inplace=True)
# df2.ELIM_SMA_pid.fillna(value='N', inplace=True)

''' the only missing age_cat belongs to subject with age = 66. so age_cat should be 2 for this subject Agecat = 2 where age >=50 years'''
df2.AGECAT_pid.fillna(value=2, inplace=True)

# df2.P_APSIDE_reaccod.fillna(value='NA', inplace=True)

'''
ATP analysis variable
'''
#df2.P_APSITE_reaccod.replace(to_replace=1.0, value='DELTOID', inplace=True)


'''
ATP analysis variable
'''
#df2.EFF_VIAL_reaccod.fillna(value=0, inplace=True)


flu_seasons = ['SEASON1_wphist', 'SEASON2_wphist', 'SEASON3_wphist']
concvac_cols = [c for c in list(df2.columns) if concvac_prefix in c]
medic_cols = [c for c in list(df2.columns) if medic_prefix in c]
diag_cols = [c for c in list(df2.columns) if diag_prefix in c]
other = [any_med, 'RAWRES', 'viral_res_binary' , 'rt_PCR', any_conc_var, any_diag]

cols = flu_seasons + concvac_cols + medic_cols + other
for label in cols:
    df2.loc[:, label].fillna(value='N', inplace=True)
    

''' fill other missing values with 'NA' '''
df2.fillna(value='NA', inplace=True)
print(df2.shape)


'''
    ############### Write to File ###################
'''
df2.to_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_joined_allvars.csv", index=False, quoting=QUOTE_NONNUMERIC)

''' drops conc vac and medic dummies '''
df2_ = df2.drop(labels=concvac_cols + medic_cols + diag_cols, axis=1, inplace=False)
df2_.to_csv("C:\\Users\\mot16\\projects\\master\\data\\" + "gsk_108134_joined_no_dummies.csv", index=False, quoting=QUOTE_NONNUMERIC)
