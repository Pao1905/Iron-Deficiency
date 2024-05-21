import numpy as np
import pandas as pd
import ast
import json

# read xlsx file
data = pd.read_excel('./Data/CC_request_DrWorthy_05092024.xlsx')
OSPAN = pd.read_csv('./Data/OSPAN_data_5.7.24.csv', header=1)

SGT = []

with open('./Data/jatos_results_Pulled_4_10_24.txt', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        SGT.append(json_data)

SGT = pd.DataFrame(SGT)

# =============================================================================
#                               Data Cleaning
# =============================================================================
# main data
# filter out excluded participants
data = data[data['Group'] != "Excluded"]
# remove columns with too many missing values
missing = data.isna().sum().sort_values(ascending=False)
missing_cols = missing[missing > 0.9 * data.shape[0]].index

# Ensure that the columns to be dropped exist in the DataFrame
drop_cols = list(set(missing_cols) & set(data.columns)) + ['Visit']
drop_cols = [col for col in drop_cols if col in data.columns]

data = data.drop(columns=drop_cols)

# detect the missing values and replace all -999 with NaN
data = data.replace(-999.0, np.nan)
data = data.replace(-9999.0, np.nan)

# # fill missing values with 0
# data = data.fillna(0)
data = data.dropna()

# # print ethnicity range
# print(data['Race'].unique())
print(data.columns)

# print the max and min values of the columns
for col in data.columns:
    print(col, data[col].max(), data[col].min())

# save cleaned data
data.to_csv('./Data/cleaned_data.csv', index=False)

# separate by group
psych_pls = data[data['Group'] == 'Psychiatric']
control_pls = data[data['Group'] == 'Control']

# remove unnecessary columns
psych_pls = psych_pls.drop(columns=['Group', 'Code', 'RAVLT_ListA_Delay_Recall_Time1', 'RAVLT_ListA_Delay_Recall_Time2',
                                    'RAVLT_ListA_RecallDelay_Trial2'])
control_pls = control_pls.drop(columns=['Group', 'Code', 'RAVLT_ListA_Delay_Recall_Time1', 'RAVLT_ListA_Delay_Recall_Time2',
                                        'RAVLT_ListA_RecallDelay_Trial2'])

# save
psych_pls.to_csv('./Data/psych_pls.csv', index=False)
control_pls.to_csv('./Data/control_pls.csv', index=False)

# SGT
SGT = SGT.groupby(SGT.index // 2).first()
# remove unnecessary columns
SGT = SGT.drop(columns=['Trial', 'QResp'])

list_to_process = ['IDNum', 'Gender', 'Ethnicity', 'Race', 'Age']
SGT[list_to_process] = SGT[list_to_process].applymap(lambda x: ast.literal_eval(x).get('Q0', None) if x else np.nan)

# explode data
SGT = SGT.explode(['React', 'Reward', 'keyResponse', 'Bank'])



