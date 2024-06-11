import numpy as np
import pandas as pd
import ast
import json
import openpyxl

# read xlsx file
OSPAN = pd.read_csv('./Data/OSPAN_data_5.7.24.csv', header=1)
data = pd.read_excel('./Data/CC_request_DrWorthy_05302024.xlsx')
workbook = openpyxl.load_workbook('./Data/CC_request_DrWorthy_DICTIONARY_05302024.xlsx')
sheet = workbook.active

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

# drop rows with missing values
data = data.dropna()

# # print ethnicity range
# print(data['Race'].unique())
print(data.columns)

# =============================================================================
# the following code is to select variables of interest
# =============================================================================
# average the left and right hemisphere volumes
data['Putamen'] = (data['r_Pu_mean'] + data['l_Pu_mean']) / 2
data['Caudate'] = (data['r_Cd_mean'] + data['l_Cd_mean']) / 2
data['Globus_Pallidus'] = (data['r_GP_mean'] + data['l_GP_mean']) / 2

# calculate the time difference between the two recall trials
data['RAVLT_ListA_Delay_Recall_Time1'] = pd.to_datetime(data['RAVLT_ListA_Delay_Recall_Time1'])
data['RAVLT_ListA_Delay_Recall_Time2'] = pd.to_datetime(data['RAVLT_ListA_Delay_Recall_Time2'])
data['RAVLT_ListA_RecallDelay_Trial2'] = (data['RAVLT_ListA_Delay_Recall_Time2'] -
                                          data['RAVLT_ListA_Delay_Recall_Time1']).dt.total_seconds() / 60

# remove the two columns
data = data.drop(columns=['RAVLT_ListA_Delay_Recall_Time1', 'RAVLT_ListA_Delay_Recall_Time2'])

# update the dropped columns
drop_cols.extend(['RAVLT_ListA_Delay_Recall_Time1', 'RAVLT_ListA_Delay_Recall_Time2'])


# Function to check if a cell is highlighted
def is_highlighted(cell):
    if cell.fill.start_color.index == '00000000':
        return False
    return True


# Iterate through the cells and find highlighted ones
highlighted_cells = []
for row in sheet.iter_rows():
    for cell in row:
        if is_highlighted(cell):
            highlighted_cells.append(cell)

# store the highlighted variables as variables of interest
variables_of_interest = [cell.value for cell in highlighted_cells]
additional_variables = ['Putamen', 'Caudate', 'Globus_Pallidus']
variables_of_interest.extend(additional_variables)

# if there are variables of interest that do not exist in the data, print and store them
missing_vars = [var for var in variables_of_interest if var not in data.columns and var not in drop_cols]

# select the variables of interest
variables_of_interest = [var for var in variables_of_interest if var in data.columns]
data = data[variables_of_interest]

# checkers - print the max and min values of the columns
for col in data.columns:
    print(col, data[col].max(), data[col].min())
# print the missing variables
print(f'Missing variables: {missing_vars}')

# save cleaned data
data.to_csv('./Data/cleaned_data.csv', index=False)

# =============================================================================
# the following code is to prepare data for MATLAB PLS implementation (not necessary unless needed)
# =============================================================================
# # separate by group
# psych_pls = data[data['Group'] == 'Psychiatric']
# control_pls = data[data['Group'] == 'Control']
#
# # remove the ID and Group columns
# psych_pls = psych_pls.drop(columns=['Code', 'Group'])
# control_pls = control_pls.drop(columns=['Code', 'Group'])
#
# # save
# psych_pls.to_csv('./Data/psych_pls.csv', index=False)
# control_pls.to_csv('./Data/control_pls.csv', index=False)

# behavioral pls
iron = data['Ferritin_ngperml']
behavioral = data[['CPT_OMI_T', 'CPT_COM_T']]

# save
iron.to_csv('./Data/iron.csv', index=False)
behavioral.to_csv('./Data/behavioral.csv', index=False)

# =============================================================================
#              SGT Data Cleaning (not necessary unless needed)
# =============================================================================
# # SGT
# SGT = SGT.groupby(SGT.index // 2).first()
# # remove unnecessary columns
# SGT = SGT.drop(columns=['Trial', 'QResp'])
#
# list_to_process = ['IDNum', 'Gender', 'Ethnicity', 'Race', 'Age']
# SGT[list_to_process] = SGT[list_to_process].applymap(lambda x: ast.literal_eval(x).get('Q0', None) if x else np.nan)
#
# # explode data
# SGT = SGT.explode(['React', 'Reward', 'keyResponse', 'Bank'])



