import sys
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from preprocessing import psych_without_neuro, psychopathology, neuro_mean_df, iron, variables_of_interest
from efa import transformed_data

# add path to the PLS results
result_dir = os.path.abspath('C:/Users/zuire/OneDrive/桌面/胡勉之/Texas A&M University/IronDeficiency/MATLAB/Result')
sys.path.append(result_dir)

# read original data
data = pd.read_csv('./Data/cleaned_data.csv')
CBCL_bi = pd.read_csv('./Data/PLS_Data/CBCL_bi.csv')

print(pearsonr(data['Ferritin_ngperml'], data['Hemoccue_Hb']))
print(pearsonr(data['PARS_AnxSymptCt'], data['l_Pu_mean']))
mean_brain = neuro_mean_df.mean(axis=1)
print(pearsonr(data['PARS_AnxSymptCt'], mean_brain))


def get_pls_results(lv_path, boot_ratio_path, original_df, method='fdr_bh', p=.05):

    # define the path
    lv_path = os.path.join(result_dir, lv_path)
    boot_ratio_path = os.path.join(result_dir, boot_ratio_path)

    # read lv_vals file
    lv_vals = sio.loadmat(lv_path)

    # read bootstrap ratio file
    boot_ratio = sio.loadmat(boot_ratio_path)

    u1 = lv_vals['u1'][:, 0]
    boot_ratio = boot_ratio['bsrs1']

    # combine the data with their respective columns
    result = np.column_stack((u1, boot_ratio))

    # name the columns
    df = pd.DataFrame(result, columns=['u1', 'boot_ratio'])

    # calculate the p values according to the boot_ratio
    df['p_value'] = 2 * (1 - norm.cdf(abs(df['boot_ratio'])))

    # adjust the p values for multiple comparisons
    df['p_value_adjusted'] = multipletests(df['p_value'], method=method)[1]

    # if boot_ratio is greater than 1.96, then the corresponding u1 value is significant
    df['significant'] = abs(df['p_value_adjusted']) < p

    # extract the variable names from the original data
    var_names = original_df.columns.tolist()

    # add the variable names to the DataFrame as the first column
    df.insert(0, 'Variable', var_names)

    return df


# get the results
# group_results = get_pls_results('./Data/PLS_Results/PLS_outputTaskPLSGroupBased_without_neuro_lv_vals.mat',
#                                 './Data/PLS_Results/PLS_outputTaskPLSGroupBased_without_neuro.mat',
#                                 psych_without_neuro)
neuro_iron_results = get_pls_results('PLS_Behav_neuro~iron_lv_vals.mat',
                                     'PLS_Behav_neuro~iron.mat',
                                     iron)
psychopathology_iron_results = get_pls_results('PLS_Behav_psychopathology~iron_lv_vals.mat',
                                               'PLS_Behav_psychopathology~iron.mat',
                                               iron)
cognition_iron_results = get_pls_results('PLS_Behav_cognition~iron_lv_vals.mat',
                                         'PLS_Behav_cognition~iron.mat',
                                         iron)
psychopathology_neuro_results = get_pls_results('PLS_Behav_psychopathology~neuro_lv_vals.mat',
                                                 'PLS_Behav_psychopathology~neuro.mat',
                                                 neuro_mean_df)
cognition_neuro_results = get_pls_results('PLS_Behav_cognition~neuro_lv_vals.mat',
                                          'PLS_Behav_cognition~neuro.mat',
                                          neuro_mean_df)
cognition_psychopathology_results = get_pls_results('PLS_Behav_cognition~psychopathology_lv_vals.mat',
                                                        'PLS_Behav_cognition~psychopathology.mat',
                                                        psychopathology)
cognition_CBCL_bi_results = get_pls_results('PLS_Behav_cognition~CBCL_bi_lv_vals.mat',
                                             'PLS_Behav_cognition~CBCL_bi.mat',
                                             CBCL_bi)

# # pls from R
# pls_result_path = './Data/PLS_Results/PLS_results_hsCRP.csv'
# pls = pd.read_csv(pls_result_path)
#
# # plotting
# x_labels = ['Control', 'Psychiatric']
# plt.figure()
# sns.barplot(x='Group', y='Comp2', data=pls, estimator='mean', errorbar='se')
# plt.ylabel('Group Score')
# plt.xticks(ticks=[0, 1], labels=x_labels)
# plt.title('PLS Significant Latent Variable')
# sns.despine()
# plt.show()
#
# # plot the line graph
# plt.figure()
# sns.lineplot(x='Ferritin_ngperml', y='Comp2', data=pls)
# plt.ylabel('Group Score')
# plt.title('PLS Significant Latent Variable')
# sns.despine()
# plt.show()

