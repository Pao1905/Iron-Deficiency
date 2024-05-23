import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# read original data
data = pd.read_csv('./Data/cleaned_data.csv')


def get_pls_results(lv_path, boot_ratio_path, method='fdr_bh', p=.05):

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
    var_names = data.columns.tolist()

    names_to_be_removed = ['Group', 'Code', 'RAVLT_ListA_Delay_Recall_Time1', 'RAVLT_ListA_Delay_Recall_Time2',
                           'RAVLT_ListA_RecallDelay_Trial2']

    # remove the unnecessary columns
    var_names = [name for name in var_names if name not in names_to_be_removed]

    # add the variable names to the DataFrame as the first column
    df.insert(0, 'Variable', var_names)

    return df


# get the results
group_results = get_pls_results('./Data/PLS_Results/lv_vals.mat', './Data/bsrs.mat')

# pls from R
pls_result_path = './Data/PLS_Results/PLS_results_hsCRP.csv'
pls = pd.read_csv(pls_result_path)

# plotting
x_labels = ['Control', 'Psychiatric']
plt.figure()
sns.barplot(x='Group', y='Comp2', data=pls, estimator='mean', errorbar='se')
plt.ylabel('Group Score')
plt.xticks(ticks=[0, 1], labels=x_labels)
plt.title('PLS Significant Latent Variable')
sns.despine()
plt.show()

# plot the line graph
plt.figure()
sns.lineplot(x='Ferritin_ngperml', y='Comp2', data=pls)
plt.ylabel('Group Score')
plt.title('PLS Significant Latent Variable')
sns.despine()
plt.show()

# correlation
from scipy.stats import pearsonr

# get the correlation
pearsonr(pls['Ferritin_ngperml'], pls['Comp2'])