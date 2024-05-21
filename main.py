import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

# read cleaned data
data = pd.read_csv('./Data/cleaned_data.csv')

# =============================================================================
#                               Kernel PLS
# =============================================================================
# split data into train and test
