import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from preprocessing import psychopathology
import matplotlib.pyplot as plt

# read cleaned data
data = psychopathology
var = data.columns

# standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Check for suitability of factor analysis
chi_square_value, p_value = calculate_bartlett_sphericity(data)
kmo_all, kmo_model = calculate_kmo(data)

# # Perform PCA to get eigenvalues
# pca = PCA()
# pca.fit(data)
# eigenvalues = pca.explained_variance_
#
# # Generate random data with the same dimensions as the original data
# random_data = np.random.normal(size=data.shape)
# pca.fit(random_data)
# random_eigenvalues = pca.explained_variance_
#
# # Plot the actual eigenvalues against the random eigenvalues
# plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', label='Actual Data')
# plt.plot(range(1, len(random_eigenvalues) + 1), random_eigenvalues, 'o-', label='Random Data')
# plt.title('Parallel Analysis Scree Plot')
# plt.xlabel('Factor')
# plt.ylabel('Eigenvalue')
# plt.legend()
# plt.show()
#
# # Determine the number of factors
# num_factors = sum(eigenvalues > random_eigenvalues)
# print(f'Number of factors: {num_factors}')

# Perform factor analysis
fa = FactorAnalyzer(rotation='varimax', n_factors=4)
fa.fit(data)
loadings = fa.loadings_
loadings = pd.DataFrame(loadings, index=var)

# transform the data
transformed_data = fa.transform(data)
transformed_data = pd.DataFrame(transformed_data, columns=['Familial-Social', 'Depressive', 'Anxiety', 'Conduct'])
transformed_data.to_csv('./Data/PLS_Data/psychopathology_factors.csv', index=False)