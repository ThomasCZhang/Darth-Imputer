import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD

def EncodeLabels(labels):
  all_labels = np.unique(labels)
  encoder = LabelEncoder()
  encoder.fit(all_labels)
  encoded = encoder.transform(labels)
  return encoded

def Implement_Random_Masking(data, sample_proportions, feature_proportion, seed):
  '''
  This function takes X and the masking portion and return the new X with the same shape with missing the given portion of random data
  Args:
  - data: full data
  - sample_proportions: float
  - feature_proportion: float 
  - seed: int
  Returns:
  - masked_data: same shape as given data, missing given portion of data
  '''
  
  np.random.seed(seed)
  cols = np.random.choice(np.arange(data.shape[1]), size=int(feature_proportion*data.shape[1]))

  mask_indices = np.random.rand(*data[:, cols].shape) < sample_proportions
  masked_data = data.copy()
  masked_data_cols = masked_data[:, cols]
  masked_data_cols[mask_indices] = np.nan
  masked_data[:, cols] = masked_data_cols

  return masked_data

def HighestVarianceFeatures(data, feature_proportion):
  '''
  This function takes data and the proportion of features to keep and returns a new matrix with the proportion of
  features with highest variance.
  Input:
  data (np.ndarray)
  feature_proportion (float)

  Returns:
  the data with reduced number of features
  '''
  variances = np.array([np.var(data[:, col]) for col in range(data.shape[1])])

  n_features = round(feature_proportion*data.shape[1])

  indexes = np.argsort(variances)[-n_features:]
  explained_ratio = np.sum(variances[indexes])/np.sum(variances)
  print(f'Percentage of explained variance is: {explained_ratio}')

  return data[:, indexes]

def HighestVarianceDecomposition(data, feature_proportion, method: str='pca'):
  '''
  This function takes data and the proportion of features to keep and returns a new matrix with the proportion of
  pca features with highest variance.
  Input:
  data (np.ndarray)
  feature_proportion (float)
  method (str): "pca" or "svd"

  Returns:
  the data with reduced number of features
  '''
  n_features = round(feature_proportion*data.shape[1])

  if n_features >= min(data.shape):
    raise Exception('Number of decomposed features cannot exceed min(nrows, ncols) '+
                    'where nrows and ncols are the number rows and columns of data')
  
  if method.lower() == 'pca':
    decomposer = PCA(n_components=n_features)
  elif method.lower() == 'svd':
    decomposer = TruncatedSVD(n_components=n_features)

  decomposer.fit(data)
  explained_ratio = np.sum(decomposer.explained_variance_ratio_)
  print(f'Percentage of explained variance is: {explained_ratio}')
  return decomposer.transform(data)
