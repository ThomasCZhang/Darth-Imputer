import numpy as np
from sklearn.preprocessing import LabelEncoder

def EncodeLabels(labels):
  all_labels = np.unique(labels)
  encoder = LabelEncoder()
  encoder.fit(all_labels)
  encoded = encoder.transform(labels)
  return encoded

def Implement_Random_Masking(data, masking_portion, num_cols, seed):
  '''
  This function takes X and the masking portion and return the new X with the same shape with missing the given portion of random data
  Args:
  - data: full data
  - Masking portion: float
  - num_cols: int 
  - seed: int
  Returns:
  - masked_data: same shape as given data, missing given portion of data
  '''
  
  np.random.seed(seed)
  cols = np.random.choice(np.arange(data.shape[1]), size=num_cols)

  mask_indices = np.random.rand(*data[:, cols].shape) < masking_portion
  masked_data = data.copy()
  masked_data_cols = masked_data[:, cols]
  masked_data_cols[mask_indices] = np.nan
  masked_data[:, cols] = masked_data_cols

  return masked_data

