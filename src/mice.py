import numpy as np
from sklearn.linear_model import LinearRegression

def TrainRegressorForSingleFeature(data, feat_num, samples_to_impute):
    """
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.

    Ouput:
    Fitted sklearn classifier.
    """
    clf = LinearRegression()
    target=data[:, feat_num]
    features=np.delete(data, feat_num, axis=1)

    target = np.delete(target, samples_to_impute, axis = 0)
    features = np.delete(features, samples_to_impute, axis = 0)
    return clf.fit(features,target)

def ImputeMissingValuesSingleFeature(data, feat_num, samples_to_impute):
    """
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.

    Output:
    New data with imputed values.
    """
    clf = TrainRegressorForSingleFeature(data, feat_num, samples_to_impute)

    feat_impute_samps = data[samples_to_impute, :]
    feat_impute_samps = np.delete(feat_impute_samps,feat_num,axis=1)
    imputed_feats = clf.predict(feat_impute_samps)

    data[samples_to_impute, feat_num] = imputed_feats
    return data

def InitializeMissingValues(data, feat_num, samples_to_impute, seed):
    """
    Initializes missing values. For a given feature, the missing values will be drawn from a uniform distribution 
    ranging from the observed minimum and maximum of the feature.
    Input:
    data (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    feat_num (int): The feature to train the classifier on.
    samples_to_impute (list[int]): List of row indexes corresponding to samples that we should impute.
    seed (int): The seed for the random number generator.

    Output:
    None
    """
    minimum, maximum = np.min(data[:,feat_num]), np.max(data[:,feat_num])
    rng = np.random.default_rng(seed)

    data[samples_to_impute, feat_num] = rng.uniform(minimum, maximum, len(samples_to_impute))
    
def ImputeData(orig_data, seed):
    """
    Input (np.ndarray): The (feature) data. N x s. N = number of samples. s = number of features.
    seed (int): The seed for the random number generator.

    Output:
    New data with imputed values.
    """
    n_row,n_col = orig_data.shape
    mask = np.isnan(orig_data)
    samples_to_impute = [np.where(mask[:, i]==True) for i in range(n_col)]

    data = orig_data.copy()
    # Initialize missing values with sort of random values.
    for col in range(n_col):
        if len(samples_to_impute[col]) == 0:
            continue
        InitializeMissingValues(data, col, samples_to_impute[col], seed)
    
    n_iters = 10
    for _ in range(n_iters):
        for col in range(n_col):
            if len(samples_to_impute[col]) == 0:
                continue
            ImputeMissingValuesSingleFeature(data, col, samples_to_impute[col])

    return data