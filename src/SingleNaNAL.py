import numpy as np

import basemodel

class SingleNaNAL(basemodel.Dataset):
    def __init__(self,
                x:np.ndarray=None, # N x d. N = number of samples. d = number of dimensions.
                y:np.ndarray=None, # N x c. N = number of samples. c = number of classes.
                seed:int=174, # The seed for generating the labeled and unlabeled sets
                test_portion:float=0.3, # Initial proportion of data to use as labeled set.
                ls_inds:np.ndarray=None, # Labeled set indicies
                batch_size:int=1, # batch size.
                classifier:any=None, # Classifier used.
                ) -> None:
        """
        Class for single feature active learning.
        Input:
        features (np.ndarray): N x d. N = number of samples. d = number of dimensions.
        labels (np.ndarray): N x c. N = number of samples. c = number of classes.
        seed (int): The seed for generating the labeled and unlabeled sets. Default is 0.
        ls_split (float): Initial proportion of data to use as labeled set. Default is 0.3.
        ls_inds (np.ndarray): Labeled set indicies. Default is None.
        batch_size (int): Number of missing features to impute each iteration.
        classifier (sklearn classifier): Classifier used.
        """
        super().__init__(x, y, seed)

        self.test_portion = len(ls_inds)/self.len() if ls_inds is not None else test_portion
        self.train_idx, self.test_idx = self.split_data(self.test_portion)        
        self.clf=classifier
        self.set_batchsize(batch_size)

    def set_batchsize(self, batch_size):
        self.batch_size = batch_size

    def LogGainOneValue(self, i:int, j:int, value:float):
        """
        Calculates the log gain from replacing missing value at position (i,j) with value.
        Input:
        i (int): The row of the missing value
        j (int): The column of the missing value
        value (float): The value to try for the missing value at (i,j)

        Output:
        The log gain from replacing the missing feature with value.
        """
        ls_mod = self.x[self.train_idx].copy() # labeled set copy
        ls_mod[i,j] = value # labeled set copy with feature[i,j] set to value
        us = self.x[self.test_idx]
        us_labels = self.y[self.test_idx]

        self.clf.fit(ls_mod, self.y[self.train_idx])
        probs = self.clf.predict_proba(us)
        log_gain = np.sum(-np.log(probs[:,us_labels]))
        return log_gain
    
    def ChooseNextFeatureToAdd(self, batch_size:int=None):
        """
        Chooses the next missing value to replace
        Input:
        batch_size(int): The number of features to impute at once. 

        Output:
        (list[tuple[int, int, float]]): List of tuples. Each tuple is of the form (i, j, val). i and j are the row 
        and column of the missing value. Float is the best value to replace the missing value with.
        """
        if batch_size is not None: self.set_batchsize(batch_size)
        ls = self.x[self.train_idx]
        querries = np.nonzero(np.isnan(ls)) # The position of the missing values. In the format of (row_inds, col_inds)
        querries = np.array([tup for tup in zip(*querries)])
        possible_vals_all_features = [np.unique(ls[~np.isnan(ls[:,i]),i]) for i in range(ls.shape[1])] # Get the possible values for each feature

        scores = []
        for (i,j) in querries: # i'th sample, j'th feature
            sample_scores = []
            possible_vals = possible_vals_all_features[j]
            for val in possible_vals:
                lg = self.LogGainOneValue(i,j,val)
                sample_scores.append(lg/len(possible_vals))
            scores.append(sample_scores)

        processed_scores = self.ConvertScoresToValScore(querries, scores, possible_vals_all_features) # List of tuples. (value, score).
        sorted_idxs = np.argsort([tup[1] for tup in processed_scores])
        next_positions = querries[sorted_idxs[-self.batch_size:]]
        next_values = [tup[0] for tup in processed_scores[-self.batch_size:]]

        res = tuple((*pos, val) for pos, val in zip(next_positions, next_values))
        breakpoint()
        return res
        
    def ConvertScoresToValScore(self, querries, scores, possible_vals_all_features):
        """
        Prepares the scores list such that it can be used for value selection via sorting and indexing.
        Input:
        querries (list[tuple]): The locations of the missing values. Each tuple is of the form (i,j) where i is the
        i'th sample and j is the j'th feature.

        scores (list[list[float]]]): A list of list of floats. Each list of floats correspond to the possible
        log gains fro one missing value.

        possible_vals_all_features list[np.ndarray]: A list of numpy ndarrays. The i'th numpy array is the collection
        of possible values for the i'th feature.

        Output:
        list[tuples(float, float)] Each tuple corresponds to the best possible (value, score) for a single missing 
        value.
        """
        new_scores = []
        for idx, (i,j) in enumerate(querries):
            possible_vals = possible_vals_all_features[j] # All the possible values that the j'th feature can take on.
            best_val_idx = np.argmax(scores[idx]) # scores[idx] is the scores for the idx'th querry
            new_scores.append((possible_vals[best_val_idx], scores[idx][best_val_idx])) 
        return new_scores

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier

    # Load Data
    # Synthetic Data
    seed = 2024
    rng = np.random.default_rng(seed)

    n_samples, n_features = 100, 15
    data = rng.integers(0, 5, size=(n_samples, n_features))
    labels = rng.integers(0, 3,size=n_samples)
    print(data.shape, labels.shape)

    mask=np.zeros(n_samples*n_features, dtype=int)
    mask[:int(n_samples*n_features*0.2)]=1
    rng.shuffle(mask)
    mask = mask.astype(bool)
    mask = mask.reshape(n_samples, n_features)
    
    data = data.astype(float)
    data[mask] = np.nan
    
    # Test making SingleFeatureAL.
    model = SingleNaNAL(data, labels, classifier=RandomForestClassifier())
    res = model.ChooseNextFeatureToAdd(3)

    print(res)