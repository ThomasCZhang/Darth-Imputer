import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SingleNaNAL():
    def __init__(self,
                features:np.ndarray=None, # N x d. N = number of samples. d = number of dimensions.
                labels:np.ndarray=None, # N x c. N = number of samples. c = number of classes.
                ls_inds:np.ndarray=None, # Labeled set indicies
                ls_split:float=0.3, # Initial proportion of data to use as labeled set.
                batch_size:int=1, # batch size.
                seed:int=0, # The seed for generating the labeled and unlabeled sets
                classifier:any=None, # Classifier used.
                ) -> None:
        """
        Class for single feature active learning.
        Input:
        features (np.ndarray): N x d. N = number of samples. d = number of dimensions.
        labels (np.ndarray): N x c. N = number of samples. c = number of classes.
        ls_inds (np.ndarray): Labeled set indicies. Default is None.
        ls_split (float): Initial proportion of data to use as labeled set. Default is 0.3.
        seed (int): The seed for generating the labeled and unlabeled sets. Default is 0.
        classifier (sklearn classifier): Classifier used.
        """
        self.rng=np.random.default_rng(seed)
        self.batch_size=batch_size

        self.features=features
        self.labels=labels

        n_samples,n_dims=self.features.shape
        self.ls_split = len(ls_inds)/n_samples if ls_inds is not None else ls_split
        self.ls_inds = ls_inds if ls_inds is not None else self.SplitData()

        mask = np.ones(n_samples, dtype=bool)
        mask[self.ls_inds] = False
        self.us_inds = np.nonzero(mask)[0]
        
        self.clf=classifier
    
    def SplitData(self):
        """
        Generate the initial indicies to assign to the labeled set.
        """
        n_samples,n_dims=self.features.shape
        ls_inds = np.arange(n_samples)
        return self.rng.choice(ls_inds, round(self.ls_split*n_samples), replace=False)

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
        ls_mod = self.features[self.ls_inds].copy() # labeled set copy
        ls_mod[i,j] = value # labeled set copy with feature[i,j] set to value
        us = self.features[self.us_inds]
        us_labels = self.labels[self.us_inds]

        self.clf.fit(ls_mod, self.labels[self.ls_inds])
        probs = self.clf.predict_proba(us)
        log_gain = np.sum(-np.log(probs[:,us_labels]))
        return log_gain
    
    def ChooseNextFeatureToAdd(self):
        """
        Chooses the next missing value to replace

        Output:
        (list[tuple[int, int, float]]): List of tuples. Each tuple is of the form (i, j, val). i and j are the row 
        and column of the missing value. Float is the best value to replace the missing value with.
        """
        ls = self.features[self.ls_inds]
        querries = np.nonzero(np.isnan(ls)) # The position of the missing values. In the format of (row_inds, col_inds)
        querries = np.array([tup for tup in zip(*querries)])
        possible_vals_all_features = [np.unique(ls[:,i]) for i in range(ls.shape[1])] # Get the possible values for each feature
        
        scores = []
        for positions in querries:
            sample_scores = []
            i,j = positions # i'th sample, j'th feature
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
        for idx, positions in enumerate(querries):
            i,j = positions # i'th sample, j'th feature
            possible_vals = possible_vals_all_features[j] # All the possible values that the j'th feature can take on.
            best_val_idx = np.argmax(scores[idx]) # scores[idx] is the scores for the idx'th querry
            new_scores.append((possible_vals[best_val_idx], scores[idx][best_val_idx])) 
        return new_scores

if __name__ == '__main__':
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
    model.batch_size = 3
    print(model.ChooseNextFeatureToAdd())