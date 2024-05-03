import numpy as np
import sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class SampleBasedImputation():

    def __init__(self, baseline_model, utility_func):
        self.model = baseline_model
        self.utility_func = utility_func

        return

    def select(self, features, labels, missing_data, test_labels):

        utilities = []
        for d in missing_data:
            utilities.append(self.utility_func(d))

        return np.argmax(utilities)

class RandomSelection():

    def __init__(self):
        return

    def select(self, features, labels, missing_data, test_labels):

        return np.random.randint(len(missing_data))


class ImputationSimulation():

    def __init__(self, features, labels, split, percent_missing, model,
    methods, evaluation_mode = "all", fold = None):
        self.features = features
        self.labels = labels
        self.percent_missing = percent_missing
        self.model = model
        self.methods = methods
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.split = split
        self.evaluation_mode = evaluation_mode

        if self.evaluation_mode == "cross validation":
            if fold is None:
                self.fold = 3
            else:
                self.fold = fold

        return

    def add_and_remove(self, feat, label, feat_full, label_full,
    unlabel, test_labels, x, indexes):

        """
        Helper function to deal with adding samples
        from the unlabeled set to the labeled set.

        L: the set of intances with labels
        U: the set of instances without labels

        Input: feat: the current set of features in L
               label: the current set of labels in L
               feat_full: all the available features
               label_full: every label (these are unknown in practice)
               unlabel: the current set of features in U (missing features)
               x: the index to remove from unlabel
               indexes: index list such that index[x] is the index of the
                        point in the feat_full and label_full objects

        Output: feat with a single sample added
                label with a single sample added
                unlabel with a single sample removed
                indexes with a single index removed

        """

        feat_shape = feat.shape[1]

        unlabel = np.delete(unlabel, x, axis = 0)
        test_labels = np.delete(test_labels, x, axis = 0)

        label_to_get = np.array(indexes)[x]
        indexes = np.delete(indexes, x)

        label_to_add = label_full[label_to_get]
        feature_to_add = feat_full[label_to_get]

        if len(label_to_add.shape) == 0:
            label_to_add = np.reshape(label_to_add, (1, 1))[0]
            feature_to_add = np.reshape(feature_to_add, (1, feat_shape))
            label = np.append(label, label_to_add, axis = 0)
            feat = np.append(feat, feature_to_add, axis = 0)
        else:
            label = np.append(label, label_to_add)
            feat = np.vstack((feat, feature_to_add))

        return feat, label, unlabel, test_labels, indexes

    def randomly_na(self, data, percent_missing):

        if percent_missing > 1:
            percent_missing = percent_missing / 100

        mask = np.random.rand(data.shape[0], data.shape[1])
        mask = mask < percent_missing
        data = (np.where(mask, np.nan, data))


        return data

    def evaluate(self, features_imputed, labels_imputed, model, features_norm, means, stdvs):

        if self.evaluation_mode == "all":
            accuracy = self.accuracy_on_all_samples(self.features,
             self.labels, features_imputed, labels_imputed, model)
            means.append(accuracy)

        if self.evaluation_mode == "cross validation":
            scores = self.cross_validation(features_norm, labels_imputed.ravel(), model)

            if np.isnan(np.mean(scores)):
                scores = self.accuracy_on_all_samples(self.features,
                 self.labels, features_imputed, labels_imputed, model)

            means.append(np.mean(scores))
            stdvs.append(np.std(scores))


        return means, stdvs




    def run_simulation(self):

        # Make initial set with no missing data
        n = len(self.features)
        initial_set_indexes, missing_data_indexes = self.split(self.features, self.labels)

        features_imputed = self.features[initial_set_indexes]
        labels_imputed = self.labels[initial_set_indexes]

        missing_features = self.randomly_na(self.features, self.percent_missing)

        missing_features = self.features[missing_data_indexes]
        test_labels = self.labels[missing_data_indexes]

        all_means = []
        all_stdvs = []

        # Run simulation with the same initial set for many
        # possible imputation query selection methods
        for selection_method in self.methods:

            model = self.model().fit(features_imputed, labels_imputed.ravel())
            indexes = missing_data_indexes.copy()

            means = []
            stdvs = []

            while len(labels_imputed) < n: # Stop at n - 1, since there will be no disagreement on the last sample

                features_norm = self.scaler.fit(features_imputed).transform(features_imputed)
                model = self.model().fit(features_norm, labels_imputed.ravel())

                means, stdvs = self.evaluate(features_imputed, labels_imputed, model, features_norm, means, stdvs)

                x = selection_method.select(features_imputed, labels_imputed,
                 missing_features, test_labels)

                # Add x to labeled_set, remove from unlabeled
                features_imputed, labels_imputed, missing_features, test_labels, indexes = self.add_and_remove(features_imputed,
                                                                                   labels_imputed,
                                                                                   self.features,
                                                                                   self.labels,
                                                                                   missing_features,
                                                                                   test_labels,
                                                                                   x,
                                                                                   indexes)


            
            all_means.append(means)
            all_stdvs.append(stdvs)

        return all_means, all_stdvs

    def cross_validation(self, features, labels, model):

        """
        Split the dataset into train and test, compute the 5 fold cross validation.

        """

        scores = cross_val_score(model, features, labels, cv = self.fold)

        return scores

    def calc_loss(self, features_imputed_norm, labels_imputed, model):
        model_ = model.fit(features_imputed_norm, labels_imputed)
        pred = model_.predict(features_imputed_norm)
        return np.sum(pred == labels_imputed) / len(pred)


    def accuracy_on_all_samples(self, missing_features_imputed,
    test_labels, known_features, train_labels, model):
        known_features_norm = self.scaler.fit(known_features).transform(known_features)
        missing_features_imputed_norm = self.scaler.fit(missing_features_imputed).transform(missing_features_imputed)
        pred = model.predict(missing_features_imputed_norm)

        return np.sum(pred == test_labels) / len(pred)


class RandomSplit():
    def __init__(self, split_percent):
        self.split = split_percent

    def __call__(self, features, labels):
        all_ind = [i for i in range(len(features))]
        n = len(all_ind)

        indexes_samples_labeled = np.random.choice(all_ind, size = n, replace = False)

        # Keep track of indexes in unlabeled data so we add the correct points
        label_set_indexes = indexes_samples_labeled[0 : int(self.split * n)]
        indexes = indexes_samples_labeled[int(self.split * n) : n]

        for label in np.unique(labels):
            if np.sum(labels[label_set_indexes] == label) == 0:
                return self.__call__(features, labels)

        return label_set_indexes, indexes

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

def str_to_int_labels(labels):

    strs = np.unique(labels)

    for i in range(len(strs)):
        labels = np.where(labels == strs[i], i, labels)

    return labels.astype(np.int32)

if __name__ == "__main__":
    data = pd.read_csv("fertility_Diagnosis.txt")

    matrix = np.array(data)
    labels = matrix[:, -1]
    features = matrix[:, 0:9]

    labels = str_to_int_labels(labels)

    split = RandomSplit(split_percent = 0.1)

    percent_missing = 0.1

    model = LogisticRegression

    methods = [RandomSelection()]

    impute = ImputationSimulation(features, labels, split, percent_missing, model, methods, "cross validation")

    predicted = LogisticRegression().fit(features, labels).predict(features)
    print(np.mean(predicted == labels))

    means, stds = impute.run_simulation()

    print(len(matrix))
    print(means)
