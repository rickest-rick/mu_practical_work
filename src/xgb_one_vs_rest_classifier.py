import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class XgbOneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter_list, n_jobs=None, tree_method="auto"):
        """

        :author: Joschka Strüber
        :param parameter_list:
        :param n_jobs:
        :tree_method:
        """
        self.parameter_list = parameter_list
        self.n_jobs = n_jobs
        self.tree_method = tree_method

        self.estimators = []
        for parameters in self.parameter_list:
            clf = xgb.XGBClassifier(parameters, tree_method=self.tree_method)
            self.estimators.append(clf)

    def fit(self, X, y):
        """

        :author; Joschka Strüber
        :param X:
        :param y:
        :return:
        """
        n_labels = y.shape[0]
        if n_labels != len(self.parameter_list):
            raise ValueError("Number of labels to predict ({0}) is not equal"
                             " to number of trained classifiers ({1})"
                             .format(n_labels, len(self.parameter_list)))

        # train a classifier for every label in y
        for label in range(n_labels):
            # remove values for which NaN labels exist
            y_label = y[label]
            is_NaN = np.isnan(y_label)
            X_clean = X[~is_NaN]
            y_clean = y[~is_NaN]

            self.estimators[label].fit(X_clean, y_clean)
        return self

    def predict(self, X):
        """

        :author: Joschka Strüber
        :param X:
        :return:
        """
        n_labels = len(self.parameter_list)
        y = []

        for label in range(n_labels):
            y.append(self.estimators[label].predict(X))
        return y

    def score(self, X, y, sample_weight=None):
        """

        :author: Joschka Strüber
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        pass

    def get_params(self, deep=True):
        """

        :author: Joschka Strüber
        :param deep:
        :return:
        """
        return self.parameter_list

    def set_params(self, parameter_list):
        """

        :author: Joschka Strüber
        :param parameter_list:
        :return:
        """
        pass

    def tune_hyperparams(self, metric, **kwargs):
        """

        :author: Joschka Strüber
        :metric:
        :param kwargs:
        :return:
        """
        pass


