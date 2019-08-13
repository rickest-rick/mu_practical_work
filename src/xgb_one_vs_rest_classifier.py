import xgboost as xgb

from sklearn.base import BaseEstimator, ClassifierMixin

from metrics import balanced_accuracy_score


class XgbOneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter_list, n_jobs=None):
        """

        :author: Joschka Strüber
        :param parameter_list:
        :param n_jobs:
        """
        self.parameter_list = parameter_list
        self.n_jobs = n_jobs

        self.estimators = []
        for parameters in self.parameter_list:
            clf = xgb.XGBClassifier(parameters)
            self.estimators.append(clf)

    def fit(self, X, y):
        """

        :author; Joschka Strüber
        :param X:
        :param y:
        :return:
        """
        if y.shape[0] != len(self.parameter_list):
            raise ValueError("Number of labels to predict ({0}) is not equal"
                             " to number of trained classifiers ({1})"
                             .format(y.shape[0], len(self.parameter_list)))

    def predict(self, X):
        """

        :author: Joschka Strüber
        :param X:
        :return:
        """
        pass

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


