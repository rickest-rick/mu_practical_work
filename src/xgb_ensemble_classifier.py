import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class XgbEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """

    """
    def __init__(self, classifiers, **kwargs):
        """

        :author: Joschka Strüber
        :param classifiers: A list of classifier models that support the scikit
            learn syntax.
        :param kwargs: A dict of parameters for the xgb ensemble classifier.
        """
        self.n_estimators = len(classifiers)
        self.classifiers = classifiers
        self.ensemble_classifier = xgb.XGBRFClassifier(**kwargs)

    def fit(self, X, y, random_state=None):
        """

        :author: Joschka Strüber
        :param X:
        :param y:
        :param random_state:
        :return:
        """
        X_train, X_ens, y_train, y_ens = train_test_split(X, y,
                                                    random_state=random_state,
                                                    stratify=y)

        ensemble_input = np.zeros([X.ens[0], self.n_estimators])
        # train classifiers on train split of X and predict probas on ensemble
        # split
        for i, clf in enumerate(self.classifiers):
            clf.fit(X_train, y_train)
            ensemble_input[:, i] = clf.predict_proba(X_ens)

        # fit xgboost ensemble classifier on predicted probabilities
        for i, clf in enumerate(self.classifiers):
            ensemble_input[:, i] = clf.fit

    def predict(self, X):
        """

        :author: Joschka Strüber
        :param X:
        :return:
        """
        return

    def score(self, X, y, sample_weights=None):
        """

        :author: Joschka Strüber
        :param X:
        :param y:
        :param sample_weights:
        :return:
        """
        return

    def get_params(self, deep=True):
        """

        :author: Joschka Strüber
        :param deep:
        :return:
        """
        return

    def set_params(self, **params):
        """

        :author: Joschka Strüber
        :param params:
        :return:
        """
        return
