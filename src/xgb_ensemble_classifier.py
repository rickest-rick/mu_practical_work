import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from copy import deepcopy

from data_handling import release_memory


class XgbEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    TODO implement parallelized version for fit and predict
    """
    def __init__(self, classifiers, n_splits=3, **kwargs):
        """
        Model to train a stacked ensemble of classifiers whose results are
        combined with an XGB classifier.

        :author: Joschka Strüber
        :param classifiers: A list of classifier models that support the scikit
            learn syntax.
        :param n_splits:
        :param kwargs: A dict of parameters for the xgb ensemble classifier.
        """
        self.n_estimators = len(classifiers)
        self.classifiers = []
        for i in range(len(classifiers)):
            self.classifiers.append(deepcopy(classifiers[i]))
        self.n_splits = n_splits
        self.ensemble_classifier = xgb.XGBRFClassifier(**kwargs)

    def fit(self, X, y, random_state=None):
        """
        Fit the model according to the given training data.

        :author: Joschka Strüber
        :param X: array, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        :param y: array, shape = [n_samples]
            Target vector relative to X
        :param random_state:
        :return: self : object
        """
        n_samples = X.shape[0]
        X_preds = np.ndarray(shape=(n_samples, 0), dtype=float)
        # train base classifiers and save predictions
        for i, clf in enumerate(self.classifiers):
            pred_column = self.get_oof_prediction(i, X, y,
                                                  random_state=random_state)
            if pred_column.ndim == 1:
                pred_column = np.reshape(pred_column, (n_samples, 1))
            X_preds = np.concatenate((X_preds, pred_column), axis=1)
        self.ensemble_classifier.fit(X_preds, y)
        self.ensemble_classifier = release_memory(self.ensemble_classifier)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        :author: Joschka Strüber
        :param X: array, shape = [n_samples, n_features]
        :return: C : array, shape [n_samples]
            Predicted class label per sample.
        """
        n_samples = X.shape[0]
        X_preds = np.ndarray(shape=(n_samples, 0), dtype=float)

        # save predicted probabilities or predictions of base classifiers
        for i, clf in enumerate(self.classifiers):
            try:
                pred_column = clf.predict_proba(X)
            except AttributeError:
                pred_column = clf.predict(X)
                pred_column = np.reshape(pred_column, (n_samples, 1))
            X_preds = np.concatenate((X_preds, pred_column), axis=1)

        return self.ensemble_classifier.predict(X_preds)

    def score(self, X, y, sample_weights=None):
        """
        Returns the mean accuracy on the given test data and labels.

        :author: Joschka Strüber
        :param X: array, shape = [n_samples, n_features]
        :param y: array, shape = [n_samples]
        :param sample_weights: array, shape = [n_samples], optional
        :return: score, float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :author: Joschka Strüber
        :param deep: boolean, optional
        :return: params : mapping of string to any
            Parameter names mapped to their values.
        """
        param_dict = {}
        for i, clf in enumerate(self.classifiers):
            param_dict[str(i)] = clf.get_params(deep=deep)
        param_dict["ensemble"] = self.ensemble_classifier.get_params(deep=deep)
        return param_dict

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :author: Joschka Strüber
        :param params:
        :return: self
        """
        if "ensemble" in params:
            self.ensemble_classifier.set_params(**params["ensemble"])
        for i in range(self.n_estimators):
            if str(i) in params:
                self.classifiers[i].set_params(**params[str(i)])

    def get_oof_prediction(self, clf_number, X_train, y_train,
                           random_state=None):
        """
        Returns out of fold predicted probabilities or predictions for the
        classifier on a k-fold split.

        :param clf_number:
        :param X_train:
        :param y_train:
        :param random_state:
        :return:
        """
        clf = self.classifiers[clf_number]
        n_samples = X_train.shape[0]
        if "predict_proba" in dir(clf):
            n_classes = np.size(np.unique(y_train))
            oof_train = np.zeros(shape=(n_samples, n_classes))
        else:
            oof_train = np.zeros(n_samples)

        kfold = StratifiedKFold(self.n_splits, random_state=random_state)
        for i, (train, test) in enumerate(kfold.split(X_train, y_train)):
            X_tr, X_tt = X_train[train], X_train[test]
            y_tr = y_train[train]

            clf.fit(X_tr, y_tr)
            # release memory of clf and load it back
            clf = release_memory(clf)
            self.classifiers[clf_number] = clf
            # save predicted probabilities and if not available save predictions
            try:
                y_pred = clf.predict_proba(X_tt)
                oof_train[test, :] = y_pred
            except AttributeError:
                oof_train[test] = clf.predict(X_tt)
        return oof_train
