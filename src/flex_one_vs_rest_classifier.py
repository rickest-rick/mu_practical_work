import numpy as np
import os
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.exceptions import ConvergenceWarning
from joblib import dump, load
from copy import deepcopy
from bayes_opt import BayesianOptimization
from threading import Thread

from data_handling import convert_to_int, release_memory

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class FlexOneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """
    The FlexOneVsRestClassifier is a Classifier to train, tune and evaluate
    multiple classifiers in a multi-label classification scenario.
    It is only a thin wrapper to allow the usage of different kinds of
    classifiers or different sets of hyperparameters. For all further tasks, the
    underlying classifiers can be directly accessed.
    TODO: implement parallelized version for train. tune and test with joblib
    :author: Joschka Strüber
    """

    def __init__(self, clf=None, n_estimators=None, classifiers=None,
                 feature_names=None, label_names=None):
        """
        This constructor expects either a list of classifiers or a single
        classifier and a number of times it should be deep copied.

        :author: Joschka Strüber
        :param clf: A single classifier with a fit and predict method.
        :n_estimators: int, the number of times a single classifier should be
            deep copied
        :classifiers: [], a list of classifiers with a fit and predict method.
        """
        if classifiers is not None and clf is None and n_estimators is None:
            self.classifiers = {}
            for i in range(len(classifiers)):
                self.classifiers[str(i)] = deepcopy(classifiers[i])
        elif clf is not None and n_estimators > 0:
            self.classifiers = {}
            for i in range(n_estimators):
                self.classifiers[str(i)] = deepcopy(clf)
        else:
            raise ValueError("Either choose a list of estimators or a single "
                             "estimator n_estimator times.")
        self.feature_names = feature_names
        self.label_names = label_names

    def fit(self, X, y, pred_expansion=False, ignore_nan=True):
        """
        Fit the model according to the given training data.

        :author: Joschka Strüber
        :param X: array, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        :param y:  array, shape = [n_samples]
            Target vector relative to X
        :param pred_expansion: Boolean, if True, expand the data set with
            predictions on the labels that were trained so far.
        :param ignore_nan: Boolean, default: True. Filter out all training
            samples for which a target label is marked as NaN. This means, we
            have potentially n_estimators different training sets X. One for
            each classifier.
        :return: self : object
        """
        n_labels = y.shape[1]
        if n_labels != len(self.classifiers):
            raise ValueError("Number of labels to fit ({0}) does not equal"
                             " the number of trained classifiers ({1})"
                             .format(n_labels, len(self.classifiers)))

        if pred_expansion:  # data set is modified, if pred_expansion
            X_train = np.copy(X)
        else:
            X_train = X
        # train a classifier for every set of labels in y
        for label in range(n_labels):
            y_train = y[:, label]
            # ignore features for which no prediction is available
            if ignore_nan:
                X_train = X
                label_is_nan = np.isnan(y_train)
                X_train = X_train[~label_is_nan]
                y_train = y_train[~label_is_nan]

            self.fit_label(label, X_train, y_train)
            # augment training data set with prediction on labels seen so far
            if pred_expansion:
                y_pred = self.classifiers[str(label)].predict(X_train)
                np.append(X_train, y_pred)
        return self

    def predict(self, X, pred_expansion=False):
        """
        Predict class labels for samples in X.

        :author: Joschka Strüber
        :param X:  array, shape = [n_samples, n_features]
            Samples.
        :param pred_expansion: Boolean, if True, expand the data set with
            predictions on the labels that were trained so far.
        :return: C: array, shape = [n_samples]
        """
        n_labels = len(self.classifiers)
        if pred_expansion:  # data set is modified, if pred_expansion
            X_test = np.copy(X)
        else:
            X_test = X

        y = []
        for label in range(n_labels):
            y_pred = self.classifiers[str(label)].predict(X_test)
            if pred_expansion:
                np.append(X_test, y_pred)
            y.append(y_pred)
        return np.array(y).T

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :author: Joschka Strüber
        :param deep: boolean, optional
        :return: params : mapping of string to any
            Parameter names mapped to their values.
        """
        param_dict = {}
        for label in range(len(self.classifiers)):
            param_dict[str(label)] = self.classifiers[str(label)].get_params(
                deep=deep)
        return param_dict

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :author: Joschka Strüber
        :param params:
        :return: self
        """
        for label in params:
            clf = self.classifiers[label]
            clf.set_params(**params[label])

    def score(self, X, y, sample_weight=None):
        """
        Compute the mean accuracy on the given test data and labels. In this
        multi-label classification case, this is the subset accuracy which is a
        harsh metric since we require for each sample that  each label set be
        correctly predicted.

        :author: Joschka Strüber
        :param X: array, shape = [n_samples, n_features]
            The test samples.
        :param y: array, shape = [n_samples, n_output]
            The true labels for X. If values are np.nan, these are ignored
            counted as correctly predicted.
        :param sample_weight: array, shape = [n_samples]
            sample weights
        :return: mean accuracy. float
        """
        y_pred = self.predict(X)
        count_correct = 0
        n_pred = y_pred.shape[1]
        # test every prediction for subset equality
        for pred_set in n_pred:
            is_nan = np.isnan(y[pred_set])
            y_pred[pred_set][is_nan] = np.nan
            if np.allclose(y_pred[pred_set], y[pred_set], equal_nan=True):
                count_correct += 1 if sample_weight is None else \
                    sample_weight[pred_set]
        return float(count_correct) / n_pred

    def tune_hyperparams(self, X, y, bounds, metric, init_points=10, n_iter=20,
                         int_params=None, groups=None):
        """
        todo: implement and comment
        :author: Joschka Strüber
        :param X:
        :param y:
        :param bounds:
        :param metric:
        :param n_iter:
        :param init_points:
        :param int_params:
        :param groups:
        :return:
        """
        for i in range(len(self.classifiers)):
            y_i = y[:, i]
            label_is_nan = np.isnan(y_i)
            X_i = X[~label_is_nan]
            y_i = y_i[~label_is_nan]
            groups_no_nan = groups[~label_is_nan]
            # check for empty target
            if y_i.size == 0:
                continue
            bayes_opt = BayesianOptimization(self.get_evaluate(i, X_i, y_i,
                                                               metric,
                                                               int_params,
                                                               groups_no_nan),
                                             pbounds=bounds)
            bayes_opt.maximize(init_points=init_points, n_iter=n_iter)
            if int_params is not None:
                params = convert_to_int(bayes_opt.max['params'], int_params)
            else:
                params = bayes_opt.max['params']
            self.classifiers[str(i)].set_params(**params)

    def get_evaluate(self, label_index, X, y, metric, int_params, groups):
        def evaluate(**kwargs):
            if int_params is not None:
                kwargs = convert_to_int(kwargs, int_params)
            clf = deepcopy(self.classifiers[str(label_index)])
            if groups is None:
                kfold = StratifiedKFold(n_splits=3, random_state=42)
            else:
                kfold = GroupKFold(n_splits=3)
            clf.set_params(**kwargs)
            sum_scores = 0.0
            for train_index, test_index in kfold.split(X, y, groups=groups):
                X_tr, X_te = X[train_index], X[test_index]
                y_tr, y_te = y[train_index], y[test_index]

                fitting_process = Thread(target=self.fitting,
                                         args=(clf, X_tr, y_tr, "tmp.joblib"))
                fitting_process.start()
                fitting_process.join()
                clf = load("tmp.joblib")
                os.remove("tmp.joblib")

                y_pred = clf.predict(X_te)
                sum_scores += metric(y_te, y_pred)
            return sum_scores / 3
        return evaluate

    def fit_label(self, label, X, y):
        """
        Method to encapsulate the training process of a single XGBoost
        Classifier.
        :param label: The label number as int. Shows which column of y
            should be used as target.
        :param X: All features as numpy array.
        :param y: A single set of target labels as numpy array.
        :return: None
        """
        clf = self.classifiers[str(label)]
        clf.fit(X, y)

        clf = release_memory(clf)
        self.classifiers[str(label)] = clf

    @staticmethod
    def fitting(clf, X, y, path):
        clf.fit(X, y)
        dump(clf, path)
        del clf

