import xgboost as xgb
import numpy as np
import shutil
import os
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from math import sqrt
from copy import deepcopy

from data_handling import load_user_data, load_some_user_data, \
    split_features_labels, user_train_test_split
from metrics import balanced_accuracy_score


class FlexOneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """
    The FlexOneVsRestClassifier is a Classifier to train, tune and evaluate
    multiple classifiers in a multilabel classification scenario.
    It is only a thin wrapper to allow the usage of different kinds of
    classifiers or different sets of hyperparameters. For all further tasks, the
    underlying classifiers can be directly accessed.

    :author: Joschka Strüber
    """

    def __init__(self, clf=None, n_estimators=None, classifiers=None):
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
            self.classifiers = classifiers
        elif clf is not None and n_estimators > 0:
            self.classifiers = []
            for i in range(n_estimators):
                self.classifiers.append(deepcopy(clf))
        else:
            raise ValueError("Either choose a list of estimators or a single "
                             "estimator n_estimator times.")

    def fit(self, X, y, pred_expansion=False, ignore_nan=False):
        """

        :author: Joschka Strüber
        :param X:
        :param y:
        :param pred_expansion: If True, expand the data set with predictions on
            the labels that were trained so far.
        :param ignore_nan:
        :return:
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
                y_pred = self.classifiers[label].predict(X_train)
                np.append(X_train, y_pred)
        return self

    def predict(self, X, pred_expansion=False):
        """

        :author: Joschka Strüber
        :param X:
        :param pred_expansion:
        :return:
        """
        n_labels = len(self.classifiers)
        if pred_expansion:  # data set is modified, if pred_expansion
            X_test = np.copy(X)
        else:
            X_test = X

        y = []
        for label in range(n_labels):
            y_pred = self.classifiers[label].predict(X_test)
            if pred_expansion:
                np.append(X_test, y_pred)
            y.append(y_pred)
        # return np.array(y).T
        return np.array(y)

    def score(self, X, y, sample_weight=None):
        """

        :author: Joschka Strüber
        :param X:
        :param y:
        :param sample_weight:
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
        model = self.classifiers[label]
        model.fit(X, y)

        # saving and loading model to release memory
        os.mkdir("tmp")
        dump(model, "tmp/xgb_model.joblib")
        del model
        model = load("tmp/xgb_model.joblib")
        if os.path.exists("tmp") and os.path.isdir("tmp"):
            shutil.rmtree("tmp")
        self.classifiers[label] = model


if __name__ == "__main__":
    start = time.time()
    # load data and reset index
    data = load_user_data()
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = list(X.index)
    labels = list(y.index)
    X = X.values
    y = y.values

    """
    # save save uuid for stratified train-test-split
    le = LabelEncoder()
    le.fit(X[:, 0])
    strat_classes = le.transform(X[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=strat_classes)
    """
    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=2)
    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2], 1)
    X_test = np.delete(X_test, [0, 1, 2], 1)

    clf = xgb.XGBClassifier(objective="binary:logistic",
                            n_jobs=-1,
                            tree_method="gpu_hist",
                            n_estimators=20,
                            learning_rate=0.02,
                            max_depth=6,
                            colsample_bytree=0.8,
                            gamma=0.5,
                            subsample=0.8)
    clf = FlexOneVsRestClassifier(clf, n_estimators=y_train.shape[1])

    # todo: remove test for perfect scale_pos_weight
    for label in range(y_train.shape[1]):
        sum_pos = np.count_nonzero(y_train[:, label] == 1.0)
        sum_neg = np.count_nonzero(y_train[:, label] == 0)
        neg_pos_ratio_train = float(sum_neg) / sum_pos if sum_pos != 0 else 1

        sum_pos = np.count_nonzero(y_test[:, label] == 1.0)
        sum_neg = np.count_nonzero(y_test[:, label] == 0)
        neg_pos_ratio_test = float(sum_neg) / sum_pos if sum_pos != 0 else 1

        clf.classifiers[label].scale_pos_weight = sqrt(neg_pos_ratio_train)
        print(label, sqrt(neg_pos_ratio_train))

    label_imputer = SimpleImputer(strategy="most_frequent")
    y_train = label_imputer.fit_transform(y_train)

    preprocess = time.time()
    print("Preprocess: {}".format(preprocess - start))

    clf.fit(X_train, y_train)

    fit_time = time.time()
    print("Fit time: {}".format(fit_time - preprocess))

    y_pred = clf.predict(X_test)
    y_pred_bias = clf.predict(X_train)

    pred_time = time.time()
    print("Prediction time: {}".format(pred_time - fit_time))
    print("Balanced accuracy: ", balanced_accuracy_score(y_test.T,
                                                         y_pred,
                                                         average="macro",
                                                         zero_default=0))
    print("Balanced accuracy bias:", balanced_accuracy_score(y_train.T,
                                                             y_pred_bias,
                                                             average="macro",
                                                             zero_default=0))
    print("---")
    score_time = time.time()
    print("Score time: {}".format(score_time - pred_time))
