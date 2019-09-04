import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump, load
from math import log, e, sqrt

from data_handling import load_user_data, load_some_user_data, \
    split_features_labels, user_train_test_split
from metrics import balanced_accuracy_score


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
            parameters["objective"] = "binary:logistic"
            parameters["n_jobs"] = self.n_jobs
            parameters["tree_method"] = self.tree_method
            clf = xgb.XGBClassifier(**parameters)
            self.estimators.append(clf)

    def fit(self, X, y, pred_expansion=False, scale_method="equal",
            ignore_nan=False):
        """

        :author; Joschka Strüber
        :param X:
        :param y:
        :param pred_expansion: If True, expand the data set with predictions on
            the labels that were trained so far.
        :param scale_method:
        :param ignore_nan:
        :return:
        """
        if scale_method not in {"equal", "log", "full", "sqrt"}:
            raise ValueError('Invalid scale method chosen: "{}"'.format(
                scale_method))
        n_labels = y.shape[1]
        if n_labels != len(self.parameter_list):
            raise ValueError("Number of labels to predict ({0}) does not equal"
                             " the number of trained classifiers ({1})"
                             .format(n_labels, len(self.parameter_list)))

        if pred_expansion:  # data set is modified, if pred_expansion
            X_train = np.copy(X)
        else:
            X_train = X
        # train a classifier for every label in y
        for label in range(n_labels):
            y_train = y[:, label]

            # ignore features for which no prediction is available
            if ignore_nan:
                X_train = X
                label_is_nan = np.isnan(y_train)
                X_train = X_train[~label_is_nan]
                y_train = y_train[~label_is_nan]
            self.fit_label(label, X_train, y_train, scale_method=scale_method)
            # augment training data set with prediction on labels seen so far
            if pred_expansion:
                y_pred = self.estimators[label].predict(X_train)
                np.append(X_train, y_pred)
        return self

    def predict(self, X, pred_expansion=False):
        """

        :author: Joschka Strüber
        :param X:
        :param pred_expansion:
        :return:
        """
        n_labels = len(self.parameter_list)
        if pred_expansion:  # data set is modified, if pred_expansion
            X_test = np.copy(X)
        else:
            X_test = X

        y = []
        for label in range(n_labels):
            y_pred = self.estimators[label].predict(X_test)
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

    def fit_label(self, label, X, y, scale_method="equal"):
        """
        Method to encapsulate the training process of a single XGBoost
        Classifier.
        :param label: The label number as int. Shows which column of y
            should be used as target.
        :param X: All features as numpy array.
        :param y: All target labels as numpy array.
        :param scale_method: One of {"equal", "log", "full"}; default: "equal"
            "equal" -> No scaling, scale_pos_weight = 1. Can result in
                unbalanced predictions for unbalanced target labels.
            "full" -> Full scaling, scale_pos_weight = sum(negative) /
                sum(positive). Can result in unbalanced predictions but the
                other way around compared to "equal" (e.g. a lot of false
                positives, instead of a lot of false negatives).
            "sqrt" -> Scale by sqrt(neg_pos_ratio). Behaves as "full" but the
                square root normalizes the scale_pos_weight towards 1 ending
                in less extreme results.
        :return: None
        """
        if scale_method == "equal":
            scale_pos_weight = 1
        else:
            """
            sum_pos = np.count_nonzero(y == 1.0)
            sum_neg = np.count_nonzero(y == 0)
            neg_pos_ratio = float(sum_neg) / sum_pos if sum_pos != 0 else 1
            if scale_method == "full":
                scale_pos_weight = neg_pos_ratio
            elif scale_method == "sqrt":
                scale_pos_weight = sqrt(neg_pos_ratio)
            else:
                assert False
            """
        class_weights = compute_class_weight('balanced', np.unique(y), y)
        print(class_weights)
        scale_pos_weight = class_weights[1]
        model = self.estimators[label]
        model.scale_pos_weight = scale_pos_weight
        model.fit(X, y)

        # saving and loading model to release memory
        dump(model, "tmp/xgb_model.joblib")
        del model
        model = load("tmp/xgb_model.joblib")
        self.estimators[label] = model


if __name__ == "__main__":
    # load data and reset index
    data = load_user_data()
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = list(X.index)
    labels = list(y.index)
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.4,
                                                             random_state=1)

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2], 1)
    X_test = np.delete(X_test, [0, 1, 2], 1)
    y_train = np.delete(y_train, -1, 1)
    y_test = np.delete(y_test, -1, 1)

    parameter_list = []
    params = {
        "n_estimators": 100,
        "learning_rate": 0.02,
        "max_depth": 6,
        "colsample_bytree": 0.9,
        "gamma": 2,
        "subsample": 0.9,
    }
    for label in range(y_train.shape[1]):
        parameter_list.append(params)
    clf = XgbOneVsRestClassifier(parameter_list=parameter_list,
                                 n_jobs=1,
                                 tree_method="gpu_hist")

    label_imputer = SimpleImputer(strategy="most_frequent")
    y_train = label_imputer.fit_transform(y_train)
    clf.fit(X_train, y_train, scale_method="full", ignore_nan=False)
    y_pred = clf.predict(X_test)
    y_pred_bias = clf.predict(X_train)
    for average in {"macro", "micro"}:
        for zero_default in {1}:
            print("full", False, average, zero_default)
            print("Balanced accuracy: ", balanced_accuracy_score(y_test.T,
                                                                 y_pred,
                                                                 average,
                                                                 zero_default))
            print("Balanced accuracy bias:", balanced_accuracy_score(y_train.T,
                                                                     y_pred_bias,
                                                                     average,
                                                                     zero_default))
            print("---")
