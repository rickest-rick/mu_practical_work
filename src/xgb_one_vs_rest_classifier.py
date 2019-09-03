import xgboost as xgb
import numpy as np
import threading

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.impute import SimpleImputer

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
            # TODO set parameters from parameter list
            clf = xgb.XGBClassifier(learning_rate=0.005,
                                    n_estimators=500,
                                    max_depth=8,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    gamma=1,
                                    objective="binary:logistic",
                                    tree_method=self.tree_method,
                                    n_jobs=self.n_jobs)
            self.estimators.append(clf)

    def fit(self, X, y, pred_expansion=False):
        """

        :author; Joschka Strüber
        :param X:
        :param y:
        :param pred_expansion: If True, expand the data set with predictions on
            the labels that were trained so far.
        :return:
        """
        n_labels = y.shape[1]
        if n_labels != len(self.parameter_list):
            raise ValueError("Number of labels to predict ({0}) is not equal"
                             " to the number of trained classifiers ({1})"
                             .format(n_labels, len(self.parameter_list)))

        if pred_expansion:  # data set is modified, if pred_expansion
            X_train = np.copy(X)
        else:
            X_train = X
        # train a classifier for every label in y
        for label in range(n_labels):
            y_train = y[:, label]
            fit_thread = threading.Thread(target=self.fit_label, args=(label,
                                                                       X_train,
                                                                       y_train))
            fit_thread.start()
            fit_thread.join()
            if pred_expansion:
                y_pred = self.estimators[label].predict(X_train)
                X_train.append(y_pred)
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
        return np.array(y).T

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

    def fit_label(self, label, X, y):
        """
        Method to encapsulate the training process of a single XGBoost
        Classifier.
        :param label: The label number as int. Shows which column of y
            should be used as target.
        :param X: All features as numpy array.
        :param y: All target labels as numpy array.
        :return: None
        """
        model = self.estimators[label]
        model.fit(X, y)
        print(model.n_estimators)

        # saving and loading model to release memory
        model.save_model("tmp/xgb.model")
        model_le = model._le
        del model
        model = xgb.XGBClassifier()
        model.load_model("tmp/xgb.model")
        model._le = model_le
        print(model.n_estimators)
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
                                                             test_size=0.2,
                                                             random_state=42)

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2], 1)
    X_test = np.delete(X_test, [0, 1, 2], 1)
    y_train = np.delete(y_train, -1, 1)
    y_test = np.delete(y_test, -1, 1)

    feature_imputer = SimpleImputer(strategy="mean")
    label_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_train = feature_imputer.fit_transform(X_train)
    X_test = feature_imputer.transform(X_test)
    y_train = label_imputer.fit_transform(y_train)

    parameter_list = []
    params = {
        "n_estimators": 10,
        "learning_rate": 0.001,
        "max_depth": 8,
        "colsample_bytree": 0.9,
        "gamma": 0.5,
        "subsample": 0.9,
    }
    for label in range(y_train.shape[1]):
        parameter_list.append(params)
    clf = XgbOneVsRestClassifier(parameter_list=parameter_list,
                                 n_jobs=1,
                                 tree_method="gpu_hist")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
