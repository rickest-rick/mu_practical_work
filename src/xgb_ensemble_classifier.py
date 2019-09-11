import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from copy import deepcopy

from data_handling import load_user_data, user_train_test_split, \
    split_features_labels, load_some_user_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from metrics import single_balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier


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
            pred_column = self.get_oof_prediction(clf, X, y,
                                                  random_state=random_state)
            if pred_column.ndim == 1:
                pred_column = np.reshape(pred_column, (n_samples, 1))
            X_preds = np.concatenate((X_preds, pred_column), axis=1)
        self.ensemble_classifier.fit(X_preds, y)

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

    def get_oof_prediction(self, clf, X_train, y_train, random_state=None):
        """
        Returns out of fold predicted probabilities or predictions for the
        classifier on a k-fold split.

        :param clf:
        :param X_train:
        :param y_train:
        :param n_splits:
        :param random_state:
        :return:
        """
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
            # save predicted probabilities and if not available save predictions
            try:
                y_pred = clf.predict_proba(X_tt)
                oof_train[test, :] = y_pred
            except AttributeError:
                oof_train[test] = clf.predict(X_tt)
        return oof_train


if __name__ == "__main__":
    # load data and reset index
    data = load_user_data("../data")
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = X.columns
    labels = y.columns
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=42)
    # UUIDs for grouped split
    uuid_groups = X_train[:, 0]
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    is_nan = np.isnan(y_train)
    X_train = X_train[~is_nan]
    y_train = y_train[~is_nan]

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2, X_train.shape[1] - 1], 1)
    X_test = np.delete(X_test, [0, 1, 2, X_test.shape[1] - 1], 1)

    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])
    X_train = preprocess_pipeline.fit_transform(X_train)
    X_test = preprocess_pipeline.transform(X_test)

    rf_clf = xgb.XGBRFClassifier(max_depth=12,
                                 n_estimators=200,
                                 n_jobs=-1,
                                 tree_method="gpu_hist")
    svc_clf = LinearSVC(tol=1e-2, max_iter=2000, C=0.1)
    ada_clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.01)
    log_reg_clf = LogisticRegression(solver="lbfgs", C=0.1, max_iter=500)
    xgb_clf = xgb.XGBClassifier(n_estimators=300,
                                max_depth=7,
                                gamma=1,
                                learning_rate=0.2,
                                n_jobs=-1,
                                tree_method="gpu_hist")

    classifiers = [xgb_clf,
                   svc_clf,
                   ada_clf,
                   log_reg_clf,
                   rf_clf]
    ensemble_clf = XgbEnsembleClassifier(classifiers=classifiers,
                                         n_splits=3,
                                         n_estimators=200,
                                         learning_rate=0.1,
                                         max_depth=5,
                                         tree_method="gpu_hist")
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print("AdaBoost:", single_balanced_accuracy_score(y_test, y_pred))

    log_reg_clf.fit(X_train, y_train)
    y_pred = log_reg_clf.predict(X_test)
    print("Logistic Regression:", single_balanced_accuracy_score(y_test,
                                                                 y_pred))

    svc_clf.fit(X_train, y_train)
    y_pred = svc_clf.predict(X_test)
    print("SVM:", single_balanced_accuracy_score(y_test, y_pred))

    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print("Random Forest:", single_balanced_accuracy_score(y_test, y_pred))

    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print("XGBoost:", single_balanced_accuracy_score(y_test, y_pred))

    ensemble_clf.fit(X_train, y_train)
    y_pred = ensemble_clf.predict(X_test)
    print("Ensemble:", single_balanced_accuracy_score(y_test, y_pred))

    gnb_clf = GaussianNB()
    voting_clf = VotingClassifier([("ada", ada_clf),
                                   ("rf", rf_clf),
                                   ("gnb", gnb_clf),
                                   ("xgb", xgb_clf),
                                   ("lr", log_reg_clf)],
                                  voting="soft",
                                  weights=[0.15, 0.25, 0.1, 0.3, 0.2],
                                  n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print("Voting:", single_balanced_accuracy_score(y_test, y_pred))