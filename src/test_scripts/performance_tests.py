import xgboost as xgb
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import make_scorer
from joblib import load

from data_handling import load_user_data, split_features_labels
from metrics import balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier


def cv_strat_ba_score(clf, X, y, groups, n_splits=5):
    sum_ba = 0.
    strat_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=random_state)
    for ii, (tr, tt) in enumerate(strat_kfold.split(X=X, y=groups)):
        X_train, X_test = X[tr], X[tt]
        y_train, y_test = y[tr], y[tt]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sum_ba += balanced_accuracy_score(y_test, y_pred)
    return sum_ba / n_splits


def cv_grouped_ba_score(clf, X, y, groups, n_splits=5):
    sum_ba = 0.
    grouped_kfold = GroupKFold(n_splits=n_splits)
    for ii, (tr, tt) in enumerate(grouped_kfold.split(X=X, y=y, groups=groups)):
        X_train, X_test = X[tr], X[tt]
        y_train, y_test = y[tr], y[tt]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        sum_ba += balanced_accuracy_score(y_test, y_pred)
    return sum_ba / n_splits


if __name__ == "__main__":
    # load data and reset index
    data = load_user_data("../../data")
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = X.columns
    labels = y.columns
    X = X.values
    y = y.values

    # UUIDs for grouped split
    uuid_groups = X[:, 0]

    # drop uuid column, the timestamps, and the label source
    X = np.delete(X, [0, 1, 2, X.shape[1] - 1], 1)
    X = np.delete(X, [0, 1, 2, X.shape[1] - 1], 1)

    n_labels = y.shape[1]
    gnb_clf = GaussianNB()
    gnb_ovr_clf = FlexOneVsRestClassifier(clf=gnb_clf, n_estimators=n_labels)
    gnb_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', gnb_ovr_clf)
    ])

    xgb_clf = xgb.XGBClassifier()
    xgb_ovr_clf = FlexOneVsRestClassifier(clf=xgb_clf, n_estimators=n_labels)
    param_dict = load("params_separated_xgb.joblib")
    xgb_ovr_clf.set_params(**param_dict)

    rf_clf = xgb.XGBRFClassifier()
    rf_ovr_clf = FlexOneVsRestClassifier(clf=rf_clf, n_estimators=n_labels)
    param_dict = load("params_separated_rf.joblib")
    rf_ovr_clf.set_params(**param_dict)

    lr_clf = LogisticRegression()
    lr_ovr_clf = FlexOneVsRestClassifier(clf=lr_clf, n_estimators=n_labels)
    param_dict = load("params_separated_lr.joblib")
    lr_ovr_clf.set_params(**param_dict)
    lr_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', lr_ovr_clf)
    ])

    clfs = [gnb_ovr_pipeline,
            xgb_ovr_clf,
            rf_ovr_clf,
            lr_ovr_pipeline]

    ba_scorer = make_scorer(balanced_accuracy_score)
    random_state = 42
    for clf in clfs:
        print(clf)
        print(cv_strat_ba_score(clf, X, y, uuid_groups))
        print(cv_grouped_ba_score(clf, X, y, uuid_groups))
