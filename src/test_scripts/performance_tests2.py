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
from sklearn.ensemble import AdaBoostClassifier
from joblib import load
from datetime import timedelta

from data_handling import load_user_data, split_features_labels
from metrics import balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier
from xgb_ensemble_classifier import XgbEnsembleClassifier


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
    data = load_user_data("../data")
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
    """
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

    svm_clf = LinearSVC()
    svm_ovr_clf = FlexOneVsRestClassifier(clf=svm_clf, n_estimators=n_labels)
    param_dict = load("params_separated_svc.joblib")
    svm_ovr_clf.set_params(**param_dict)
    svm_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', svm_ovr_clf)
    ])
    """
    # build classifiers with saved hyperparameters
    xgb_params = load("params_separated_xgb.joblib")
    rf_params = load("params_separated_rf.joblib")
    svc_params = load("params_separated_svc.joblib")
    lr_params = load("params_separated_lr.joblib")

    classifiers = []
    for label in range(n_labels):
        xgb_param = xgb_params[str(label)]
        print("xgb", xgb_param)
        xgb_clf = xgb.XGBClassifier(**xgb_param)

        rf_param = rf_params[str(label)]
        rf_clf = xgb.XGBRFClassifier(**rf_param)
        print("rf", rf_param)

        #svc_param = svc_params[str(label)]
        #svc_clf = LinearSVC(**svc_param)
        #svc_clf.set_params(tol=1e-2)

        lr_param = lr_params[str(label)]
        lr_clf = LogisticRegression(**lr_param)
        print()

        ada_clf = AdaBoostClassifier(n_estimators=50,
                                     learning_rate=0.9)

        classifiers = [xgb_clf,
                       rf_clf,
                       #svc_clf,
                       lr_clf,
                       ada_clf]
        ensemble_clf = XgbEnsembleClassifier(classifiers=classifiers,
                                             n_splits=3,
                                             n_estimators=200,
                                             learning_rate=0.1,
                                             max_depth=5,
                                             gamma=1,
                                             tree_method="gpu_hist")
        classifiers.append(ensemble_clf)

    # add list of classifiers to flexible OneVsRestClassifier
    ensemble_clf_ovr = FlexOneVsRestClassifier(classifiers=classifiers)

    ensemble_clf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', ensemble_clf_ovr)
    ])

    xgb_clf = xgb.XGBClassifier()
    xgb_ovr = FlexOneVsRestClassifier(xgb_clf, n_estimators=n_labels)

    ensemble_clf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', xgb_ovr)
    ])


    """
    clfs = [svm_ovr_pipeline,
            gnb_ovr_pipeline,
            xgb_ovr_clf,
            rf_ovr_clf,
            lr_ovr_pipeline,
            ensemble_clf_pipeline]
    """

    clfs = [ensemble_clf_pipeline]

    ba_scorer = make_scorer(balanced_accuracy_score)
    random_state = 42
    for clf in clfs:
        print(clf)
        start = time.time()
        print(cv_strat_ba_score(clf, X, y, uuid_groups))
        strat_time = time.time()
        print("Time Stratified: ", str(timedelta(strat_time - start)))
        print(cv_grouped_ba_score(clf, X, y, uuid_groups))
        grouped_time = time.time()
        print("Grouped Time: ", str(timedelta(grouped_time - strat_time)))