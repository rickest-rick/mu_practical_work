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
from mlxtend.classifier import StackingCVClassifier

from data_handling import load_user_data, split_features_labels
from metrics import balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier


def cv_strat_ba_score(clf, X, y, groups, n_splits=5, random_state=None):
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

    gnb_clf = GaussianNB()
    gnb_ovr_clf = FlexOneVsRestClassifier(clf=gnb_clf, n_estimators=n_labels)
    gnb_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', gnb_ovr_clf)
    ])

    xgb_clf = xgb.XGBClassifier()
    xgb_ovr_clf = FlexOneVsRestClassifier(clf=xgb_clf, n_estimators=n_labels)
    xgb_params = load("params_separated_xgb.joblib")
    xgb_ovr_clf.set_params(**xgb_params)

    rf_clf = xgb.XGBRFClassifier()
    rf_ovr_clf = FlexOneVsRestClassifier(clf=rf_clf, n_estimators=n_labels)
    rf_params = load("params_separated_rf.joblib")
    rf_ovr_clf.set_params(**rf_params)

    lr_clf = LogisticRegression()
    lr_ovr_clf = FlexOneVsRestClassifier(clf=lr_clf, n_estimators=n_labels)
    lr_params = load("params_separated_lr.joblib")
    lr_ovr_clf.set_params(**lr_params)
    lr_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', lr_ovr_clf)
    ])

    svm_clf = LinearSVC()
    svm_ovr_clf = FlexOneVsRestClassifier(clf=svm_clf, n_estimators=n_labels)
    svc_params = load("params_separated_svc.joblib")
    svm_ovr_clf.set_params(**svc_params)
    svm_ovr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', svm_ovr_clf)
    ])

    # build base classifiers with saved hyperparameters
    classifiers = []
    for label in range(n_labels):
        xgb_param = xgb_params[str(label)]
        xgb_clf = xgb.XGBClassifier(**xgb_param)

        rf_param = rf_params[str(label)]
        rf_clf = xgb.XGBRFClassifier(**rf_param)

        svc_param = svc_params[str(label)]
        svc_clf = LinearSVC(**svc_param)
        svc_clf.set_params(tol=1e-2)

        lr_param = lr_params[str(label)]
        lr_clf = LogisticRegression(**lr_param)

        ada_clf = AdaBoostClassifier(n_estimators=50,
                                     learning_rate=0.9)

        internal_clfs = [xgb_clf,
                         rf_clf,
                         lr_clf]
        meta_clf = xgb.XGBClassifier(n_estimators=200,
                                     learning_rate=0.1,
                                     max_depth=5,
                                     gamma=1,
                                     n_jobs=-1,
                                     tree_method="gpu_hist")
        ensemble_clf = StackingCVClassifier(classifiers=internal_clfs,
                                            meta_classifier=meta_clf,
                                            use_probas=True,
                                            drop_last_proba=True,
                                            random_state=42,
                                            n_jobs=-1)
        classifiers.append(ensemble_clf)

    # add list of classifiers to flexible OneVsRestClassifier
    ensemble_clf_ovr = FlexOneVsRestClassifier(classifiers=classifiers)

    ensemble_clf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', ensemble_clf_ovr)
    ])

    clfs = [#svm_ovr_pipeline,
            #gnb_ovr_pipeline,
            #xgb_ovr_clf,
            #rf_ovr_clf,
            #lr_ovr_pipeline,
            ensemble_clf_pipeline]
    clf_names = [#"SVM",
                 #"Naive Bayes",
                 #"XGB",
                 #"Random Forest",
                 #"Logistic Regression",
                 "Ensemble"]

    ba_scorer = make_scorer(balanced_accuracy_score)
    random_state = 42
    for clf, label in zip(clfs, clf_names):
        print(label)
        #start = time.time()
        #print(cv_strat_ba_score(clf, X, y, uuid_groups, n_splits=5,
        #                        random_state=42))
        strat_time = time.time()
        #print("Time Stratified: ", str(timedelta(seconds=strat_time - start)))
        print(cv_grouped_ba_score(clf, X, y, uuid_groups, n_splits=5))
        grouped_time = time.time()
        print("Grouped Time: ", str(timedelta(seconds=grouped_time - strat_time)))
