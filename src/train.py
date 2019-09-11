import sys
import xgboost as xgb
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import load, dump

from flex_one_vs_rest_classifier import FlexOneVsRestClassifier
from xgb_ensemble_classifier import XgbEnsembleClassifier
from data_handling import load_user_data, split_features_labels


if __name__ == "__main__":
    args = sys.argv
    data_path = args[1]
    classifier_path = args[2]

    # read and process training data
    data = load_user_data(data_path)
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attr_names = X.columns
    label_names = y.columns
    X = X.values
    y = y.values

    # drop uuid column, the timestamps and the label source
    X_train = np.delete(X, [0, 1, 2, X.shape[1] - 1], 1)

    # build classifiers with saved hyperparameters
    xgb_params = load("params_separated_xgb.joblib")
    rf_params = load("params_separated_rf.joblib")
    svc_params = load("params_separated_svc.joblib")
    lr_params = load("params_separated_lr.joblib")

    classifiers = []
    for label in range(y.shape[1]):
        xgb_param = xgb_params[str(label)]
        xgb_clf = xgb.XGBClassifier(**xgb_param)

        rf_param = rf_params[str(label)]
        rf_clf = xgb.XGBRFClassifier(**rf_param)

        lr_param = lr_params[str(label)]
        lr_clf = LogisticRegression(**lr_param)

        ada_clf = AdaBoostClassifier(n_estimators=50,
                                     learning_rate=0.9)

        clfs_internal = [xgb_clf,
                         rf_clf,
                         lr_clf,
                         ada_clf]
        ensemble_clf = XgbEnsembleClassifier(classifiers=classifiers,
                                             n_splits=3,
                                             n_estimators=200,
                                             learning_rate=0.1,
                                             max_depth=5,
                                             gamma=1,
                                             tree_method="gpu_hist",
                                             n_jobs=-1)
        classifiers.append(ensemble_clf)

    # add list of classifiers to flexible OneVsRestClassifier
    clf = FlexOneVsRestClassifier(classifiers=classifiers,
                                  label_names=label_names,
                                  feature_names=attr_names)

    clf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
        ('clf', clf)
    ])

    # train and save classifier
    clf_pipeline.fit(X_train, y)
    dump(clf_pipeline, classifier_path)
