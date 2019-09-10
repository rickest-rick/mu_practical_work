import xgboost as xgb
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from data_handling import load_user_data, load_some_user_data, \
    split_features_labels, user_train_test_split
from metrics import balanced_accuracy_score, single_balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier

if __name__ == "__main__":
    start = time.time()
    # load data and reset index
    data = load_user_data("../../data")
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = X.columns
    labels = y.columns
    X = X.values
    y = y.values

    # save save uuid for stratified train-test-split
    le = LabelEncoder()
    le.fit(X[:, 0])
    strat_classes = le.transform(X[:, 0])

    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=None)
    # UUIDs for grouped split
    uuid_groups = X_train[:, 0]

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2, X_train.shape[1] - 1], 1)
    X_test = np.delete(X_test, [0, 1, 2, X_test.shape[1] - 1], 1)

    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])
    X_train_clean = preprocess_pipeline.fit_transform(X_train)
    X_test_clean = preprocess_pipeline.transform(X_test)

    clf = xgb.XGBClassifier(objective="binary:logistic",
                            n_jobs=-1,
                            tree_method="gpu_hist",
                            n_estimators=400,
                            learning_rate=0.02,
                            max_depth=7,
                            colsample_bytree=0.9,
                            gamma=1,
                            subsample=0.9)

    clf = FlexOneVsRestClassifier(clf, n_estimators=y_train.shape[1])


    # todo: remove test for perfect scale_pos_weight
    
    for label in range(y_train.shape[1]):
        sum_pos = np.count_nonzero(y_train[:, label] == 1.0)
        sum_neg = np.count_nonzero(y_train[:, label] == 0)
        neg_pos_ratio_train = float(sum_neg) / sum_pos if sum_pos != 0 else 1
        
        class_weights = {0: 1,
                         1: neg_pos_ratio_train}
        params = {"class_weights": class_weights}
        clf.classifiers[label].set_params(**params)
        
        # clf.classifiers[label].scale_pos_weight = sqrt(neg_pos_ratio_train)

    # save the parameter bounds and a list of parameters that must be integers

    def ba(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        return single_balanced_accuracy_score(conf_matrix)

    bounds = {"max_depth": (6, 8.5),
              "learning_rate": (0.01, 0.5),
              "gamma": (0, 10)}
    int_params = ["max_depth"]
    
    clf.tune_hyperparams(X=X_train,
                         y=y_train,
                         bounds=bounds,
                         metric=ba,
                         init_points=6,
                         n_iter=9,
                         int_params=int_params,
                         groups=uuid_groups)

    dump(clf.get_params(), "params_separated.joblib")

    params = load("params_separated.joblib")
    new_params = dict()
    for key in range(len(clf.classifiers)):
        new_params[str(key)] = params[key]
    clf.set_params(**new_params)

    bounds = {"scale_pos_weight": (0.7, 1.4)}

    clf.tune_hyperparams(X=X_train,
                         y=y_train,
                         bounds=bounds,
                         metric=ba,
                         init_points=10,
                         n_iter=15,
                         groups=uuid_groups)

    nb_clf = GaussianNB()
    clf = FlexOneVsRestClassifier(nb_clf, n_estimators=y_train.shape[1])

    clf.fit(X_train_clean, y_train)
    y_pred = clf.predict(X_test_clean)
    print("Balanced accuracy NB: ", balanced_accuracy_score(y_test.T,
                                                            y_pred,
                                                            average="macro",
                                                            zero_default=0))

    lr_clf = LogisticRegression(solver="lbfgs", tol=1e-3, max_iter=500)
    bounds = {"C": (0.0001, 1)}
    clf = FlexOneVsRestClassifier(lr_clf, n_estimators=y_train.shape[1])

    clf.tune_hyperparams(X=X_train_clean,
                         y=y_train,
                         bounds=bounds,
                         metric=ba,
                         init_points=6,
                         n_iter=9,
                         groups=uuid_groups)

    dump(clf.get_params(), "params_separated_lr.joblib")
    param_dict = load("params_separated_lr.joblib")
    print(param_dict)
    clf.fit(X_train_clean, y_train)
    y_pred = clf.predict(X_test_clean)
    print("Balanced accuracy LR: ", balanced_accuracy_score(y_test.T,
                                                            y_pred,
                                                            average="macro",
                                                            zero_default=0))

    rf_clf = xgb.XGBRFClassifier(max_depth=12,
                                 n_estimators=200,
                                 tree_method="gpu_hist",
                                 objective="binary:logistic")

    clf = FlexOneVsRestClassifier(rf_clf, n_estimators=y_train.shape[1])
    bounds = {"max_depth": (8, 15),
              "colsample_bynode": (0.5, 0.9)}
    clf.tune_hyperparams(X=X_train,
                         y=y_train,
                         bounds=bounds,
                         metric=ba,
                         init_points=6,
                         n_iter=9,
                         int_params=["max_depth"],
                         groups=uuid_groups)

    dump(clf.get_params(), "params_separated_rf.joblib")
    param_dict = load("params_separated_rf.joblib")
    print(param_dict)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Balanced accuracy RF: ", balanced_accuracy_score(y_test.T,
                                                            y_pred,
                                                            average="macro",
                                                            zero_default=0))

    svc_clf = LinearSVC(tol=3e-3)
    clf = FlexOneVsRestClassifier(svc_clf, n_estimators=y_train.shape[1])
    bounds = {"C": (0.01, 5)}
    clf.tune_hyperparams(X=X_train_clean,
                         y=y_train,
                         bounds=bounds,
                         metric=ba,
                         init_points=6,
                         n_iter=9,
                         groups=uuid_groups)

    dump(clf.get_params(), "params_separated_svc.joblib")
    param_dict = load("params_separated_svc.joblib")
    print(param_dict)
    clf.fit(X_train_clean, y_train)
    y_pred = clf.predict(X_test_clean)
    print("Balanced accuracy SVC: ", balanced_accuracy_score(y_test.T,
                                                             y_pred,
                                                             average="macro",
                                                             zero_default=0))

    preprocess = time.time()
    print("Preprocess: {}".format(preprocess - start))

    clf.fit(X_train, y_train, ignore_nan=True)

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
    score_time = time.time()
    print("Score time: {}".format(score_time - pred_time))
