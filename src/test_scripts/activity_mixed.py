import xgboost as xgb
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from joblib import dump, load

from data_handling import load_user_data, load_some_user_data, \
    split_features_labels
from metrics import balanced_accuracy_score, single_balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier

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

    # save save uuid for stratified train-test-split
    le = LabelEncoder()
    le.fit(X[:, 0])
    strat_classes = le.transform(X[:, 0])

    # UUIDs for grouped split
    uuid_groups = X[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=None,
                                                        stratify=strat_classes)

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2, X_train.shape[1] - 1], 1)
    X_test = np.delete(X_test, [0, 1, 2, X_test.shape[1] - 1], 1)

    clf = xgb.XGBClassifier(objective="binary:logistic",
                            n_jobs=-1,
                            tree_method="gpu_hist",
                            n_estimators=400,
                            learning_rate=0.02,
                            max_depth=7,
                            colsample_bytree=0.8,
                            gamma=1,
                            subsample=0.8)
    clf = FlexOneVsRestClassifier(clf, n_estimators=y_train.shape[1])

    # save the parameter bounds and a list of parameters that must be integers
    params = {"max_depth": (6, 8.5),
              "learning_rate": (0.01, 0.5),
              "gamma": (0, 10),
              "colsample_bytree": (0.5, 1),
              "subsample": (0.5, 1)}
    int_params = ["max_depth"]

    def ba(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        return single_balanced_accuracy_score(conf_matrix)

    clf.tune_hyperparams(X=X_train,
                         y=y_train,
                         bounds=params,
                         metric=ba,
                         init_points=10,
                         n_iter=25,
                         int_params=int_params)

    dump(clf.get_params(), "params_separated.joblib")
    param_dict = load("params_separated.joblib")
    print(param_dict)

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