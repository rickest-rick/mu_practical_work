import sys
import xgboost as xgb
import numpy as np

from flex_one_vs_rest_classifier import FlexOneVsRestClassifier
from data_handling import load_user_data, split_features_labels
from joblib import dump


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
    # todo
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                                n_jobs=-1,
                                tree_method="gpu_hist")
    clf = FlexOneVsRestClassifier(clf=xgb_clf,
                                  n_estimators=y.shape[1])

    # train and save classifier
    clf.fit(X_train, y)
    dump(clf, classifier_path)
