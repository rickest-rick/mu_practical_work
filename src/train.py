import sys
import xgboost as xgb
import numpy as np
import os
import pandas as pd

from joblib import load, dump

from flex_one_vs_rest_classifier import FlexOneVsRestClassifier
from data_handling import load_user_data, split_features_labels

from matplotlib import pyplot as plt


if __name__ == "__main__":
    args = sys.argv
    data_path = args[1]
    classifier_path = args[2]

    # read and process training data
    data = load_user_data(data_path)
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)

    # drop uuid column, the timestamps and the label source
    X_train = X.drop(['level_0',
                      'level_1',
                      'timestamp',
                      'label_source'], axis=1)

    attr_names = X_train.columns.tolist()
    label_names = y.columns.tolist()
    X_train = X_train.values
    y = y.values

    # build classifiers with saved hyperparameters
    xgb_params = load("params_separated_xgb.joblib")

    # add list of classifiers to flexible OneVsRestClassifier
    clf = FlexOneVsRestClassifier(clf=xgb.XGBClassifier(),
                                  n_estimators=y.shape[1],
                                  label_names=label_names,
                                  feature_names=attr_names)
    clf.set_params(**xgb_params)

    # train and save classifier
    clf.fit(X_train, y)

    # save feature importance plots
    if not os.path.exists("plots"):
        os.mkdir("plots")
    for label, xgb_clf in clf.classifiers.items():
        xgb_clf.get_booster().feature_names = attr_names
        feature_importances = xgb.plot_importance(xgb_clf, max_num_features=15)
        path = os.path.join("plots", label_names[int(label)] + ".png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    dump(clf, classifier_path)
