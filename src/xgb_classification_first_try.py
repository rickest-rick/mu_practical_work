import gc

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
import xgboost as xgb
import pandas as pd
import numpy as np

from data_handling import load_user_data, split_features_labels


if __name__ == '__main__':
    load_model = True

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # load data and reset index
    user_data = load_user_data()
    features_df, labels_df = split_features_labels(user_data)
    user_data = None
    features_df.reset_index(inplace=True)
    labels_df.reset_index(inplace=True)

    # remove uuid column, the timestamps and the label_source from dataframe
    labels_df.drop(labels_df.columns[[0, 1]], axis=1, inplace=True)
    y = labels_df.values
    index_cols = features_df.columns[[0, 1, 2, -1]]
    features_df.drop(index_cols, axis=1, inplace=True)
    X = features_df.values
    features_df = None
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # preprocess the data by setting NaN values to the mean and standard scaling
    preprocess_feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

    preprocess_label_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value=0.0))])

    X_train = preprocess_feature_pipeline.fit_transform(X_train)
    X_test = preprocess_feature_pipeline.transform(X_test)

    y_train = preprocess_label_pipeline.fit_transform(y_train)
    y_test = preprocess_label_pipeline.transform(y_test)

    # 5-fold CV with random search of hyperparams for random forest classifier
    param_dist = {'bootstrap': [True, False],
                  'max_depth': [20, 30, 40, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [30, 50, 100, 150]
                  }
    if not load_model:
        clf = OneVsRestClassifier(xgb.XGBClassifier(max_depth=15, verbosity=2, n_jobs=-1))
        # clf = OneVsRestClassifier(RandomForestClassifier(verbose=2, max_depth=30,
        #                                                 n_estimators=100, n_jobs=-1))

        # mlb = MultiLabelBinarizer()
        # y = mlb.fit_transform(y_train)
        # print(np.shape(y))
        clf.fit(X_train, y_train)
        dump(clf, 'RandomForest_1.joblib')
    else:
        clf = load('RandomForest_1.joblib')

    preds = clf.predict(X_test)
    print(np.shape(preds))
    print(np.shape(y_test))

    score_zero_one = zero_one_loss(y_test, preds, normalize=True)
    print("Zero-One-Loss: ", score_zero_one)

    score_hamming = hamming_loss(y_test, preds)
    print("Hamming Loss: ", score_hamming)

    score_f1 = f1_score(y_test, preds, average="micro")
    print("F1 score: ", score_f1)

    score_recall = recall_score(y_test, preds, average="micro")
    print("Recall: ", score_recall)

    matrices = multilabel_confusion_matrix(y_test, preds)
    support = list()
    specificity = list()
    for i in range(np.shape(matrices)[0]):
        m = matrices[i]
        spec = float(m[0, 0])/(m[0, 1]+m[0, 0])
        supp = np.count_nonzero(y_test[:i])
        specificity.append(spec)
        support.append(supp)

    true_neg = neg = true_pos = 0

    for i in range(np.shape(matrices)[0]):
        m = matrices[i]
        true_neg += m[0, 0]
        neg += m[0, 1] + m[0, 0]

    print("True Neg and Neg: {0} {1}".format(true_neg, neg))

    score_specificity_2 = float(true_neg) / neg

    score_specificity = float(sum([support[i]*specificity[i] for i in range(len(support))]))/sum(support)
    print("Specificity: ", score_specificity)
    print("Specificity 2: ", score_specificity_2)

    score_balanced_accuracy = (score_specificity + score_recall) / 2
    print("Balanced accuracy: ", score_balanced_accuracy)

    score_balanced_accuracy_2 = (score_specificity_2 + score_recall) / 2
    print("Balanced accuracy 2: ", score_balanced_accuracy_2)