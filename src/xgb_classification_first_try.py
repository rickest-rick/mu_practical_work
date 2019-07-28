import xgboost as xgb
import pandas as pd
import numpy as np
import gc

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from scipy.stats import uniform, randint, reciprocal

from data_handling import load_user_data, split_features_labels
from metrics import balanced_accuracy_score

ba_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

if __name__ == '__main__':
    load_model = False

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
    y = labels_df.values[:, 0:5]
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

    if not load_model:
        # 5-fold CV with random search of hyperparams for xgboost classifier
        param_dist = {
            "estimator__colsample_bytree": [0.8],
            "estimator__gamma": [0],
            "estimator__learning_rate": reciprocal(0.05, 0.3),
            "estimator__max_depth": [4, 6, 8],
            "estimator__n_estimators": [500],
            "estimator__subsample": [0.8]
        }
        xgb_clf = xgb.XGBClassifier(objective="binary:logistic", nthread=1,
                                    tree_method="gpu_hist")
        ovr_clf = OneVsRestClassifier(xgb_clf)
        random_search = RandomizedSearchCV(ovr_clf, param_dist, n_iter=10, cv=3,
                                           verbose=3, n_jobs=1,
                                           scoring=ba_scorer)
        # clf = OneVsRestClassifier(RandomForestClassifier(verbose=2, max_depth=30,
        #                                                 n_estimators=100, n_jobs=-1))

        # mlb = MultiLabelBinarizer()
        # y = mlb.fit_transform(y_train)
        # print(np.shape(y))
        random_search.fit(X_train, y_train)
        print(random_search.cv_results_)
        for estimator in random_search.best_estimator_.estimators_:
            print("Best n_tree limit:", estimator.get_booster().best_ntree_limit)
        train_preds = random_search.predict(X_train)
        print("Balanced Accuracy Train:", balanced_accuracy_score(y_train, train_preds))
        print("F1-Score:", f1_score(y_train, train_preds, average="micro"))
        dump(random_search.best_estimator_, 'RandomForest_1.joblib')
    else:
        random_search = load('RandomForest_1.joblib')

    print(random_search.best_estimator_)

    preds = random_search.predict(X_test)

    score_zero_one = zero_one_loss(y_test, preds, normalize=True)
    print("Zero-One-Loss: ", score_zero_one)

    score_hamming = hamming_loss(y_test, preds)
    print("Hamming Loss: ", score_hamming)

    score_f1 = f1_score(y_test, preds, average="micro")
    print("F1 score: ", score_f1)

    score_recall = recall_score(y_test, preds, average="micro")
    print("Recall: ", score_recall)

    print("Balanced accuracy: ", balanced_accuracy_score(y_test, preds))
