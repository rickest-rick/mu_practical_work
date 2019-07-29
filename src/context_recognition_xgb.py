import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from scipy.stats import uniform, randint, reciprocal

from data_handling import *
from metrics import balanced_accuracy_score


if __name__ == '__main__':
    load_model = False

    # load data and reset index
    data = load_user_data()
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = list(X.index)
    labels = list(y.index)
    X = X.values
    random_labels = np.random.randint(0, 50, 6)
    y = y.values[:, random_labels]
    print(np.shape(X), np.shape(y))

    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=42)

    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2], 1)
    X_test = np.delete(X_test, [0, 1, 2], 1)
    y_train = np.delete(y_train, -1, 1)
    y_test = np.delete(y_test, -1, 1)

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
        ba_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
        # 5-fold CV with random search of hyperparams for xgboost classifier
        param_dist = {
            "estimator__colsample_bytree": [0.7],
            "estimator__gamma": [0],
            "estimator__alpha": [0],
            "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__max_depth": [6, 8, 10],
            "estimator__n_estimators": [100, 1000, 2000],
            "estimator__subsample": [0.9]
        }
        xgb_clf = xgb.XGBClassifier(objective="binary:logistic", nthread=-1,
                                    tree_method="gpu_hist", n_iter_no_change=5,
                                    verbosity=2)
        ovr_clf = OneVsRestClassifier(xgb_clf)
        random_search = RandomizedSearchCV(ovr_clf, param_dist, n_iter=27, cv=3,
                                           verbose=3, n_jobs=1,
                                           scoring=ba_scorer)
        random_search.fit(X_train, y_train)
        print(random_search.cv_results_)
        for estimator in random_search.best_estimator_.estimators_:
            print("Best iteration:", estimator.get_booster().best_ntree_limit)
        train_preds = random_search.predict(X_train)
        print("Balanced Accuracy Train:", balanced_accuracy_score(y_train, train_preds))
        dump(random_search.best_estimator_, 'context_rec.joblib')
    else:
        random_search = load('context_rec.joblib')

    print(random_search.best_estimator_)

    preds = random_search.predict(X_test)

    score_f1 = f1_score(y_test, preds, average="micro")
    print("F1 score: ", score_f1)

    score_recall = recall_score(y_test, preds, average="micro")
    print("Recall: ", score_recall)

    print("Balanced accuracy: ", balanced_accuracy_score(y_test, preds))