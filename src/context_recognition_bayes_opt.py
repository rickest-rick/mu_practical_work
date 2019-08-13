import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from bayes_opt import BayesianOptimization
from joblib import dump, load

from data_handling import *
from metrics import balanced_accuracy_score


def xgb_evaluate(max_depth, gamma, learning_rate):
    xgb_classifier = xgb.XGBClassifier(max_depth=int(max_depth),
                                       subsample=0.8,
                                       learning_rate=learning_rate,
                                       gamma=gamma,
                                       colsample_bytree=0.8,
                                       n_estimators=500,
                                       objective="binary:logistic",
                                       nthread=-1,
                                       tree_method="gpu_hist",
                                       n_iter_no_change=5,
                                       verbosity=2)
    ovr_classifier = OneVsRestClassifier(xgb_classifier)
    kf = KFold(n_splits=3, shuffle=True)
    sum_ba_accuracy = 0
    for train_index, test_index in kf.split(X_train):
        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        ovr_classifier.fit(X_tr, y_tr)
        preds = ovr_classifier.predict(X_te)
        sum_ba_accuracy += balanced_accuracy_score(y_te, preds)
    return sum_ba_accuracy / 3


if __name__ == '__main__':
    # load data and reset index
    data = load_user_data()
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = list(X.index)
    labels = list(y.index)
    X = X.values
    random_labels = np.random.randint(0, 50, 30)
    y = y.values[:, random_labels]
    #y = y.values

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

    """
    xgb_bo = BayesianOptimization(xgb_evaluate, {"max_depth": (5, 8),
                                                 "gamma": (0, 1),
                                                 "learning_rate": (0.08, 0.25)})
    xgb_bo.maximize(init_points=10, n_iter=20, acq='ei')

    print(xgb_bo.res)
    """
    xgb_classifier = xgb.XGBClassifier(max_depth=8,
                                       subsample=0.8,
                                       learning_rate=0.2,
                                       gamma=1,
                                       reg_lambda=2,
                                       colsample_bytree=0.8,
                                       n_estimators=100,
                                       objective="binary:logistic",
                                       nthread=-1,
                                       tree_method="auto",
                                       n_iter_no_change=5,
                                       verbosity=1)
    ovr_clf = OneVsRestClassifier(xgb_classifier)
    ovr_clf.fit(X_train, y_train)
    for estimator in ovr_clf.estimators_:
        print("Best n_tree limit:", estimator.get_booster().best_iteration)

    train_preds = ovr_clf.predict(X_train)
    print("Balanced Accuracy Train:", balanced_accuracy_score(y_train, train_preds))
    print("F1-Score:", f1_score(y_train, train_preds, average="micro"))
    dump(ovr_clf, 'context_rec.joblib')
    preds = ovr_clf.predict(X_test)

    score_zero_one = zero_one_loss(y_test, preds, normalize=True)
    print("Zero-One-Loss: ", score_zero_one)

    score_hamming = hamming_loss(y_test, preds)
    print("Hamming Loss: ", score_hamming)

    score_f1 = f1_score(y_test, preds, average="micro")
    print("F1 score: ", score_f1)

    score_recall = recall_score(y_test, preds, average="micro")
    print("Recall: ", score_recall)

    print("Balanced accuracy: ", balanced_accuracy_score(y_test, preds))