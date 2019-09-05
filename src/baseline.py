import numpy as np
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import reciprocal

from data_handling import load_user_data, load_some_user_data, \
    split_features_labels, user_train_test_split
from metrics import balanced_accuracy_score
from flex_one_vs_rest_classifier import FlexOneVsRestClassifier


if __name__ == "__main__":
    # load data and reset index
    data = load_user_data()
    data.reset_index(inplace=True)
    X, y = split_features_labels(data)
    attrs = list(X.index)
    labels = list(y.index)
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = user_train_test_split(X, y,
                                                             test_size=0.2,
                                                             random_state=42)
    # drop uuid column, the timestamps, and the label source
    X_train = np.delete(X_train, [0, 1, 2], 1)
    X_test = np.delete(X_test, [0, 1, 2], 1)
    y_train = np.delete(y_train, -1, 1)
    y_test = np.delete(y_test, -1, 1)

    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])
    X_train = preprocess_pipeline.fit_transform(X_train)
    X_test = preprocess_pipeline.transform(X_test)

    """
    label_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    y_train = label_imputer.fit_transform(y_train)

    C_values = {
        "estimator__C": reciprocal(0.001, 1)
    }

    log_reg_clf = LogisticRegression(solver="lbfgs",
                                     max_iter=500,
                                     verbose=0,
                                     tol=3e-3,
                                     class_weight="balanced")
    clf = OneVsRestClassifier(log_reg_clf, n_jobs=1)
    ba_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True,
                            average="macro", zero_default=0)
    random_search = RandomizedSearchCV(clf, C_values, n_iter=20, cv=5,
                                       verbose=3, n_jobs=11, scoring=ba_scorer)
    random_search.fit(X_train, y_train)
    print(random_search.cv_results_)

    y_pred = random_search.predict(X_test)
    y_pred_bias = random_search.predict(X_train)
    """
    n_estimators = y_train.shape[1]
    log_reg_clf = LogisticRegression(solver="lbfgs",
                                     C=1,
                                     max_iter=500,
                                     verbose=0,
                                     tol=5e-4)
    log_reg_ovr = FlexOneVsRestClassifier(clf=log_reg_clf,
                                          n_estimators=n_estimators)

    log_reg_bal_clf = LogisticRegression(solver="lbfgs",
                                         C=1,
                                         max_iter=500,
                                         verbose=0,
                                         tol=5e-4,
                                         class_weight="balanced")
    log_reg_bal_ovr = FlexOneVsRestClassifier(clf=log_reg_bal_clf,
                                              n_estimators=n_estimators)

    random_forest_clf = xgb.XGBRFClassifier(max_depth=20,
                                            n_estimators=100,
                                            tree_method="gpu_hist",
                                            objective="binary:logistic")
    random_forest_ovr = FlexOneVsRestClassifier(clf=random_forest_clf,
                                                n_estimators=n_estimators)
    classifiers = {log_reg_ovr,
                   log_reg_bal_ovr,
                   random_forest_ovr}

    for clf in classifiers:
        print(clf.classifiers[0])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_bias = clf.predict(X_train)

        print("Balanced accuracy: ", balanced_accuracy_score(y_test.T,
                                                             y_pred,
                                                             average="macro",
                                                             zero_default=0))
        print("Balanced accuracy bias:", balanced_accuracy_score(y_train.T,
                                                                 y_pred_bias,
                                                                 average="macro",
                                                                 zero_default=0))
