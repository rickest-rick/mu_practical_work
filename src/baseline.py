import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_handling import load_user_data, load_some_user_data, \
    split_features_labels, user_train_test_split
from src.metrics import balanced_accuracy_score


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

    label_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    y_train = label_imputer.fit_transform(y_train)

    log_reg_clf = LogisticRegression(class_weight="balanced", solver="lbfgs")
    clf = OneVsRestClassifier(log_reg_clf, n_jobs=-1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_bias = clf.predict(X_train)

    print("Balanced accuracy: ", balanced_accuracy_score(y_test.T, y_pred))
    print("Balanced accuracy bias:", balanced_accuracy_score(y_train.T, y_pred_bias))