import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score

from src.data_handling import load_user_data, split_features_labels


if __name__ == '__main__':
    # load data and reset index
    user_data = load_user_data()
    features_df, labels_df = split_features_labels(user_data)
    features_df.reset_index(inplace=True)

    # get uuid column and remove it from the labels
    uuid_df = features_df.iloc[:, 0].copy()
    index_cols = features_df.columns[[0, 1]]
    features_df.drop(index_cols, axis=1, inplace=True)

    X = features_df.values
    y = uuid_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp = imp.fit(X_train)
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)

    scaler = StandardScaler()
    scaler = scaler.fit(X_train_imp)
    X_train_sc = scaler.transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    param_dist = {'bootstrap': [True, False],
                  'max_depth': [10],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [30, 100, 200]
                  }

    clf = RandomForestClassifier(class_weight="balanced")
    random_search = RandomizedSearchCV(clf, param_dist, n_iter=3, cv=3,
                                       verbose=10, n_jobs=1,
                                       scoring='f1_weighted')

    random_search.fit(X_train_sc, y_train)
    final_clf = random_search.best_estimator_

    final_clf.fit(X_train_sc, y_train)
    preds = final_clf.predict(X_test_sc)

    score_f1 = f1_score(y_test, preds, average="weighted")
    print("F1 score: ", score_f1)


