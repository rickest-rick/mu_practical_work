import gc

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.data_handling import load_user_data, split_features_labels


if __name__ == '__main__':
    # load data and reset index
    user_data = load_user_data()
    features_df, labels_df = split_features_labels(user_data)
    user_data = None
    features_df.reset_index(inplace=True)

    # get uuid column and remove them, the timestamps and the label_source from
    # the labels
    y = features_df.iloc[:, 0].values
    index_cols = features_df.columns[[0, 1, 2, -1]]
    features_df.drop(index_cols, axis=1, inplace=True)
    X = features_df.values
    features_df = None
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # preprocess the data by setting NaN values to the mean and standard scaling
    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

    X_train = preprocess_pipeline.fit_transform(X_train)
    X_test = preprocess_pipeline.transform(X_test)

    # 5-fold CV with random search of hyperparams for random forest classifier
    param_dist = {'bootstrap': [True, False],
                  'max_depth': [20, 30, 40, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [30, 50, 100, 150]
                  }

    clf = RandomForestClassifier(class_weight="balanced")
    random_search = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5,
                                       verbose=10, n_jobs=3,
                                       scoring='f1_weighted')

    random_search.fit(X_train, y_train)
    final_clf = random_search.best_estimator_
    print(final_clf)

    final_clf.fit(X_train, y_train)
    preds = final_clf.predict(X_test)

    score_f1 = f1_score(y_test, preds, average="weighted")
    print("F1 score: ", score_f1)
