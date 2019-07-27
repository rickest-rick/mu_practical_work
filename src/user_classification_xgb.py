import gc
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
from joblib import dump, load

from data_handling import load_some_user_data, split_features_labels


if __name__ == '__main__':
    # load data and reset index
    user_data = load_some_user_data()
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

    param_dist = {
        "colsample_bytree": uniform(0.3, 0.7),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.02, 0.3),
        "max_depth": randint(2, 9),
        "n_estimators": randint(100, 200),
        "subsample": uniform(0.4, 0.6)
    }

    clf = xgb.XGBClassifier(objective="multi:softprob")
    random_search = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5,
                                       verbose=3, n_jobs=-1,
                                       scoring='f1_weighted')
    random_search.fit(X_train, y_train)
    print(random_search.best_estimator_)
    dump(random_search.best_estimator_, "user_class_xgb.joblib")

    preds = random_search.predict(X_test)

    score_f1 = f1_score(y_test, preds, average="weighted")
    print("F1 score: ", score_f1)
    print(confusion_matrix(y_test, preds))
