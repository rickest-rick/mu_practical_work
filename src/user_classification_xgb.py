import gc

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb

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

    clf = xgb.XGBClassifier(objective="multi:softprob")
    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    score_f1 = f1_score(y_test, preds, average="weighted")
    print("F1 score: ", score_f1)
    print(confusion_matrix(y_test,preds))
