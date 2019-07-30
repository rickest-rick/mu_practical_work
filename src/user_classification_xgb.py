import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score

from src.data_handling import load_user_data, split_features_labels


if __name__ == '__main__':
    # load data and reset index
    user_data = load_user_data()
    features_df, labels_df = split_features_labels(user_data)
    features_df.reset_index(inplace=True)

    # get uuid column and remove them, the timestamps and the label_source from
    # the labels
    y = features_df.iloc[:, 0].values
    index_cols = features_df.columns[[0, 1, 2, -1]]
    features_df.drop(index_cols, axis=1, inplace=True)
    X = features_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", max_depth=2,
                                  random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    score_f1 = f1_score(y_test, y_pred, average="weighted")
    print("F1 score: ", score_f1)
