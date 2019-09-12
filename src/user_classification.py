import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import dump, load

from data_handling import load_user_data, split_features_labels


if __name__ == '__main__':
    # load data and reset index
    user_data = load_user_data()
    features_df, labels_df = split_features_labels(user_data)
    del user_data
    features_df.reset_index(inplace=True)

    # get uuid column and remove them, the timestamps and the label_source from
    # the labels
    y = features_df.iloc[:, 0].values
    index_cols = features_df.columns[[0, 1, 2, -1]]
    features_df.drop(index_cols, axis=1, inplace=True)
    feature_names = features_df.columns
    X = features_df

    # train-test split and wrap output in data frame to save column names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=41)

    # preprocess the data by setting NaN values to the mean and standard scaling
    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

    X_train = pd.DataFrame(preprocess_pipeline.fit_transform(X_train),
                           columns=feature_names)
    X_test = pd.DataFrame(preprocess_pipeline.transform(X_test),
                          columns=feature_names)

    clf = xgb.XGBClassifier(colsample_bytree=0.9,
                            n_estimators=200,
                            learning_rate=0.04,
                            max_depth=8,
                            subsample=0.9,
                            gamma=0.05,
                            objective="multi:softprob",
                            tree_method="gpu_hist")

    clf.fit(X_train, y_train)
    xgb.plot_importance(clf, max_num_features=20)
    xgb.plot_tree(clf)
    fig = plt.gcf()
    fig.set_size_inches(80, 8)
    plt.show()

    preds = clf.predict(X_test)
    score_f1 = f1_score(y_test, preds, average="weighted")
    print("F1 score: ", score_f1)
    print(confusion_matrix(y_test, preds))
