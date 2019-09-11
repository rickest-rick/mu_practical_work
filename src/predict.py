import sys
import numpy as np
import pandas as pd

from joblib import load
from data_handling import load_user_data

if __name__ == "__main__":
    args = sys.argv
    data_path = args[1]
    classifier_path = args[2]
    output_path = args[3]

    # read and process training data
    data = load_user_data(data_path)
    data.reset_index(inplace=True)
    feature_names = data.columns
    X = data.values

    # drop uuid column, the timestamps and the label source
    X = np.delete(X, [0, 1, 2, X.shape[1] - 1], 1)

    clf = load(classifier_path)
    y_pred = clf.predict(X)

    print(clf["clf"].label_names)

    df = pd.DataFrame(y_pred, columns=clf.label_names)
    df.to_csv(output_path)
