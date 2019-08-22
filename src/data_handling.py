import pandas as pd
import numpy as np
import os
import fancyimpute
import pickle
from random import randint

from sklearn.model_selection import GroupShuffleSplit

data_path = "../data"


def load_user_data(path=data_path):
    """
    Load the csv files of all users and combine them in a single pandas data
    frame.
    :author: Joschka Strüber
    :param path: The directory of the csv files.
    :return: The combined data frame.
    """
    user_keys = []
    user_frames = [process_user(f, user_keys) for f in list_file_paths(path)]
    combined_user_frames = pd.concat(user_frames, keys=user_keys,sort=False)
    return combined_user_frames


def load_some_user_data(path=data_path):
    """
    Load the csv files of some (5) users and combine them in a single pandas data
    frame.
    :author: Daniel Beckmann
    :param path: The directory of the csv files.
    :return: The combined data frame.
    """
    user_keys = []
    all_users = list_file_paths(path)
    some_users = all_users[0:5]
    user_frames = [process_user(f, user_keys) for f in some_users]
    combined_user_frames = pd.concat(user_frames, keys=user_keys)
    return combined_user_frames


def process_user(csv_file, keys):
    """
    Reads a user's csv file into a data frame and saves the user's UUID as a
    key.
    :author: Joschka Strüber
    :param csv_file: The path to the csv file.
    :param keys: A list of user UUIDs.
    :return: The read file as pandas data frame.
    """
    file = pd.read_csv(csv_file)
    base = os.path.basename(csv_file)
    keys.append(base.split(os.extsep)[0])
    return file


def list_file_paths(input_path):
    """
    Return all files in a given directory with their paths.
    :author: Joschka Strüber
    """
    return [os.path.join(path, file)
            for (path, dirs, files) in os.walk(input_path)
            for file in files]


def split_features_labels(frame):
    """
    Split a data frame into a frame containing features and another one with
    labels, indicated by starting with 'label:'
    :author: Joschka Strüber
    :param frame: A data frame with user data and labels.
    :return: (Features, Labels)
    """
    label_cols = [col for col in frame.columns if col.startswith("label:")]
    features = frame.drop(label_cols, axis=1)
    labels = frame[label_cols]
    return features, labels


def user_train_test_split(X, y, test_size=0.2, random_state=2):
    user_group = X[:, 0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    for ii, (tr, tt) in enumerate(splitter.split(X=X, groups=user_group)):
        X_train = X[tr]
        y_train = y[tr]
        X_test = X[tt]
        y_test = y[tt]
    return X_train, X_test, y_train, y_test

def impute_missing_labels(label_matrix):
    """
    Impute missing labels (NaN) by using fancyimpute by Alex Rubinsteyn (github.com/iskandr)
    :author: Daniel Beckmann
    :param label_matrix: pandas data frame containing the labels for each data sample
    :return: imputed label matrix without NaN-values
    """
    sparse_label_matrix = label_matrix.values
    size = np.shape(sparse_label_matrix)[0] * np.shape(sparse_label_matrix)[1]
    print("Label matrix shape: {}".format(np.shape(sparse_label_matrix)))


    ones = np.where(sparse_label_matrix == 1, sparse_label_matrix, 0)
    print("Non-zero values in label matrix: {}".format(float(np.count_nonzero(ones)) / size))

    # full_label_matrix = fancyimpute.KNN(k=3).fit_transform(sparse_label_matrix)

    #full_label_matrix_normalized = fancyimpute.BiScaler().fit_transform(sparse_label_matrix)
    # full_label_matrix = fancyimpute.SoftImpute(max_iters=500, min_value=0,max_value=1).fit_transform(sparse_label_matrix)
    # with open("matrix","wb") as f:
        # pickle.dump(full_label_matrix,f)

    with open("matrix","rb") as f:
        full_label_matrix = pickle.load(f)
    sparse_label_matrix_imputed = fancyimpute.SimpleFill(fill_method="zero").fit_transform(sparse_label_matrix)
    sparse_ratios = [np.average(sparse_label_matrix_imputed[:,x],0) for x in range(51)]


    full_label_matrix = np.where(full_label_matrix < 10e-2, full_label_matrix, 1)
    full_label_matrix = np.where(full_label_matrix >= 10e-2, full_label_matrix, 0)

    print("")
    print("Full Label matrix shape: {}".format(np.shape(full_label_matrix)))
    ones = np.where(full_label_matrix == 1,full_label_matrix, 0)
    print("Non-zero values in full label matrix: {}".format(float(np.count_nonzero(ones)) / size))
    full_ratios = [np.average(full_label_matrix[:, x], 0) for x in range(51)]

    print("")
    print("Factors between label appearance ratio:")
    print(sparse_ratios)
    print(full_ratios)
    print([full_ratios[i]/sparse_ratios[i] for i in range(51)])




if __name__ == "__main__":
    data = load_user_data()
    attr, labels = split_features_labels(data)
    attr_values = attr.values
    labels_values = labels.values
    # user_train_test_split(attr_values, labels_values)

    impute_missing_labels(labels)
