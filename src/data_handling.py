import pandas as pd
import os

import fancyimpute

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
    combined_user_frames = pd.concat(user_frames, keys=user_keys)
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


def user_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split a set of training and target data into training and test data sets
    based on grouping by uuids in the first column.
    :author: Joschka Strüber
    :param X: Training data as array
    :param y: Target data as 1d array
    :param test_size: Portion of the users used as test data. Default = 0.2
    :param random_state:
    :return: X_train, X_test, y_train, y_test
    """
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
    :return: imputed label matrix without NaN-values (as numpy array)
    """
    sparse_label_matrix = label_matrix.values

    full_label_matrix = fancyimpute.SoftImpute(max_iters=500, min_value=0,max_value=1).fit_transform(sparse_label_matrix)

    # threshold to convert imputed real values into binary ones. Threshold value is chosen empirically.
    full_label_matrix = np.where(full_label_matrix < 10-2, full_label_matrix, 1)
    full_label_matrix = np.where(full_label_matrix >= 10e-2, full_label_matrix, 0)

    return full_label_matrix


def convert_to_int(dictionary, int_keys):
    """
    # todo
    :author: Joschka Strüber
    :param dictionary:
    :param int_keys:
    :return:
    """
    for int_key in int_keys:
        dictionary[int_key] = int(dictionary[int_key])
    return dictionary
