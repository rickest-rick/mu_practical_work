import pandas as pd
import os


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


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#
# File handling provided by author
# Modified by TP because was written in python 2
#
#######################################################################################################################
import numpy as np;
from io import StringIO;



def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:', '');
        pass;

    return (feature_names, label_names);


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(csv_str, delimiter=',', skiprows=1);

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int);

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)];

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1];  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat);  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.;  # Y is the label matrix

    return (X, Y, M, timestamps);


'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''


def read_user_data(user_data_file):

    # Read the entire csv file of the user:
    with open(user_data_file, 'r') as fid:
        csv_str = fid.read();
        pass;

    (feature_names, label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    # ToDo: Fix this! the first paramter was a binary before, but this could no longer be handles in parse header of csv. THerefore
    #  here again the file path is given and the file loaded a second time. This might be bad for perfomance
    (X, Y, M, timestamps) = parse_body_of_csv(user_data_file, n_features);

    return (X, Y, M, timestamps, feature_names, label_names);


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'

    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;

    label = label.replace('__', ' (').replace('_', ' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m', 'I\'m');
    return label;

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
















