import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


def balanced_accuracy_score(y_true, y_pred):
    """
    Compute the balanced accuracy for a binary multilabel classification. The BA
    is computed as (Recall + Specifity) / 2. NaN values are ignored.
    :author: Daniel Beckmann, Joschka Strüber
    :param test_labels: The true labels as a 2d array.
    :param preds: Predicted labels by a classifier as a 2d array.
    :return: The balanced accuracy as float.
    """
    true_neg = 0
    true_pos = 0
    neg = 0
    pos = 0

    for label_set in range(y_true.shape[0]):
        label_true = y_true[label_set]
        label_pred = y_pred[label_set]

        # remove NaN values
        is_NaN = np.isnan(label_true)
        label_true = label_true[~is_NaN]
        label_pred = label_pred[~is_NaN]

        conf_matrix = confusion_matrix(label_true, label_pred)
        # todo delete debug print
        print(label_set, "\n", conf_matrix)
        # count true negatives, true positives, all negatives and all positives
        # for each attribute
        if conf_matrix.shape == (0, 0):  # empty confusion matrix
            continue
        if conf_matrix.shape == (2, 2):  # not all positive or negative
            true_neg += conf_matrix[0, 0]
            true_pos += conf_matrix[1, 1]
            neg += conf_matrix[0, 0] + conf_matrix[1, 0]
            pos += conf_matrix[1, 1] + conf_matrix[0, 1]
            # todo delete debug print
            #print(label_set)
            #print(float(conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[1, 0]))
            #print(float(conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1]))
        else:
            if label_true[0] == 0:  # all negative
                true_neg += conf_matrix[0, 0]
                neg += conf_matrix[0, 0]
            else:  # all positive
                true_pos += conf_matrix[0, 0]
                pos += conf_matrix[0, 0]
    if neg != 0:
        specificity = float(true_neg) / neg
    else:
        specificity = 1
    if pos != 0:
        recall = float(true_pos) / pos
    else:
        recall = 1
    print("Neg: {}, True Neg: {}".format(neg, true_neg))
    print("Pos: {}, True Pos: {}".format(pos, true_pos))
    return (recall + specificity) / 2

