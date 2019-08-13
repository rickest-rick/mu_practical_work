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

    for label in range(y_true.shape[0]):
        label_true = y_true[label]
        label_pred = y_pred[label]

        # remove NaN values
        is_NaN = np.isnan(label_true)
        label_true = label_true[~is_NaN]
        label_pred = label_pred[~is_NaN]

        conf_matrix = confusion_matrix(label_true, label_pred)
        # count true negatives, true positives, all negatives and all positives
        # for each attribute
        if conf_matrix.shape == (2, 2):  # not all positive or negative
            true_neg += conf_matrix[0, 0]
            true_pos += conf_matrix[1, 1]
            neg += conf_matrix[0, 0] + conf_matrix[0, 1]
            pos += conf_matrix[1, 1] + conf_matrix[1, 0]
        else:
            if label_true[0] == 0:  # all negative
                true_neg += conf_matrix[0, 0]
                neg += conf_matrix[0, 0]
            else:  # all positive
                true_pos += conf_matrix[0, 0]
                pos += conf_matrix[0, 0]

    specificity = float(true_neg) / neg
    recall = float(true_pos) / pos
    print("P: {}, N: {}, TP: {}, TN: {}".format(pos, neg, true_pos, true_neg))
    print("Spec: {0}, Recall: {1}".format(specificity, recall))
    return (recall + specificity) / 2

