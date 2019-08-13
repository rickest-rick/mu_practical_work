import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix


def balanced_accuracy_score(test_labels, preds):
    """
    Compute the balanced accuracy for a multilabel classification. The BA is
    computed as (Recall + Specifity) / 2.
    :author: Daniel Beckmann, Joschka Strüber
    :param test_labels: The true labels as a 2d array.
    :param preds: Predicted labels by a classifier as a 2d array.
    :return: The balanced accuracy as float.
    """
    true_neg = 0
    neg = 0

    ml_conf_matrix = multilabel_confusion_matrix(test_labels, preds)
    # count true negatives and all negatives for each attribute
    for i in range(np.shape(ml_conf_matrix)[0]):
        conf_matrix = ml_conf_matrix[i]
        true_neg += conf_matrix[0, 0]
        neg += conf_matrix[0, 0] + conf_matrix[0, 1]

    specificity = float(true_neg) / neg
    recall = recall_score(test_labels, preds, average="micro")

    return (recall + specificity) / 2