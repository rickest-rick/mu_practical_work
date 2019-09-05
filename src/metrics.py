import numpy as np

from sklearn.metrics import confusion_matrix


def balanced_accuracy_score(y_true, y_pred, average="micro", zero_default=1):
    """
    Compute the balanced accuracy for a binary multilabel classification. The BA
    is computed as (Recall + Specifity) / 2. NaN values are ignored.
    :author: Daniel Beckmann, Joschka Strüber
    :param test_labels: The true labels as a 2d array.
    :param preds: Predicted labels by a classifier as a 2d array.
    :param average: str, "micro" (default) or "macro"
        "micro" -> compute the global recall and specificity by incorporating
            the sum of all confusion matrices for every label
        "macro" -> compute the global recall and specificty as the mean of the
            individual recalls and specificities for every label
    :param zero_default: number, 1 (default)
        Which value should be used as default for the recall and specificity, if
        the sum of positive or negative samples is 0.
    :return: The balanced accuracy as float.
    """
    if average == "micro":
        return balanced_accuracy_score_micro(y_true, y_pred, zero_default)
    elif average == "macro":
        return balanced_accuracy_score_macro(y_true, y_pred, zero_default)
    else:
        raise ValueError('Invalid average method chosen: "{}"'.format(average))


def balanced_accuracy_score_micro(y_true, y_pred, zero_default=1):
    true_neg = 0
    true_pos = 0
    neg = 0
    pos = 0

    n_labels = y_true.shape[0]
    for label_set in range(n_labels):
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
            # print(label_set)
            # print(float(conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[1, 0]))
            # print(float(conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1]))
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
        specificity = zero_default
    if pos != 0:
        recall = float(true_pos) / pos
    else:
        recall = zero_default
    print("Neg: {}, True Neg: {}".format(neg, true_neg))
    print("Pos: {}, True Pos: {}".format(pos, true_pos))
    return (recall + specificity) / 2


def balanced_accuracy_score_macro(y_true, y_pred, zero_default=1):
    sum_specificity = 0.0
    sum_recall = 0.0

    n_labels = y_true.shape[0]
    for label_set in range(n_labels):
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
            true_neg = conf_matrix[0, 0]
            true_pos = conf_matrix[1, 1]
            neg = conf_matrix[0, 0] + conf_matrix[1, 0]
            pos = conf_matrix[1, 1] + conf_matrix[0, 1]

            # compute the specificity and recall or set them to the default
            # value, if this is not possible
            specificity = float(true_neg) / neg if neg != 0 else zero_default
            recall = float(true_pos) / pos if pos != 0 else zero_default
        else:
            if label_true[0] == 0:  # all negative
                true_neg = conf_matrix[0, 0]
                neg = conf_matrix[0, 0]

                specificity = float(true_neg) / neg
                recall = zero_default
            else:  # all positive
                true_pos = conf_matrix[0, 0]
                pos = conf_matrix[0, 0]

                specificity = zero_default
                recall = float(true_pos) / pos
        # todo delete debug print
        # print(label_set)
        # print(specificity)
        # print(recall)
        sum_specificity += specificity
        sum_recall += recall
    # micro average is average of all specificity and recall values
    specificity = sum_specificity / n_labels
    recall = sum_recall / n_labels

    print("Specificity: {:.3f}".format(specificity))
    print("Recall: {:.3f}".format(recall))
    return (recall + specificity) / 2


