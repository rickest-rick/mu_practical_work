import numpy as np

from sklearn.metrics import confusion_matrix


def balanced_accuracy_score(y_true, y_pred, average="macro", zero_default=1):
    """
    Compute the balanced accuracy for a binary multilabel classification. The BA
    is computed as (Recall + Specifity) / 2. NaN values are ignored.
    :author: Daniel Beckmann, Joschka Strüber
    :param test_labels: The true labels as a 2d array.
    :param preds: Predicted labels by a classifier as a 2d array.
    :param average: str, "macro" (default) or "micro"
        "micro" -> compute the global recall and specificity by incorporating
            the sum of all confusion matrices for every label
        "macro" -> compute the global recall and specificty as the mean of the
            individual recalls and specificities for every label
    :param zero_default: number, 1 (default)
        Which value should be used as default for the recall and specificity, if
        the sum of positive or negative samples is 0.
    :return: The balanced accuracy as float.
    """
    if y_true.ndim == 1:
        is_NaN = np.isnan(y_true)
        y_true = y_true[~is_NaN]
        y_pred = y_pred[~is_NaN]
        return single_balanced_accuracy_score(y_true, y_pred)

    if average == "micro":
        return balanced_accuracy_score_micro(y_true, y_pred, zero_default)
    elif average == "macro":
        return balanced_accuracy_score_macro(y_true, y_pred)
    else:
        raise ValueError('Invalid average method chosen: "{}"'.format(average))


def balanced_accuracy_score_micro(y_true, y_pred, zero_default=0.5):
    true_neg = 0
    true_pos = 0
    neg = 0
    pos = 0

    n_labels = y_true.shape[1]
    for label_set in range(n_labels):
        label_true = y_true[:, label_set]
        label_pred = y_pred[:, label_set]

        # remove NaN values
        is_NaN = np.isnan(label_true)
        label_true = label_true[~is_NaN]
        label_pred = label_pred[~is_NaN]

        conf_matrix = confusion_matrix(label_true, label_pred)
        # count true negatives, true positives, all negatives and all positives
        # for each attribute
        if conf_matrix.shape == (0, 0):  # empty confusion matrix
            continue

        if conf_matrix.shape == (2, 2):  # not all positive or negative
            true_neg += conf_matrix[0, 0]
            true_pos += conf_matrix[1, 1]
            neg += conf_matrix[0, 0] + conf_matrix[1, 0]
            pos += conf_matrix[1, 1] + conf_matrix[0, 1]
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
    return (recall + specificity) / 2


def balanced_accuracy_score_macro(y_true, y_pred):
    sum_balanced_accuracy = 0.0
    n_labels = y_true.shape[1]
    n_labels_nonempty = n_labels
    for label_set in range(n_labels):
        label_true = y_true[:, label_set]
        label_pred = y_pred[:, label_set]

        if np.count_nonzero(~np.isnan(label_true)) == 0:  # all nan
            n_labels_nonempty -= 1
        else:
            balanced_accuracy = single_balanced_accuracy_score(label_true,
                                                               label_pred)
            sum_balanced_accuracy += balanced_accuracy
    # micro average is average of all specificity and recall values
    if n_labels_nonempty != 0:
        balanced_accuracy = sum_balanced_accuracy / n_labels_nonempty
    else:
        balanced_accuracy = 1

    return balanced_accuracy


def single_balanced_accuracy_score(y_true, y_pred):
    # remove NaN values
    is_nan = np.isnan(y_true)
    y_true = y_true[~is_nan]
    y_pred = y_pred[~is_nan]
    if y_true.size == 0:
        return 1

    conf_matrix = confusion_matrix(y_true, y_pred)
    if conf_matrix.shape == (2, 2):  # not all positive or negative
        true_neg = conf_matrix[0, 0]
        true_pos = conf_matrix[1, 1]
        neg = conf_matrix[0, 0] + conf_matrix[1, 0]
        pos = conf_matrix[1, 1] + conf_matrix[0, 1]

        # compute the specificity and recall or set them to the default
        # value, if this is not possible
        if neg != 0 and pos != 0:
            specificity = float(true_neg) / neg
            recall = float(true_pos) / pos
        else:
            specificity = 0.5
            recall = 0.5
    else:
        if y_true[0] == 0:  # all negative
            true_neg = conf_matrix[0, 0]
            neg = conf_matrix[0, 0]

            specificity = float(true_neg) / neg
            recall = 1
        else:  # all positive
            true_pos = conf_matrix[0, 0]
            pos = conf_matrix[0, 0]

            specificity = 1
            recall = float(true_pos) / pos
    return (specificity + recall) / 2.
