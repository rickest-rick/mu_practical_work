import tensorflow as tf


class GmcLoss(tf.keras.losses.Loss):
    """
    Binary crossentropy loss modified by a generalized maximal correlation (GMC)
    regularization term.
    :author: Joschka Strüber
    """
    def __init__(self, y, alpha=1, **kwargs):
        self.coexist_counts = get_coexist_counts(y)
        self.alpha = alpha
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Call the generalized maximal correlation loss function for a 2d tensor
        of predictions and one of true labels.
        :author: Joschka Strüber
        :param y_true:
        :param y_pred:
        :return: 1d tensor, GMC loss for every input
        """
        # Compute masked binary crossentropy, setting the expected labels to the
        # predictions where they are nan is the same as masking, because the
        # log loss is zero where true and predicted label are the same.
        is_not_nan = tf.logical_not(tf.math.is_nan(y_true))
        y_true_clean = tf.where(is_not_nan, y_true, y_pred)
        bce = tf.keras.losses.BinaryCrossentropy()
        bce_loss = bce.call(y_true_clean, y_pred)

        # compute gmc regularization term omega
        dot_matrix = tf.matmul(tf.transpose(y_pred), y_pred)
        diag = - 0.5 * tf.linalg.diag_part(dot_matrix)
        dot_matrix = tf.linalg.set_diag(dot_matrix, diag)
        #todo: change coexist factor back
        dot_matrix = tf.math.multiply(self.coexist_counts, dot_matrix)
        #coexist_counts = get_coexist_counts(y_true)
        #dot_matrix = tf.math.multiply(coexist_counts, dot_matrix)
        pred_count = tf.dtypes.cast(tf.shape(y_pred)[0], dtype=dot_matrix.dtype)
        omega = - tf.divide(tf.reduce_sum(dot_matrix), pred_count)

        return bce_loss + self.alpha * omega

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


def get_coexist_counts(y):
    """
    Returns co-occurence matrix of y with itself as 2d tensor. NaN values are
    treated as zeros.
    :author: Joschka Strüber
    :param y: Labels as 1d or 2d tensor.
    :return:
    """
    is_not_nan = tf.math.logical_not(tf.math.is_nan(y))
    y_clean = tf.where(is_not_nan, y, 0)
    return tf.matmul(tf.transpose(y_clean), y_clean)
