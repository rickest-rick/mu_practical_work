import tensorflow as tf

from tensorflow import keras


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
        epsilon = tf.convert_to_tensor(keras.backend.epsilon(), y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # compute crossentropy
        bce = tf.math.negative(y_true * tf.math.log(y_pred + 
                                                    keras.backend.epsilon()))
        bce -= (1 - y_true) * tf.math.log(1 - y_pred + keras.backend.epsilon())
        # mask nan_values with 0 and reduce to mean in each row
        is_not_nan = tf.math.logical_not(tf.math.is_nan(bce))
        nans_per_row = tf.math.count_nonzero(is_not_nan, axis=1, dtype=bce.dtype)
        bce = tf.where(is_not_nan, bce, 0)
        bce = tf.math.reduce_mean(bce, axis=1)
        
        # compute gmc regularization term omega
        dot_matrix = tf.matmul(tf.transpose(y_pred), y_pred)
        diag = tf.divide(tf.linalg.diag_part(dot_matrix), -2)
        dot_matrix = tf.linalg.set_diag(dot_matrix, diag)
        dot_matrix = tf.math.multiply(self.coexist_counts, dot_matrix)
        pred_count = tf.dtypes.cast(tf.shape(y_pred)[0], dtype=dot_matrix.dtype)
        omega = tf.divide(tf.reduce_sum(dot_matrix), pred_count)
        
        return bce - self.alpha * omega

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
    y = y + keras.backend.epsilon()
    is_not_nan = tf.math.logical_not(tf.math.is_nan(y))
    y_clean = tf.where(is_not_nan, y, 0)
    return tf.matmul(tf.transpose(y_clean), y_clean)
