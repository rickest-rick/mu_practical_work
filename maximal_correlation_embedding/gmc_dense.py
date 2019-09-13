import tensorflow as tf

from tensorflow import keras


class GmcDense(keras.layers.Dense):
    """
    Modified dense layer which can return a GMC loss function based on a batch
    of embedding vectors.
    :author: Joschka Strüber
    """
    def __init__(self, units, alpha, y, **kwargs):
        super(GmcDense, self).__init__(units, **kwargs)
        coexist_counts = get_coexist_counts(y)
        self.coexist_counts = tf.Variable(initial_value=coexist_counts,
                                          trainable=False, dtype=tf.float32)
        self.alpha = tf.Variable(initial_value=alpha, trainable=False,
                                 dtype=tf.float32)
        self.n_samples = tf.Variable(initial_value=y.shape[0], trainable=False,
                                     dtype=tf.float32)

    def gmc_loss(self, y_true, y_pred):
        """
        Call the generalized maximal correlation loss function for a 2d tensor
        of predictions and one of true labels.
        :author: Joschka Strüber
        :param y_true: 1d tensor, target data
        :param y_pred: 1d tensor, predictions
        :return: 1d tensor, GMC loss for every input
        """
        epsilon = tf.convert_to_tensor(keras.backend.epsilon(),
                                       y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # compute crossentropy
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce = tf.negative(bce)
        bce -= (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        # mask nan_values with 0 and reduce to mean in each row
        is_not_nan = tf.math.logical_not(tf.math.is_nan(bce))
        bce = tf.where(is_not_nan, bce, 0)
        bce = tf.math.reduce_mean(bce, axis=1)

        # compute gmc regularization term omega
        dot_matrix = tf.matmul(tf.transpose(self.kernel), self.kernel)
        diag = tf.divide(tf.linalg.diag_part(dot_matrix), -2)
        dot_matrix = tf.linalg.set_diag(dot_matrix, diag)
        dot_matrix = tf.math.multiply(self.coexist_counts, dot_matrix)
        weighted_dot_sum = tf.reduce_sum(dot_matrix)
        omega = tf.negative(tf.divide(weighted_dot_sum, self.n_samples))

        return bce + self.alpha * omega
    
    
class MCAlphaDropout(keras.layers.AlphaDropout):
    """
    Wrapper class that always enables training mode in an alpha dropout layer.
    Used for Monte Carlo Dropout method
    """
    def call(self, inputs):
        return super().call(inputs, training=True)
    
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
