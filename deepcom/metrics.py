import tensorflow as tf
# =============================
# Define Metrics
# =============================
def ber(y, y_pred):
    """Measure Bit Error Rate (BER)
    
    Args:
        y - tf.Tensor - shape (batch_size, K, 1)
        y_pred - tf.Tensor - shape (batch_size, K, 1)

    Returns:
        ber - a tf.float - represents bit error rate
            in a batch.
    """
    errors =  tf.cast(tf.where(y != y_pred), tf.int32)
    ber = tf.reduce_sum(errors) / tf.size(y)
    return ber

def bler(y, y_pred):
    """Measure Bit Block Error Rate (BER)
    
    Args:
        y - tf.Tensor - shape (batch_size, K, 1)
        y_pred - tf.Tensor - shape (batch_size, K, 1)

    Returns:
        bler - a tf.float - represents bit block error rate
            in a batch.
    """
    assert y.shape.as_list() == y_pred.shape.as_list(), \
    "Prediction and Ground truth must have same shape.\n"\
    "GT:{0} Prediction:{1}".format(y.shape.as_list(), y_pred.shape.as_list())

    errors = tf.where(y != y_pred)
    bler = float(len(error[0])) / tf.size(y)
    return bler

# def accuracy(y, y_pred):

#     # Make sure shapes are matched
#     assert y.shape == y_pred.shape, \
#     "Prediction and Ground truth must have same shape.\n"\
#     "Expected:{0} Actual:{1}".format(y.shape, y_pred.shape)

#     return np.sum(y == y_pred) / np.product(y.shape)
