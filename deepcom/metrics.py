import tensorflow as tf

def ber(y, y_pred):
    """Measure Bit Error Rate (BER)
    
    Args:
        y - tf.Tensor - shape (batch_size, K, 1)
        y_pred - tf.Tensor - shape (batch_size, K, 1)

    Returns:
        ber - a tf.float - represents bit error rate
            in a batch.
    """
    hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
    ber = tf.reduce_sum(hamming_distances) / tf.size(y)
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
    num_blocks_per_batch = tf.cast(tf.shape(y)[0], tf.int64)
    hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
    bler = tf.count_nonzero(tf.reduce_sum(hamming_distances, axis=1)) / num_blocks_per_batch
    return bler
