"""Implementation of Neural Decoder for Convolutional Codes

Reference:
[1] Kim, Hyeji, et al. "Communication Algorithms via Deep Learning." ICLR (2018)
"""
from tensorflow.python.keras.layers import GRU, Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Bidirectional, TimeDistributed


def neural_decoder(inputs, is_training=False):
  """Neural Network Decoder Definition

  Args:
    inputs: - tf.Tensor - shape [batch, sequence_length, 2] represents the
      noisy data.
    is_training:

  Returns:
    x - tf. Tensor - shape [batch, sequence_length, 1] 
      decoded output
  """
  x = Bidirectional(GRU(
    units=400,
    return_sequences=True,
    trainable=is_training
  ))(inputs)

  x = BatchNormalization(trainable=is_training)(x)

  x = Bidirectional(GRU(
    units=400,
    return_sequences=True,
    trainable=is_training
  ))(x)

  x = BatchNormalization(trainable=is_training)(x)

  x = TimeDistributed(Dense(
    units=1,
    activation='sigmoid'),
    trainable=is_training,
  )(x)

  return x

