"""Implementation of Neural Decoder for rate-1/2 Recursive Systematic Convolutional (RSC)
 Codes. Mentioned in  second paragraph of 4th page in this paper:

Reference:
[1] Kim, Hyeji, et al. "Communication Algorithms via Deep Learning." ICLR (2018)
"""
from tensorflow.python.keras.layers import GRU, Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Bidirectional, TimeDistributed


def NRSCDecoder(inputs, is_training=False):
  """Definition of Neural Network Decoder for rate-1/2 (2 parity bits/ message bit)
  Recursive Systematic Convolutional Codes, aka "N-RSC" Decoder.

  Args:
    inputs: - tf.Tensor - shape [batch, sequence_length, 2] represents the
      noisy signals.
    is_training: - a boolean

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

