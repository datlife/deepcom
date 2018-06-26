"""Implementation of Neural Decoder for rate-1/2 Recursive Systematic Convolutional (RSC)
 Codes. Mentioned in  second paragraph of 4th page in this paper:

Reference:
[1] Kim, Hyeji, et al. "Communication Algorithms via Deep Learning." ICLR (2018)
"""
from tensorflow.python.keras.layers import GRU, LSTM, Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Bidirectional, TimeDistributed


def NRSCDecoder(x, 
                is_training=False, 
                layer='gru', 
                direction='bidirectional', 
                num_layers=2, 
                hidden_units=400, 
                dropout=0.5):
  """Definition of Neural Network Decoder for rate-1/2 (2 parity bits/ message bit)
  Recursive Systematic Convolutional Codes, aka "N-RSC" Decoder.

  Args:
    x: - tf.Tensor - shape [batch, sequence_length, 2] represents the
      noisy signals.
    is_training: - a boolean
    layer: str - type of rnn layer (only 'gru' or 'lstm')
    direction: str   'bidirectional' or 'unidirectional'
    num_layers - int - number of hidden layers
    hidden_units: int - number of hidden units per layer
    dropout: -float - drop out rate during training

  Returns:
    x - tf. Tensor - shape [batch, sequence_length, 1] 
      decoded output

  Raise:
    ValueError: if `layer` or `direction` is invalid input
  """
  for _ in range(num_layers):
    
    if layer=='gru':
      inner_layer = GRU(
        units=hidden_units,
        return_sequences=True,
        trainable=is_training,
        recurrent_dropout=dropout)
    elif layer=='lstm':
      inner_layer = GRU(
        units=hidden_units,
        return_sequences=True,
        trainable=is_training,
        recurrent_dropout=dropout)
    else:
      raise ValueError('Invalid `layer` parameter'
          '(only GRU or LSTM).')

    if direction=='bidirectional':
      x = Bidirectional(inner_layer)(x)
    elif direction=='unidirectional':
      x = inner_layer(x)
    else:
      raise ValueError('Invalid `direction` parameter'
          '(only bidirectional or unidirectional).')
    x = BatchNormalization(trainable=is_training)(x)
  x = TimeDistributed(Dense(
      units=1, 
      activation='sigmoid',
      trainable=is_training))(x)
  return x
