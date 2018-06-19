"""Similuate the Convolutional Encoder to encode message bits"""
import numpy as np
from commpy.channelcoding import convcode as cc
from ..utils import g_matrix_generator


class ConvolutionalCodeEncoder(object):
  """Encode message using Convolutional Code Scheme

  It generates sequences of parity bits from sequences of message bits
    by a `convolution` operation:
  
  """
  def __init__(self, constraint_length, data_rate=1/2):
    """Initialzie Encoder

    Args:
      constraint_length: 1D ndarray ints 
        The larger `constraint_length` K, the more times a particular 
        message bit is used when calculating parity bits.
  
      data_rate: 
    """

    memory = np.array([constraint_length - 1])
    #  A Generator Matrix G for Conv. Code Encoder.
    #  shape = 2-D arrays of ints (octal representation)
    g_matrix = g_matrix_generator(constraint_length, data_rate)
    self.trellis = cc.Trellis(memory, g_matrix)

  def encode(self, message_bits):
    """Encode message bits
    Args:
      message_bits: 1D ndarray - with elements containg {0,1}.

    Returns:
      coded_bits: 1D ndarray - convolution encoded message bits.
    """
    coded_bits = cc.conv_encode(message_bits, trellis=self.trellis)
    return coded_bits


