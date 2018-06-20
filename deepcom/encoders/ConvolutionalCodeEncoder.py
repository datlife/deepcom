"""Similuate the Convolutional Encoder to encode message bits"""
import numpy as np
from commpy.channelcoding import convcode as cc

class ConvolutionalCodeEncoder(object):
  """Encode message bits using Convolutional Code Scheme
  """
  def __init__(self, generator_matrix, memory, code_type='default'):
    """Initialzie Convolution Code Encoder

    Args:
      constraint_length: 1D ndarray ints 
        The larger `constraint_length` K, the more times a particular 
        message bit is used when calculating parity bits.
      data_rate: number of parity bits for each message bit
    """
    self.trellis = cc.Trellis(memory, generator_matrix, code_type=code_type)

  def encode(self, message_bits):
    """Encode message bits
    Args:
      message_bits: 1D ndarray - with elements containg {0,1}.

    Returns:
      coded_bits: 1D ndarray - convolution encoded message bits.
    """
    coded_bits = cc.conv_encode(message_bits, trellis=self.trellis)
    return coded_bits


