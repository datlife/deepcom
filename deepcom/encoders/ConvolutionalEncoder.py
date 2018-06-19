"""Similuate the encoder to generate noisy data"""
import numpy as np 
import commpy.channelcoding.convcode as cc


def ConvolutionalEncoder(message_bits, generator_matrix, memory):
  """Encode a bitstream using Convolutional Code Scheme
  
  Args:
    message_bits:
    generator_matrix:
    memory:

  Returns:

  """
  trellis = cc.Trellis(memory, generator_matrix)
  coded_bits = cc.conv_encode(message_bits, trellis)
  return coded_bits


