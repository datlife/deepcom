"""Implementation of Viterbi Algorithm for Decoding Convolutional Codes

"""
import numpy as np
import commpy.channelcoding.convcode as cc
from ..utils import g_matrix_generator

def ViterbiDecoder(input_signal, constrain_length, rate=1/2):
  """Viterbi Decoder for Convolutional codes
  """
  memory = np.array([constrain_length - 1])
  g_matrix = g_matrix_generator(constrain_length, rate)
  trellis = cc.Trellis(memory, g_matrix)
  
  decoded_seqs = cc.viterbi_decode(
    coded_bits=input_signal, 
    trellis=trellis)

  return decoded_seqs

