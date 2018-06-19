"""Implementation of Viterbi Algorithm for Decoding Convolutional Codes

"""
import numpy as np
import commpy.channelcoding.convcode as cc
from ..utils import g_matrix_generator

def ViterbiDecoder(corrupted_signals, constrain_length, rate=1/2):
  """Viterbi Decoder for Convolutional codes
  """
  memory = np.array([constrain_length - 1])
  g_matrix = g_matrix_generator(constrain_length, rate)
  trellis = cc.Trellis(memory, g_matrix)
  
  decoded_seqs = [None for _ in range(len(corrupted_signals))]
  for i, coded_bits in enumerate(corrupted_signals):
    decoded_seqs[i] = cc.viterbi_decode(
        coded_bits=coded_bits.astype(float), 
        trellis=trellis)

  return decoded_seqs

