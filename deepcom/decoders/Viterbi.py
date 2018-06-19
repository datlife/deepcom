"""Viterbi Algorithm for Decoding Convolutional Codes"""
import numpy as np
import commpy.channelcoding.convcode as cc

#TODO: add test
def viterbi_decoder(encoded_bitsream, block_length, rate=1/2):
  """Viterbi Decoder for Convolutional codes

  """
  decoded_seqs = [None for _ in range(len(encoded_bitsream))]
  generator_matrix = gen_gmatrix(block_length, rate)
  memory = np.array([block_length - 1])
  trellis = cc.Trellis(memory, generator_matrix)
  
  for i, encoded_sequence in enumerate(encoded_bitsream):
    decoded_seq = cc.viterbi_decode(encoded_sequence.astype(float), trellis)
    decoded_seqs[i] = decoded_seq

  return decoded_seqs