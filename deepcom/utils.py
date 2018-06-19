import math
import numpy as np

#######################################
# Noise Helper Function
#######################################
def corrupt_signal(input_signal, noise_type, sigma = 1.0, vv =5.0):
    '''Simulate data corruption over a channel'''

     # input_signal has to be a numpy array.
    data_shape = input_signal.shape 
    if noise_type == 'awgn':
        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 't-dist':
        noise = sigma * math.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    return corrupted_signal


def generate_random_binary_sequence(seq_len, p=0.5):
  """Generate message bits length `seq_len` of a random binary 
  sequence, where each bit picked is a one with probability p.

  Args:
    seq_len: - int - length of message bit
    p - float - probability

  Return:
    seq: - 1D ndarray - represent a message bits
  """
  seq = np.zeros(seq_len)
  for i in range(seq_len):
    seq[i] = 1 if (np.random.random() < p) else 0
  return seq

def g_matrix_generator(L, data_date):
  """
  For a rate of 1/2 or of 1/3, and a given constraint length L, generates the
  commonly used G matrix. The default memory assumed is L-1.
  This function has the functionality equivalent of a lookup table.
  [Credits to pg. 789-790, Proakis and Salehi 2nd edition]
  In such conv coding, the azssumed k is 1, and the n is k * rate. Further, such
  sequences are worked on one symbol at a time.
  """
  if data_date == 1/3:
    dict_l_generator = {
      3: [5, 7, 7],
      4: [13, 15, 17],
      5: [25, 33, 37],
      6: [47, 53, 75],
      7: [133, 145, 175],
      8: [255, 331, 367],
      9: [557, 663, 711]
      }

  elif data_date == 1/2:
    dict_l_generator = {
      3: [5, 7],
      4: [15, 17],
      5: [23, 35],
      6: [53, 75],
      7: [133, 171],
      8: [247, 371],
      9: [561, 753]
    }

  else:
    assert False, "This rate is currently not supported, {}".format(str(data_date))

  return np.array([dict_l_generator[L]])

