import numpy as np
import commpy as cp
import tensorflow as tf
from .utils import awgn_channel



def data_generator(inputs, labels, batch_size, shuffle=True):
    """Construct a data generator using tf.Dataset"""
    dataset  = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset  = dataset.batch(batch_size)
    dataset  = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset    
    

def create_dataset(num_sequences, block_length, trellis, snr, seed):
    X = []
    Y = []
    # Init seed
    np.random.seed(seed)
    for _ in range(num_sequences):
        ground_truth = generate_message_bits(block_length)

        # Simulates data sent over AWGN channel
        coded_bits = cp.channelcoding.conv_encode(ground_truth, trellis)
        noisy_bits = awgn_channel(coded_bits, snr)

        # Ignore the last 4 bits        
        X.append(noisy_bits[: 2*block_length])
        Y.append(ground_truth)

    np.random.seed()

    X = np.reshape(X, (-1, block_length, 2))
    Y = np.reshape(Y, (-1, block_length, 1))
    return X, Y


def generate_message_bits(seq_len, p=0.5):
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
  

# #######################################
# # Build RNN Feed Helper Function (for Turbo Code only, need to refactor)
# #######################################

# def build_rnn_data_feed(num_block, block_len, noiser, codec, is_all_zero = False ,is_same_code = False, **kwargs):
#     '''
#     :param num_block:
#     :param block_len:
#     :param noiser: list, 0:noise_type, 1:sigma,     2:v for t-dist, 3:radar_power, 4:radar_prob
#     :param codec:  list, 0:trellis1,   1:trellis2 , 2:interleaver
#     :param kwargs:
#     :return: X_feed, X_message
#     '''

#     # Unpack Noiser
#     noise_type  = noiser[0]
#     noise_sigma = noiser[1]
#     vv          = 5.0
#     radar_power = 20.0
#     radar_prob  = 5e-2
#     denoise_thd = 10.0
#     snr_mix     = [0, 0, 0]

#     if noise_type == 't-dist':
#         vv = noiser[2]
#     elif noise_type == 'awgn+radar' or noise_type == 'hyeji_bursty':
#         radar_power = noiser[3]
#         radar_prob  = noiser[4]

#     elif noise_type == 'awgn+radar+denoise' or noise_type == 'hyeji_bursty+denoise':
#         radar_power = noiser[3]
#         radar_prob  = noiser[4]
#         denoise_thd = noiser[5]

#     elif noise_type == 'mix_snr_turbo' or noise_type == 'random_snr_turbo':
#         snr_mix = noiser[6]

#     elif noise_type == 'customize':
#         '''
#         TBD, noise model shall be open to other user, for them to train their own decoder.
#         '''

#         print '[Debug] Customize noise model not supported yet'
#     else:  # awgn
#         pass

#     #print '[Build RNN Data] noise type is ', noise_type, ' noiser', noiser

#     # Unpack Codec
#     trellis1    = codec[0]
#     trellis2    = codec[1]
#     interleaver = codec[2]


#     p_array     = interleaver.p_array

#     X_feed = []
#     X_message = []

#     same_code = np.random.randint(0, 2, block_len)

#     for nbb in range(num_block):
#         if is_same_code:
#             message_bits = same_code
#         else:
#             if is_all_zero == False:
#                 message_bits = np.random.randint(0, 2, block_len)
#             else:
#                 message_bits = np.random.randint(0, 1, block_len)

#         X_message.append(message_bits)

#         rnn_feed_raw = np.stack([sys_r, par1_r, np.zeros(sys_r.shape), intleave(sys_r, p_array), par2_r], axis = 0).T
#         rnn_feed = rnn_feed_raw

#         X_feed.append(rnn_feed)

#     X_feed = np.stack(X_feed, axis=0)

#     X_message = np.array(X_message)
#     X_message = X_message.reshape((-1,block_len, 1))

#     return X_feed, X_message
