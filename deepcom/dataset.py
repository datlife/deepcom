"""Data Loader"""
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import commpy as cp
from .utils import corrupt_signal

def data_genenerator(x, y, batch_size, shuffle=True):
  """A Tensorflow way to load data for training.

  Args:
    x - ndarray - inputs 
    y - ndarray - ground truths
    batch_size - int - number of inputs to load 
      into model per run.
    shuffle - a boolean - to shuffle data or not
      should be off during Evaluation/Testing.

  Return
    dataset - a tf.data.Dataset
  """
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.prefetch(batch_size)
  if shuffle:
    dataset = dataset.shuffle(1000)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset


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


def create_dataset(num_sequences, block_length, trellis, seed, 
                   noise_type='awgn',
                   snr=0.0,
                   error_prob=0.01,
                   num_cpus=4):
  """Generate synthetic message bits for training RNN"""
  # Init seed
  np.random.seed(seed)
  snr = snr + 10 * np.log10(1./2.)
  sigma = np.sqrt(1. / (2. * 10 **(snr / 10.)))
  with mp.Pool(processes=num_cpus) as pool:
    result = pool.starmap(
        func,
        [(block_length, trellis, noise_type, sigma, error_prob) for _ in range(num_sequences)])
    X, Y = zip(*result)
  np.random.seed()
  X = np.reshape(X, (-1, block_length, 2))
  Y = np.reshape(Y, (-1, block_length, 1))
  return X, Y


def func(block_length, trellis, noise_type, sigma, error_prob):
  """Helper function to generate a pair of (input, label)
  for training Neural Decoder.
  """

  ground_truth = generate_message_bits(block_length)
  kwargs = {
    'noise_type': noise_type,
    'sigma': sigma,
    'error_prob': error_prob
  }
  # Simulates data sent over AWGN channel
  coded_bits = cp.channelcoding.conv_encode(ground_truth, trellis)
  noisy_bits = corrupt_signal(coded_bits, **kwargs)

  # Ignore the last 4 bits
  input_signal = noisy_bits[: 2*block_length]
  return input_signal, ground_truth