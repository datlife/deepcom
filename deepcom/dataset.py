import numpy as np
import commpy as cp
import multiprocessing as mp
from .utils import awgn_channel, generate_message_bits

import tensorflow as tf


def data_generator(inputs, labels, batch_size, shuffle=True):
    """Construct a data generator using tf.Dataset"""
    dataset  = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset  = dataset.batch(batch_size)
    dataset  = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset    
    

def create_dataset(num_sequences, block_length, trellis, snr, seed, num_cpus=4):
    # Init seed
    np.random.seed(seed)
    snr = snr + 10 * np.log10(1./2.)
    sigma = np.sqrt(1. / (2. * 10 **(snr / 10.)))
      
    with mp.Pool(processes=num_cpus) as pool:
      X, Y = zip(*pool.starmap(func, 
        [(block_length,trellis, sigma) for _ in range(num_sequences)]))

    np.random.seed()
    X = np.reshape(X, (-1, block_length, 2))
    Y = np.reshape(Y, (-1, block_length, 1))
    return X, Y


def func(block_length, trellis, sigma):
    ground_truth = generate_message_bits(block_length)
    # Simulates data sent over AWGN channel
    coded_bits = cp.channelcoding.conv_encode(ground_truth, trellis)
    noisy_bits = awgn_channel(coded_bits, sigma)
    # Ignore the last 4 bits        
    input_signal = noisy_bits[: 2*block_length]
    return input_signal, ground_truth

