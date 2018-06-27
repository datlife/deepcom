"""Evaluate a neural decoder at multiple SNR (Signal to Noise) values.

python evaluate.py \
--checkpoint_dir ./reports/logs/BiGRU-2-400::dropout-0.3::epochs-50 \
--dataset ./rnn_120k_bl100_snr0.dataset \
--batch_size 600
"""
import os
import argparse
import pickle
import numpy as np
import commpy as cp
import multiprocessing as mp
import tensorflow as tf

from commpy.channelcoding import Trellis
from deepcom.metrics import BER, BLER
from deepcom.utils import corrupt_signal
from deepcom.dataset import data_genenerator

def parse_args():
  """Parse Arguments for evaluating Neural-RSC"""
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--dataset', type=str, required=True)
  args.add_argument('--checkpoint_dir', type=str, required=True)
  args.add_argument('--batch_size', type=int, required=True)
  return args.parse_args()

def main(args):
  experiment_log = args.checkpoint_dir
  # ####################################
  # Load Dataset for Testing
  # ####################################
  with open(args.dataset, 'rb') as f:
      _, _, X_test, Y_test = pickle.load(f)  # ignore trnaining
  print('Number of testing sequences {}\n'.format(len(X_test)))
  
  # #####################################
  # Load pre-trained Neural Decoder Model
  # #####################################
  try:
    model_path = os.path.join(experiment_log, 'BiGRU.hdf5')
    print(model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'BER': BER, 'BLER': BLER})
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print('Pre-trained model is loaded.')
  except Exception as e:
    print(e)
    raise ValueError('Pre-trained model is not found.')
    
  # ####################################
  # Start Eval Pipeline
  # ####################################
  # Create a Trellis structure for Conv. Code Encoder
  G = np.array([[0o7, 0o5]])    #  Generator Matrix (octal representation)
  M = np.array([3 - 1])         # Number of delay elements in the convolutional encoder
  trellis = Trellis(M, G, feedback=0o7, code_type='rsc')

  block_length = np.shape(Y_test)[1]
  labels = np.reshape(Y_test, (-1, block_length)).astype(int)
  pool = mp.Pool(processes=mp.cpu_count())
  try:
    SNRs  = np.linspace(0, 7.0, 8)
    for snr in SNRs:
      # Compute noise variance `sigma`
      snr_linear = snr + 10 * np.log10(1./2.)
      sigma = np.sqrt(1. / (2. * 10 **(snr_linear / 10.)))

      # #################################################################
      # For every SNR_db, we generates new noisy signals
      # #################################################################
      result = pool.starmap(
          func=generate_noisy_input, 
          iterable=[(msg_bits, trellis, sigma) for msg_bits in labels])
      X, Y =  zip(*result)

      # #################################################################
      # BENCHMARK NEURAL DECODER 
      # #################################################################
      Y = np.reshape(Y, (-1, block_length, 1))
      X = np.reshape(np.array(X)[:, :2*block_length], (-1, block_length, 2))
      test_set = data_genenerator(X, Y, args.batch_size, shuffle=False)
      
      decoded_bits = model.predict(
          test_set.make_one_shot_iterator(), 
          steps=len(Y_test) // args.batch_size)

      # Compute Hamming Distances between message bits and decoded bits
      decoded_bits = np.reshape(np.round(decoded_bits), (-1, block_length)).astype(int)
      original_bits = np.reshape(Y, (-1, block_length)).astype(int)
      hamming_dist = np.sum(np.not_equal(original_bits, decoded_bits),axis=1)

      # Compute Bit Error Rate (BER) and Bit Block Error Rate (BLER)
      ber = sum(hamming_dist) / np.product(np.shape(Y))
      bler = np.count_nonzero(hamming_dist) / len(Y)
      print('[SNR]={:.2f} [BER]={:5.7f} [BLER]={:5.7f}'.format(snr, ber, bler))
      
  except Exception as e:
      print(e)
  finally:
      pool.close()


def generate_noisy_input(message_bits, trellis, sigma):
    # Encode message bit
    coded_bits = cp.channelcoding.conv_encode(message_bits, trellis)
    # Corrupt message on BAWGN Channel
    coded_bits = corrupt_signal(coded_bits, noise_type='awgn', sigma=sigma)
    return coded_bits, message_bits

if __name__ == '__main__':
  arguments = parse_args()
  main(arguments)