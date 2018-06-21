"""Train a neural decoder"""
import argparse
import numpy as np
import tensorflow as tf
from commpy.channelcoding import Trellis

from deepcom.metrics import ber
from deepcom.NeuralDecoder import NRSCDecoder
from deepcom.dataset import create_dataset, data_generator


def main(args):
  #  Generator Matrix (octal representation)
  G = np.array([[0o7, 0o5]]) 
  M = np.array([3 - 1])
  trellis = Trellis(M, G, feedback=0o7, code_type='rsc')

  # ####################################
  # Generate Dataset for training/eval
  # ####################################
  X_train, Y_train = create_dataset(
      num_sequences=1000,
      block_length=args.block_length,
      trellis=trellis,
      snr=0.01,
      seed=2018)

  X_test, Y_test = create_dataset(
      num_sequences=40,
      block_length=args.block_length,
      trellis=trellis,
      snr=0.01,
      seed=1111)

  print('Number of training sequences {}'.format(len(X_train)))
  print('Number of testing sequences {}'.format(len(X_test)))
  print(X_train.shape, Y_train.shape)

  # ####################################
  # Define Neural Decoder Model
  # ####################################
  inputs = tf.keras.Input(shape=(None, 2))
  outputs = NRSCDecoder(inputs, is_training=True)
  model = tf.keras.Model(inputs, outputs)

  model.summary()
  model.compile(
      optimizer=tf.keras.optimizers.Adam(args.learning_rate), 
      loss='binary_crossentropy',
      metrics=['acc'])


  # ####################################
  # Start Training/Eval Pipeline
  # ####################################
  model.fit(
      X_train, 
      Y_train, 
      validation_split=0.2, 
      batch_size=args.batch_size)


def parse_args():
  """Parse Arguments for training Neural-RSC.
    NOTE: Default hyper-parameters are from suggested the original paper.
  """
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--batch_size', type=int, default=4)
  args.add_argument('--block_length', type=int, default=100)
  args.add_argument('--num_examples', type=int, default=12000)
  args.add_argument('--learning_rate', type=float, default=1e-3)
  return args.parse_args()


if __name__ == '__main__':
  arguments = parse_args()
  main(arguments)
