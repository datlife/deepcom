"""Train a neural decoder"""
import argparse
import pickle
import numpy as np
import tensorflow as tf

from deepcom.metrics import ber, bler
from deepcom.NeuralDecoder import NRSCDecoder
from deepcom.dataset import data_genenerator


def parse_args():
  """Parse Arguments for training Neural-RSC"""
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--dataset_path', type=str, required=True)
  args.add_argument('--batch_size', type=int, default=2)
  args.add_argument('--block_length', type=int, default=100)
  args.add_argument('--num_examples', type=int, default=12000)
  args.add_argument('--learning_rate', type=float, default=1e-3)
  return args.parse_args()

def main(args):
  # ####################################
  # Load Dataset for training/eval
  # ####################################
  with open(args.dataset_path, 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)
  print('Number of training sequences {}'.format(len(X_train)))
  print('Number of testing sequences {}\n'.format(len(X_test)))

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
      metrics=[ber, bler])

  # ####################################
  # Start Training/Eval Pipeline
  # ####################################
  train_set = data_genenerator(X_train, Y_train, args.batch_size, shuffle=True)
  test_set = data_genenerator(X_test, Y_test, args.batch_size, shuffle=False)

  model.fit(
      train_set.make_one_shot_iterator(), 
      steps_per_epoch= len(X_train) // args.batch_size, 
      validation_data=test_set.make_one_shot_iterator(),
      validation_steps= len(X_test) // args.batch_size,
      epochs=5)

if __name__ == '__main__':
  arguments = parse_args()
  main(arguments)
