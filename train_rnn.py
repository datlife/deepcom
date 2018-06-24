"""Train a neural decoder"""
import os
import argparse
import pickle
import tensorflow as tf

from deepcom.model import NRSCDecoder
from deepcom.dataset import data_genenerator
from deepcom.metrics import BER, BLER, TrainValTensorBoard


def parse_args():
  """Parse Arguments for training Neural-RSC"""
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--dataset', type=str, required=True)

  args.add_argument('--epochs', type=int, default=20)
  args.add_argument('--batch_size', type=int, default=500)
  args.add_argument('--learning_rate', type=float, default=1e-3)
  args.add_argument('--dropout_rate', type=float, default=0.3)

  args.add_argument('--num_layers', type=int, default=2)
  args.add_argument('--num_hidden_units', type=int, default=400)
  return args.parse_args()


def training_pipeline(args):
  """Main Tranining Pipeline"""
  # Define path to save training log for visualization later
  experiment_log = './logs/BiGRU-{}-{}::dropout-{}::epochs-{}'.format(
      args.num_layers, args.num_hidden_units, args.dropout_rate, args.epochs)

  # ####################################
  # Load Dataset for training/eval
  # ####################################
  with open(args.dataset, 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)
  print('Number of training sequences {}'.format(len(X_train)))
  print('Number of testing sequences {}\n'.format(len(X_test)))

  # ####################################
  # Define Neural Decoder Model
  # ####################################
  # Construct Model
  inputs = tf.keras.Input(shape=(None, 2))
  outputs = NRSCDecoder(
      inputs,
      is_training=True,
      num_layers=args.num_layers,
      hidden_units=args.num_hidden_units)
  model = tf.keras.Model(inputs, outputs)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(args.learning_rate),
      loss='binary_crossentropy',
      metrics=[BER, BLER])

  # Attempt to load pretrained model if available.
  try:
    model_path = os.path.join(experiment_log, 'BiGRU.hdf5')
    pretrained = tf.keras.models.load_model(
        model_path,
        custom_objects={'BER': BER, 'BLER': BLER})
    model = pretrained
    print('Pre-trained weights are loaded.')
  except Exception as e:
    print(e)

  # Setup some callbacks to help training better
  summary = TrainValTensorBoard(experiment_log, write_graph=False)
  backup = tf.keras.callbacks.ModelCheckpoint(
      filepath='%s/BiGRU.hdf5' % experiment_log,
      monitor='val_BER',
      save_best_only=True)

  # Stop training early if the model seems to overfit
  early_stoping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      min_delta=0.0,
      patience=3,
      verbose=0, mode='auto')

  # ####################################
  # Start Training/Eval Pipeline
  # ####################################
  train_set = data_genenerator(X_train, Y_train, args.batch_size, shuffle=True)
  test_set = data_genenerator(X_test, Y_test, args.batch_size, shuffle=False)

  model.fit(
      train_set.make_one_shot_iterator(),
      steps_per_epoch=len(Y_train) // args.batch_size,
      validation_data=test_set.make_one_shot_iterator(),
      validation_steps=len(Y_test) // args.batch_size,
      callbacks=[summary, backup, early_stoping],
      epochs=args.epochs)
  print('Training is completed.')


if __name__ == '__main__':
  arguments = parse_args()
  training_pipeline(arguments)
