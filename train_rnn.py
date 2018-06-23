"""Train a neural decoder"""
import argparse
import pickle
import tensorflow as tf

from deepcom.dataset import data_genenerator
from deepcom.NeuralDecoder import NRSCDecoder
from deepcom.metrics import BER, BLER, TrainValTensorBoard


def parse_args():
  """Parse Arguments for training Neural-RSC"""
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--dataset_path', type=str, required=True)

  args.add_argument('--epochs', type=int, default=20)
  args.add_argument('--batch_size', type=int, default=600)
  args.add_argument('--learning_rate', type=float, default=1e-3)
  args.add_argument('--dropout_rate', type=float, default=0.3)

  args.add_argument('--num_layers', type=int, default=2)
  args.add_argument('--num_hidden_units', type=int, default=400)
  return args.parse_args()

def main(args):

  # ####################################
  # Load Dataset for training/eval
  # ####################################
  with open(args.dataset_path, 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)
  train_set = data_genenerator(X_train, Y_train, args.batch_size, shuffle=True)
  test_set = data_genenerator(X_test, Y_test, args.batch_size, shuffle=False)
  print('Number of training sequences {}'.format(len(X_train)))
  print('Number of testing sequences {}\n'.format(len(X_test)))

  # ####################################
  # Define Neural Decoder Model
  # ####################################
  inputs = tf.keras.Input(shape=(None, 2))
  outputs = NRSCDecoder(inputs, 
      is_training=True, 
      num_layers=args.num_layers, 
      hidden_units=args.num_hidden_units)
  model = tf.keras.Model(inputs, outputs)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(args.learning_rate), 
      loss='binary_crossentropy',
      metrics=[BER, BLER])
  model.summary()

  # ####################################
  # Start Training/Eval Pipeline
  # ####################################
  experiment_log = './logs/Bi-GRU-{}-{}::dropout-{}::epochs-{}'.format(
      args.num_layers, args.num_hidden_units, args.dropout_rate, args.epochs)

  summary = TrainValTensorBoard(experiment_log, write_graph=False)
  backup = tf.keras.callbacks.ModelCheckpoint('%s/BiGRU.weights' % experiment_log,
    monitor='val_loss', save_best_only=True, save_weights_only=True)

  try:
    model.load_weights('%s/BiGRU.weights' % experiment_log)
  except:
    pass

  model.fit(
      train_set.make_one_shot_iterator(), 
      steps_per_epoch=50, 
      validation_data=test_set.make_one_shot_iterator(),
      validation_steps=len(X_test) // args.batch_size,
      callbacks=[summary, backup],
      epochs=args.epochs)

  print('Training is completed.')

if __name__ == '__main__':
  arguments = parse_args()
  main(arguments)
