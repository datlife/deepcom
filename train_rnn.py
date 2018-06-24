"""Train a neural decoder

Example Usage:

# For CPU
>> python train_rnn.py \
--dataset ./rnn_0k_bl50_snr0.dataset \
--batch_size 4
--epochs 50
--dropout_Rate 0.7

# For GPU
>> python train_rnn.py \
--dataset ./rnn_120k_bl50_snr0.dataset \
--batch_size 500
--epochs 50
--dropout_Rate 0.7
"""
import os
import argparse
import pickle
import tensorflow as tf

from deepcom.model import NRSCDecoder
from deepcom.dataset import data_genenerator
from deepcom.metrics import BER, BLER, TrainValTensorBoard


def training_pipeline(args):
  """Main Tranining Pipeline"""
  # Define path to save training log for visualization later
  experiment_log = './reports/logs/BiGRU-{}-{}::dropout-{}::epochs-{}'.format(
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
  if args.pretrained_model is not None:
    try:
      model_path = os.path.join(experiment_log, 'BiGRU.hdf5')
      pretrained = tf.keras.models.load_model(
          model_path,
          custom_objects={'BER': BER, 'BLER': BLER})
      model = pretrained
      print('Pre-trained weights are loaded.')
    except Exception as e:
      print(e)
      print('\nFailed to load pretrained model. Start training from stratch')

  # ####################################
  # Start Training/Eval Pipeline
  # ####################################
  train_set = data_genenerator(X_train, Y_train, args.batch_size, shuffle=True)
  test_set = data_genenerator(X_test, Y_test, args.batch_size, shuffle=False)

  print("To monitor training in Tensorboard, execute this command line in terminal:\n" \
      "tensorboard --logdir=%s" \
      "\nThen, open http://0.0.0.0:6006/ into your web browser.\n" % experiment_log)

  # Setup some callbacks to help training better
  summary = TrainValTensorBoard(experiment_log, write_graph=False) # TensorBoard
  backup = tf.keras.callbacks.ModelCheckpoint(                     # Backup best model
      filepath='%s/BiGRU.hdf5' % experiment_log,
      monitor='val_BER',
      save_best_only=True)
  # Stop training early if the model seems to overfit
  early_stoping = tf.keras.callbacks.EarlyStopping(                # Early stopping
      monitor='val_loss',
      min_delta=0.0,
      patience=3,
      verbose=0, mode='auto')

  model.fit(
      train_set.make_one_shot_iterator(),
      steps_per_epoch=len(Y_train) // args.batch_size,
      validation_data=test_set.make_one_shot_iterator(),
      validation_steps=len(Y_test) // args.batch_size,
      callbacks=[summary, backup, early_stoping],
      epochs=args.epochs)
  print('Training is completed.')


def parse_args():
  """Parse Arguments for training Neural-RSC"""
  args = argparse.ArgumentParser(description='Train a Neural Decoder')
  args.add_argument('--dataset', type=str, required=True,
      help='Path to rnn_*.dataset file generated from `generate_synthetic_dataset.py`')

  # Training Hyper-Parameters
  args.add_argument('--epochs', type=int, default=20)
  args.add_argument('--batch_size', type=int, default=500)
  args.add_argument('--learning_rate', type=float, default=1e-3)
  args.add_argument('--dropout_rate', type=float, default=0.7)
 
  # For experiments different network architectures
  args.add_argument('--num_layers', type=int, default=2)
  args.add_argument('--num_hidden_units', type=int, default=400)
  args.add_argument('--pretrained_model', type=str, help='Path to pretrained .hdf5 model')
  return args.parse_args()


if __name__ == '__main__':
  arguments = parse_args()
  training_pipeline(arguments)
