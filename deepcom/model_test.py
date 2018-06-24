"""Tests for Neural Coder Definition"""
import tensorflow as tf
from .NeuralDecoder import NRSCDecoder

def test_model_constructor():
  batch_size = 4
  sequence_len = 100
  inputs = tf.random_uniform(shape=(batch_size, sequence_len, 2))
  outputs = NRSCDecoder(inputs, is_training=True)
  assert outputs.get_shape().as_list() == [4, 100, 1]

def test_long_sequence_inputs():
  batch_size = 4
  sequence_len = 10000
  inputs = tf.random_uniform(shape=(batch_size, sequence_len, 2))
  outputs = NRSCDecoder(inputs, is_training=True)
  assert outputs.get_shape().as_list() == [4, 10000, 1]
