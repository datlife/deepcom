"""Test for Neural Coder Definition"""
import tensorflow as tf
from model import neural_decoder


class NeuralCoderTest(tf.test.TestCase):
  """Simple Tests to make sure my poorly written codes do not break anything"""

  def testNeuralCoderConstruction(self):
    batch_size = 4
    sequence_len = 100
    inputs = tf.random_uniform(shape=(batch_size, sequence_len, 2))

    outputs = neural_decoder(inputs, is_training=True)
    # Match sure output shape is correct.
    self.assertEqual(
      [4, 100, 1],
      outputs.get_shape().as_list())

  def testLongSequenceInput(self):
    batch_size = 4
    sequence_len = 10000
    inputs = tf.random_uniform(shape=(batch_size, sequence_len, 2))

    outputs = neural_decoder(inputs, is_training=True)
    # Match sure output shape is correct.
    self.assertEqual(
      [4, 10000, 1],
      outputs.get_shape().as_list())

  def testShortSequenceInput(self):
    batch_size = 4
    sequence_len = 1000
    inputs = tf.random_uniform(shape=(batch_size, sequence_len, 2))

    outputs = neural_decoder(inputs, is_training=True)
    # Match sure output shape is correct.
    self.assertEqual(
      [4, 1000, 1],
      outputs.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
