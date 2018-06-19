import numpy as np

def bitstream_generator(sequence_length=8):
  """Randomly generate a bitstream of 1's and 0's."""
  return np.random.randint(2, size=sequence_length)

