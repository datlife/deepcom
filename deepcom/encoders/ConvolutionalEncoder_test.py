import numpy as np
from encode import bitstream_generator

# test bitstream_generator.py
def test_bitstream_generator():
    np.random.seed(2018)
    generated = bitstream_generator(8)
    expected = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    assert np.all(generated == expected)