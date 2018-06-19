import numpy as np
from .ViterbiDecoder import ViterbiDecoder
from ..encoders import ConvolutionalCodeEncoder
from ..utils import generate_random_binary_sequence, corrupt_signal

def test_viterbi_decoder_on_conv_codes():

    NUM_SEQS = 5
    BLOCK_LEN = 10
    DATA_RATE = 1/2
    CONSTRAINT_LEN = 3

    # For reproducability
    np.random.seed(2018)

    # Create a random message bits
    msg_bit_sequences = [None for _ in range(NUM_SEQS)]
    for i in range(NUM_SEQS):
        message_bits = generate_random_binary_sequence(BLOCK_LEN)
        msg_bit_sequences[i] = message_bits
    
    # Define a Conv Code Encoding Scheme
    encoder = ConvolutionalCodeEncoder(
        constraint_length=CONSTRAINT_LEN,
        data_rate=DATA_RATE)

    # Encode data
    encoded_sequences = [None for _ in range(NUM_SEQS)]
    for idx, message_bits in enumerate(msg_bit_sequences):
        coded_bits = encoder.encode(message_bits)
        encoded_sequences[idx] = coded_bits

    # Simulate data corruption over AWGN Channel
    corrupted_signals = corrupt_signal(
        input_signal=np.asarray(encoded_sequences), 
        noise_type='awgn')

    # Decode signals using Viterbi 
    decoded_signals = ViterbiDecoder(
        corrupted_signals, 
        constrain_length=CONSTRAINT_LEN, 
        rate=DATA_RATE)

    exptected_encoded_bits =  np.array(
    [[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


    assert np.all(np.asarray(decoded_signals) == exptected_encoded_bits)
