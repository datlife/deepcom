import numpy as np
import commpy as comm
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

    # Create not-random message bit sequences
    msg_bit_sequences = [None for _ in range(NUM_SEQS)]
    for i in range(NUM_SEQS):
        message_bits = generate_random_binary_sequence(BLOCK_LEN)
        msg_bit_sequences[i] = message_bits
    
    # Define an Encoder using Conv Code Encoding Scheme
    encoder = ConvolutionalCodeEncoder(
        constraint_length=CONSTRAINT_LEN,
        data_rate=DATA_RATE)

    # Encode data
    encoded_sequences = [None for _ in range(NUM_SEQS)]
    for idx, message_bits in enumerate(msg_bit_sequences):
        coded_bits = encoder.encode(message_bits)
        encoded_sequences[idx] = coded_bits

    # Simulates data corruption over AWGN Channel
    # encoded_sequences = corrupt_signal(
    #     input_signal=np.asarray(encoded_sequences), 
    #     noise_type='awgn')
    # noisy_signals = comm.channels.awgn(np.asarray(encoded_sequences), snr_dB)

    # Decode signals using Viterbi Decoder
    decoded_signals = ViterbiDecoder(
        encoded_sequences, 
        constrain_length=CONSTRAINT_LEN, 
        rate=DATA_RATE)
    
    # Extract only BLOCK_LEN size from Viterbi outputs
    decoded_signals = np.array(decoded_signals)[:, :BLOCK_LEN]

    # print(np.shape(decoded_signals))
    # print(decoded_signals)

    expected_msg_bit_sequences = np.array(
    [[0, 1, 0, 1, 1, 0, 0, 0, 0, 1], 
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0], 
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 0], 
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0], 
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

    assert np.all(np.asarray(decoded_signals) == expected_msg_bit_sequences)

    # Because there are noises, we accept True 
    # if most decode bits are matched with original msgs bit sequences.