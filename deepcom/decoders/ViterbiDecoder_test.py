import numpy as np
from commpy.channels import awgn
from .ViterbiDecoder import ViterbiDecoder
from ..encoders import ConvolutionalCodeEncoder
from ..utils import generate_random_binary_sequence
from ..metrics import compute_accuracy, compute_BER

def test_viterbi_decoder_on_noiseless_signals():
    """Test Viterbi Decoder on Noiseless Convolutional Coded Signal"""
    NUM_SEQS = 3
    BLOCK_LEN = 20
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

    # Decode signals using Viterbi Decoder
    decoded_signals = ViterbiDecoder(
        np.array(encoded_sequences).ravel(), 
        constrain_length=CONSTRAINT_LEN, 
        rate=DATA_RATE)

    # Reshape into original sequences
    decoded_signals =  np.reshape(decoded_signals, (-1, (BLOCK_LEN + 2)))

    # Extract only BLOCK_LEN size from Viterbi outputs
    decoded_signals = np.array(decoded_signals)[:, :BLOCK_LEN]
    
    assert np.all(decoded_signals == msg_bit_sequences)


def test_viterbi_decoder_on_noisy_signals():
    """Test Viterbi Decoder on Convolutional Coded Signal
    over AWGN Channel.
    """
    NUM_SEQS = 3
    BLOCK_LEN = 20
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
    encoded_sequences = awgn(
        input_signal=np.array(encoded_sequences).ravel(), 
        snr_dB=8.0)

    # Decode signals using Viterbi Decoder
    decoded_signals = ViterbiDecoder(
        encoded_sequences, 
        constrain_length=CONSTRAINT_LEN, 
        rate=DATA_RATE)
    
    # Reshape into original messgage bit sequences
    decoded_signals =  np.reshape(decoded_signals, (-1, (BLOCK_LEN + 2)))

    # Extract only BLOCK_LEN size from Viterbi outputs
    decoded_signals = decoded_signals[:, :BLOCK_LEN]
    
    # Because there are noises, we accept True 
    # if most decode bits are matched with original msgs bit sequences. 
    # print(accuracy)
    acc = compute_accuracy(np.array(msg_bit_sequences), decoded_signals)
    assert acc - 0.683 <= 0.05

    ber = compute_BER(np.array(msg_bit_sequences), decoded_signals)
    assert ber - 0.317 <=  0.001
