"""Benchmark the results of Viterbi Decoder and Neural Network Decoder
on Convolutional Codes.
"""
import numpy as np
from commpy.channels import awgn
from deepcom.decoders import ViterbiDecoder
from deepcom.encoders import ConvolutionalCodeEncoder


def main():

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

def run_benchmark(
    expected_message_bits_sequences,
    noisy_encoded_bits_over_awgn_channel,
    decoder, 
    block_length,
    signal_to_noise_ratio):

 
    
    # Reshape into original message bit sequences
    decoded_signals =  np.reshape(decoded_signals, (-1, (BLOCK_LEN + 2)))

    # Extract only BLOCK_LEN size from Viterbi outputs
    decoded_signals = decoded_signals[:, :BLOCK_LEN]
    