# THIS IS NOT AN OFFICIAL IMPLEMENTATION

This repository is an implmentation of "*Communication Algorithms via Deep Learning*". Paper: https://arxiv.org/abs/1805.09317.


## Result
---

This repository validates that, indeed, an RNN can learn to decode convolution coded signals over AWGN Channel. Moreover, this RNN can generalize well to decode at different Signal To Noise (SNR) values as good as Viterbi Algorithm Decoder. We use `Bit Error Rate` (BER) and `Block Error Rate` (BLER).


Block Length = 100         |  Block Length = 1000 
:-------------------------:|:-------------------------:
![](reports/week2/result_ber_block_length_100_snr0.png)| ![](reports/week2/result_ber_block_length_1000_snr0.png)
