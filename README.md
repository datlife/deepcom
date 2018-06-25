# Communication Algorithms via Deep Learning


This repository is an implmentation of "*Communication Algorithms via Deep Learning*". Paper: https://arxiv.org/abs/1805.09317. Here is the current result:

<p align="center">
  <img src=reports/result_conv_code.png />
</p>

 * This paper claims that a Recurrent Neural Network can **learn from data to decode** noisy coded signal over Additive White Gaussian Noise (AWGN) Channel **as good as** Viterbi and BCJR algorithm. 

## Network Architecture:

<p align="center">
  <img src=reports/network_architecture.png />
</p>

* **Why Bi-directional, and not uni-directional, RNN?** Similar to dynamic programming, it usually consists of a forward and backward steps. The Bi-directional RNN architecture allows the network to learn the feature representation in both direction. I demonstrated a fail case, when using Uni-directional RNN, in [`unidirection_fail_not_converge.ipynb`](reports/unidirection_fail_not_converge.ipynb) notebook.

![](reports/results/week2_unidirectional_failed_to_converge.png)

* **Proper training data matters.** In addition, the paper provides an emperical method to determine the Signal-to-Noise (SNR) for generating training dataset. It helps the network generalize better during testing. Given message bit sequence `K`, transmitted codeword sequence of length `c` and data rate `r`. Then, `SNR_train` is computed as:

<p align="center">
<img src=https://latex.codecogs.com/gif.latex?%24%24SNR_%7Btrain%7D%3Dmin%5C%7BSNR_%7Btest%7D%2C%2010log_%7B10%7D%282%5E%7B2r%7D%20-%201%29%5C%7D%5Cspace%5Cspace%20%5Ctext%7B%28Appendix%20D%29%7D%24%24 /></p>
  
* For example, the paper uses `r=1/2` and block length `c=2K`. Then `SNR_{train} =min(SNR_{test}, 0)`.

## Usage
### Install dependencies
```
conda env create -f dependencies.yml
source activate deepcom
```
### (Recommended) IPython Notebook for training and benchmarking model with Viterbi Decoder.

* [reproduce_result.ipynb](reproduce_result.ipynb): A Jypyter notebook demonstrates how to train a Neural Decoder and compare  the performance with  Viterbi Decoder.

### (Optional) Steps to preproduce the result yourself.

* Generate synthetic data for training/testing. This script will generate a pickle file `rnn_12k_bl100_snr0.dataset`

    ```shell
    python generate_synthetic_dataset.py \
    --snr 0 \
    --block_length 100 \
    --num_training_sequences 12000\
    --num_testing_sequences  10000  \
    --num_cpu_cores 8 \
    --training_seed 2018 \
    --testing_seed 1111
    ```

* Train the network
  * For GPU supported machine
  ```
  python train_rnn.py \
  --dataset ./rnn_12k_bl100_snr0.dataset \
  --batch_size 200
  --epochs 50
  --dropout_Rate 0.7
  ```
  * For CPU, properly take a long time to converge
  ```
  python train_rnn.py \
  --dataset ./rnn_12k_bl100_snr0.dataset \
  --batch_size 4
  --epochs 50
  --dropout_Rate 0.7
  ```

* Benchmark the result, there are two ways
  * Use a script to only benchmark the Neural Decoder (over multiple SNRs).
  ```
  python evaluate.py \
  --checkpoint_dir ./reports/logs/BiGRU-2-400::dropout0.7::epochs-50
  --dataset ./rnn_12k_bl100_snr0.dataset \
  --batch_size 200 \
  ```
   * Use an existing benchmark notebook in `reports/benchmark.ipynb` 
