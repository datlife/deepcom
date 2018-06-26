"""Contains helper functions for DeepCom
"""
import math
import numpy as np
import commpy as cp


def corrupt_signal(input_signal, noise_type, sigma = 1.0, error_prob=0.01,
                   vv =5.0, radar_power = 20.0, radar_prob = 5e-2, denoise_thd = 10.0,
                   modulate_mode = 'bpsk', snr_mixture = [0, 0, 0]):
  """Simulate data corruption over a channel
  Reference: Author's code
  https://github.com/yihanjiang/Sequential-RNN-Decoder
  """
  data_shape = input_signal.shape  # input_signal has to be a numpy array.
  if noise_type == 'bsc':
    corruptted_signal = cp.channels.bsc(input_signal, error_prob)

  if noise_type == 'awgn':
    noise = sigma * np.random.standard_normal(data_shape) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise

  elif noise_type == 't-dist':
    noise = sigma * math.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
    corrupted_signal = 2.0*input_signal-1.0 + noise

  elif noise_type == 'awgn+radar':
    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
    add_poscomp = np.ones(data_shape) - abs(add_pos)
    corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

  elif noise_type == 'radar':
    noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
    corrupted_signal = 2.0*input_signal-1.0 + noise

  elif noise_type == 'awgn+radar+denoise':
    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
    add_poscomp = np.ones(data_shape) - abs(add_pos)
    corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos
    corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

  elif noise_type == 'hyeji_bursty':
    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    add_pos     = np.random.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])
    corrupted_signal = bpsk_signal + radar_power* np.random.standard_normal( size = data_shape ) * add_pos


  elif noise_type == 'hyeji_bursty+denoise' or noise_type == 'hyeji_bursty+denoise0' or noise_type == 'hyeji_bursty+denoise1':

    def denoise_thd_func():
      sigma_1 = sigma
      sigma_2 = radar_power
      optimal_thd = math.sqrt( (2*(sigma_1**2)*(sigma_1**2 + sigma_2**2)/(sigma_2**2)) * math.log(math.sqrt(sigma_1**2 + sigma_2**2)/sigma_1))
      return optimal_thd

    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    add_pos     = np.random.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])
    corrupted_signal = bpsk_signal + radar_power* np.random.standard_normal( size = data_shape ) * add_pos

    #a = denoise_thd
    if denoise_thd == 10.0:
      a = denoise_thd_func() + 1
      print(a)
    else:
      a = denoise_thd
      print(a)

    if noise_type == 'hyeji_bursty+denoise' or noise_type == 'hyeji_bursty+denoise0':
      corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-a, threshmax=a, newval=0.0)
    else:
      corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-a, newval=-a)
      corrupted_signal  = stats.threshold(corrupted_signal, threshmax=a, newval=a)

  elif noise_type == 'mixture-normalized':
    ref_snr = 0
    ref_sigma= 10**(-ref_snr*1.0/20)# reference is always 0dB.
    noise = sigma * np.random.standard_normal(data_shape) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise
    bpsk_signal_ref = 2.0*input_signal-1.0 + ref_sigma * np.random.standard_normal(data_shape)
    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    pstate1 = 0.5
    add_pos     = np.random.choice([0, 1.0], data_shape, p=[pstate1,1-pstate1])
    add_poscomp = np.ones(data_shape) - abs(add_pos)
    corrupted_signal = bpsk_signal_ref * add_poscomp *1.0/(ref_sigma**2) + bpsk_signal * add_pos *1.0/(sigma**2)
    return corrupted_signal

  elif noise_type == 'mixture':
    ref_snr = 0
    ref_sigma= 10**(-ref_snr*1.0/20)# reference is always 0dB.
    noise = sigma * np.random.standard_normal(data_shape) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise
    bpsk_signal_ref = 2.0*input_signal-1.0 + ref_sigma * np.random.standard_normal(data_shape)
    bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
    pstate1 = 0.5
    add_pos     = np.random.choice([0, 1.0], data_shape, p=[pstate1,1-pstate1])
    add_poscomp = np.ones(data_shape) - abs(add_pos)
    corrupted_signal = bpsk_signal_ref * add_poscomp *1.0 + bpsk_signal * add_pos *1.0
    return corrupted_signal

  elif noise_type == 'mix_snr_turbo':
    noise = snr_mixture[0] * np.random.standard_normal(data_shape) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise

  elif noise_type == 'random_snr_turbo':
    this_snr = np.random.uniform(snr_mixture[2],snr_mixture[0], data_shape)
    noise = np.multiply(this_snr, np.random.standard_normal(data_shape)) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise

  else:
    print('[Warning][Noise Generator]noise_type noty specified!')
    noise = sigma * np.random.standard_normal(data_shape)
    corrupted_signal = 2.0*input_signal-1.0 + noise

  return corrupted_signal

