import math
import numpy as np


def generate_random_binary_sequence(seq_len, p=0.5):
  """Generate message bits length `seq_len` of a random binary 
  sequence, where each bit picked is a one with probability p.

  Args:
    seq_len: - int - length of message bit
    p - float - probability

  Return:
    seq: - 1D ndarray - represent a message bits
  """
  seq = np.zeros(seq_len)
  for i in range(seq_len):
    seq[i] = 1 if (np.random.random() < p) else 0
  return seq

def g_matrix_generator(L, data_date):
  """
  Refence: TODO from @GanTu github
  For a rate of 1/2 or of 1/3, and a given constraint length L, generates the
  commonly used G matrix. The default memory assumed is L-1.
  This function has the functionality equivalent of a lookup table.
  [Credits to pg. 789-790, Proakis and Salehi 2nd edition]
  In such conv coding, the azssumed k is 1, and the n is k * rate. Further, such
  sequences are worked on one symbol at a time.
  """
  if data_date == 1/3:
    dict_l_generator = {
      3: [5, 7, 7],
      4: [13, 15, 17],
      5: [25, 33, 37],
      6: [47, 53, 75],
      7: [133, 145, 175],
      8: [255, 331, 367],
      9: [557, 663, 711]
      }

  elif data_date == 1/2:
    dict_l_generator = {
      3: [5, 7],
      4: [15, 17],
      5: [23, 35],
      6: [53, 75],
      7: [133, 171],
      8: [247, 371],
      9: [561, 753]
    }

  else:
    assert False, "This rate is currently not supported, {}".format(str(data_date))

  return np.array([dict_l_generator[L]])

def corrupt_signal(input_signal, noise_type, sigma = 1.0,
                    vv =5.0, radar_power = 20.0, radar_prob = 5e-2, denoise_thd = 10.0,
                    modulate_mode = 'bpsk', snr_mixture = [0, 0, 0]):
    '''
    Documentation TBD.
    only bpsk is allowed, but other modulation is allowed by user-specified modulation.
    :param noise_type: required, choose from 'awgn', 't-dist'
    :param sigma:
    :param data_shape:
    :param vv: parameter for t-distribution.
    :param radar_power:
    :param radar_prob:
    :return:
    '''

    data_shape = input_signal.shape  # input_signal has to be a numpy array.

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

        # noise = sigma * np.random.standard_normal(data_shape) + \
        #         np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        #
        # corrupted_signal = 2.0*input_signal-1.0  + noise

    elif noise_type == 'radar':
        noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'awgn+radar+denoise':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        add_poscomp = np.ones(data_shape) - abs(add_pos)
        corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

        # noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        # corrupted_signal = 2.0*input_signal-1.0 + noise
        # corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

    elif noise_type == 'hyeji_bursty':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])
        corrupted_signal = bpsk_signal + radar_power* np.random.standard_normal( size = data_shape ) * add_pos

        #
        # burst = np.random.randint(0, data_shape[0], int(radar_prob*data_shape[0]))
        # bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        # corrupt_signal[burst] = bpsk_signal[burst] + radar_power * np.random.standard_normal(data_shape)

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
