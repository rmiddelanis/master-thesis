import numpy as np
from scipy import signal
import pandas as pd
import math
import json


def _fir_rand_input(signal_length, response_length, num_channels=1):
    print("Generating FIR random data with ", signal_length, " samples and ", num_channels, "channels...")
    t = np.zeros((signal_length, 1))
    t[:, 0] = [i for i in range(signal_length)]
    x = np.random.randn(num_channels, signal_length)
    b = [1 / (i+1) for i in range(response_length)]
    a = [1]
    y = signal.lfilter(b, a, x, zi=None)
    data_id = np.zeros((signal_length, 1))
    for i in range(0, signal_length, int(signal_length/100)):
        data_id[i:i+int(signal_length/100)].fill(i)
    return t, np.transpose(x), np.transpose(y), data_id


def _generate_fir_response(signal_input, response_length):
    b = [np.random.rand()*(1-i/response_length) for i in range(response_length)]
    a = [1]
    return signal.lfilter(b, a, signal_input, zi=None, axis=0)


def _sine(signal_length, frequency, amp, offset, delta_t):
    print("Generating Sine data with {} samples, frequency {} Hz, Amplitude {}, offset {}, delta_t {}".format(
        signal_length, frequency, amp, offset, delta_t))
    t = np.zeros((signal_length+delta_t, 1))
    t[:, 0] = np.arange(0, signal_length+delta_t, 1)
    x = np.sin(frequency*2*math.pi*t)*amp + offset
    y = x[delta_t:]
    x = x[:-delta_t]
    t = t[:-delta_t]
    data_id = np.zeros((signal_length, 1))
    for i in range(0, signal_length, int(signal_length / 100)):
        data_id[i:i + int(signal_length / 100)].fill(i)
    return t, x, y, data_id


def _rand_sine_sum(signal_length, ground_frequency, amp, offset, num_superpos, delta_t):
    signal_params = {}
    signal_params['signal_length'] = signal_length
    signal_params['ground_frequency'] = ground_frequency
    signal_params['amp'] = amp
    signal_params['offset'] = offset
    signal_params['num_superpos'] = num_superpos
    signal_params['delta_t'] = delta_t

    print("Generating Sine data with {} samples, ground frequency {} Hz, Amplitude {}, offset {}, delta_t {}".format(
        signal_length, ground_frequency, amp, offset, delta_t))

    t = np.zeros((signal_length+delta_t, 1))
    t[:, 0] = np.arange(0, signal_length+delta_t, 1)
    # x = np.sin(ground_frequency*2*math.pi*t)*amp + offset
    x = np.zeros((signal_length+delta_t, 1))

    f_max = 0.5
    base = math.pow(f_max*signal_length, 1/(num_superpos-1))

    for i in range(num_superpos):
        f_i = 1/(signal_length/math.pow(base, i))
        superpos_freq = np.random.normal(f_i, (1/3)*f_i)
        superpos_amp = np.random.normal(0, 1/3*amp)
        x = x + np.sin(superpos_freq*2*math.pi*t)*superpos_amp + np.random.rand(signal_length+delta_t,1)*0.05
        print("Superimposed frequency:", superpos_freq, "Hz and Amplitude", superpos_amp)
        signal_params['freq_'+str(i)] = superpos_freq
        signal_params['amp_'+str(i)] = superpos_amp

    y = x[delta_t:]
    x = x[:-delta_t]
    t = t[:-delta_t]
    data_id = np.zeros((signal_length, 1))
    for i in range(0, signal_length, int(signal_length / 100)):
        data_id[i:i + int(signal_length / 100)].fill(i)
    return t, x, y, data_id, signal_params


if __name__ == "__main__":
    output_path = "./data/Synthetic/"
    signal_length_ = 10000
    ground_frequency_ = 1e-5
    amp_ = 1
    offset_ = 0
    num_superpos_ = 1000
    delta_t_ = 1

    response_length_ = 50

    num_filterings_ = 0

    t1, x1, y1, id1, signal_params_1 = _rand_sine_sum(signal_length=signal_length_, ground_frequency=ground_frequency_,
                                                      amp=amp_, offset=offset_, num_superpos=num_superpos_,
                                                      delta_t=delta_t_)
    t2, x2, y2, id2, signal_params_2 = _rand_sine_sum(signal_length=signal_length_, ground_frequency=ground_frequency_,
                                                      amp=amp_, offset=offset_, num_superpos=num_superpos_,
                                                      delta_t=delta_t_)

    y1_filtered = y1
    y2_filtered = y2

    for filtering in range(num_filterings_):
        y1_filtered = y1_filtered + _generate_fir_response(y1, response_length_)
        y2_filtered = y2_filtered + _generate_fir_response(y2, response_length_)

    x = np.concatenate((x1, x2), axis=1)
    x_reverse = np.concatenate((x2, x1), axis=1)
    y = np.concatenate((y1_filtered, y2_filtered), axis=1)
    y_reverse = np.concatenate((y2_filtered, y1_filtered), axis=1)

    columns = ['t[s]'] + ['Ist']*x.shape[1] + ['Soll']*y.shape[1] + ['data_id']

    df = pd.DataFrame(np.concatenate((t1, x, y, id1), axis=1), columns=columns)
    df.to_pickle(output_path + "synthetic_data.p")

    df_reverse = pd.DataFrame(np.concatenate((t1, x_reverse, y_reverse, id1), axis=1), columns=columns)
    df_reverse.to_pickle(output_path + "synthetic_data_reverse.p")

    with open(output_path + "config_Dataset_A.txt", "w") as config_file:
        json.dump(signal_params_1, config_file)
    config_file.close()

    with open(output_path + "config_Dataset_B.txt", "w") as config_file:
        json.dump(signal_params_2, config_file)
    config_file.close()


