import torch
import torch.nn as nn
import os
import pickle
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from synthetic_data_generation import _rand_sine_sum
from utils import data_generator, make_experiment_dir, save_model, plot_and_save
from model import TCN
import scipy.stats as stats
from matplotlib import colors
import matplotlib
import warnings

experiment_config = {
    'cuda': True,
    'dropout': 0.05,
    'clip': 0.15,
    # 'epochs': 60,
    'epochs': 120,
    # 'epochs': 1,
    'ksize': 6,
    'levels': 4,
    'lr_init': 0.0001,
    'lr_tune': 0.0001,
    'optim': 'Adam',
    'nhid': 50,
    'validseqlen': 1,
    # 'seqstepwidth_train': 5,
    'seqstepwidth_train': 1,
    # 'seqstepwidth_tune': 50,  # currently not in use
    'seq_len': 150,
    'batch_size': 12,
    'log_interval': 1,
    'num_repetitions': 1,
    # 'num_repetitions': 10,
    # 'store_models': False,
    'store_models': True,
    # 'lr_adapt': 'schedule:40:50',
    'lr_adapt': 'schedule:95:105',
    'lr_adapt_dense_only': 'schedule:10:15',
    # 'lr_adapt': None,
    'plot_svg': False,
}

dataset_config = {
    'test_size': 0.2,
    'train_valid_split': [1, 0],
    'sampling_size': 1,
    'data_file_name': r'./data/Synthetic/synthetic_data.p',
    'sample_frequency': 1,
    'max_data': 9999999,
    'target_variables': ['Soll'],
    'seed': 1,
    'series_x': '11',
    'series_y': '11',
    'pca': False,
    'batch_size': experiment_config['batch_size'],
    'cuda': experiment_config['cuda'],
}

# Plot and figure settings
markersize = 30
figure_y_limits = {
    # 'BnA': [(0.01, 0.28), (0.03, 0.27)],
    # 'BnA': [(None, None), (None, None)],
    'BnA': [(None, None), (0.03, 0.15)],
    # 'BnA_plus': [(0.035, 0.1), (0.0455, 0.09)],
    'BnA_plus': [(None, None), (None, None)],
    # 'full_weight_init_w_partial_freeze': [(0.035, 0.33), (0.04, 0.3)],
    # 'full_weight_init_w_partial_freeze': [(None, None), (None, None)],
    'full_weight_init_w_partial_freeze': [(None, None), (0.03, 0.1)],
    # 'random_reference': [(None, None), (0, 0.25)],
    'random_reference': [(None, None), (None, None)],
    'quantitative_experiment': [(None, None), (None, None)],
    # 'all': (0.03, 0.11),
    # 'all': (0.045, 0.15),
    'all': (0.03, 0.08),
}
global_font_size = 20
matplotlib.rcParams.update({'font.size': global_font_size})
plt.rcParams['svg.fonttype'] = 'none'
if not experiment_config['plot_svg']:
    matplotlib.use("pgf")
    pgf_with_rc_fonts = {
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": ["DejaVu Sans"],
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)
    matplotlib.rcParams.update({'pgf.rcfonts': False})

experiment_name_solver = {
    'BnA': 'Hard Feature Extraction',
    'BnA_plus': 'Soft Feature Extraction',
    'full_weight_init_w_partial_freeze': 'Full Weight init, partial Freeze',
    'random_reference': 'Random Reference',
}

if torch.cuda.is_available():
    if not experiment_config['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def calc_data_split(splitsize_ref_path):
    initial_split_params = dataset_config['train_valid_split']
    best_losses_a = []
    best_losses_b = []
    all_losses = []
    real_train_sizes_rel = []
    for train_size in np.arange(0.05, 0.55, 0.05):
        iteration_folder = splitsize_ref_path + "TuneSize_" + str(train_size) + "/"
        if not os.path.exists(iteration_folder):
            os.makedirs(iteration_folder)
        dataset_config['train_valid_split'] = [1-train_size, train_size]
        load_data()

        # swap train and tune set so that train_models(0,...) can be used for training on the tune set
        global x_train
        global x_tune
        global y_train
        global y_tune
        temp = x_train
        x_train = x_tune
        x_tune = temp
        temp = y_train
        y_train = y_tune
        y_tune = temp
        real_train_sizes_rel.append(x_train.shape[0]*x_train.shape[2]/(x_train.shape[0]*x_train.shape[2] +
                                                                       x_tune.shape[0]*x_tune.shape[2]))
        model_a = TCN(1, 1, cuda=experiment_config['cuda'], ksize=experiment_config['ksize'],
                      dropout=experiment_config['dropout'], clip=experiment_config['clip'],
                      epochs=experiment_config['epochs'], levels=experiment_config['levels'],
                      log_interval=experiment_config['log_interval'], lr=experiment_config['lr_init'],
                      optim=experiment_config['optim'], nhid=experiment_config['nhid'],
                      validseqlen=experiment_config['validseqlen'],
                      seqstepwidth=experiment_config['seqstepwidth_train'],
                      seq_len=experiment_config['seq_len'], batch_size=experiment_config['batch_size'])
        model_b = TCN(1, 1, cuda=experiment_config['cuda'], ksize=experiment_config['ksize'],
                      dropout=experiment_config['dropout'], clip=experiment_config['clip'],
                      epochs=experiment_config['epochs'], levels=experiment_config['levels'],
                      log_interval=experiment_config['log_interval'], lr=experiment_config['lr_init'],
                      optim=experiment_config['optim'], nhid=experiment_config['nhid'],
                      validseqlen=experiment_config['validseqlen'],
                      seqstepwidth=experiment_config['seqstepwidth_train'],
                      seq_len=experiment_config['seq_len'], batch_size=experiment_config['batch_size'])
        losses_a, losses_b = train_models(0, model_a, model_b, iteration_folder, is_quantitative_experiment=False)
        all_losses.append([losses_a, losses_b])
        if not experiment_config['store_models']:
            os.remove(iteration_folder + "0_base_A/model.pt")
            os.remove(iteration_folder + "0_base_B/model.pt")
        best_losses_a.append(min(losses_a))
        best_losses_b.append(min(losses_b))
    plot_kwargs = {'marker': 'o'}
    plot_and_save([best_losses_a, best_losses_b],
                  ["Best Testloss over TuneSet Size [Model A]", "Best Testloss over TuneSet Size [Model B]"],
                  splitsize_ref_path + "testlosses.svg", x_names=["TuneSet Size", "TuneSet Size"],
                  x_ranges=[real_train_sizes_rel, real_train_sizes_rel], **plot_kwargs)
    pickle.dump(all_losses, open(splitsize_ref_path + "all_results_array.p", "wb"))
    dataset_config['train_valid_split'] = initial_split_params


def do_experiment_series(experiment_series_dir, num_repetitions, is_quantitative_experiment, freeze, random_init,
                         diff_learning_rate, train_base):
    experiment_final_results = []
    experiment_best_results = []
    for repetition in range(num_repetitions):
        repetition_path = experiment_series_dir + "Iteration_" + str(repetition) + "/"
        os.makedirs(repetition_path)
        final_model_results, best_model_results = do_experiment(repetition_path, is_quantitative_experiment, freeze,
                                                                random_init, diff_learning_rate, train_base)
        experiment_final_results.append(final_model_results)
        experiment_best_results.append(best_model_results)
    experiment_final_results = np.array(experiment_final_results)
    experiment_best_results = np.array(experiment_best_results)
    pickle.dump(experiment_final_results, open(experiment_series_dir + "final_results_array.p", "wb"))
    pickle.dump(experiment_best_results, open(experiment_series_dir + "best_results_array.p", "wb"))


def do_experiment(experiment_folder, is_quantitative_experiment, freeze, random_init, diff_learning_rate, train_base):
    model_a = TCN(1, 1, cuda=experiment_config['cuda'], ksize=experiment_config['ksize'],
                  dropout=experiment_config['dropout'], clip=experiment_config['clip'],
                  epochs=experiment_config['epochs'], levels=experiment_config['levels'],
                  log_interval=experiment_config['log_interval'], lr=experiment_config['lr_init'],
                  optim=experiment_config['optim'], nhid=experiment_config['nhid'],
                  validseqlen=experiment_config['validseqlen'], seqstepwidth=experiment_config['seqstepwidth_train'],
                  seq_len=experiment_config['seq_len'], batch_size=experiment_config['batch_size'])
    model_b = TCN(1, 1, cuda=experiment_config['cuda'], ksize=experiment_config['ksize'],
                  dropout=experiment_config['dropout'], clip=experiment_config['clip'],
                  epochs=experiment_config['epochs'], levels=experiment_config['levels'],
                  log_interval=experiment_config['log_interval'], lr=experiment_config['lr_init'],
                  optim=experiment_config['optim'], nhid=experiment_config['nhid'],
                  validseqlen=experiment_config['validseqlen'], seqstepwidth=experiment_config['seqstepwidth_train'],
                  seq_len=experiment_config['seq_len'], batch_size=experiment_config['batch_size'])
    if train_base:
        train_models(0, model_a, model_b, experiment_folder, is_quantitative_experiment)
    else:
        notraining_models(0, model_a, model_b, experiment_folder)

    final_model_results = []
    best_model_results = []
    all_results = []
    for round_ in range(1, experiment_config['levels'] * 2 + 1):
        model_a_frozen, model_b_frozen = prepare_qualitative_experiment_models(experiment_folder, round_, freeze,
                                                                               random_init, diff_learning_rate)
        if is_quantitative_experiment:
            model_a_frozen, model_b_frozen = prepare_quantitative_experiment_models(round_, model_a_frozen,
                                                                                    model_b_frozen)
        model_a_losses, model_b_losses = train_models(round_, model_a_frozen, model_b_frozen, experiment_folder,
                                                      is_quantitative_experiment)
        all_results.append([model_a_losses, model_b_losses])
        best_model_results.append([min(model_a_losses), min(model_b_losses)])
        final_model_results.append([model_a_losses[-1], model_b_losses[-1]])
        print("Round {:d} Testlosses - Final Model A (selffer): {:f}, "
              "Final Model B (transfer): {:f}".format(round_, final_model_results[-1][0], final_model_results[-1][1]))
        print("Round {:d} Testlosses - Best Model A (selffer): {:f}, "
              "Best Model B (transfer): {:f}".format(round_, best_model_results[-1][0], best_model_results[-1][1]))

    if not experiment_config['store_models']:
        os.remove(experiment_folder + "0_base_A/model.pt")
        os.remove(experiment_folder + "0_base_B/model.pt")

    final_model_results = np.array(final_model_results)
    best_model_results = np.array(best_model_results)
    all_results = np.array(all_results)
    pickle.dump(all_results, open(experiment_folder + "all_results_array.p", "wb"))

    x = np.arange(1, experiment_config['levels'] * 2 + 1)
    plot_results(x, final_model_results, experiment_folder, "final_model_plot")
    plot_results(x, best_model_results, experiment_folder, "best_model_plot")
    return final_model_results, best_model_results


def prepare_qualitative_experiment_models(experiment_folder, round_, freeze, random_init, diff_learning_rate):
    model_a_frozen = torch.load(open(experiment_folder + "0_base_A/model.pt", "rb"))
    model_b_frozen = torch.load(open(experiment_folder + "0_base_B/model.pt", "rb"))
    model_a_frozen.set_learning_rate(experiment_config['lr_tune'])
    model_a_frozen.linear.weight.data.normal_(0, 0.01)
    model_b_frozen.set_learning_rate(experiment_config['lr_tune'])
    model_b_frozen.linear.weight.data.normal_(0, 0.01)

    optim_params_a = []
    optim_params_b = []
    for level in range(experiment_config['levels']):
        optim_params_a.append({
            'params': model_a_frozen.tcn.network[level].parameters(),
            'lr': experiment_config['lr_tune'] * math.pow(1e-1, experiment_config['levels'] - (level + 1))
        })
        optim_params_b.append({
            'params': model_b_frozen.tcn.network[level].parameters(),
            'lr': experiment_config['lr_tune'] * math.pow(1e-1, experiment_config['levels'] - (level + 1))
        })
        for layer in range(1, 3):
            if level * 2 + layer <= round_:
                if freeze:
                    for param in getattr(model_a_frozen.tcn.network[level], 'conv' + str(layer)).parameters():
                        param.requires_grad = False
                    for param in getattr(model_b_frozen.tcn.network[level], 'conv' + str(layer)).parameters():
                        param.requires_grad = False
            else:
                if random_init:
                    getattr(model_a_frozen.tcn.network[level], 'conv' + str(layer)).weight.data.normal_(0, 0.01)
                    getattr(model_b_frozen.tcn.network[level], 'conv' + str(layer)).weight.data.normal_(0, 0.01)
    if diff_learning_rate:
        model_a_frozen.optimizer = getattr(torch.optim, experiment_config['optim'])(optim_params_a,
                                                                                    experiment_config['lr_tune'])
        model_b_frozen.optimizer = getattr(torch.optim, experiment_config['optim'])(optim_params_a,
                                                                                    experiment_config['lr_tune'])
    return model_a_frozen, model_b_frozen


def prepare_quantitative_experiment_models(round_, model_a_frozen, model_b_frozen):
    num_resblocks_to_keep = int(round_/2)
    if round_ > 1:
        model_a_frozen.tcn.network = nn.Sequential(*list(model_a_frozen.tcn.network.children())[:num_resblocks_to_keep])
        model_b_frozen.tcn.network = nn.Sequential(*list(model_b_frozen.tcn.network.children())[:num_resblocks_to_keep])
    elif round_ == 1:
        model_a_frozen.tcn.network = nn.Sequential(*list(model_a_frozen.tcn.network.children())[:1])
        model_b_frozen.tcn.network = nn.Sequential(*list(model_b_frozen.tcn.network.children())[:1])
        model_a_frozen.tcn.network[0].net = nn.Sequential(*[])
        model_b_frozen.tcn.network[0].net = nn.Sequential(*[])

    # remove second conv, relu and dropout
    if round_ % 2 == 0:
        model_a_frozen.tcn.network[-1].net = nn.Sequential(
            *list(model_a_frozen.tcn.network[-1].net.children())[:int(len(model_a_frozen.tcn.network[-1].net) / 2)])
        model_b_frozen.tcn.network[-1].net = nn.Sequential(
            *list(model_b_frozen.tcn.network[-1].net.children())[:int(len(model_b_frozen.tcn.network[-1].net) / 2)])
    return model_a_frozen, model_b_frozen


def train_models(experiment_round, model_a, model_b, experiment_folder, is_quantitative_experiment):
    path_a, path_b, save_path_a, save_path_b = make_iteration_dirs(experiment_folder, experiment_round)

    lr_adapt = experiment_config['lr_adapt']
    if experiment_round == 1 and is_quantitative_experiment:
        lr_adapt = experiment_config['lr_adapt_dense_only']

    if experiment_round == 0:
        train_losses_a, __, test_losses_a = model_a.fit_and_test(x_train=x_train[:, 0:1, :], y_train=y_train[:, 0:1, :],
                                                                 x_test=x_test[:, 0:1, :], y_test=y_test[:, 0:1, :],
                                                                 experiment_folder=save_path_a,
                                                                 lr_adapt=lr_adapt)
        train_losses_b, __, test_losses_b = model_b.fit_and_test(x_train=x_train[:, 1:2, :], y_train=y_train[:, 1:2, :],
                                                                 x_test=x_test[:, 1:2, :], y_test=y_test[:, 1:2, :],
                                                                 experiment_folder=save_path_b,
                                                                 lr_adapt=lr_adapt)
    else:
        train_losses_a, __, test_losses_a = model_a.fit_and_test(x_train=x_train[:, 0:1, :], y_train=y_train[:, 0:1, :],
                                                                 x_test=x_test[:, 0:1, :], y_test=y_test[:, 0:1, :],
                                                                 experiment_folder=save_path_a,
                                                                 lr_adapt=lr_adapt)
        train_losses_b, __, test_losses_b = model_b.fit_and_test(x_train=x_train[:, 0:1, :], y_train=y_train[:, 0:1, :],
                                                                 x_test=x_test[:, 0:1, :], y_test=y_test[:, 0:1, :],
                                                                 experiment_folder=save_path_b,
                                                                 lr_adapt=lr_adapt)
    train_losses_a = np.array(train_losses_a)
    test_losses_a = np.array(test_losses_a)
    train_losses_b = np.array(train_losses_b)
    test_losses_b = np.array(test_losses_b)
    plot_and_save([train_losses_a, test_losses_a],
                  ["Training Loss over Epochs", "Test Loss over Epochs"],
                  path_a + "losses.svg", x_names=["Epoch", "Epoch"], x_ranges=[])
    plot_and_save([train_losses_b, test_losses_b],
                  ["Training Loss over Epochs", "Test Loss over Epochs"],
                  path_b + "losses.svg", x_names=["Epoch", "Epoch"], x_ranges=[])
    pickle.dump(test_losses_a, open(path_a + "test_losses_array.p", "wb"))
    pickle.dump(test_losses_b, open(path_b + "test_losses_array.p", "wb"))

    return test_losses_a, test_losses_b


def notraining_models(experiment_round, model_a, model_b, experiment_folder):
    path_a, path_b, save_path_a, save_path_b = make_iteration_dirs(experiment_folder, experiment_round=experiment_round)
    save_model(model_a, save_path_a)
    save_model(model_b, save_path_b)
    test_loss_a = np.array(model_a.evaluate(x_test[:, 0:1, :], y_test[:, 0:1, :]))
    test_loss_b = np.array(model_b.evaluate(x_test[:, 1:2, :], y_test[:, 1:2, :]))
    test_loss_a_avg = np.array([sum(test_loss_a) / len(test_loss_a)])
    test_loss_b_avg = np.array([sum(test_loss_b) / len(test_loss_b)])
    pickle.dump(test_loss_a_avg, open(path_a + "test_losses_array.p", "wb"))
    pickle.dump(test_loss_b_avg, open(path_b + "test_losses_array.p", "wb"))


def fix_base_a_losses(experiment_ensemble_dir):
    """
    function to fix a previous error in the way the base losses were stored by function "notraining_models":
    previously, notraining_models stored the base model losses of all batches and not the average
    """
    experiment_dir = os.path.join(experiment_ensemble_dir, "random_reference/")
    for experiment_dir_entry in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, experiment_dir_entry)):
            base_a_losses_iter = pickle.load(
                open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_A", "test_losses_array.p"), "rb"))
            base_b_losses_iter = pickle.load(
                open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_B", "test_losses_array.p"), "rb"))
            if len(base_a_losses_iter) == 1:
                print("Base A loss in path {} seems to be fixed already. Doing nothing!".format(
                    os.path.join(experiment_dir, experiment_dir_entry)))
            else:
                pickle.dump(base_a_losses_iter, open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_A",
                                                                  "test_losses_array_old.p"), "wb"))
                base_a_losses_iter = np.array([sum(base_a_losses_iter) / len(base_a_losses_iter)])
                pickle.dump(base_a_losses_iter, open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_A",
                                                                  "test_losses_array.p"), "wb"))
                print("Replaced test_losses_array.p in path {} with corrected array. "
                      "Old values are stored in as test_losses_array_old.p".format(os.path.join(experiment_dir,
                                                                                                experiment_dir_entry)))
            if len(base_b_losses_iter) == 1:
                print("Base B loss in path {} seems to be fixed already. Doing nothing!".format(
                    os.path.join(experiment_dir, experiment_dir_entry)))
            else:
                pickle.dump(base_a_losses_iter, open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_B",
                                                                  "test_losses_array_old.p"), "wb"))
                base_a_losses_iter = np.array([sum(base_a_losses_iter) / len(base_a_losses_iter)])
                pickle.dump(base_a_losses_iter, open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_B",
                                                                  "test_losses_array.p"), "wb"))
                print("Replaced test_losses_array.p in path {} with corrected array. "
                      "Old values are stored in as test_losses_array_old.p".format(os.path.join(experiment_dir,
                                                                                                experiment_dir_entry)))


def make_iteration_dirs(experiment_folder, experiment_round):
    if experiment_round == 0:
        path_a = os.path.join(experiment_folder, "0_base_A/")
        path_b = os.path.join(experiment_folder, "0_base_B/")
    else:
        path_a = os.path.join(experiment_folder, "A_" + str(experiment_round) + "_A/")
        path_b = os.path.join(experiment_folder, "B_" + str(experiment_round) + "_A/")
    if not os.path.exists(path_a):
        os.makedirs(path_a)
    if not os.path.exists(path_b):
        os.makedirs(path_b)
    if experiment_config['store_models'] or experiment_round == 0:
        save_path_a = path_a
        save_path_b = path_b
    else:
        save_path_a = None
        save_path_b = None
    return path_a, path_b, save_path_a, save_path_b


def train_reference(experiment_dir, num_repetitions, freeze_conv_layers):
    all_test_losses = []
    for repetition in range(num_repetitions):
        testlosses = []
        repetition_path = experiment_dir + "Iteration_" + str(repetition) + "/"
        os.makedirs(repetition_path)
        linear_model = TCN(1, 1, cuda=experiment_config['cuda'], ksize=experiment_config['ksize'],
                           dropout=experiment_config['dropout'], clip=experiment_config['clip'],
                           epochs=experiment_config['epochs'], levels=experiment_config['levels'],
                           log_interval=experiment_config['log_interval'], lr=experiment_config['lr_init'],
                           optim=experiment_config['optim'], nhid=experiment_config['nhid'],
                           validseqlen=experiment_config['validseqlen'],
                           seqstepwidth=experiment_config['seqstepwidth_train'], seq_len=experiment_config['seq_len'],
                           batch_size=experiment_config['batch_size'])
        if freeze_conv_layers:
            for param in linear_model.tcn.parameters():
                param.requires_grad = False
        for epoch in range(experiment_config['epochs']):
            linear_model.train_epoch(x_tune[:, 0:1, :], y_tune[:, 0:1, :], epoch)
            test_loss = linear_model.evaluate(x_test[:, 0:1, :], y_test[:, 0:1, :])
            testlosses.append(sum(test_loss)/len(test_loss))
        if experiment_config['store_models']:
            save_model(linear_model, repetition_path)
        all_test_losses.append(testlosses)
    all_test_losses = np.array(all_test_losses)
    pickle.dump(all_test_losses, open(experiment_dir + "all_results_array.p", "wb"))

    x = np.zeros(num_repetitions)
    y_final = np.zeros((num_repetitions, 1))
    y_best = np.zeros((num_repetitions, 1))
    for repetition in range(num_repetitions):
        y_final[repetition] = all_test_losses[repetition][-1]
        y_best[repetition] = min(all_test_losses[repetition])
    plot_results(x, y_final, experiment_dir, "final_model_plot")
    plot_results(x, y_best, experiment_dir, "best_model_plot")


def produce_synthetic_data(path):
    dataset_config['data_file_name'] = path + "synthetic_data.p"
    dataset_config['target_variables'] = '[Soll]'
    t1, x1, y1, id1, signal_params_1 = _rand_sine_sum(1000000, 1e-5, 1, 0, 30, 1)
    t2, x2, y2, id2, signal_params_2 = _rand_sine_sum(1000000, 1e-5, 1, 0, 30, 1)
    x = np.concatenate((x1, x2), axis=1)
    y = np.concatenate((y1, y2), axis=1)
    columns = ['t[s]'] + ['Ist'] * x.shape[1] + ['Soll'] * y.shape[1] + ['data_id']
    df = pd.DataFrame(np.concatenate((t1, x, y, id1), axis=1), columns=columns)
    df.to_pickle(dataset_config['data_file_name'])
    with open(path + "synthetic_dataset_A_config.txt", "w") as config_file:
        json.dump(signal_params_1, config_file)
    config_file.close()
    with open(path + "synthetic_dataset_B_config.txt", "w") as config_file:
        json.dump(signal_params_2, config_file)
    config_file.close()


def load_data():
    global data
    global x_train
    global x_test
    global x_tune
    global y_train
    global y_test
    global y_tune
    data, pca_scaler = data_generator(dataset_config)
    x_train = data.x_train
    x_test = data.x_test
    x_tune = data.x_valid[0]
    y_train = data.y_train
    y_test = data.y_test
    y_tune = data.y_valid[0]
    if experiment_config['cuda']:
        x_train = x_train.cuda()
        x_test = x_test.cuda()
        x_tune = x_tune.cuda()
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        y_tune = y_tune.cuda()


def plot_results(x, y, path, name):
    plt.figure()
    plt.scatter(x, y[:, 0], marker='o', c='b', label='Loss Model A (selffer)')
    if y.shape[1] > 1:
        plt.scatter(x, y[:, 1], marker='x', c='r', label='Loss Model B (transfer)')
    plt.legend()
    plt.title("Model Losses")
    plt.xlabel("Transfer Level")
    plt.ylabel("Loss")
    if experiment_config['plot_svg']:
        plt.savefig(path + name + ".svg", format = 'svg')
    else:
        plt.savefig(path + name + ".pgf", transparent=True)
    print("Plot(s) saved as %s" % path + name + ".svg")
    plt.close('all')


def make_figures(experiment_ensemble_dir):
    all_best_model_losses_mean = []
    all_best_base_a_losses_means = []
    idx_to_name_mapping = []
    name_to_idx_mapping = {}
    with open(os.path.join(experiment_ensemble_dir, 'dataset_config.txt'), 'rb') as config_file:
        stored_dataset_config = json.load(config_file)
    if stored_dataset_config['data_file_name'] == r'./data/Synthetic/synthetic_data.p':
        selffer_name = r'A$\rightarrow$A'
        transfer_name = r'B$\rightarrow$A'
        rnd_transfer_name = r'Rnd$\rightarrow$A'
    elif stored_dataset_config['data_file_name'] == r'./data/Synthetic/synthetic_data_reverse.p':
        selffer_name = r'B$\rightarrow$B'
        transfer_name = r'A$\rightarrow$B'
        rnd_transfer_name = r'Rnd$\rightarrow$B'
    for experiment_ensemble_dir_entry in os.listdir(experiment_ensemble_dir):
        experiment_dir = os.path.join(experiment_ensemble_dir, experiment_ensemble_dir_entry) + "/"
        experiment_name = experiment_ensemble_dir_entry
        if os.path.isdir(experiment_dir):
            all_losses_a, all_losses_b, best_model_losses, final_model_losses, base_a_losses = load_experiment_losses(experiment_dir)
            best_model_losses_mean = np.mean(best_model_losses, axis=0)
            final_model_losses_mean = np.mean(final_model_losses, axis=0)
            best_base_a_losses = np.min(base_a_losses, axis=1)
            best_base_a_losses_mean = np.mean(best_base_a_losses, axis=0)
            all_best_model_losses_mean.append(best_model_losses_mean)
            if experiment_name in ['BnA', 'BnA_plus', 'full_weight_init_w_partial_freeze']:
                all_best_base_a_losses_means.append(best_base_a_losses_mean)
            idx_to_name_mapping.append(experiment_name)
            name_to_idx_mapping[experiment_name] = len(idx_to_name_mapping)-1
            if experiment_name in ['BnA', 'BnA_plus', 'full_weight_init_w_partial_freeze', 'random_reference']:
                make_probplot(experiment_dir, experiment_name, best_model_losses, best_base_a_losses, selffer_name)
                make_qualitative_experiment_lossplot(experiment_dir, experiment_name, best_model_losses,
                                                     best_base_a_losses, selffer_name, transfer_name,
                                                     idx_to_name_mapping)
                make_duration_plot(experiment_dir, experiment_name, all_losses_a, all_losses_b, base_a_losses,
                                   selffer_name, transfer_name)
            elif experiment_name in ['quantitative_experiment']:
                make_quantitative_experiment_lossplot(experiment_dir, experiment_name, best_model_losses,
                                                      best_base_a_losses, selffer_name, transfer_name,
                                                      idx_to_name_mapping)
    if len(idx_to_name_mapping) >= 3:
        make_all_experiments_lossplot(experiment_ensemble_dir, all_best_base_a_losses_means, all_best_model_losses_mean,
                                      idx_to_name_mapping, name_to_idx_mapping, selffer_name, transfer_name,
                                      rnd_transfer_name)
    else:
        warnings.warn("Not all experiments found. Cannot make comparison plots...")


def load_experiment_losses(experiment_dir):
    best_model_losses = pickle.load(open(experiment_dir + "best_results_array.p", "rb"))
    final_model_losses = pickle.load(open(experiment_dir + "final_results_array.p", "rb"))
    base_a_losses = []
    num_iterations = 0
    for experiment_dir_entry in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, experiment_dir_entry)):
            base_a_losses_iter = pickle.load(open(os.path.join(experiment_dir, experiment_dir_entry, "0_base_A",
                                                      "test_losses_array.p"), "rb"))
            num_iterations = num_iterations + 1
            base_a_losses.append(base_a_losses_iter)
    base_a_losses = np.array(base_a_losses)
    all_losses_a = []
    all_losses_b = []
    for i in range(num_iterations):
        a_array_iter = []
        b_array_iter = []
        for j in range(1, 2 * experiment_config['levels'] + 1):
            a_array = pickle.load(open(os.path.join(experiment_dir, "Iteration_" + str(i), "A_" + str(j)
                                                    + "_A", "test_losses_array.p"), "rb"))
            b_array = pickle.load(open(os.path.join(experiment_dir, "Iteration_" + str(i), "B_" + str(j)
                                                    + "_A", "test_losses_array.p"), "rb"))
            b_array_iter.append(b_array)
            a_array_iter.append(a_array)
        all_losses_a.append(a_array_iter)
        all_losses_b.append(b_array_iter)
    all_losses_a = np.array(all_losses_a)
    all_losses_b = np.array(all_losses_b)
    return all_losses_a, all_losses_b, best_model_losses, final_model_losses, base_a_losses


def make_duration_plot(experiment_dir, experiment_name, all_losses_a, all_losses_b, base_a_losses, selffer_name,
                       transfer_name):
    best_model_losses_indices_a, best_model_losses_indices_b, best_model_losses_indices_base = \
        get_best_losses_indices(all_losses_a, all_losses_b, base_a_losses)
    best_model_losses_mean_indices_a = best_model_losses_indices_a.mean(axis=0)
    best_model_losses_mean_indices_b = best_model_losses_indices_b.mean(axis=0)
    best_model_losses_mean_indices_base = best_model_losses_indices_base.mean(axis=0)
    conf_interval_selffer = get_conf_intervals(best_model_losses_indices_base, best_model_losses_indices_a, alpha=0.05)
    conf_interval_transfer = get_conf_intervals(best_model_losses_indices_base, best_model_losses_indices_b, alpha=0.05)
    num_layers = all_losses_a.shape[1]

    fig_objects = {}
    fig_object_labels = {}
    fig, ax = plt.subplots()
    fig_objects['base'], = ax.plot([0, num_layers],
                                   [best_model_losses_mean_indices_base, best_model_losses_mean_indices_base],
                                   '--', color='k', markersize=markersize,
             linewidth=2, label='Mean Base Training Time')
    fig_object_labels['base'] = 'Base {}'.format(selffer_name[-1:])
    ax.fill_between(np.arange(num_layers)+1, conf_interval_transfer[0], conf_interval_transfer[1], color='r',
                    alpha=0.25)
    fig_objects['transfer'], = ax.plot(np.arange(num_layers+1), np.concatenate(([best_model_losses_mean_indices_base], best_model_losses_mean_indices_b)), color='r')
    fig_object_labels['transfer'] = 'transfer {}'.format(transfer_name)
    ax.set_ylim((0, 120))
    if selffer_name[-1] == "B":
        location_loc = 'lower left'
    elif selffer_name[-1] == "A":
        location_loc = 'upper left'
    ax.legend([fig_objects[name] for name in fig_objects.keys()],
              [fig_object_labels[name] for name in fig_object_labels.keys()],
              numpoints=1,
              loc=location_loc)
    ax.set_xlabel('Transfer Level')
    ax.set_ylabel('Training Time [Epcohs]')
    plt.tight_layout()
    if experiment_config['plot_svg']:
        fig.savefig(os.path.join(experiment_dir, "duration_plot_" + experiment_name + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        fig.savefig(os.path.join(experiment_dir, "duration_plot_" + experiment_name + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')


def get_best_losses_indices(all_losses_a, all_losses_b, base_a_losses):
    num_iterations = all_losses_a.shape[0]
    num_layers = all_losses_a.shape[1]
    best_model_losses_indices_a = np.zeros((all_losses_a.shape[0], all_losses_a.shape[1]))
    best_model_losses_indices_b = np.zeros((all_losses_b.shape[0], all_losses_b.shape[1]))
    best_model_losses_indices_base = np.zeros(base_a_losses.shape[0])
    for it in range(num_iterations):
        for layer in range(num_layers):
            best_model_losses_indices_a[it, layer] = np.where(all_losses_a[it, layer, :] == np.min(all_losses_a[it, layer, :]))[0]
            best_model_losses_indices_b[it, layer] = np.where(all_losses_b[it, layer, :] == np.min(all_losses_b[it, layer, :]))[0]
        best_model_losses_indices_base[it] = np.where(base_a_losses[it, :] == np.min(base_a_losses[it, :]))[0]
    return best_model_losses_indices_a, best_model_losses_indices_b, best_model_losses_indices_base


def make_probplot(experiment_dir, experiment_name, best_model_losses, best_base_a_losses, selffer_name):
    fig_probplot = plt.figure(2, figsize=[10, 10])
    gs_probplot_01 = gridspec.GridSpec(5, 4)
    ax_probplot_01 = fig_probplot.add_subplot(gs_probplot_01[0, 0])
    stats.probplot(best_base_a_losses, plot=ax_probplot_01)
    ax_probplot_01.set_title('Base Model')
    for layer in range(best_model_losses.shape[1]):
        ax_layer_selffer = fig_probplot.add_subplot(gs_probplot_01[1 + int(layer / 4), layer % 4])
        ax_layer_transfer = fig_probplot.add_subplot(gs_probplot_01[3 + int(layer / 4), layer % 4])
        stats.probplot(best_model_losses[:, layer, 0], plot=ax_layer_selffer)
        stats.probplot(best_model_losses[:, layer, 1], plot=ax_layer_transfer)
        ax_layer_selffer.set_title('Selffer Model (l = {})'.format((layer + 1)))
        ax_layer_transfer.set_title('Transfer Model (l = {})'.format((layer + 1)))
    plt.tight_layout()
    if experiment_config['plot_svg']:
        fig_probplot.savefig(os.path.join(experiment_dir, "probplot_" + experiment_name + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        fig_probplot.savefig(os.path.join(experiment_dir, "probplot_" + experiment_name + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')


def make_qualitative_experiment_lossplot(experiment_dir, experiment_name, best_model_losses, best_base_a_losses,
                                         selffer_name, transfer_name, idx_to_name_mapping):
    best_model_losses_mean = np.mean(best_model_losses, axis=0)
    base_a_mean = np.mean(best_base_a_losses, axis=0)

    fig_objects = {}
    fig_object_labels = {}
    conf_interval_selffer = get_conf_intervals(best_base_a_losses, best_model_losses[:, :, 0], alpha=0.05)
    conf_interval_transfer = get_conf_intervals(best_base_a_losses, best_model_losses[:, :, 1], alpha=0.05)
    horizontal_marker_offset = 0.1
    if experiment_name == 'random_reference':
        selffer_name = u'Rnd\u2192{}'.format(selffer_name[-1])
        transfer_name = u'Rnd\u2192{}'.format(transfer_name[-1])
        base_name = 'Rnd'
        horizontal_marker_offset = 0
    else:
        base_name = selffer_name[-1]
    fig1 = plt.figure(1, figsize=[10, 10])
    gs0 = gridspec.GridSpec(2, 1)
    ax01 = fig1.add_subplot(gs0[0, :])
    fig_objects['base'] = ax01.scatter(0 * best_base_a_losses, best_base_a_losses, marker='o', c='k',
                                       s=markersize)
    fig_object_labels['base'] = 'Base {}'.format(base_name)
    if experiment_name != 'random_reference':
        for layer in range(best_model_losses.shape[1]):
            vals = best_model_losses[:, layer, 0]
            fig_objects['selffer_best_sc'] = ax01.scatter(1 + layer + 0 * vals - horizontal_marker_offset, vals,
                                                          marker='+',
                                                          c='b',
                                                          s=markersize)
            fig_object_labels['selffer_best_sc'] = 'selffer {}'.format(selffer_name)
    for layer in range(best_model_losses.shape[1]):
        vals = best_model_losses[:, layer, 1]
        fig_objects['transfer_best_sc'] = ax01.scatter(1 + layer + 0 * vals + horizontal_marker_offset, vals,
                                                       marker='x', c='r',
                                                       s=markersize)
        fig_object_labels['transfer_best_sc'] = 'transfer {}'.format(transfer_name)
    ax01.legend([fig_objects[name] for name in fig_objects.keys()],
                [fig_object_labels[name] for name in fig_objects.keys()],
                numpoints=1)
    ax01.set_xlabel('Transfer Level')
    ax01.set_ylabel(r'Prediction Loss $\lambda$ (lower is better)')
    ax01.set_ylim(figure_y_limits[idx_to_name_mapping[-1]][0])

    ax02 = fig1.add_subplot(gs0[1, :])
    fig_objects = {}
    fig_object_labels = {}
    fig_objects['base'] = ax02.scatter(0, base_a_mean, marker='o', c='k', s=markersize)
    fig_object_labels['base'] = 'Base {}'.format(base_name)
    ax02.plot([0, len(best_model_losses_mean)], [base_a_mean, base_a_mean], '--', color='k',
              markersize=markersize,
              linewidth=2)
    if experiment_name != 'random_reference':
        ax02.fill_between(range(1, len(best_model_losses_mean) + 1),
                                                                conf_interval_selffer[0], conf_interval_selffer[1],
                                                                color='b', alpha=0.25)
        fig_objects['selffer_best_ln'], = ax02.plot(range(len(best_model_losses_mean) + 1),
                                                    np.concatenate(([base_a_mean], best_model_losses_mean[:, 0])),
                                                    color='b', linewidth=2)
        fig_object_labels['selffer_best_ln'] = 'selffer {}'.format(selffer_name)
    ax02.fill_between(range(1, len(best_model_losses_mean) + 1),
                                                             conf_interval_transfer[0], conf_interval_transfer[1],
                                                             color='r', alpha=0.25)
    fig_objects['transfer_best_ln'], = ax02.plot(range(len(best_model_losses_mean) + 1),
                                                 np.concatenate(([base_a_mean], best_model_losses_mean[:, 1])),
                                                 color='r', linewidth=2)
    fig_object_labels['transfer_best_ln'] = 'transfer {}'.format(transfer_name)
    ax02.legend([fig_objects[name] for name in fig_objects.keys()],
                [fig_object_labels[name] for name in fig_objects.keys()],
                numpoints=1)
    ax02.set_xlabel('Transfer Level')
    ax02.set_ylabel('Prediction Loss $\lambda$ (lower is better)')
    ax02.set_ylim(figure_y_limits[idx_to_name_mapping[-1]][1])
    fig1.tight_layout()
    if experiment_config['plot_svg']:
        plt.savefig(os.path.join(experiment_dir, "best_model_plot__" + experiment_name + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        plt.savefig(os.path.join(experiment_dir, "best_model_plot__" + experiment_name + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')


def make_quantitative_experiment_lossplot(experiment_dir, experiment_name, best_model_losses, best_base_a_losses,
                                          selffer_name, transfer_name, idx_to_name_mapping):
    base_a_mean = np.mean(best_base_a_losses, axis=0)
    best_model_losses_mean = np.mean(best_model_losses, axis=0)
    all_losses_mean = np.concatenate((best_model_losses_mean, np.array([base_a_mean, base_a_mean]).reshape((1,2))), axis=0)
    all_losses_mean_diff = all_losses_mean[:-1, :] - all_losses_mean[1:, :]
    all_losses_mean_diff = all_losses_mean_diff / np.array([best_model_losses_mean[0, 0] - base_a_mean,
                                                            best_model_losses_mean[0, 1] - base_a_mean]) * 100

    fig_objects = {}
    fig1 = plt.figure(1, figsize=[10, 10])
    gs0 = gridspec.GridSpec(3, 1)
    ax01 = fig1.add_subplot(gs0[0, :])
    fig_objects['base'] = ax01.scatter([8] * len(best_base_a_losses), best_base_a_losses, marker='o', c='k',
                                       s=markersize, label='Base {}'.format(selffer_name[-1:]))
    horizontal_marker_offset = 0.1
    for layer in range(best_model_losses.shape[1]):
        loss_vals = best_model_losses[:, layer, 0]
        fig_objects['selffer_best_sc'] = ax01.scatter(layer + 0 * loss_vals - horizontal_marker_offset,
                                                      loss_vals, marker='+', c='b', s=markersize)
    for layer in range(best_model_losses.shape[1]):
        loss_vals = best_model_losses[:, layer, 1]
        fig_objects['transfer_best_sc'] = ax01.scatter(layer + 0 * loss_vals + horizontal_marker_offset,
                                                       loss_vals, marker='x', c='r', s=markersize)
    ax01.legend([fig_objects['base'], fig_objects['selffer_best_sc'], fig_objects['transfer_best_sc']],
                ['Base {}'.format(selffer_name[-1:]),
                 'Reduced Base {}'.format(selffer_name[-1:]),
                 'Reduced Base {}'.format(transfer_name[0])], numpoints=1)
    ax01.set_xlabel('Number of Convolutional Layers')
    ax01.set_ylabel('Prediction Loss $\lambda$\n (lower is better)')
    ax01.set_ylim(figure_y_limits[idx_to_name_mapping[-1]][0])
    ax02 = fig1.add_subplot(gs0[1:, :])
    fig_objects['selffer_best_ln'] = ax02.scatter(range(len(best_model_losses_mean)),
                                                  best_model_losses_mean[:, 0], marker='o', c='b')
    ax02.plot(range(len(best_model_losses_mean)+1), np.concatenate((best_model_losses_mean[:, 0], [base_a_mean])),
              c='b')
    fig_objects['transfer_best_ln'] = ax02.scatter(range(len(best_model_losses_mean)), best_model_losses_mean[:, 1],
                                                   marker='o', c='r')
    ax02.plot(range(len(best_model_losses_mean)), best_model_losses_mean[:, 1], c='r')
    fig_objects['base'] = ax02.scatter(8, base_a_mean, marker='o', c='k', s=markersize)
    ax02.legend([fig_objects['base'], fig_objects['selffer_best_ln'], fig_objects['transfer_best_ln']],
                ['Base {}'.format(selffer_name[-1:]), 'Reduced Base {}'.format(selffer_name[-1:]),
                 'Reduced Base {}'.format(transfer_name[0])], numpoints=1)
    ax02.set_xlabel('Number of Convolutional Layers')
    ax02.set_ylabel('Prediction Loss $\lambda$\n (lower is better)')
    ax02.set_ylim(figure_y_limits[idx_to_name_mapping[-1]][1])
    ax02.set_ylim((0, ax02.get_ylim()[1] * 1.8))

    matplotlib.rcParams.update({'font.size': global_font_size*0.8})

    ax02_ydim = ax02.get_ylim()[1] - ax02.get_ylim()[0]
    ax02_inset_xpos = 1
    ax02_inset_ypos = np.max(best_model_losses_mean[1, :]) + 0.15*ax02_ydim
    ax02_inset_width = 7
    ax02_inset_height = ax02.get_ylim()[1] - 0.08*ax02_ydim - ax02_inset_ypos
    ax02_inset = ax02.inset_axes([ax02_inset_xpos, ax02_inset_ypos, ax02_inset_width, ax02_inset_height],
                                 transform=ax02.transData)
    ax02_inset.set_xlim((1, 8))
    ax02_inset.spines['top'].set_visible(False)
    ax02_inset.spines['right'].set_visible(False)
    ax02_inset.set_ylabel('layer\n knowledge'+ r' $\kappa$ ' + '[%]')
    ax02_inset.fill_between(np.arange(len(all_losses_mean_diff[:, 0])) + 1, all_losses_mean_diff[:, 0], color='b',
                            alpha=0.25, label='Layer Knowledge Model {}'.format(selffer_name[-1:]))
    ax02_inset.fill_between(np.arange(len(all_losses_mean_diff[:-1, 1])) + 1, all_losses_mean_diff[:-1, 1], color='r',
                      alpha=0.25, label='Layer Knowledge Model {}'.format(transfer_name[0]))
    matplotlib.rcParams.update({'font.size': global_font_size})
    fig1.tight_layout()
    if experiment_config['plot_svg']:
        plt.savefig(os.path.join(experiment_dir, "best_model_plot__" + experiment_name + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        plt.savefig(os.path.join(experiment_dir, "best_model_plot__" + experiment_name + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')


def make_all_experiments_lossplot(experiment_ensemble_dir, all_best_base_a_losses_means, all_best_model_losses_mean,
                                  idx_to_name_mapping, name_to_idx_mapping, selffer_name, transfer_name,
                                  rnd_transfer_name):
    all_best_base_a_losses_mean = np.mean(all_best_base_a_losses_means)
    all_best_model_losses_mean = np.array(all_best_model_losses_mean)
    all_best_base_a_losses_mean_array = np.full((all_best_model_losses_mean.shape[0], 1,
                                                 all_best_model_losses_mean.shape[2]), all_best_base_a_losses_mean)
    all_best_model_losses_mean = np.concatenate((all_best_base_a_losses_mean_array, all_best_model_losses_mean), axis=1)

    num_levels = all_best_model_losses_mean.shape[1]
    num_experiments = len(idx_to_name_mapping)
    hsv_color_mappings = np.array(
        [[[60 / 96, 1, 96 / 96], [52 / 96, 1, 60 / 96], [70 / 96, 1, 78 / 96], [70 / 96, 1, 78 / 96]],
         [[0 / 96, 1, 96 / 96], [90 / 96, 1, 80 / 96], [4 / 96, 1, 80 / 96], [3 / 96, 1, 78 / 96]]])
    rgb_color_mappings = [[colors.hsv_to_rgb(hsv_color_mappings[0][c]) for c in range(len(hsv_color_mappings[0]))],
                          [colors.hsv_to_rgb(hsv_color_mappings[1][c]) for c in range(len(hsv_color_mappings[1]))]]
    shadow_factor = 0.15

    fig_objects = {}
    fig2 = plt.figure(5, figsize=[10, 7])
    legend_handles = [[], []]
    fig_objects['all_base'] = plt.scatter(0, all_best_base_a_losses_mean, marker='o', c='k', s=markersize)
    legend_handles[0].append(fig_objects['all_base'])
    legend_handles[1].append('Base {}'.format(selffer_name[-1:]))
    plt.plot([0, num_levels - 1], [all_best_base_a_losses_mean, all_best_base_a_losses_mean], '--', color='k',
             markersize=markersize, linewidth=2, label='Base {}'.format(selffer_name[-1:]))
    for _exp in range(num_experiments):
        if idx_to_name_mapping[_exp] in ['BnA', 'BnA_plus', 'full_weight_init_w_partial_freeze']:
            # if idx_to_name_mapping[_exp] in ['BnA']:
            fig_objects[str(_exp + 1) + '_selffer_best_ln'], = plt.plot(range(num_levels),
                                                                        all_best_model_losses_mean[_exp, :, 0],
                                                                        color=rgb_color_mappings[0][_exp], linewidth=2)
            fig_objects[str(_exp + 1) + '_transfer_best_ln'], = plt.plot(range(num_levels),
                                                                         all_best_model_losses_mean[_exp, :, 1],
                                                                         color=rgb_color_mappings[1][_exp], linewidth=2)
            legend_handles[0].append(fig_objects[str(_exp + 1) + '_selffer_best_ln'])
            legend_handles[1].append(
                '{}: selffer ({})'.format(experiment_name_solver[idx_to_name_mapping[_exp]], selffer_name))
            legend_handles[0].append(fig_objects[str(_exp + 1) + '_transfer_best_ln'])
            legend_handles[1].append(
                '{}: transfer ({})'.format(experiment_name_solver[idx_to_name_mapping[_exp]], transfer_name))
        elif idx_to_name_mapping[_exp] in ['random_reference']:
            fig_objects[str(_exp + 1) + '_transfer_best_ln'], = \
                plt.plot(range(num_levels), all_best_model_losses_mean[_exp, :, 1],
                         color='g', linewidth=2)
            legend_handles[0].append(fig_objects[str(_exp + 1) + '_transfer_best_ln'])
            legend_handles[1].append(
                '{}: transfer ({})'.format(experiment_name_solver[idx_to_name_mapping[_exp]], rnd_transfer_name))
    plt.xlabel('Transfer Level')
    plt.ylabel('Prediction Loss $\lambda$ (lower is better)')
    plt.ylim(figure_y_limits['all'])
    plt.legend(legend_handles[0], legend_handles[1], numpoints=1)
    fig2.tight_layout()
    if experiment_config['plot_svg']:
        plt.savefig(os.path.join(experiment_ensemble_dir, "best_model_plot_line" + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        plt.savefig(os.path.join(experiment_ensemble_dir, "best_model_plot_line" + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')

    fig3 = plt.figure(6, figsize=[10, 5])
    legend_handles = [[], []]
    fig_objects['all_base'] = plt.scatter(0, all_best_base_a_losses_mean, marker='o', c='k', s=markersize)
    legend_handles[0].append(fig_objects['all_base'])
    legend_handles[1].append('Base {}'.format(selffer_name[-1:]))
    plt.plot([0, num_levels - 1], [all_best_base_a_losses_mean, all_best_base_a_losses_mean], '--', color='k',
             markersize=markersize, linewidth=2, label='Base {}'.format(selffer_name[-1:]))
    for _exp in [name_to_idx_mapping[name] for name in ['random_reference', 'BnA', 'BnA_plus']]:
        __transfer_name = transfer_name
        if idx_to_name_mapping[_exp] in ['BnA_plus', 'BnA']:
            fig_objects[str(_exp + 1) + '_transfer_best_ln'], = plt.plot(range(num_levels),
                                                                         all_best_model_losses_mean[_exp, :, 1],
                                                                         color=rgb_color_mappings[1][_exp], linewidth=2)
        elif idx_to_name_mapping[_exp] == 'random_reference':
            fig_objects[str(_exp + 1) + '_transfer_best_ln'], = plt.plot(range(num_levels),
                                                                         all_best_model_losses_mean[_exp, :, 1],
                                                                         color='g', linewidth=2)
        if idx_to_name_mapping[_exp] == 'random_reference':
            __transfer_name = rnd_transfer_name
            plt.fill_between(range(num_levels), all_best_model_losses_mean[_exp, :, 1],
                             all_best_base_a_losses_mean,
                             color=colors.hsv_to_rgb([1 / 3, 0.09, 75 / 96]))
        elif idx_to_name_mapping[_exp] == 'BnA_plus':
            plt.fill_between(range(num_levels), all_best_model_losses_mean[_exp, :, 1], all_best_base_a_losses_mean,
                             color=colors.hsv_to_rgb(np.insert(hsv_color_mappings[1][_exp][[0, 2]], 1, shadow_factor)))
        elif idx_to_name_mapping[_exp] == 'BnA':
            plt.fill_between(range(2, num_levels), all_best_model_losses_mean[_exp, :, 1][2:],
                             all_best_base_a_losses_mean, interpolate=True,
                             color=colors.hsv_to_rgb(np.insert(hsv_color_mappings[1][_exp][[0, 2]], 1, shadow_factor)))
        if idx_to_name_mapping[_exp] not in ['quantitative_experiment', 'full_weight_init_w_partial_freeze']:
            legend_handles[0].append(fig_objects[str(_exp + 1) + '_transfer_best_ln'])
            legend_handles[1].append('{}: transfer ({})'.format(experiment_name_solver[idx_to_name_mapping[_exp]],
                                                                __transfer_name))
    # if selffer_name[-1] == 'A':
    #     plt.text(2, 0.0372, 'A')
    #     plt.text(7.5, 0.055, 'B')
    #     plt.text(4, 0.055, 'C')
    # elif selffer_name[-1] == 'B':
    #     plt.text(5, 0.0355, 'A')
    #     plt.text(7, 0.06, 'B')
    #     plt.text(4.5, 0.072, 'C')
    if selffer_name[-1] == 'A':
        plt.text(4.5, 0.034, 'A')
        plt.text(7, 0.05, 'B')
        plt.text(3.5, 0.045, 'C')
    elif selffer_name[-1] == 'B':
        plt.text(2.91, 0.032, 'A')
        plt.text(7, 0.05, 'B')
        plt.text(3.5, 0.045, 'C')
    plt.xlabel('Transfer Level')
    plt.ylabel('Prediction Loss $\lambda$ (lower is better)')
    plt.ylim(figure_y_limits['all'])
    plt.legend(legend_handles[0], legend_handles[1], numpoints=1, loc='upper left')
    fig3.tight_layout()
    # plt.show(block=True)
    if experiment_config['plot_svg']:
        plt.savefig(os.path.join(experiment_ensemble_dir, "best_model_plot_line_shadowed" + get_fig_name_suffix(selffer_name) + ".svg"), format='svg')
    else:
        plt.savefig(os.path.join(experiment_ensemble_dir, "best_model_plot_line_shadowed" + get_fig_name_suffix(selffer_name) + ".pgf"), transparent=True)
    plt.close('all')


def get_conf_intervals(base_losses, model_losses, alpha):
    n = base_losses.shape[0]
    m = model_losses.shape[0]
    mean_base = np.mean(base_losses)
    mean_model = np.mean(model_losses, axis=0)
    sd_base = np.std(base_losses)
    sd_model = np.std(model_losses, axis=0)
    lower_bounds = []
    upper_bounds = []
    for layer in range(model_losses.shape[1]):
        df = int((sd_base**2 / n + sd_model[layer]**2 / m)**2 /
                 ((sd_base**2 / n)**2 / (n-1) + (sd_model[layer]**2 / m)**2 / (m-1)))
        t_value = stats.t.ppf(alpha / 2, df=df)
        denominator = math.sqrt(sd_base**2 / m + sd_model[layer]**2 / n)
        confidence_margin = t_value * denominator
        lower_bounds.append(mean_model[layer]-confidence_margin)
        upper_bounds.append(mean_model[layer]+confidence_margin)
    return [lower_bounds, upper_bounds]


def make_model_loss_curve(path):
    losses = pickle.load(open(path + "test_losses_array.p", "rb"))
    matplotlib.rcParams.update({'font.size': global_font_size*0.8})
    fig = plt.figure(figsize=[10, 3.75])
    plt.plot(np.arange(1, len(losses)+1), losses, label="Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss (lower is better)')
    plt.legend()
    fig.tight_layout()
    if experiment_config['plot_svg']:
        plt.savefig(os.path.join(path, "losses.svg"), format='svg')
    else:
        plt.savefig(os.path.join(path, "losses.pgf"), transparent=True)
    plt.close('all')
    matplotlib.rcParams.update({'font.size': global_font_size})


def get_fig_name_suffix(selffer_name):
    if selffer_name[-1] == 'A':
        return ""
    elif selffer_name[-1] == 'B':
        return "_reverse"


def load_experiment_config(path):
    global experiment_config
    with open(path, "r") as read_file:
        global experiment_config
        experiment_config = json.load(read_file)


def calc_duration_variance(experiment_dir):
    all_losses_a, all_losses_b, best_model_losses, final_model_losses, base_a_losses = load_experiment_losses(experiment_dir)
    best_model_losses_indices_a, best_model_losses_indices_b, best_model_losses_indices_base = get_best_losses_indices(all_losses_a, all_losses_b, base_a_losses)
    print('Standard deviations for base model in experiment in directory {}'.format(experiment_dir))
    print('Base A Loss Std: {}'.format(np.std(best_model_losses_indices_a)))


# make two one-sided t-tests (TOST) as a test for equivalence of sample means
def calc_tost(a, b, delta, alpha):
    n = len(a)
    m = len(b)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    sd_a = np.std(a)
    sd_b = np.std(b)
    delta_u = np.abs(delta)
    delta_l = -np.abs(delta)
    df = int((sd_a ** 2 / n + sd_b ** 2 / m) ** 2 / ((sd_a ** 2 / n) ** 2 / (n - 1) + (sd_b ** 2 / m) ** 2 / (m - 1)))
    t_value = np.abs(stats.t.ppf(alpha, df=df))
    denominator = math.sqrt(sd_a ** 2 / m + sd_b ** 2 / n)
    t_upper = (mean_a - mean_b - delta_u) / denominator
    t_lower = (mean_a - mean_b - delta_l) / denominator
    if t_lower >= t_value and t_upper <= -t_value:
        return True
    return False


if __name__ == "__main__":
    experiment_ensemble_folder = make_experiment_dir("./Transferability_Investigation/")
    print("Saving Parameters...")
    with open(experiment_ensemble_folder + 'experiment_config.txt', 'w') as file:
        file.write(json.dumps(experiment_config))
    print('Saved as %s' % experiment_ensemble_folder + "experiment_config.txt")
    with open(experiment_ensemble_folder + 'dataset_config.txt', 'w') as file:
        file.write(json.dumps(dataset_config))
    print('Saved as %s' % experiment_ensemble_folder + "dataset_config.txt")
    load_data()

    # experiment BnA like in paper "How transferable are features in deep neural networks"
    bna_path = experiment_ensemble_folder + "BnA/"
    os.makedirs(bna_path)
    do_experiment_series(experiment_series_dir=bna_path, num_repetitions=experiment_config['num_repetitions'],
                         is_quantitative_experiment=False, freeze=True, random_init=True, diff_learning_rate=False,
                         train_base=True)
    # make_figures(experiment_ensemble_folder)

    # experiment BnA+ like in paper "How transferable are features in deep neural networks"
    bna_plus_path = experiment_ensemble_folder + "BnA_plus/"
    os.makedirs(bna_plus_path)
    do_experiment_series(experiment_series_dir=bna_plus_path, num_repetitions=experiment_config['num_repetitions'],
                         is_quantitative_experiment=False, freeze=False, random_init=True, diff_learning_rate=False,
                         train_base=True)
    # make_figures(experiment_ensemble_folder)

    # additional experiment with full weight initialization and partial freeze
    full_weight_init_w_partial_freeze_path = experiment_ensemble_folder + "full_weight_init_w_partial_freeze/"
    os.makedirs(full_weight_init_w_partial_freeze_path)
    do_experiment_series(experiment_series_dir=full_weight_init_w_partial_freeze_path, num_repetitions=experiment_config['num_repetitions'],
                         is_quantitative_experiment=False, freeze=True, random_init=False, diff_learning_rate=False,
                         train_base=True)
    # make_figures(experiment_ensemble_folder)

    # random reference model with random weights, gradually freeze conv layers
    # exclude possibility that all knowledge is simply learned in the unfrozen random layers and transferred
    # layers have no effect
    random_ref_path = experiment_ensemble_folder + "random_reference/"
    os.makedirs(random_ref_path)
    do_experiment_series(experiment_series_dir=random_ref_path, num_repetitions=experiment_config['num_repetitions'],
                         is_quantitative_experiment=False, freeze=True, random_init=False, diff_learning_rate=False,
                         train_base=False)
    # make_figures(experiment_ensemble_folder)

    # quantitative experiment
    # l layers are transferred and directly mapped to the linear layer
    # transferred layers are frozen --> only the linear layer is trained
    quantitative_experiment_path = experiment_ensemble_folder + "quantitative_experiment/"
    os.makedirs(quantitative_experiment_path)
    do_experiment_series(experiment_series_dir=quantitative_experiment_path,
                         num_repetitions=experiment_config['num_repetitions'],
                         is_quantitative_experiment=True, freeze=True, random_init=False, diff_learning_rate=False,
                         train_base=True)
    make_figures(experiment_ensemble_folder)

    # # linear reference model with random weights, all convolutional layers frozen and only the linear layer can
    # # learn
    # # exclude possibility that all knowledge is simply learned in the linear layer
    # linear_ref_path = experiment_ensemble_folder + "linear_reference/"
    # os.makedirs(linear_ref_path)
    # train_reference(linear_ref_path, experiment_config['num_repetitions'], True)
    #
    # # random reference model with random weights, all convolutional layers and linear layer unfrozen
    # # assess possibility if all knowledge can be learned only from the tune set
    # random_ref_path = experiment_ensemble_folder + "random_reference/"
    # os.makedirs(random_ref_path)
    # train_reference(random_ref_path, experiment_config['num_repetitions'], False)
