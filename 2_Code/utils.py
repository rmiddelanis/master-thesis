import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing import DataCleaner
from sklearn.decomposition import PCA


def data_generator(dataset_config):
    datacleaner = DataCleaner(dataset_config)
    data = datacleaner.load()
    data.train_valid_split(dataset_config['train_valid_split'])

    series_x_selection = [int(i) == True for i in list(dataset_config['series_x'])]
    series_y_selection = [int(i) == True for i in list(dataset_config['series_y'])]
    selected_columns_x = [i for i in range(len(series_x_selection)) if series_x_selection[i]]
    selected_columns_y = [i for i in range(len(series_y_selection)) if series_y_selection[i]]
    data.x_train = data.x_train[:, selected_columns_x]
    for i in range(len(data.x_valid)):
        data.x_valid[i] = data.x_valid[i][:, selected_columns_x]
    data.x_test = data.x_test[:, selected_columns_x]
    pca_scaler = None

    if dataset_config['pca']:
        data.x_train, data.x_valid, data.x_test, pca_scaler = pca_x_(data.x_train, data.x_valid, data.data.x_test)

    data.x_train = batchify(torch.from_numpy(data.x_train).type('torch.FloatTensor'), dataset_config['batch_size'])
    for i in range(len(data.x_valid)):
        data.x_valid[i] = batchify(torch.from_numpy(data.x_valid[i]).type('torch.FloatTensor'),
                                   dataset_config['batch_size'])
    data.x_test = batchify(torch.from_numpy(data.x_test).type('torch.FloatTensor'), dataset_config['batch_size'])
    data.y_train = batchify(torch.from_numpy(data.y_train[:, selected_columns_y]).type('torch.FloatTensor'),
                            dataset_config['batch_size'])
    for i in range(len(data.y_valid)):
        data.y_valid[i] = batchify(torch.from_numpy(data.y_valid[i][:, selected_columns_y]).type('torch.FloatTensor'),
                                   dataset_config['batch_size'])
    data.y_test = batchify(torch.from_numpy(data.y_test[:, selected_columns_y]).type('torch.FloatTensor'),
                           dataset_config['batch_size'])

    return data, pca_scaler


def pca_x_(x_train, x_valid, x_test):
    pca_scaler = PCA()
    pca_scaler.fit(np.concatenate((x_train, x_valid, x_test)))
    explained_variance_ratio_cum = 0.0
    num_components = 0
    for explained_variance in pca_scaler.explained_variance_ratio_:
        if explained_variance_ratio_cum < 0.9:
            explained_variance_ratio_cum = explained_variance_ratio_cum + explained_variance
            num_components = num_components + 1
        else:
            break
    pca_scaler = PCA(num_components)
    pca_scaler.fit(np.concatenate([x_train, x_valid, x_test]))
    return pca_scaler.transform(x_train), pca_scaler.transform(x_valid), pca_scaler.transform(x_test), pca_scaler


def batchify(data, batch_size):
    len_batches = data.size(0) // batch_size
    num_signals = data.size(1)
    data = data.narrow(0, 0, len_batches * batch_size)
    data = data.view(batch_size, len_batches, num_signals).transpose(1, 2)
    return data


def debatchify(data):
    num_signals = data.size(1)
    # return data.view(num_signals,-1).detach()
    return data.transpose(0, 1).contiguous().view(num_signals, -1).detach()


def get_batch(source, target, start_index, seq_len):
    seq_len = min(seq_len, source.size(2) - 1 - start_index)
    end_index = start_index + seq_len
    x = source[:, :, start_index:end_index].contiguous()
    # y = target[:, :, start_index+1:end_index+1].contiguous()  # x[n] and y[n+1]
    y = target[:, :, start_index:end_index].contiguous()  # x[n] and y[n]
    return x, y


def save_model(model, path):
    print("Saving Model...")
    torch.save(model, os.path.join(path, 'model.pt'))
    print('Saved as %s' % path + 'model.pt')


def plot_and_save_prediction(model, source, target, args, clip=1e5, path="./figures/"):
    source = debatchify(source)
    target = debatchify(target)
    input_length = int(
        (min(source.size(1) - 1, target.size(1) - 1, clip + args.seq_len) // args.seq_len) * args.seq_len)
    source = source.narrow(1, 0, input_length)
    target = target.narrow(1, 0, input_length)
    num_channels_x = source.size(0)
    num_channels_y = target.size(0)
    # prediction = np.zeros((num_channels,(input_length-args.seq_len)//stepsize))
    prediction = torch.zeros((num_channels_y, (input_length - args.seq_len) + 1))
    truth = target[:, args.seq_len - 1:]
    num_sequences = ((input_length - args.seq_len) // args.seq_len)
    for offset in range(0, args.seq_len):
        x = torch.zeros(num_sequences, num_channels_x, args.seq_len)  # .type('torch.FloatTensor')
        if args.cuda:
            x = x.cuda()
        for channel in range(num_channels_x):
            x[:, channel, :] = source[channel, offset:offset + num_sequences * args.seq_len].contiguous().view(
                num_sequences, args.seq_len)
        for counter, batch_idx in enumerate(range(0, x.size(0), args.batch_size)):
            if batch_idx + args.batch_size <= x.size(0):
                batch = x[batch_idx:batch_idx + args.batch_size, :, :]
            else:
                batch = x[batch_idx:, :, :]
            batch_prediction = model(batch).detach()
            for i in range(batch_prediction.size(0)):
                prediction[:, offset + batch_idx * args.seq_len + i * args.seq_len] = batch_prediction[i, :, -1]
        if offset % 10 == 0:
            print("Calculated offset ", offset, "of ", (args.seq_len - 1))
    source_plot = source[:, args.seq_len - 1:]
    if args.cuda:
        truth = truth.cpu()
        prediction = prediction.cpu()
        source_plot = source_plot.cpu()

    criterion = torch.nn.MSELoss()
    overall_loss = criterion(prediction, truth)
    naive_overall_loss = criterion(torch.zeros(prediction.shape), truth)
    print("Overall Loss [nn.MSELoss()]: ", overall_loss)

    figure = plt.figure(figsize=(25, 8))
    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, num_channels_y, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(1, num_channels_x, subplot_spec=gs0[1])
    plt.gcf().text(0.1, 0.95, "Model Loss: {0} \nNaive Loss: {1}".format(overall_loss, naive_overall_loss), fontsize=14)
    for channel in range(num_channels_y):
        ax = plt.subplot(gs00[:, channel])
        ax.plot(prediction[channel, :].numpy(), label='Model')
        ax.plot(truth[channel, :].numpy(), label='Real')
        ax.legend(loc="upper right")
        ax.set_title("Target Channel {}".format(channel))
        loss = criterion(prediction[channel, :], truth[channel, :])
        naive_loss = criterion(torch.zeros(prediction[channel, :].shape), truth[channel, :])
        ax.text(0, max(torch.max(prediction[channel, :]).item(), torch.max(truth[channel, :]).item()) * 0.9,
                "Model Loss: {0} \nNaive Loss: {1}".format(loss, naive_loss))
    for channel in range(num_channels_x):
        ax = plt.subplot(gs01[:, channel])
        ax.plot(source_plot[channel, :].numpy(), label='Input')
        ax.legend(loc="upper right")
        ax.set_title("Feature Channel {}".format(channel))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    figure.savefig(path + "prediction.svg", format='svg')
    print("Plot(s) saved as %s" % path + "prediction.svg")
    plt.show()
    return prediction, truth


def selfplot(model, source, target, args):
    stepsize = 1
    max_steps = 30000
    source = debatchify(source)
    target = debatchify(target)
    input_length = min(len(source), len(target))
    num_channels = min(source.size(1), target.size(1))
    prediction = torch.zeros((num_channels, input_length))
    truth = torch.zeros((num_channels, input_length))
    prediction[:, 0:args.seq_len] = source[0:args.seq_len, :].transpose(0, 1)
    truth[:, 0:args.seq_len] = target[0:args.seq_len, :].transpose(0, 1)
    for time in range(10, min(input_length - args.seq_len - stepsize, max_steps), stepsize):
        x = prediction[:, time:time + args.seq_len].unsqueeze(0)
        y = target[time + 1:time + args.seq_len + 1, :].unsqueeze(0).transpose(1, 2)
        current_prediction = model(x).detach()
        prediction[:, args.seq_len + int(time / stepsize)] = current_prediction.view(num_channels, args.seq_len)[:, -1:]
        truth[:, int(args.seq_len + time / stepsize)] = y.view(num_channels, args.seq_len)[:, -1:]
        if time % 1000 * stepsize == 0:
            print("Calculating time step ", time)
    return prediction, truth


def plot_and_save(y_ranges, y_names, path, x_ranges=[], x_names=[], texts=[], **plot_kwargs):
    if len(x_ranges) == 0:
        for y_range in y_ranges:
            x_ranges.append([i for i in range(len(y_range))])
    if len(x_names) == 0:
        for index in range(len(x_ranges)):
            x_names.append("Input Feature %i" % index)
    if len(texts) < len(y_ranges):
        for i in range(len(y_ranges) - len(texts)):
            texts.append("")
    figure = plt.figure(figsize=(25, 8))
    for channel in range(len(y_ranges)):
        ax = plt.subplot(2, int(len(y_ranges) / 2 + 0.5), channel + 1)
        ax.plot(x_ranges[channel], y_ranges[channel], **plot_kwargs)
        ax.legend()
        ax.set_title(y_names[channel])
        ax.text(0, 0, texts[channel])
    if mpl.get_backend() != 'pgf':
        figure.savefig(path, format='svg')
    else:
        figure.savefig(path)
    plt.close('all')


def make_experiment_dir(path):
    exp_folder = path + (str(datetime.now()) + "/").replace(":", "-")
    os.makedirs(exp_folder)
    return exp_folder


def save_args(path, args):
    print("Saving Parameters...")
    with open(path + "config.txt", "w") as config_file:
        for arg in vars(args):
            line = str(arg) + "     " + str(getattr(args, arg)) + "\n"
            config_file.write(line)
    config_file.close()
    print('Saved as %s' % path + "config.txt")
