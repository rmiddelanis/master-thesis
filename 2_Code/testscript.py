import argparse

import numpy as np
import torch
from model import NetworkEvaluator
from model import TCN
from sklearn.model_selection import RandomizedSearchCV
from utils import data_generator
from utils import make_experiment_dir
from utils import plot_and_save
from utils import plot_and_save_prediction
from utils import save_args

parser = argparse.ArgumentParser(description='Sequence Modeling - Simply Cozy')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval (default: 1')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--lr_adapt', type=str, default='valid',
                    help='learn rate adaptation (default: valid)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--series_x', type=str, default='10000000000000000000000000',
                    help='the series of the dataset to use for training (default: 10000000000000000000000000)')
parser.add_argument('--series_y', type=str, default='10000000000000000000000000',
                    help='the series of the dataset to predict (default: 10000000000000000000000000)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (default: -1 = random seed)')
parser.add_argument('--validseqlen', type=int, default=1,
                    help='valid sequence length (# of Samples for Loss computation) (default: 1)')
parser.add_argument('--seqstepwidth', type=int, default=1,
                    help='stepwidth for batch drawing (default: 1)')
parser.add_argument('--seq_len', type=int, default=100,
                    help='total sequence length (default: 200)')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size (default: 12)')
parser.add_argument('--documentation', action='store_false',
                    help='enable documentation of experiments (default: True)')
parser.add_argument('--pca', action='store_true',
                    help='use PCA on exegenous series (default: False)')
parser.add_argument('--dataset_path', type=str, default=r'./data/Synthetic/synthetic_data.p',
                    help='path to the dataset (default: ./data/Synthetic/synthetic_data.p)')

args = parser.parse_args()
print(args)

dataset_config = {
    'test_size': 0.2,
    'sampling_size': 1,
    'data_file_name': args.dataset_path,
    'sample_frequency': 1,
    'max_data': 9999999,
    'target_variables': ['Soll'],
    'seed': None if args.seed == -1 else args.seed,
    'train_valid_split': [0.8, 0.2],
    'series_x': args.series_x,
    'series_y': args.series_y,
    'pca': args.pca,
    'batch_size': args.batch_size,
    'cuda': args.cuda,
}

test_hyperparams = {
    "dropout": [-1] + list(np.linspace(0.05, 0.6, 12)),
    "clip": [-1] + list(np.linspace(0.05, 0.6, 12)),
    "ksize": [2, 4, 6, 8, 10],
    "levels": [2, 4, 6, 8],
    "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "nhid": [50, 150, 200, 250],
    "batch_size": [16, 32, 64, 96],
    "seq_len": [100, 300, 500, 700],
    "validseqlen": [1]
}


def run(optimize=False):
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.cuda.seed_all()
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    data, pca_scaler = data_generator(dataset_config)
    data.x_valid = data.x_valid[0]
    data.y_valid = data.y_valid[0]
    if args.cuda:
        data.x_train = data.x_train.cuda()
        data.x_test = data.x_test.cuda()
        data.x_valid = data.x_valid.cuda()
        data.y_train = data.y_train.cuda()
        data.y_test = data.y_test.cuda()
        data.y_valid = data.y_valid.cuda()
    input_size = data.x_train.size(1)
    output_size = data.y_train.size(1)

    if optimize:
        optimizations_path = "./optimizations/"
        optimization_experiment_folder = make_experiment_dir(optimizations_path)
        save_args(optimization_experiment_folder, args)
        print("Saving best Random Search Field...")
        with open(optimization_experiment_folder + "random_search_field.txt", "w") as file:
            for key, value in test_hyperparams.items():
                line = str(key) + "     " + str(value) + "\n"
                file.write(line)
        file.close()
        print('Saved as %s' % optimization_experiment_folder + "random_search_field.txt")
        rs = RandomizedSearchCV(NetworkEvaluator(input_size, output_size, args.cuda), test_hyperparams, n_jobs=1)
        rs.fit(data.x_train.cpu().numpy(), y=data.y_train.cpu().numpy())
        best_estimator = rs.best_estimator_
        print("Best Params: \n")
        print(rs.best_params_)
        print("Saving best Parameters...")
        with open(optimization_experiment_folder + "best_params.txt", "w") as file:
            for key, value in rs.best_params_.items():
                line = str(key) + "     " + str(value) + "\n"
                file.write(line)
        file.close()
        print('Saved as %s' % optimization_experiment_folder + "best_params.txt")
    else:
        experiment_path = "./experiments/"
        experiment_folder = make_experiment_dir(experiment_path)

        model = TCN(input_size, output_size, cuda=args.cuda, ksize=args.ksize, dropout=args.dropout, clip=args.clip,
                    epochs=args.epochs, levels=args.levels, log_interval=args.log_interval, lr=args.lr,
                    optim=args.optim, nhid=args.nhid, validseqlen=args.validseqlen, seqstepwidth=args.seqstepwidth,
                    seq_len=args.seq_len, batch_size=args.batch_size)
        all_epoch_train_losses, all_epoch_valid_losses, all_epoch_test_losses = model.fit_and_test(x_train=data.x_train,
                                                                                                   y_train=data.y_train,
                                                                                                   x_test=data.x_test,
                                                                                                   y_test=data.y_test,
                                                                                                   x_valid=data.x_valid,
                                                                                                   y_valid=data.y_valid,
                                                                                                   lr_adapt=args.lr_adapt,
                                                                                                   experiment_folder=experiment_folder)
        save_args(experiment_folder, args)
        plot_and_save_prediction(model, data.x_test, data.y_test, args, path=experiment_folder)
        plot_and_save([all_epoch_train_losses, all_epoch_valid_losses, all_epoch_test_losses],
                      ['Train Loss over Epochs', 'Validation Loss over Epochs', 'Test Loss over Epochs'],
                      experiment_folder + 'losses.svg',
                      x_names=['Epoch', 'Epoch', 'Epoch'])


if __name__ == "__main__":
    run(False)
