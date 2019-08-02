import sys
import math
import torch
import torch.optim as opt
import time
from utils import get_batch, save_model
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, cuda=True, ksize=5, dropout=0.25, clip=0.2, epochs=10, levels=4,
                 log_interval=5, lr=1e-3, optim='Adam', nhid=150, validseqlen=320, seqstepwidth=50, seq_len=400,
                 batch_size=32):
        super(TCN, self).__init__()
        print("Initializing TCN with input_size={0}, output_size={1}, cuda={2}, ksize={3}, dropout={4}, "
              "clip={5}, epochs={6}, levels={7}, log_interval={8}, lr={9}, optim={10}, nhid={11}, "
              "validseqlen={12}, seq_len={13}, batch_size={14} ".format(input_size, output_size, cuda, ksize,
                                                                        dropout, clip, epochs, levels, log_interval,
                                                                        lr, optim, nhid, validseqlen, seq_len,
                                                                        batch_size))
        self.config = ModelConfig()
        self.config.input_size = input_size
        self.config.output_size = output_size
        self.config.clip = clip
        self.config.epochs = epochs
        self.config.cuda = cuda
        self.config.ksize = ksize
        self.config.dropout = dropout
        self.config.levels = levels
        self.config.log_interval = log_interval
        self.config.lr = lr
        self.config.optim = optim
        self.config.nhid = nhid
        self.config.validseqlen = validseqlen
        self.config.seqstepwidth = seqstepwidth
        self.config.seq_len = seq_len
        self.config.batch_size = batch_size
        self.config.n_channels = [self.config.nhid] * self.config.levels
        self.tcn = TemporalConvNet(input_size, self.config.n_channels, kernel_size=ksize, dropout=dropout)
        self.linear = nn.Linear(self.config.n_channels[-1], output_size)
        self.init_weights()
        if cuda:
            self.cuda()
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.optimizer = getattr(opt, self.config.optim)(self.parameters(), lr=lr)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2)).transpose(1, 2)

    def fit_and_test(self, x_train, y_train, x_test, y_test, x_valid=None, y_valid=None, lr_adapt=None,
                     experiment_folder=None):
        if lr_adapt == 'valid' and (x_valid is None or y_valid is None):
            raise ValueError('Valid Set should be provided if Valid Set Learn Rate Adaptation is specified!')
        global lr
        try:
            print("Training for %d epochs..." % self.config.epochs)
            all_epoch_train_losses = []
            all_epoch_valid_losses = []
            all_epoch_test_losses = []
            for epoch in range(1, self.config.epochs + 1):
                self.train_epoch(x_train, y_train, epoch)
                epoch_train_losses = self.evaluate(x_train, y_train)
                epoch_train_loss_avg = sum(epoch_train_losses) / len(epoch_train_losses)
                all_epoch_train_losses.append(epoch_train_loss_avg)
                if x_valid is not None and y_valid is not None:
                    epoch_valid_losses = self.evaluate(x_valid, y_valid)
                    epoch_valid_loss_avg = sum(epoch_valid_losses) / len(epoch_valid_losses)
                    all_epoch_valid_losses.append(epoch_valid_loss_avg)
                    self.print_train_status(pre_sep=True, text='| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
                        epoch, epoch_valid_loss_avg, epoch_valid_loss_avg / math.log(2)), post_sep=False)
                else:
                    all_epoch_valid_losses.append(-1)
                epoch_test_losses = self.evaluate(x_test, y_test)
                epoch_test_loss_avg = sum(epoch_test_losses) / len(epoch_test_losses)
                all_epoch_test_losses.append(epoch_test_loss_avg)
                self.print_train_status(pre_sep=False, text='| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}'.format(
                    epoch, epoch_test_loss_avg, epoch_test_loss_avg / math.log(2)), post_sep=True)
                if lr_adapt is not None:
                    self.adapt_learn_rate(epoch, lr_adapt, all_epoch_valid_losses)
                    if self.config.lr < 1e-6:
                        break
                if all_epoch_test_losses[-1] <= min(all_epoch_test_losses) and experiment_folder is not None:
                    save_model(self, experiment_folder)
        except KeyboardInterrupt:
            self.print_train_status(True, '', False)
            if experiment_folder is not None:
                save_model(self, experiment_folder)
        self.print_train_status(True, '| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
            min(all_epoch_test_losses), min(all_epoch_test_losses) / math.log(2)), True)
        return all_epoch_train_losses, all_epoch_valid_losses, all_epoch_test_losses

    def train_epoch(self, X_train, Y_train, epoch):
        self.train()
        total_loss = 0
        source = X_train
        target = Y_train
        source_len = source.size(2)
        start_time = time.time()
        losses = []
        # for batch_idx, i in enumerate(range(0, source_len - 1, self.config.validseqlen)):
        for batch_idx, i in enumerate(range(0, source_len - self.config.seq_len, self.config.seqstepwidth)):
            if i + self.config.seq_len - self.config.validseqlen >= source_len:
                print("Skipping batch due to end of sequence")
                continue
            x, y = get_batch(source, target, i, self.config.seq_len)
            self.optimizer.zero_grad()
            y_pred = self(x)
            final_y_pred = y_pred[:, :, -self.config.validseqlen].contiguous().view(-1)
            final_y = y[:, :, -self.config.validseqlen].contiguous().view(-1)
            loss = self.criterion(final_y_pred, final_y)
            loss.backward()
            if self.config.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)
            self.optimizer.step()
            total_loss += loss.item()
            losses.append(loss.item())
            if (batch_idx+1) % self.config.log_interval == 0:
                cur_loss = total_loss / self.config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.3f} | bpc {:5.3f}'.format(
                    epoch, batch_idx+1, int((source_len-self.config.seq_len) / self.config.seqstepwidth), self.config.lr,
                    elapsed * 1000 / self.config.log_interval, cur_loss, cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()
        return losses

    def evaluate(self, source, target):
        self.eval()
        losses = []
        source_len = source.size(2)
        target_len = target.size(2)
        if source_len != target_len:
            print("ERROR: Source and Target time series should have the same length!")
        # TODO: s.o. train_epoch() bzgl. validseqlen
        for batch, i in enumerate(range(0, source_len, self.config.seqstepwidth)):
            if i + self.config.seq_len - self.config.validseqlen >= source_len:
                continue
            x, y = get_batch(source, target, i, self.config.seq_len)
            y_pred = self(x)
            final_y_pred = y_pred[:, :, -self.config.validseqlen].contiguous().view(-1)
            final_y = y[:, :, -self.config.validseqlen].contiguous().view(-1)
            loss = self.criterion(final_y_pred, final_y)
            losses.append(loss.item())
        return losses

    def adapt_learn_rate(self, epoch, lr_adapt, valid_losses=[]):
        if lr_adapt is None:
            return
        if lr_adapt == 'valid':
            if epoch > 5 and valid_losses[-1] > max(valid_losses[-5:-1]):
                self.config.lr = self.config.lr / 10.
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.lr
        elif lr_adapt[:8] == 'schedule':
            schedule_epochs = lr_adapt.split(':')
            schedule_epochs = [int(schedule_epoch) for schedule_epoch in schedule_epochs[1:]]
            if epoch in schedule_epochs:
                self.config.lr = self.config.lr / 10.
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.lr

    def print_train_status(self, pre_sep, text, post_sep):
        if pre_sep:
            print('-' * 89)
        print(text)
        if post_sep:
            print('-' * 89)

    def set_learning_rate(self, _lr):
        self.config.lr = _lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr


class ModelConfig:
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.dropout = None
        self.clip = None
        self.epochs = None
        self.ksize = None
        self.levels = None
        self.log_interval = None
        self.lr = None
        self.optim = None
        self.nhid = None
        self.validseqlen = None
        self.seq_len = None
        self.batch_size = None
        self.n_channels = None
        self.cuda = None


class NetworkEvaluator(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, output_size, cuda=True, ksize=5, dropout=0.25, clip=0.3, epochs=150, levels=4,
                 log_interval=5, lr=1e-3, optim='Adam', nhid=150, validseqlen=1, seqstepwidth=50, seq_len=100, batch_size=16):
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.clip = clip
        self.epochs = epochs
        self.cuda = cuda
        self.ksize = ksize
        self.levels = levels
        self.log_interval = log_interval
        self.lr = lr
        self.optim = optim
        self.nhid = nhid
        self.validseqlen = validseqlen
        self.seqstepwidth = seqstepwidth
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.model = TCN(input_size, output_size, cuda=cuda, ksize=ksize, dropout=dropout, clip=clip, epochs=epochs,
                         levels=levels, log_interval=log_interval, lr=lr, optim=optim, nhid=nhid, validseqlen=validseqlen,
                         seqstepwidth=seqstepwidth, seq_len=seq_len, batch_size=batch_size)

    def fit(self, x_train, y_train):
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        if self.cuda:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        for epoch in range(1, self.epochs + 1):
            self.model.train_epoch(x_train, y_train, epoch)

    def score(self, x_test, y_test):
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        if self.cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        losses = self.model.evaluate(x_test,y_test)
        average_loss = sum(losses) / len(losses)
        return 1/average_loss