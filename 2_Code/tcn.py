import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TCN Residual Block as described in figure 1(b) of the paper
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # torch.nn.Conv1d(...)
        # input dimensions (N, C_in, L) and output size (N, C_out, L_out)
        # N = batch size, C = # of channels, L = length of the sequence
        # groups default := 1 --> all input channels are convolved to all output channels

        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.pad1 = nn.ReplicationPad1d(padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.pad2 = nn.ReplicationPad1d(padding)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.pad1, self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.pad2, self.conv2, self.chomp2, self.relu2, self.dropout2)

        # optional 1x1 Convolution to have matching channel dimensions
        # TODO changed downsample weights to constant 1 without gradient!!
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
            for param in self.downsample.parameters():
                param.requires_grad = False
            self.downsample.weight.data.fill_(1)
            self.downsample.bias.data.fill_(0)
        else:
            self.downsample = None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # out = the stacked path through the residual block (figure 1(b))
        out = self.net(x)

        # out = the straight path through the residual block (figure 1(b)) with
        # optional 1x1 convolution
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# creates a stack of multiple TemporalBlocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # number of layers = num_levels;
        # num_channels is a list with the number of neurons (i.e. channels) for each layer
        # num_channels[0] = number of input features for each sample
        num_levels = len(num_channels)

        # Create a Residual Block (=TemporalBlock) for each level (=layer)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(n_inputs=in_channels, n_outputs=out_channels, kernel_size=kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)