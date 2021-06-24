import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM
from torch.nn.parameter import Parameter

from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree


class GeneralConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels,
                 n_heads=[1, 1], iscat=[False, False], dropout=0.):
        super(GeneralConv, self).__init__()
        self.task = task

        if conv_name == 'gcn_conv':
            self.conv = GCNConv(in_channels, out_channels)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.lin = nn.Linear(in_channels, out_channels)

        elif conv_name == 'gat_conv':
            if iscat[0]:
                in_channels = in_channels * n_heads[0]
            self.conv = GATConv(in_channels=in_channels,
                                out_channels=out_channels,
                                heads=n_heads[1],
                                concat=iscat[1],
                                dropout=dropout)
            if self.task == 'inductive':  # if transductive, we dont use linear
                if iscat[1]:
                    out_channels = out_channels * n_heads[1]
                self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        if self.task == 'transductive':
            return self.conv(x, edge_index)
        elif self.task == 'inductive':
            return self.conv(x, edge_index) + self.lin(x)


class JumpingKnowledge(torch.nn.Module):
    def __init__(self, mode, att_mode=None, channels=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        self.att_mode = att_mode
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            self.weight = Parameter(torch.FloatTensor((num_layers * channels) // 2))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'lstm':
            x = torch.stack(xs, dim=1)  # x (n, l, d)
            # alpha (n, l, dl), _[0] (n, dl/2) for h and c
            alpha, _ = self.lstm(x)

            if(self.att_mode in ['sd', 'mx', 'sd+']):  # SD or MX
                dim = alpha.size()[-1]
                alpha_f, alpha_b = alpha[:, :, :dim//2], alpha[:, :, dim//2:]

                sd = alpha_f * alpha_b
                if(self.att_mode == 'sd'):  # SD
                    alpha = sd.sum(dim=-1) / math.sqrt(dim//2)
                elif(self.att_mode == 'sd+'):
                    sd = (sd * self.weight).sum(dim=-1)
                    alpha = sd / math.sqrt(dim//2)
                else:  # MX
                    alpha = self.att(alpha).squeeze(-1)
                    alpha = alpha * torch.sigmoid(sd.sum(dim=-1))

            else:  # GO
                alpha = self.att(alpha).squeeze(-1)

            alpha = torch.softmax(alpha, dim=-1)
            # (n, l, d) * (n, l, 1) = (n, l, d), -> (n, d)
            return (x * alpha.unsqueeze(-1)).sum(dim=1), alpha


