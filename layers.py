import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='add', add_self_loops=True):
        super(GCNConv, self).__init__(aggr=aggr)

        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class JumpingKnowledge(torch.nn.Module):
    def __init__(self, mode, channels=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            # lstm(d,dl/2), att(dl,1) , n_layer=6, hid=32
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
            alpha, _ = self.lstm(x) # alpha (n, l, dl), _ (2, n, dl/2) for h and c
            alpha = self.att(alpha).squeeze(-1) # alpha (n, l, 1) -> (n, l)
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1) # (n, l, d) * (n, l, 1) = (n, l, d), -> (n, d)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
