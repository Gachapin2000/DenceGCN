import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM

from torch_geometric.nn import MessagePassing, GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree


class GeneralConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, n_heads=1, iscat=False, dropout=0.):
        super(GeneralConv, self).__init__()

        self.conv_name = conv_name
        self.task = task

        if self.conv_name == 'gcn_conv':
            self.conv = GCNConv(in_channels, out_channels)
            if(self.task == 'inductive'): # if transductive, we dont use linear
                self.lin  = nn.Linear(in_channels, out_channels)

        elif self.conv_name == 'gat_conv':
            if iscat:
                in_channels = n_heads * in_channels
            else:
                in_channels = in_channels
            self.conv = GATConv(in_channels, out_channels, n_heads, iscat, dropout)
            if self.task == 'inductive': # if transductive, we dont use linear
                if iscat:
                    self.lin = nn.Linear(in_channels, out_channels * n_heads)
                else:
                    self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        if self.task == 'inductive':
            return self.conv(x, edge_index) + self.lin(x)
        elif self.task == 'transductive':
            return self.conv(x, edge_index)


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
                channels, channels,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2* channels, 1)
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
            alpha, _ = self.lstm(x) # alpha (n, l, dl), _[0] (n, dl/2) for h and c
            
            if(self.att_mode in ['sd', 'mx']): # SD or MX
                dim = alpha.size()[-1]
                alpha_f, alpha_b = alpha[:,:,:dim//2], alpha[:,:,dim//2:]

                sd = (alpha_f * alpha_b).sum(dim=-1)
                if(self.att_mode == 'sd'): # SD
                    alpha = sd / math.sqrt(dim//2)
                else: # MX
                    alpha = self.att(alpha).squeeze(-1)
                    alpha = alpha * torch.sigmoid(sd)
            
            else: # GO
                alpha = self.att(alpha).squeeze(-1)
            
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1) # (n, l, d) * (n, l, 1) = (n, l, d), -> (n, d)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)


class GCN(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()
        self.task = task
        self.dropout = dropout
        
        n_layers = [n_feat] + list(n_hid) + [n_class]
        self.convs = torch.nn.ModuleList()
        self.lins  = torch.nn.ModuleList()
        for idx in range(len(n_layers)-1):
            self.convs.append(GCNConv(n_layers[idx], n_layers[idx+1]))
            self.lins.append(torch.nn.Linear(n_layers[idx], n_layers[idx+1]))

    def forward(self, x, edge_index):
        for conv, lin in zip(self.convs[0:-1], self.lins[0:-1]):
            if self.task == 'inductive':
                x = conv(x, edge_index) + lin(x)
            elif self.task == 'transductive':
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # iff the last convolutional layer, we don't use relu and dropout
        if(self.task == 'inductive'):
            return self.convs[-1](x, edge_index) + self.lins[-1](x)
        elif self.task == 'transductive':
            return self.convs[-1](x, edge_index)