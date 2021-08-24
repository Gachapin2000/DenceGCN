import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM
from torch.nn.parameter import Parameter

from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, conv
from torch_geometric.utils import add_self_loops, degree


class GeneralConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, self_node,
                 n_heads=[1, 1], iscat=[False, False], dropout=0.):
        super(GeneralConv, self).__init__()
        self.task = task

        if conv_name == 'gcn_conv':
            self.conv = GCNConv(in_channels, out_channels, add_self_loops=self_node)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.lin = nn.Linear(in_channels, out_channels)

        elif conv_name == 'sage_conv':
            self.conv = SAGEConv(in_channels, out_channels, root_weight=self_node)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.lin = nn.Linear(in_channels, out_channels)

        elif conv_name == 'gat_conv':
            if iscat[0]:
                in_channels = in_channels * n_heads[0]
            self.conv = GATConv(in_channels=in_channels,
                                out_channels=out_channels,
                                heads=n_heads[1],
                                concat=iscat[1],
                                dropout=dropout,
                                add_self_loops=self_node)
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
    def __init__(self, mode, att_mode=None, channels=None, num_layers=None, temparature=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        self.summary_mode = 'lstm'
        self.att_mode = att_mode
        self.att_temparature = temparature
        assert self.mode in ['cat', 'max', 'attention']

        if mode == 'attention':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            # self.att = Linear(2 * channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, hs):
        assert isinstance(hs, list) or isinstance(hs, tuple)

        if self.mode == 'cat':
            return torch.cat(hs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(hs, dim=-1).max(dim=-1)[0]
            
        elif self.mode == 'attention':
            h = torch.stack(hs, dim=1)  # h is (n, L, d).

            # summary takes h as input, query and key vector as output
            # query and key shape must be (n, L, hid). hid size depends on summary_mode
            if self.summary_mode == 'vanilla':
                query = h.clone() # query's l-th row is h_i^l
                n_layers = h.size()[1]
                key = query[:, -1, :].repeat(n_layers, 1, 1).permute(1,0,2) # key's all row is h_i^L

            elif self.summary_mode == 'roll':
                query = h.clone()              # query's l-th row is h_i^l
                key = torch.roll(query, -1, 1) # key's l-th row is h_i^(l+1)
                query, key, h = query[:, :-1, :], key[:, :-1, :], h[:, :-1, :]

            elif self.summary_mode == 'lstm':
                alpha, _ = self.lstm(h) # alpha (n, L, dl). dl/2 is hid_channels of forward or backward LSTM
                out_channels = alpha.size()[-1]
                query, key = alpha[:, :, :out_channels//2], alpha[:, :, out_channels//2:]


            if self.att_mode == 'dp':  # att_mode is DP
                alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])

            elif self.att_mode == 'ad':  # att_mode is AD
                query_key = torch.cat([query, key], dim=-1)
                alpha = self.att(query_key).squeeze(-1)
            
            else: # att_mode is MX
                query_key = torch.cat([query, key], dim=-1)
                alpha_ad = self.att(query_key).squeeze(-1)
                alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

            alpha = torch.softmax(alpha/self.att_temparature, dim=-1)
            # (n, L, d) * (n, L, 1) = (n, L, d), -> (n, d)
            return (h * alpha.unsqueeze(-1)).sum(dim=1), alpha



class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    
    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = False):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd